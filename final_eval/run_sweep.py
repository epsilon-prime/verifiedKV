"""Harness check: one model load per job, resumable, over a (dataset, task, arm) list.

    SWEEP=sweeps/tierA_llama.yaml sbatch -p a40 --qos=a40 job_check.sbatch

Structure copied from phase2/experiment6/run_sweep.py so the two behave identically: a run holding
a metrics.json is skipped, which makes a requeued job resume, and the model is loaded once and
monkeypatched onto each runner so a long job amortises the load.

Yaml keys: group, model, cells (list of {dataset, task}), arms, ratio, n_samples / fraction /
n_contexts, max_context_length.
"""

import argparse
import json
import os
import re
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
HOME = os.path.expanduser("~")
sys.path.insert(0, os.path.join(HOME, "src", "kvpress-e5", "evaluation"))
sys.path.insert(0, HERE)

KEPT_RE = re.compile(r"kept fraction ([0-9.]+)")

ap = argparse.ArgumentParser()
ap.add_argument("--sweep", required=True)
args = ap.parse_args()

SWEEP_PATH = os.path.join(HERE, args.sweep)  # relative to this folder, or absolute
SWEEP = yaml.safe_load(open(SWEEP_PATH))
MODEL = SWEEP["model"]
# <experiment>/sweeps/x.yaml writes to <experiment>/results/<group>, so other experiments reuse this runner
OUTDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(SWEEP_PATH))), "results", SWEEP["group"])
os.makedirs(OUTDIR, exist_ok=True)
T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:8.1f}s] {m}", flush=True)


import evaluate  # noqa: E402
import evaluate_registry  # noqa: E402

# Offline loading. `load_dataset(repo, data_dir=...)` cannot resolve from a local cache, only a
# config can, so LongBench and RULER-4096 fail under HF_DATASETS_OFFLINE with
# "Couldn't find cache for <repo> for config 'default-data_dir=...'". Phase 2 solved this with a
# dispatcher that reads the hub snapshot's parquet and jsonl directly; import it rather than
# reimplement it, so the rows are byte-identical to what Phase 2 scored.
#
# ruler32k / ruler64k / ruler128k do NOT go through this: evaluate dispatches them to
# load_ruler_length_dataset, which uses the config form and already resolves offline.
sys.path.insert(0, os.path.join(HOME, "phase2", "experiment4"))
import data as phase2_data  # noqa: E402

_real_load_dataset = evaluate.load_dataset


def _scbench_local(config: str):
    """Read one SCBench task straight from the hub snapshot's jsonl.

    ``load_dataset`` is not reliable for this. It succeeds when its prepared-arrow-cache lookup
    hits, and otherwise falls back to ``snapshot_download``, which refuses the snapshot: it was
    fetched with ``allow_patterns``, so ``local_files_only=True`` raises IncompleteSnapshotError
    over 19 missing files that are all images and ``.gitattributes``. Phase 2 hit the same thing as
    AUDIT #4 in experiment2, which is why its loader exists at all.

    Measured, not assumed: it FAILED on cn17-dgx and SUCCEEDED on cn11-dgx and on the login node,
    same env, same shared cache. Reading the jsonl directly has no snapshot validation and no
    arrow-cache dependency, so it does not vary by node.

    Returns the RAW rows (``context``, ``multi_turns``, ``id``), deliberately NOT flattened, so
    that evaluate's ``n_samples`` selects whole contexts and its own ``flatten_multi_turns`` then
    expands the turns. Phase 2's loader pre-flattens, which silently breaks both.
    """
    from datasets import Dataset

    root = Path(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")))
    snaps = root / "hub" / "datasets--microsoft--SCBench" / "snapshots"
    assert snaps.exists(), f"SCBench not downloaded: {snaps}"
    data = sorted(snaps.iterdir(), key=lambda d: d.stat().st_mtime)[-1] / "data"
    jsonl = data / f"{config}.jsonl"
    assert jsonl.exists(), f"no such SCBench task file: {jsonl}"
    with jsonl.open() as handle:
        return Dataset.from_list([json.loads(line) for line in handle])


def _load(path, *a, **kw):
    """Route each benchmark to the loader that is actually correct for it.

    SCBench goes to the REAL ``load_dataset``. It publishes its tasks as configs, which resolve
    from the local cache offline (verified directly), so it never needed the patch. Sending it to
    Phase 2's loader is wrong twice over: that loader takes BARE task names while
    ``EvaluationConfig`` normalises ``summary`` to ``scbench_summary``, and it returns rows already
    flattened to one per turn, so ``n_samples`` would select 5 random *turns* rather than 5
    *contexts* and ``flatten_multi_turns`` would then run a second time on a frame with no
    ``multi_turns`` column. It also silently applies experiment3's own length filter and
    ``E3_N_CTX``.

    Everything else keeps the patch: ``load_dataset(repo, data_dir=...)`` cannot resolve from a
    local cache, only a config can, so LongBench and RULER-4096 fail under ``HF_DATASETS_OFFLINE``
    with "Couldn't find cache for <repo> for config 'default-data_dir=...'".

    ruler32k / ruler64k / ruler128k reach neither: evaluate dispatches them to
    load_ruler_length_dataset, which uses its own import and already resolves offline.
    """
    if "SCBench" in str(path):
        config = a[0] if a else (kw.get("name") or kw.get("data_dir"))
        return _scbench_local(str(config))
    return phase2_data.load(path, *a, **kw)


evaluate.load_dataset = _load

# LOFT: kvpress's loader returns the dev (10) AND test (100) questions of a task's one shared context,
# so n_samples / fraction would mix them. Score one split, named in the sweep: test by default, dev only
# for tuning (final_eval/SETUP.md P9). Patched on evaluate's module globals, where run_evaluation looks
# the loader up at call time.
LOFT_SPLIT = SWEEP.get("loft_split", "test")
_real_load_loft = evaluate.load_loft_dataset


def _load_loft(data_dir):
    df = _real_load_loft(data_dir)
    df = df[df["split"] == LOFT_SPLIT].reset_index(drop=True)
    assert len(df), f"LOFT {data_dir}: no rows in split {LOFT_SPLIT!r}"
    return df


evaluate.load_loft_dataset = _load_loft

# RULER dev rows: test sets are df.sample(n, random_state=42), a prefix of one permutation, so rows from
# beyond that prefix ("150:152" at 32k, 200 rows a task; "400:401" at 128k, 500 rows) never overlap any
# test sample. Set ruler_rows for readiness checks and tuning only, never with n_samples.
RULER_ROWS = SWEEP.get("ruler_rows")
if RULER_ROWS:
    assert not SWEEP.get("n_samples") and not SWEEP.get("fraction"), "ruler_rows replaces n_samples / fraction"
    _real_load_ruler = evaluate.load_ruler_length_dataset
    _a, _b = (int(x) for x in str(RULER_ROWS).split(":"))

    def _load_ruler_dev(dataset, task):
        df = _real_load_ruler(dataset, task)
        assert len(df) >= _b, f"{dataset}/{task}: {len(df)} rows, ruler_rows {RULER_ROWS}"
        return df.sample(frac=1, random_state=42).iloc[_a:_b]

    evaluate.load_ruler_length_dataset = _load_ruler_dev

import logging  # noqa: E402


class _Capture(logging.Handler):
    """Collect the press's own debug lines so H3 and H4 do not have to be inferred from outputs.

    BernoulliPress logs "c=..., kept fraction ..." at debug level and nowhere else. Setting the
    logger's level is not enough, because the handler evaluate attaches filters at INFO, so this
    adds a handler of its own. Capturing the press's own numbers is the only way to tell
    "compression ran and changed nothing" from "compression never ran", which is the Phase 2 bug
    that made nine runs score exactly no_press.
    """

    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.records = []

    def emit(self, record):
        self.records.append(record.getMessage())


_press_logger = logging.getLogger("kvpress.presses.bernoulli_press")  # also the BernoulliPress inside NeedleFetchPress
_press_logger.setLevel(logging.DEBUG)
_press_logger.propagate = False
_capture = _Capture()
_press_logger.addHandler(_capture)

REG = evaluate_registry.PRESS_REGISTRY
assert evaluate.PRESS_REGISTRY is REG, "two registries: the arms would not be the ones we set"

RATIO = float(SWEEP.get("ratio", 0.875))
ARMS = SWEEP.get("arms", ["no_press", "bernoulli_kvzip"])
todo = [(c["dataset"], c["task"], arm) for c in SWEEP["cells"] for arm in ARMS]

log(f"harness check {Path(args.sweep).stem}: {MODEL}, {len(todo)} runs, "
    f"GPU {torch.cuda.get_device_name(0)}, outdir {OUTDIR}")


def run_dirs(cfg):
    """kvpress's get_results_dir CREATES a numbered subdir when the base exists, so it cannot be
    used to test whether a run finished. Rebuild the base name without side effects."""
    comp = [cfg.dataset, str(cfg.data_dir), cfg.model.replace("/", "--"), cfg.press_name,
            f"{cfg.compression_ratio:.2f}"]
    if cfg.n_samples is not None:
        comp.append(f"nsamples{cfg.n_samples}")
    elif cfg.fraction < 1.0:
        comp.append(f"fraction{cfg.fraction:.3f}")
    if cfg.max_context_length is not None:
        comp.append(f"max_context{cfg.max_context_length}")
    base = Path(OUTDIR) / "__".join(comp)
    if not base.exists():
        return []
    return [base] + sorted((d for d in base.glob("[0-9]*") if d.is_dir()), key=lambda d: int(d.name))


BASE = asdict(evaluate.EvaluationConfig())
pipe = None
failures = []
for i, (dataset, task, arm) in enumerate(todo):
    over = {
        "dataset": dataset, "data_dir": task, "model": MODEL,
        "press_name": arm, "compression_ratio": 0.0 if arm == "no_press" else RATIO,
        "seed": 42, "output_dir": OUTDIR,
        "max_new_tokens": SWEEP.get("max_new_tokens"),
        "max_context_length": SWEEP.get("max_context_length"),
        # An additive-bias press must run under sdpa: flash attention would silently ignore it, and
        # under eager transformers bypasses ALL_ATTENTION_FUNCTIONS so the patch never runs.
        "model_kwargs": {"attn_implementation": "sdpa"},
    }
    if SWEEP.get("n_samples"):
        over["n_samples"] = int(SWEEP["n_samples"])
    if SWEEP.get("fraction"):
        over["fraction"] = float(SWEEP["fraction"])
    cfg = evaluate.EvaluationConfig(**{**BASE, **over})

    evaluate.logger.handlers.clear()  # each runner adds one; they accumulate
    runner = evaluate.EvaluationRunner(cfg)
    if any((d / "metrics.json").exists() for d in run_dirs(cfg)):
        log(f"{i + 1}/{len(todo)} {dataset}/{task} {arm}: done, skipping")
        continue
    if pipe is None:
        runner._setup_press()
        runner._setup_model_pipeline()
        pipe = runner.pipeline
        assert pipe.model.config._attn_implementation == "sdpa", pipe.model.config._attn_implementation
        cfgm = getattr(pipe.model.config, "text_config", pipe.model.config)  # VLM-style configs nest it
        log(f"model loaded: {MODEL} ({cfgm.num_hidden_layers} layers, "
            f"{cfgm.num_attention_heads}q/{cfgm.num_key_value_heads}kv heads)")
    runner._setup_model_pipeline = lambda r=runner: setattr(r, "pipeline", pipe)

    log(f"{i + 1}/{len(todo)} {dataset}/{task} {arm}")
    _capture.records.clear()
    if hasattr(REG.get(arm), "reset_stats"):
        REG[arm].reset_stats()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    try:
        runner.run_evaluation()
    except Exception as exc:  # a failing cell must not take the rest of the sweep with it
        failures.append({"dataset": dataset, "task": task, "arm": arm, "error": f"{type(exc).__name__}: {exc}"})
        log(f"   FAILED: {type(exc).__name__}: {exc}")
        continue
    dt, peak = time.time() - t0, torch.cuda.max_memory_allocated() / 1e9

    # H6: where do contexts actually land under THIS model's tokenizer? RULER 64k and 128k are
    # tokenised with Llama-3.2, so their nominal lengths need not hold for Qwen3 or Mistral.
    ctx = {}
    try:
        df = getattr(runner, "df", None)
        if df is not None and "context" in df:
            tok = pipe.tokenizer
            lens = sorted(len(tok(c, add_special_tokens=False).input_ids) for c in df["context"].head(8))
            ctx = {"ctx_len_min": lens[0], "ctx_len_median": lens[len(lens) // 2], "ctx_len_max": lens[-1],
                   "ctx_len_n": len(lens)}
    except Exception as exc:
        ctx = {"ctx_len_error": f"{type(exc).__name__}: {exc}"}

    # H3 and H4 straight from the press, not inferred from the outputs.
    kept = []
    for record in _capture.records:
        found = KEPT_RE.search(record)
        if found:
            kept.append(float(found.group(1)))
    press_stats = {"press_log_lines": len(_capture.records),
                   "kept_frac_min": min(kept) if kept else None,
                   "kept_frac_max": max(kept) if kept else None}
    if hasattr(REG.get(arm), "fetch_stats"):  # NeedleFetchPress: how often and how much it fetched
        press_stats["fetch_stats"] = dict(REG[arm].fetch_stats)
        press_stats["theta"] = REG[arm].theta
        press_stats["sketch_frac"] = REG[arm].sketch_frac

    dirs = run_dirs(cfg)
    dest = dirs[-1] if dirs else None
    if dest is not None:
        json.dump({"seconds": dt, "peak_gb": peak, "gpu": torch.cuda.get_device_name(0),
                   "press": arm, "ratio": cfg.compression_ratio, "dataset": dataset,
                   "task": task, "model": MODEL, **ctx, **press_stats},
                  open(dest / "cost.json", "w"), indent=2)
    log(f"   press log lines {press_stats['press_log_lines']}, kept_frac {press_stats['kept_frac_min']}")
    log(f"   done in {dt:.0f}s, peak {peak:.1f} GB -> {dest}")

if failures:
    json.dump(failures, open(os.path.join(OUTDIR, "failures.json"), "w"), indent=2)
    log(f"sweep complete with {len(failures)} FAILED cells; see failures.json")
else:
    log("sweep complete, no failures")
