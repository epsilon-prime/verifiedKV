"""Quality harness: one model load per job, resumable, over the (dataset, task, arm) cells of one sweep file.

    python -u run_grid.py --sweep sweeps/<name>.yaml

- RULER test rows are df.sample(n_samples, random_state=42), that is rows 0 to n-1 of the seed-42 permutation.
- LOFT uses one shared context per task and the split named by loft_split.
- A cell whose folder holds metrics.json is skipped, so a resubmitted job resumes.
- Each cell writes results/<group>/<dataset>__<task>__<model>__<arm>__<ratio>__.../ with predictions.csv,
  metrics.json, config.yaml and cost.json.

Sweep keys: group, model, ratio, arms, cells (list of {dataset, task}), n_samples (RULER), loft_split,
max_context_length, ruler_rows "a:b" (development rows, smoke tests only), require_clean (refuse a modified checkout).
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "evaluation"))
sys.path.insert(0, str(HERE))

ap = argparse.ArgumentParser()
ap.add_argument("--sweep", required=True)
ap.add_argument("--results", default=os.environ.get("VKV_GRID_RESULTS", str(HERE / "results")),
                help="results root (default experiments/quality/results, or $VKV_GRID_RESULTS)")
args = ap.parse_args()

import yaml  # noqa: E402

SWEEP_PATH = Path(args.sweep).resolve()
SWEEP = yaml.safe_load(open(SWEEP_PATH))
MODEL = SWEEP["model"]
OUTDIR = os.path.join(args.results, SWEEP["group"])
T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:8.1f}s] {m}", flush=True)


def git(*a):
    try:
        return subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as exc:  # no git on the node: record that instead of failing
        return f"unavailable ({type(exc).__name__})"


# ---- environment checks, before anything heavy ------------------------------------------------------------------
COMMIT = git("rev-parse", "HEAD")
DIRTY = [ln for ln in git("status", "--porcelain", "--untracked-files=no").splitlines() if ln.strip()]
if SWEEP.get("require_clean") and DIRTY:
    sys.exit(f"require_clean: the checkout has local changes, commit or stash them first:\n" + "\n".join(DIRTY))
cache = os.environ.get("KVPRESS_SCORE_CACHE")
assert cache and os.path.isdir(cache), f"set KVPRESS_SCORE_CACHE to an existing folder (got {cache!r}); see README"
ARMS = SWEEP["arms"]
if any("fastkvzip" in a for a in ARMS):
    gates = os.environ.get("KVPRESS_FASTKVZIP_GATES")
    assert gates and os.path.isdir(gates), f"set KVPRESS_FASTKVZIP_GATES (got {gates!r}); see README"
for c in SWEEP["cells"]:
    assert c["dataset"] in ("ruler32k", "ruler128k", "loft"), f"dataset {c['dataset']} is not in the final grid"
os.makedirs(OUTDIR, exist_ok=True)

import torch  # noqa: E402
import transformers  # noqa: E402

import evaluate  # noqa: E402
import evaluate_registry  # noqa: E402
import arms  # noqa: E402

import kvpress  # noqa: E402

assert Path(kvpress.__file__).resolve().is_relative_to(REPO), f"kvpress imported from {kvpress.__file__}, not {REPO}"
PRESSES = arms.register()
REG = evaluate_registry.PRESS_REGISTRY
assert evaluate.PRESS_REGISTRY is REG, "two registries: the arms would not be the ones we set"
for a in ARMS:
    assert a in REG, f"unknown arm {a}"

if any("fastkvzip" in a for a in ARMS):
    from kvpress.presses.fastkvzip_press import get_gate_id
    gate = os.path.join(os.environ["KVPRESS_FASTKVZIP_GATES"], get_gate_id(MODEL))
    assert os.path.isfile(gate), f"no Fast KVzip gate for {MODEL}: {gate} (see README: gates)"

# LOFT: keep one split; patched where run_evaluation looks it up
LOFT_SPLIT = SWEEP.get("loft_split", "test")
_real_load_loft = evaluate.load_loft_dataset


def _load_loft(data_dir):
    df = _real_load_loft(data_dir)
    df = df[df["split"] == LOFT_SPLIT].reset_index(drop=True)
    assert len(df), f"LOFT {data_dir}: no rows in split {LOFT_SPLIT!r}"
    return df


evaluate.load_loft_dataset = _load_loft

# RULER dev rows for smoke tests: rows a:b of the seed-42 permutation. The test sets are its prefix (n_samples
# rows from 0), so dev rows from 150 (32k, 200 rows a task) or 400 (128k, 500 rows) never overlap them.
RULER_ROWS = SWEEP.get("ruler_rows")
if RULER_ROWS:
    assert not SWEEP.get("n_samples"), "ruler_rows replaces n_samples"
    _real_load_ruler = evaluate.load_ruler_length_dataset
    _a, _b = (int(x) for x in str(RULER_ROWS).split(":"))

    def _load_ruler_dev(dataset, task):
        df = _real_load_ruler(dataset, task)
        assert len(df) >= _b, f"{dataset}/{task}: {len(df)} rows, ruler_rows {RULER_ROWS}"
        return df.sample(frac=1, random_state=42).iloc[_a:_b]

    evaluate.load_ruler_length_dataset = _load_ruler_dev

RATIO = float(SWEEP["ratio"])
todo = [(c["dataset"], c["task"], arm) for c in SWEEP["cells"] for arm in ARMS]
ENV = {"commit": COMMIT, "dirty": DIRTY, "host": platform.node(), "kvpress": str(Path(kvpress.__file__).parent),
       "torch": torch.__version__,
       "transformers": transformers.__version__, "cuda": torch.version.cuda, "arm_settings": arms.settings(),
       "score_cache": cache, "gates": os.environ.get("KVPRESS_FASTKVZIP_GATES"),
       "sweep": str(SWEEP_PATH.relative_to(HERE)) if SWEEP_PATH.is_relative_to(HERE) else str(SWEEP_PATH)}
log(f"{SWEEP_PATH.name}: {MODEL}, ratio {RATIO}, {len(todo)} cells, GPU {torch.cuda.get_device_name(0)}, "
    f"commit {COMMIT[:10]}{' (MODIFIED)' if DIRTY else ''}, outdir {OUTDIR}")


def run_dirs(cfg):
    """kvpress's get_results_dir CREATES a numbered subdir when the base exists, so it cannot be used to test
    whether a run finished. Rebuild the base name without side effects ."""
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
        # an additive-bias press must run under sdpa (flash attention would ignore the bias)
        "model_kwargs": {"attn_implementation": "sdpa"},
    }
    if SWEEP.get("n_samples") and dataset != "loft":
        over["n_samples"] = int(SWEEP["n_samples"])
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
        cfgm = getattr(pipe.model.config, "text_config", pipe.model.config)
        log(f"model loaded: {MODEL} ({cfgm.num_hidden_layers} layers, "
            f"{cfgm.num_attention_heads}q/{cfgm.num_key_value_heads}kv heads)")
    runner._setup_model_pipeline = lambda r=runner: setattr(r, "pipeline", pipe)

    log(f"{i + 1}/{len(todo)} {dataset}/{task} {arm}")
    press = REG.get(arm)
    if hasattr(press, "kept_log"):
        press.kept_log.clear()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    try:
        runner.run_evaluation()
    except Exception as exc:  # a failing cell must not take the rest of the sweep with it
        failures.append({"dataset": dataset, "task": task, "arm": arm, "error": f"{type(exc).__name__}: {exc}"})
        log(f"   FAILED: {type(exc).__name__}: {exc}")
        continue
    dt, peak = time.time() - t0, torch.cuda.max_memory_allocated() / 1e9

    ctx = {}
    try:  # context lengths under THIS model's tokenizer (RULER is Llama-tokenised)
        df = getattr(runner, "df", None)
        if df is not None and "context" in df:
            lens = sorted(len(pipe.tokenizer(c, add_special_tokens=False).input_ids)
                          for c in df["context"].drop_duplicates().head(8))
            ctx = {"ctx_len_min": lens[0], "ctx_len_median": lens[len(lens) // 2], "ctx_len_max": lens[-1],
                   "ctx_len_n": len(lens), "n_answers": int(len(df)), "n_contexts": int(df["context"].nunique())}
    except Exception as exc:
        ctx = {"ctx_len_error": f"{type(exc).__name__}: {exc}"}

    stats = {}
    if isinstance(press, arms.VkvFinalGridPress):  # the draw actually made: kept fraction per context
        kl = list(press.kept_log)
        stats = {"draw_frac": press.draw_frac, "sketch_frac": press.sketch_frac,
                 "kept_frac_min": min(kl) if kl else None, "kept_frac_max": max(kl) if kl else None,
                 "kept_frac_mean": sum(kl) / len(kl) if kl else None, "n_draws": len(kl)}

    dirs = run_dirs(cfg)
    dest = dirs[-1] if dirs else None
    if dest is not None:
        json.dump({"seconds": dt, "peak_gb": peak, "gpu": torch.cuda.get_device_name(0),
                   "press": arm, "ratio": cfg.compression_ratio, "dataset": dataset, "task": task, "model": MODEL,
                   **ctx, **stats, **ENV}, open(dest / "cost.json", "w"), indent=2)
    log(f"   done in {dt:.0f}s, peak {peak:.1f} GB, draw {stats.get('kept_frac_mean')} -> {dest}")

if failures:
    json.dump(failures, open(os.path.join(OUTDIR, f"failures_{SWEEP_PATH.stem}.json"), "w"), indent=2)
    log(f"sweep complete with {len(failures)} FAILED cells; see failures_{SWEEP_PATH.stem}.json")
    sys.exit(1)
log("sweep complete, no failures")
