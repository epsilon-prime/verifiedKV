"""Vkv1 downstream: RULER accuracy in real generation, with Vkv1's online statistics (fetches, transfers, realised
denominator error), next to reference arms. One model load; every arm runs on every row; resumable.

Arms:
    full                                no compression
    bern:<kept>                         BernoulliPress on CachedKVzipPress, kept fraction <kept>
    needle_final_c:<ratio>[:<tq>:<td>]  the registry's needle fetch at compression ratio <ratio> (sketch charged)
    vkv1:<kept>:<eps0>:<delta>[:flags]  Vkv1Press (phase4/vkv1/press.py), residency <kept>; flags:
                                        tight (Part III check; default simple), grad (eps-gradient fetch order;
                                        default needle order), ht (no truncation), sketch (needle's 2-bit sketch;
                                        default perfect), nopersist, sink=<N> / win=<N> (first / last N
                                        context entries at r = 1; default 64 each; sink=4:win=0 = BernoulliPress)
Per arm: out/<run>/<arm>/rows.jsonl (prediction, gold, task, row, statistics). Metrics: summarize.py.

    python -u run_downstream.py --rows 152:153 --tasks qa_1 --arms full vkv1:0.0625:0.3:0.05 --run smoke
"""

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import torch

HOME = os.path.expanduser("~")
sys.path.insert(0, os.path.join(HOME, "src", "kvpress-e5", "evaluation"))
sys.path.insert(0, os.path.join(HOME, "phase4"))

import evaluate  # noqa: E402  (also patches attention through kvpress)
from kvpress.presses.bernoulli_press import BernoulliPress  # noqa: E402
from kvpress.presses.cached_kvzip_press import CachedKVzipPress  # noqa: E402
from kvpress.presses.needle_fetch_press import NeedleFetchPress  # noqa: E402

from vkv1.press import Vkv1Press  # noqa: E402  (routes kvpress's needle hook for Vkv1 states)

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
ap.add_argument("--dataset", default="ruler32k")
ap.add_argument("--tasks", nargs="+", required=True)
ap.add_argument("--rows", required=True, help="rows a:b of the seed-42 permutation (dev rows: a >= 152)")
ap.add_argument("--arms", nargs="+", required=True)
ap.add_argument("--run", required=True)
ap.add_argument("--max-new-tokens", type=int, default=0, help="cap on generated tokens (0 = the task's own)")
ap.add_argument("--max-context-length", type=int, default=200000)
args = ap.parse_args()
OUT = Path("out") / args.run
T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:8.1f}s] {m}", flush=True)


def make_press(arm: str):
    kind, *rest = arm.split(":")
    if kind == "full":
        return None
    if kind == "bern":
        return BernoulliPress(press=CachedKVzipPress(compression_ratio=1 - float(rest[0])))
    if kind == "needle_final_c":
        p = copy.deepcopy(evaluate.PRESS_REGISTRY["needle_final_c"])
        p.compression_ratio = float(rest[0])
        if len(rest) > 1:
            p.theta_question, p.theta = float(rest[1]), float(rest[2])
        return p
    if kind == "vkv1":
        kept, eps0, delta = float(rest[0]), float(rest[1]), float(rest[2])
        flags = set(rest[3:])
        win = next((int(f.split("=")[1]) for f in flags if f.startswith("win=")), 64)
        sink = next((int(f.split("=")[1]) for f in flags if f.startswith("sink=")), 64)
        known = {"tight", "grad", "ht", "sketch", "nopersist"}
        bad = {f for f in flags if f not in known and not f.startswith(("win=", "sink="))}
        assert not bad, f"unknown flags {bad}"
        return Vkv1Press(bern=BernoulliPress(press=CachedKVzipPress()), kept=kept, eps0=eps0, delta=delta,
                         check="tight" if "tight" in flags else "simple",
                         order="grad" if "grad" in flags else "needle", trunc="ht" not in flags,
                         sketch="needle" if "sketch" in flags else "perfect", window=win, n_sink=sink,
                         persist="nopersist" not in flags)
    raise ValueError(arm)


a, b = (int(x) for x in args.rows.split(":"))
assert a >= 152 or args.dataset != "ruler32k", "rows 150-151 are the extract / calibration contexts; use 152 onwards"
cfg = evaluate.EvaluationConfig(dataset=args.dataset, data_dir=args.tasks[0], model=args.model,
                                press_name="no_press", compression_ratio=0.0,
                                model_kwargs={"attn_implementation": "sdpa"})
runner = evaluate.EvaluationRunner(cfg)
runner.press = None
runner._setup_model_pipeline()
pipe = runner.pipeline
presses = {arm: make_press(arm) for arm in args.arms}
done = {}
for arm in args.arms:
    (OUT / arm).mkdir(parents=True, exist_ok=True)
    f = OUT / arm / "rows.jsonl"
    done[arm] = {(j["task"], j["row"]) for j in map(json.loads, f.read_text().splitlines())} if f.exists() else set()
log(f"{args.model}; arms {args.arms}; rows {a}:{b}; tasks {args.tasks}")

for task in args.tasks:
    df = evaluate.load_ruler_length_dataset(args.dataset, task).sample(frac=1, random_state=42).iloc[a:b]
    for ridx, (_, row) in zip(range(a, b), df.iterrows()):
        for arm, press in presses.items():
            if (task, ridx) in done[arm]:
                continue
            t0 = time.time()
            if isinstance(press, Vkv1Press):
                press.reset()
            elif isinstance(press, NeedleFetchPress):
                press.reset_stats()
            max_new = int(row["max_new_tokens"])
            if args.max_new_tokens:
                max_new = min(max_new, args.max_new_tokens)
            torch.cuda.reset_peak_memory_stats()
            with torch.inference_mode():
                out = pipe(row["context"], question=row["question"], answer_prefix=row["answer_prefix"],
                           press=press, max_new_tokens=max_new, max_context_length=args.max_context_length)
            rec = {"task": task, "row": ridx, "pred": out["answer"], "gold": list(row["answer"]),
                   "sec": round(time.time() - t0, 1), "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
            extra = ""
            if isinstance(press, Vkv1Press):
                st = {k: v for k, v in press.stats.items() if not k.startswith("_")}
                rec["stats"], rec["kept"] = st, (press.kept_log[-1] if press.kept_log else None)
                q, d = max(st["question_decisions"], 1), max(st["decode_decisions"], 1)
                extra = (f" kept {rec['kept']:.4f} | Q fetch {st['question_fetch_any'] / q:.3f} "
                         f"k {st['question_k_sum'] / max(st['question_fetch_any'], 1):.0f} "
                         f"fail {st['question_fail'] / q:.4f} (T0 {st['question_fail_T0'] / q:.4f}) "
                         f"fwd% {100 * st['question_fwd_xfer'] / max(st['question_fwd_ctx'], 1):.3f} | "
                         f"D fetch {st['decode_fetch_any'] / d:.3f} k {st['decode_k_sum'] / max(st['decode_fetch_any'], 1):.0f} "
                         f"fail {st['decode_fail'] / d:.4f} (T0 {st['decode_fail_T0'] / d:.4f}) "
                         f"fwd% {100 * st['decode_fwd_xfer'] / max(st['decode_fwd_ctx'], 1):.4f} | "
                         f"persist% {100 * st['persisted_extra'] / max(st['persisted_slots'], 1):.3f}")
            elif isinstance(press, NeedleFetchPress):
                rec["needle_stats"] = {k: v for k, v in press.fetch_stats.items()
                                       if not k.startswith("_") and k != "per_layer"}
            with open(OUT / arm / "rows.jsonl", "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            log(f"{task}_{ridx} {arm}: {rec['sec']}s pred={out['answer'][:60]!r} gold={rec['gold'][:2]}{extra}")
            torch.cuda.empty_cache()
log("done")
