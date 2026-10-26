"""Downstream (real generation) runner for vkv_final_algo on RULER, with the press's online statistics.
One model load; every arm runs on every row; resumable (rows already in out/<run>/<arm>/rows.jsonl are skipped).

Arms:
    full                                    no compression
    bern:<kept>                             BernoulliPress on CachedKVzipPress, kept fraction <kept> (baseline)
    vkv_final:<ratio>:<eps0>:<delta>[:flags]
                                            vkv_final_algo(ratio, eps0, delta); flags:
                                              slh=<M>   shortlist size per KV head (default 128; 0 = off)
                                              perfect   plan on the exact weights (reference; nothing is charged)
                                              exact     exact float64 Chernoff curve instead of the fast k*
    e.g. vkv_final:0.9375:0.3:0.05 is the default configuration at 16x.

kvpress comes from this repository (<repo>/kvpress). Set KVPRESS_SCORE_CACHE to share KVzip scores across runs.
Output: out/<run>/<arm>/rows.jsonl (prediction, gold, task, row, seconds, peak memory, the press's statistics).

    python -u run_downstream.py --rows 152:154 --tasks niah_single_3 qa_1 --arms full vkv_final:0.9375:0.3:0.05 --run smoke
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "kvpress"))
sys.path.insert(0, str(REPO / "kvpress" / "evaluation"))
sys.path.insert(0, str(HERE))
import evaluate  # noqa: E402  (also patches attention through kvpress)
from kvpress.presses.bernoulli_press import BernoulliPress  # noqa: E402
from kvpress.presses.cached_kvzip_press import CachedKVzipPress  # noqa: E402
from press_final import VkvFinalPress, vkv_final_algo  # noqa: E402  (routes kvpress's needle hook)

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
ap.add_argument("--dataset", default="ruler32k")
ap.add_argument("--tasks", nargs="+", required=True)
ap.add_argument("--rows", required=True, help="rows a:b of the seed-42 permutation (RULER dev rows: a >= 152)")
ap.add_argument("--arms", nargs="+", required=True)
ap.add_argument("--run", required=True)
ap.add_argument("--max-new-tokens", type=int, default=0, help="cap on generated tokens (0 = the task's own)")
ap.add_argument("--max-context-length", type=int, default=200000)
args = ap.parse_args()
OUT = HERE / "out" / args.run
T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:8.1f}s] {m}", flush=True)


def make_press(arm: str):
    kind, *rest = arm.split(":")
    if kind == "full":
        return None
    if kind == "bern":
        return BernoulliPress(press=CachedKVzipPress(compression_ratio=1 - float(rest[0])))
    if kind == "vkv_final":
        kw = {}
        for f in rest[3:]:
            if f.startswith("slh="):
                kw["shortlist_head"] = int(f[4:])
            elif f == "perfect":
                kw["plan_weights"] = "perfect"
            elif f == "exact":
                kw["fast"] = False
            else:
                raise ValueError(f"unknown vkv_final flag {f!r}")
        return vkv_final_algo(float(rest[0]), float(rest[1]), float(rest[2]), **kw)
    raise ValueError(arm)


a, b = (int(x) for x in args.rows.split(":"))
assert args.dataset.startswith("ruler"), "this runner covers the RULER length datasets (ruler16k / 32k / 64k / 128k)"
assert a >= 152 or args.dataset != "ruler32k", "ruler32k rows 150-151 are calibration contexts; use 152 onwards"
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
log(f"{args.model}; arms {args.arms}; rows {a}:{b}; tasks {args.tasks}; GPU {torch.cuda.get_device_name(0)}")

for task in args.tasks:
    df = evaluate.load_ruler_length_dataset(args.dataset, task).sample(frac=1, random_state=42).iloc[a:b]
    for ridx, (_, row) in zip(range(a, b), df.iterrows()):
        for arm, press in presses.items():
            if (task, ridx) in done[arm]:
                continue
            t0 = time.time()
            if isinstance(press, VkvFinalPress):
                press.reset()
            max_new = int(row["max_new_tokens"])
            if args.max_new_tokens:
                max_new = min(max_new, args.max_new_tokens)
            torch.cuda.reset_peak_memory_stats()
            with torch.inference_mode():
                out = pipe(row["context"], question=row["question"], answer_prefix=row["answer_prefix"],
                           press=press, max_new_tokens=max_new, max_context_length=args.max_context_length)
            rec = {"task": task, "row": ridx, "pred": out["answer"], "gold": list(row["answer"]),
                   "sec": round(time.time() - t0, 1), "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                   "gpu": torch.cuda.get_device_name(0)}
            extra = ""
            if isinstance(press, VkvFinalPress):
                st = {k: v for k, v in press.stats.items() if not k.startswith("_")}
                rec.update(stats=st, kept=press.kept_log[-1] if press.kept_log else None, kept_draw=press.kept_draw,
                           sketch_frac=press.sketch_frac if press.plan_weights == "sketch" else 0.0)
                q, d = max(st["question_decisions"], 1), max(st["decode_decisions"], 1)
                extra = (f" draw {rec['kept']:.4f} | Q fetch {st['question_fetch_any'] / q:.3f} "
                         f"fail {st['question_fail'] / q:.4f} (@{press.eval_eps}: {st['question_fail_eval'] / q:.4f}) "
                         f"| D fetch {st['decode_fetch_any'] / d:.3f} fail {st['decode_fail'] / d:.4f} "
                         f"(@{press.eval_eps}: {st['decode_fail_eval'] / d:.4f})")
            with open(OUT / arm / "rows.jsonl", "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            log(f"{task}_{ridx} {arm}: {rec['sec']}s pred={out['answer'][:60]!r} gold={rec['gold'][:2]}{extra}")
            torch.cuda.empty_cache()
log("done")
