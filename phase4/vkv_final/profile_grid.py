"""Where does vkv_grid's time go? One model load, a few RULER contexts, full cache vs vkv_grid; synchronised timers
around each stage of the press (wrapping its building blocks, the press itself is not edited), split by phase.

    python -u profile_grid.py --cases ruler32k:niah_single_3:152 ruler32k:cwe:152 ruler64k:niah_single_3:0 \
        --eps0 0.3 --delta 0.1

Stages (per forward, per layer, summed over the answer):
    bias_total   the whole vkv_grid_bias call
    prepare      vkv1/algo.prepare: T* (sort by jump) + grad order (sort) + suffix curves, float64
    kstar        chernoff.kstar_fast: Bernstein screen + Chernoff curves (32-point lambda grid)
    fetch_bias   algo.fetch_mask + algo.attention_bias
    rest         bias_total minus the above: score matmuls, the shortlist, exp, masks
Also: rows planned, rows that needed a Chernoff curve, CUDA kernel launches in one decode-step layer.
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "kvpress"))
sys.path.insert(0, str(REPO / "kvpress" / "evaluation"))
sys.path.insert(0, str(HERE))
import evaluate  # noqa: E402
import press_grid as pg  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
ap.add_argument("--cases", nargs="+", required=True, help="dataset:task:row (row = position in the seed-42 permutation)")
ap.add_argument("--eps0", type=float, default=0.3)
ap.add_argument("--delta", type=float, default=0.1)
ap.add_argument("--out", default="out/profile.json")
args = ap.parse_args()

T = defaultdict(float)
N = defaultdict(int)
PHASE = {"p": "question"}


def sync():
    torch.cuda.synchronize()


def timed(name, fn):
    def wrap(*a, **k):
        sync()
        t0 = time.perf_counter()
        out = fn(*a, **k)
        sync()
        T[f"{PHASE['p']}:{name}"] += time.perf_counter() - t0
        N[f"{PHASE['p']}:{name}"] += 1
        return out
    return wrap


_bias = pg.vkv_grid_bias


def bias_wrapped(query, key, state, scaling):
    PHASE["p"] = "decode" if query.shape[2] == 1 else "question"
    N[f"{PHASE['p']}:rows"] += query.shape[1] * query.shape[2]
    return timed("bias_total", _bias)(query, key, state, scaling)


_kstar = pg.chernoff.kstar_fast


def kstar_wrapped(*a, **k):
    out = _kstar(*a, **k)
    N[f"{PHASE['p']}:rows_curved"] += out[2]
    return out


pg.vkv_grid_bias = bias_wrapped  # the route looks the name up at call time
pg.algo.prepare = timed("prepare", pg.algo.prepare)
pg.chernoff.kstar_fast = timed("kstar", kstar_wrapped)
pg.algo.fetch_mask = timed("fetch_bias", pg.algo.fetch_mask)
pg.algo.attention_bias = timed("fetch_bias", pg.algo.attention_bias)

cfg = evaluate.EvaluationConfig(dataset="ruler32k", data_dir="qa_1", model=args.model, press_name="no_press",
                                compression_ratio=0.0, model_kwargs={"attn_implementation": "sdpa"})
runner = evaluate.EvaluationRunner(cfg)
runner.press = None
runner._setup_model_pipeline()
pipe = runner.pipeline
gpu = torch.cuda.get_device_name(0)
results = []
for case in args.cases:
    ds, task, ridx = case.split(":")
    ridx = int(ridx)
    row = evaluate.load_ruler_length_dataset(ds, task).sample(frac=1, random_state=42).iloc[ridx]
    for arm in ("full", "vkv_grid"):
        press = None if arm == "full" else pg.vkv_grid(0.9375, args.eps0, args.delta)
        T.clear()
        N.clear()
        sync()
        t0 = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            out = pipe(row["context"], question=row["question"], answer_prefix=row["answer_prefix"], press=press,
                       max_new_tokens=int(row["max_new_tokens"]), max_context_length=200000)
        sync()
        total = time.perf_counter() - t0
        rec = {"case": case, "arm": arm, "gpu": gpu, "total_s": round(total, 2),
               "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2), "pred": out["answer"][:80],
               "timers": {k: round(v, 2) for k, v in sorted(T.items())}, "counts": dict(sorted(N.items()))}
        results.append(rec)
        print(json.dumps(rec), flush=True)
    # kernel launches in one decode-step layer of vkv_grid (CUDA kernels counted by the profiler)
    del press

# one decode-step layer under torch.profiler, on the last case, to count kernel launches
from torch.profiler import ProfilerActivity, profile  # noqa: E402

captured = {}


def capture_once(query, key, state, scaling):
    if query.shape[2] == 1 and "done" not in captured:
        captured["done"] = True
        with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
            o = _bias(query, key, state, scaling)
            sync()
        ev = prof.key_averages()
        captured["cuda_kernels"] = int(sum(e.count for e in ev if e.device_type.name == "CUDA"))
        captured["top"] = [(e.key[:60], e.count, round(e.self_device_time_total / 1e3, 2))
                           for e in sorted(ev, key=lambda e: -e.self_device_time_total)[:12]]
        return o
    return _bias(query, key, state, scaling)


pg.vkv_grid_bias = capture_once
ds, task, ridx = args.cases[0].split(":")
row = evaluate.load_ruler_length_dataset(ds, task).sample(frac=1, random_state=42).iloc[int(ridx)]
with torch.inference_mode():
    pipe(row["context"], question=row["question"], answer_prefix=row["answer_prefix"],
         press=pg.vkv_grid(0.9375, args.eps0, args.delta), max_new_tokens=2, max_context_length=200000)
print(json.dumps({"decode_layer_kernels": captured.get("cuda_kernels"), "top_kernels_ms": captured.get("top")}),
      flush=True)
results.append({"decode_layer_profile": captured})
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
json.dump(results, open(args.out, "w"), indent=1)
