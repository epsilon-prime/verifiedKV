"""The vectorised certificate returns the same k* as the loop version (GPU if available)."""
import time

import torch

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from verifiedkv import chernoff_vec as cv  # noqa: E402
from verifiedkv.chernoff_vec import algo, chernoff  # noqa: E402

dev = "cuda"
F64 = torch.float64
mism = 0
t_old = t_new = 0.0
for seed in range(12):
    g = torch.Generator(device=dev).manual_seed(seed)
    n = [600, 20000, 33000, 70000][seed % 4]
    N = 32 if seed % 3 else 176                      # decode-like / question-like batches
    wh = torch.rand(N, n, generator=g, dtype=F64, device=dev) ** 8
    wh[:, 7] *= 50
    r = (torch.rand(N, n, generator=g, dtype=F64, device=dev) * 0.12).clamp_min(1e-3)
    r[:, :64] = 1.0
    tail = r < 1
    plan = algo.prepare(wh, r, tail, wh.sum(-1) * 0.2, 0.05, order="grad")
    for eps0, delta in ((0.3, 0.05), (0.3, 0.1), (0.5, 0.05)):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        a = chernoff.kstar_fast(plan, wh, r, tail, eps0, delta)
        torch.cuda.synchronize(); t1 = time.perf_counter()
        b = cv.kstar_fast(plan, wh, r, tail, eps0, delta)
        torch.cuda.synchronize(); t2 = time.perf_counter()
        t_old += t1 - t0; t_new += t2 - t1
        same = torch.equal(a[0], b[0]) and a[1:] == b[1:]
        mism += not same
        print(f"seed {seed} n {n} N {N} eps0 {eps0} delta {delta}: identical {same}, curved {a[2]}", flush=True)
print(f"time old {t_old:.2f}s new {t_new:.2f}s ({t_old / t_new:.2f}x)")
print("ALL IDENTICAL" if mism == 0 else f"{mism} MISMATCHES")
