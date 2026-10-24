"""Summarise diag.py: per residency, eps0 and method, pooled over contexts and layers.

    python summarize.py out/<run>

Columns: Q / D = question pass / decode.
    fetch    share of decisions with k > 0          k|f    mean entries per fetching decision
    fwd%     per forward and KV head: |union over its rows of F \\ S| as % of the context (Q: the whole pass, D: per step)
    xNdl     fwd% relative to the needle rule on the same decisions (1.0 = needle-level traffic)
    fail     realised P(|B hat - B| > eps0 B) after fetching (target <= delta, except oracle / needle which promise none)
    fail0    the same with no fetch (truncated estimator; HT for needle)
    give     share of rows the Remark-2 allocation leaves with P > delta (avg_* only)
    pers%    extra resident entries at the end of the answer (persisted fetches the draw had dropped), % of context
    MB/ans   entries read from the host per answer x 2 (K, V) x head_dim 128 x 2 bytes, summed over layers and KV heads
"""

import glob
import sys
from pathlib import Path

import torch

run = sys.argv[1]
files = sorted(glob.glob(f"{run}/*.pt"))
bs = [torch.load(f) for f in files]
b0 = bs[0]
M = {m: i for i, m in enumerate(b0["metrics"])}
print(f"{run}: {len(bs)} contexts ({', '.join(Path(f).stem for f in files)}), layers {len(b0['layers'])}, "
      f"delta {b0['delta']}, MC draws {b0['mc_draws']}, needle theta {b0['theta']}, sinks {b0['sinks']} window {b0['window']}")
for res in b0["res"]:
    for e in b0["eps0"]:
        acc = {m: sum(b["acc"][res][f"{m}|{e}"] for b in bs) for m in b0["methods"]}  # [phase, layer, metric]
        pers = {m: sum(float(b["persist"][res][f"{m}|{e}"].mean()) for b in bs) / len(bs) for m in b0["methods"]}
        ref = acc.get("needle")

        def fwdp(a, ph):
            x = a[ph].sum(0)
            return 100 * float(x[M["fwd_xfer"]] / max(float(x[M["fwd_ctx"]]), 1))

        print(f"\n=== kept {res}, eps0 {e} ===")
        print(f"{'method':>19} | {'Q fetch':>7} {'k|f':>7} {'fwd%':>7} {'xNdl':>6} {'fail':>6} {'fail0':>6} {'give':>5} | "
              f"{'D fetch':>7} {'k|f':>6} {'fwd%':>7} {'xNdl':>6} {'fail':>6} {'fail0':>6} {'give':>5} | {'pers%':>6} {'MB/ans':>7}")
        for m in b0["methods"]:
            a = acc[m]
            cols = []
            for ph in (0, 1):
                x = a[ph].sum(0)
                n, nf = max(float(x[M["n"]]), 1), max(float(x[M["fetch"]]), 1)
                rel = fwdp(a, ph) / fwdp(ref, ph) if ref is not None and fwdp(ref, ph) > 0 else float("nan")
                cols.append(f"{x[M['fetch']] / n:7.3f} {x[M['k']] / nf:{7 if ph == 0 else 6}.0f} "
                            f"{fwdp(a, ph):7.{3 if ph == 0 else 4}f} {rel:6.1f} {x[M['fail']] / n:6.4f} "
                            f"{x[M['fail0']] / n:6.4f} {x[M['giveup']] / n:5.3f}")
            mb = float(a[:, :, M["fwd_xfer"]].sum()) * 2 * 128 * 2 / 1e6 / len(bs)
            print(f"{m:>19} | {cols[0]} | {cols[1]} | {100 * pers[m]:6.2f} {mb:7.1f}")
