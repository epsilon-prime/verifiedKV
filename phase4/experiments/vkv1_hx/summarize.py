"""Summarise diag_hx.py (from vkv1_nopersist/summarize.py): per residency, eps0 and method, pooled over contexts and layers.

    python summarize.py out/<run>

Columns: Q / D = question pass / decode.
    fetch    share of decisions with k > 0          k|f    mean entries per fetching decision
    fwd%     per forward and KV head: |union over its rows of F \\ S| as % of the context (Q: the whole pass, D: per step)
    xNdl     fwd% relative to the needle rule in the same residency (1.0 = needle-level traffic)
    fail     realised P(|B hat - B| > eps0 B) after fetching (target <= delta, except oracle / needle which promise none)
    bound    mean certified P(fail) at the chosen k (chern / hx: rigorous bounds; mc: an estimate)
    peak%    largest single-forward transfer of one (layer, KV head), % of context (the fetch buffer to reserve)
    store%   largest store memory actually used (store entries the draw had dropped), % of context (<= rho)
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
      f"delta {b0['delta']}, MC draws {b0['mc_draws']}, hx m {b0['hx_m']} bins {b0['hx_bins']}, "
      f"needle theta {b0['theta']}, sinks {b0['sinks']} window {b0['window']}")
for res in b0["res"]:
    kap, rho = b0["res_cfg"][res]
    for e in b0["eps0"]:
        acc = {m: sum(b["acc"][res][f"{m}|{e}"] for b in bs) for m in b0["methods"]}  # [phase, layer, metric]
        mx = {m: [[max(float(b["acc"][res][f"{m}|{e}"][ph, :, M[c]].max()) for b in bs) for ph in (0, 1)]
                  for c in ("fwd_peak", "store_occ")] for m in b0["methods"]}  # maxima (sums would add them)
        ref = acc.get("needle")

        def fwdp(a, ph):
            x = a[ph].sum(0)
            return 100 * float(x[M["fwd_xfer"]] / max(float(x[M["fwd_ctx"]]), 1))

        print(f"\n=== {res}: draw {kap - rho:.4f} + store {rho:.4f}, eps0 {e} ===")
        print(f"{'method':>10} | {'Q fetch':>7} {'k|f':>6} {'fwd%':>7} {'xNdl':>6} {'fail':>6} {'bound':>6} {'peak%':>6} | "
              f"{'D fetch':>7} {'k|f':>5} {'fwd%':>7} {'xNdl':>6} {'fail':>6} {'bound':>6} {'peak%':>6} {'store%':>6} | {'MB/ans':>7}")
        for m in b0["methods"]:
            a = acc[m]
            cols = []
            for ph in (0, 1):
                x = a[ph].sum(0)
                n = max(float(x[M["n"]]), 1)
                nf = max(float(x[M["fetch"]]), 1)
                rel = fwdp(a, ph) / fwdp(ref, ph) if ref is not None and fwdp(ref, ph) > 0 else float("nan")
                s = (f"{x[M['fetch']] / n:7.3f} {x[M['k']] / nf:{6 if ph == 0 else 5}.0f} {fwdp(a, ph):7.{3 if ph == 0 else 4}f} "
                     f"{rel:6.1f} {x[M['fail']] / n:6.4f} {x[M['bound']] / n:6.4f} {100 * mx[m][0][ph]:6.2f}")
                if ph == 1:
                    s += f" {100 * mx[m][1][ph]:6.2f}"
                cols.append(s)
            mb = float(a[:, :, M["fwd_xfer"]].sum()) * 2 * 128 * 2 / 1e6 / len(bs)
            print(f"{m:>10} | {cols[0]} | {cols[1]} | {mb:7.1f}")
