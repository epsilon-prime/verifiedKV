"""Summarise sim.py outputs: the checks of vkv_draft0.pdf section 10, per kept fraction, delta, config, eps0, phase.

    python summarize.py out/<run> [--by-layer]

Columns (rates are over decisions = (layer, query head, position) rows, pooled over contexts, layers, draws):
    fetch     share of decisions with k* > 0
    k*|f      mean k* over fetching decisions; p50 / p90 from log2 histograms (bin upper edges)
    xfer|f    mean |F \\ S| (entries read from the host) per fetching decision
    fwd%      per forward and KV head: union over its rows of F \\ S, as % of the context (question: whole pass)
    fwd_any   share of (forward, KV head) with any transfer
    fail      P(|B hat - B| > eps0 B) after fetching (<= delta expected with true weights)
    T0 / HT0  the same with no fetch (truncated / plain Horvitz-Thompson), on the same persisted state
    k*/orc    sum k* / sum oracle k over decisions whose oracle k > 0 (>= 1: the certificate's slack)
    beta%     beta(k*) / B tilde;  |T*| mean truncated set;  b+ before -> after truncation (/ B tilde)
    L1bad     decisions violating Lemma 1 (must be 0);  nonmono  tight rows passing at k and failing at k + 1
    t>s       tight k* > simple k* on the same plan (must be 0);  persist%  extra resident entries at answer end
"""

import argparse
import glob
from pathlib import Path

import torch

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--by-layer", action="store_true")
args = ap.parse_args()
files = sorted(glob.glob(f"{args.run}/*.pt"))
blobs = [torch.load(f) for f in files]
b0 = blobs[0]
M = {m: i for i, m in enumerate(b0["metrics"])}
print(f"{args.run}: {len(blobs)} contexts ({', '.join(Path(f).stem for f in files)}), weights {b0['weights']}, "
      f"draws {b0['draws']}, layers {len(b0['layers'])}, sinks {b0.get('sinks', 4)}, window {b0['window']}, persist {b0['persist_on']}")


def pq(h, q):
    """Quantile from a log2 histogram [NB]: upper edge of the bin (bin b holds [2^(b-1), 2^b))."""
    c = h.cumsum(0)
    if c[-1] <= 0:
        return float("nan")
    b = int((c >= q * c[-1]).nonzero()[0])
    return 0 if b == 0 else 2 ** b - 1


def orc(a):
    """sum k* / sum oracle k over decisions with oracle k > 0 ('-' when there are none)."""
    return f"{a[M['k_on_oracle']] / a[M['oracle']]:.2f}" if a[M["oracle"]] > 0 else "-"


def l1(a, cname):
    """Lemma 1 violations; '-' without truncation (the lemma is about the truncated estimator)."""
    return "-" if cname.endswith("-ht") else str(int(a[M["lemma1_bad"]]))


for res in b0["res"]:
    acc = sum(b["acc"][res] for b in blobs)  # [cfg, delta, eps0, phase, layer, metric]
    hist = sum(b["hist"][res] for b in blobs)  # [cfg, delta, hist, phase, eps0, NB]
    pers = sum(b["persist"][res].sum(-1) / (b["draws"] * b["n_kv"] * b["n"] * len(b["layers"])) for b in blobs) \
        / len(blobs)  # [cfg, delta, eps0]
    kept = sum(b["kept"][res] for b in blobs) / len(blobs)
    for di, delta in enumerate(b0["delta"]):
        print(f"\n=== kept {res} (mean r {kept:.4f}, replayed layers {sum(b.get('kept_sel', b['kept'])[res] for b in blobs) / len(blobs):.4f}), delta {delta} ===")
        hdr = (f"{'config':>22} {'eps0':>4} {'ph':>2} | {'fetch':>5} {'k*|f':>7} {'p50':>5} {'p90':>6} {'xfer|f':>7} "
               f"{'fwd%':>6} {'fwd_any':>7} | {'fail':>6} {'T0':>6} {'HT0':>6} | {'k*/orc':>6} {'beta%':>5} "
               f"{'|T*|':>6} {'b+bef':>6} {'b+aft':>6} | {'L1bad':>5} {'nonmono':>7} {'t>s':>4} {'pers%':>6}")
        print(hdr)
        for ci, cname in enumerate(b0["configs"]):
            for ei, e0 in enumerate(b0["eps0"]):
                for ph, pname in enumerate(("Q", "D")):
                    a = acc[ci, di, ei, ph].sum(0)
                    n, nf = max(a[M["n"]], 1), max(a[M["fetch"]], 1)
                    hk = hist[ci, di, 0, ph, ei]
                    print(f"{cname:>22} {e0:>4.2f} {pname:>2} | {a[M['fetch']] / n:5.3f} {a[M['k']] / nf:7.1f} "
                          f"{pq(hk, .5):5.0f} {pq(hk, .9):6.0f} {a[M['xfer']] / nf:7.1f} "
                          f"{100 * a[M['fwd_xfer']] / max(a[M['fwd_ctx']], 1):6.3f} "
                          f"{a[M['fwd_any']] / max(a[M['fwd']], 1):7.3f} | {a[M['fail']] / n:6.4f} "
                          f"{a[M['fail_T0']] / n:6.4f} {a[M['fail_HT0']] / n:6.4f} | "
                          f"{orc(a):>6} {100 * a[M['beta_rel']] / n:5.2f} "
                          f"{a[M['T_size']] / n:6.1f} {a[M['bplus_before_rel']] / n:6.3f} "
                          f"{a[M['bplus_after_rel']] / n:6.3f} | {l1(a, cname):>5} "
                          f"{a[M['nonmono']] / n:7.4f} {int(a[M['tight_gt_simple']]):4d} "
                          f"{100 * pers[ci, di, ei]:6.3f}" if ph == 1 else
                          f"{cname:>22} {e0:>4.2f} {pname:>2} | {a[M['fetch']] / n:5.3f} {a[M['k']] / nf:7.1f} "
                          f"{pq(hk, .5):5.0f} {pq(hk, .9):6.0f} {a[M['xfer']] / nf:7.1f} "
                          f"{100 * a[M['fwd_xfer']] / max(a[M['fwd_ctx']], 1):6.3f} "
                          f"{a[M['fwd_any']] / max(a[M['fwd']], 1):7.3f} | {a[M['fail']] / n:6.4f} "
                          f"{a[M['fail_T0']] / n:6.4f} {a[M['fail_HT0']] / n:6.4f} | "
                          f"{orc(a):>6} {100 * a[M['beta_rel']] / n:5.2f} "
                          f"{a[M['T_size']] / n:6.1f} {a[M['bplus_before_rel']] / n:6.3f} "
                          f"{a[M['bplus_after_rel']] / n:6.3f} | {l1(a, cname):>5} "
                          f"{a[M['nonmono']] / n:7.4f} {int(a[M['tight_gt_simple']]):4d}")
        if args.by_layer:
            print("\n  fetch rate / fail by layer (first config, each eps0, decode):")
            for li, layer in enumerate(b0["layers"]):
                a = acc[0, di, :, 1, li]
                print(f"  layer {layer:>2}: " + "  ".join(
                    f"e{e0:g} {a[ei, M['fetch']] / max(a[ei, M['n']], 1):.3f}/{a[ei, M['fail']] / max(a[ei, M['n']], 1):.4f}"
                    for ei, e0 in enumerate(b0["eps0"])))
