"""Vkv1 upstream: Algorithm 2 of vkv_draft0.pdf replayed offline on the all-layer extracts, with the realised error.

Data: phase3/experiments/algo_improvements/out/extracts/<model>_all/<task>_<row>.pt (13 RULER-32k tasks x dev rows
150-151, all layers, exact scaled queries of the question pass and of every step of the full-cache greedy answer,
post-RoPE context keys, question / generated keys, KVzip scores). Decode is replayed along that fixed reference answer
(Assumption A1 at decode).

Residency (Algorithm 1, algo.residency, the same function as the press): one global c over all layers and KV heads with
sum min(1, c tau) = kept * size, tau = KVzip score, the first --sinks (64) and last --window (64) entries at r = 1 and paid
from the budget; one Bernoulli draw per (layer, KV head, draw index).

Per forward over the cache (the question pass: every question position; each decode step) and per KV head, for every
config (check x order x truncation) and every eps0:
    tail = {r < 1} minus entries persisted by earlier forwards of this answer (fetched entries stay resident, exact);
    per row (query head, position): prepare -> k* -> F (algo.py); the estimate uses the per-row F;
    after the forward the union of F over its rows persists (the question pass starts from nothing).
Weights: "true" (perfect sketch: exact weights in the plan) or "sketch" (needle fetch's 2-bit 16-pair sketch).

Counts per (res, config, eps0, delta, phase, layer) in METRICS (sums; summarize.py divides), plus log2 histograms of
k*, transfers and oracle k, and the persisted extra entries per (layer, KV head) at the end of the answer.

    python -u sim.py <xdir> --out out/<run> [--layers 0,18,35] [--max-steps 32] [--draws 2] [--shard i/k]
"""

import argparse
import glob
import os
import sys
import time
from pathlib import Path

import torch

HOME = os.path.expanduser("~")
sys.path.insert(0, os.path.join(HOME, "phase4"))
sys.path.insert(0, os.path.join(HOME, "phase3", "experiments", "algo_improvements"))
from vkv1 import algo  # noqa: E402
from cert import HeadSketch  # noqa: E402  (read-only import)

ap = argparse.ArgumentParser()
ap.add_argument("xdir")
ap.add_argument("--out", required=True)
ap.add_argument("--res", nargs="+", default=["g16", "g22.37", "g32"], help="g<x>: kept fraction 1/x")
ap.add_argument("--eps0", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.5])
ap.add_argument("--delta", type=float, nargs="+", default=[0.05])
ap.add_argument("--draws", type=int, default=2)
ap.add_argument("--weights", choices=["true", "sketch"], default="true")
ap.add_argument("--checks", nargs="+", default=list(algo.CHECKS))
ap.add_argument("--orders", nargs="+", default=list(algo.ORDERS))
ap.add_argument("--truncs", nargs="+", default=["trunc", "ht"])
ap.add_argument("--sinks", type=int, default=64, help="first context entries forced to r = 1")
ap.add_argument("--window", type=int, default=64, help="local window (last context entries) forced to r = 1")
ap.add_argument("--layers", default="all", help="'all' or comma-separated indices into the extract's layer list")
ap.add_argument("--max-steps", type=int, default=0, help="decode steps replayed (0 = all)")
ap.add_argument("--no-persist", action="store_true")
ap.add_argument("--chunk", type=int, default=8_000_000, help="max eps0 x rows x n elements per block")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--shard", default="0/1")
args = ap.parse_args()
OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
F64 = torch.float64
CONFIGS = [(c, o, t) for o in args.orders for t in args.truncs for c in args.checks]
CNAMES = [f"{c}-{o}-{t}" for c, o, t in CONFIGS]
METRICS = ("n", "fetch", "k", "xfer", "fail", "fail_T0", "fail_HT0", "oracle", "k_on_oracle", "oracle_pos",
           "under_oracle", "beta_rel", "T_size", "n_tail", "bplus_before_rel", "bplus_after_rel", "lemma1_bad",
           "nonmono", "eps_cert0", "tight_gt_simple", "fwd", "fwd_any", "fwd_xfer", "fwd_ctx")
MI = {m: i for i, m in enumerate(METRICS)}
HISTS = ("k", "xfer", "oracle", "fwd_xfer")
NB = 20
E0 = torch.tensor(args.eps0, dtype=F64, device=dev)
NE = len(args.eps0)


def log2bin(x: torch.Tensor) -> torch.Tensor:
    x = x.double()
    return torch.where(x > 0, x.clamp_min(1).log2().floor() + 1, torch.zeros_like(x)).clamp(max=NB - 1).long()


def hist_add(h: torch.Tensor, x: torch.Tensor, sel: torch.Tensor):
    """h [NE, NB] += histogram of x [NE, ...] over sel [NE, ...] per eps0."""
    b = log2bin(x)
    for e in range(NE):
        h[e] += torch.bincount(b[e][sel[e]].reshape(-1), minlength=NB).to(h)


def forward(w, wh, wx, r, Z, P, cfg, delta, acc, hist, U):
    """One block of rows of one forward. w, wh [R, n] exact / plan weights; wx [R]; r [n]; Z [n] bool;
    P [NE, n] persisted (exact) entries per eps0 state. Adds to acc [NE, n_metrics], hist [n_hist, NE, NB];
    ORs the per-forward transfer set into U [NE, n] and returns the union of F over the rows [NE, n]."""
    check, order, trunc = cfg
    R, n = w.shape
    zero = torch.zeros((), dtype=F64, device=w.device)
    tail = (r < 1)[None] & ~P  # [NE, n]
    tail3 = tail[:, None]  # [NE, 1, n]
    sure = torch.where(tail3, zero, w[None]).sum(-1) + wx[None]  # [NE, R]: resident exact mass incl. w^x
    plan = algo.prepare(wh[None].expand(NE, R, n), r, tail3, sure, delta, trunc=trunc == "trunc", order=order)
    k, nonmono = algo.kstar(plan, E0[:, None], check)
    F = algo.fetch_mask(plan, k) & tail3
    B = w.sum(-1) + wx  # [R]
    Zt = (tail & Z[None])[:, None]  # sampled tail [NE, 1, n]
    no = torch.zeros_like(F)
    bh = algo.estimate(w[None], r, tail3, sure, Z, plan.T, F)
    bh0 = algo.estimate(w[None], r, tail3, sure, Z, plan.T, no)
    bht = algo.estimate(w[None], r, tail3, sure, Z, tail3 & (r <= 0), no)
    e = (bh - B).abs() / B
    errc = algo.error_curve(plan, w[None], r, tail3, Z, bh0 - B)
    ko = algo.oracle_k(errc, B[None].expand(NE, R), E0[:, None], plan.n_tail)
    xf = F & ~Zt
    xfer = xf.sum(-1)
    U |= xf.any(1)
    pos = tail3 & (r > 0)
    ht_jump = torch.where(pos & ~plan.T, wh[None] * (1 - r) / r.clamp_min(1e-300), zero).amax(-1)
    fetch = k > 0
    eo = E0[:, None]
    vals = {
        "n": torch.full_like(e, 1.0), "fetch": fetch, "k": k, "xfer": xfer, "fail": e > eo,
        "fail_T0": (bh0 - B).abs() / B > eo, "fail_HT0": (bht - B).abs() / B > eo, "oracle": ko,
        "k_on_oracle": torch.where(ko > 0, k, 0), "oracle_pos": ko > 0, "under_oracle": k < ko,
        "beta_rel": algo.curve_at(plan.beta, k) / plan.Bt, "T_size": (plan.T & tail3).sum(-1),
        "n_tail": plan.n_tail, "bplus_before_rel": plan.b_plus_before / plan.Bt,
        "bplus_after_rel": plan.up[..., 0] / plan.Bt, "lemma1_bad": ht_jump > 2 * plan.beta_T * (1 + 1e-9),
        "nonmono": nonmono, "eps_cert0": plan.eps_simple()[..., 0],
    }
    if check == "tight":
        ks, _ = algo.kstar(plan, E0[:, None], "simple")
        vals["tight_gt_simple"] = k > ks
    for m, v in vals.items():
        acc[:, MI[m]] += v.double().sum(-1)
    hist_add(hist[0], k, fetch)
    hist_add(hist[1], xfer, fetch)
    hist_add(hist[2], ko, ko > 0)
    return F.any(1)


def end_forward(U, acc, hist, n):
    """Per-forward (per KV head) transfer: |union over rows of F \\ S|."""
    u = U.sum(-1).double()
    acc[:, MI["fwd"]] += 1
    acc[:, MI["fwd_any"]] += (u > 0).double()
    acc[:, MI["fwd_xfer"]] += u
    acc[:, MI["fwd_ctx"]] += n
    hist_add(hist[3], u[:, None], (u > 0)[:, None])


si, sk = (int(x) for x in args.shard.split("/"))
files = sorted(glob.glob(f"{args.xdir}/*.pt"))[: args.limit or None][si::sk]
T0 = time.time()
for f in files:
    tag = Path(f).stem
    if (OUT / f"{tag}.pt").exists():
        print(f"{tag}: exists, skipping", flush=True)
        continue
    blob = torch.load(f, map_location="cpu")
    n, nq, G, nkv = blob["n"], blob["nq"], blob["n_heads"] // blob["n_kv"], blob["n_kv"]
    tau_all = blob["tau_kvzip"].double()
    lidx = list(range(len(blob["layers"]))) if args.layers == "all" else [int(x) for x in args.layers.split(",")]
    # Algorithm 1: one global c over all layers and KV heads, sinks and local window at r = 1 (the press's residency)
    r_all = {res: algo.residency(tau_all, 1 / float(res[1:]), args.sinks, args.window) for res in args.res}

    def resid(li, h, res):
        return r_all[res][li, h].to(dev)

    shape = (len(CONFIGS), len(args.delta), NE, 2, len(lidx))
    acc = {res: torch.zeros(*shape, len(METRICS), dtype=F64, device=dev) for res in args.res}
    hist = {res: torch.zeros(len(CONFIGS), len(args.delta), len(HISTS), 2, NE, NB, dtype=F64, device=dev) for res in args.res}
    persist = {res: torch.zeros(len(CONFIGS), len(args.delta), NE, len(lidx), dtype=F64, device=dev) for res in args.res}
    # kept: over all layers and KV heads (the global budget); kept_sel: over the replayed layers only
    kept = {res: float(torch.stack([resid(li, h, res) for li in range(len(blob["layers"])) for h in range(nkv)]).mean())
            for res in args.res}
    kept_sel = {res: float(torch.stack([resid(li, h, res) for li in lidx for h in range(nkv)]).mean())
                for res in args.res}
    steps_total = 0 if blob["q_decode"] is None else blob["q_decode"].shape[2]
    steps = steps_total if not args.max_steps else min(args.max_steps, steps_total)
    print(f"{tag}: n={n} nq={nq} decode steps {steps}/{steps_total} layers {len(lidx)} kept "
          + " ".join(f"{k}={v:.4f} (replayed layers {kept_sel[k]:.4f})" for k, v in kept.items()), flush=True)
    for lj, li in enumerate(lidx):
        t_l = time.time()
        for h in range(nkv):
            keys = blob["keys"][li, h].to(dev).float()
            kx = blob["keys_extra"][li, h].to(dev).float()
            qq = blob["q_question"][li, h * G:(h + 1) * G].to(dev).float().reshape(G * nq, -1)  # row = g * nq + t
            tq = torch.arange(nq, device=dev).repeat(G)
            qd = None if steps == 0 else blob["q_decode"][li, h * G:(h + 1) * G, :steps].to(dev).float()
            if args.weights == "sketch":
                sk_ = HeadSketch(blob["keys"][li, h])
                S_idx, k_hat = sk_.S.to(dev), sk_.k_hat.to(dev)

            def weights(q, tpos, r):
                """q [R, d] -> exact w, plan wh [R, n] and w^x [R] (shared shift m per row)."""
                s = q @ keys.T
                se = (q @ kx.T).masked_fill(torch.arange(kx.shape[0], device=dev)[None] > tpos[:, None],
                                            -float("inf"))
                if args.weights == "true":
                    m = torch.maximum(s.amax(-1), se.amax(-1))[:, None]
                    w = torch.exp((s - m).double())
                    return w, w, torch.exp((se - m).double()).sum(-1)
                s_hat = q[:, S_idx] @ k_hat.T
                m = torch.maximum(torch.where(r >= 1, s, s_hat).amax(-1), se.amax(-1))[:, None]
                return (torch.exp((s - m).double()), torch.exp((s_hat - m).double()),
                        torch.exp((se - m).double()).sum(-1))

            for res in args.res:
                r = resid(li, h, res)
                wq, whq, wxq = weights(qq, tq, r)
                wd = [weights(qd[:, j], torch.full((G,), nq + j, device=dev), r) for j in range(steps)]
                for d in range(args.draws):
                    Z = torch.rand(n, device=dev, dtype=F64) < r
                    for ci, cfg in enumerate(CONFIGS):
                        for di, delta in enumerate(args.delta):
                            a_q, a_d = acc[res][ci, di, :, 0, lj], acc[res][ci, di, :, 1, lj]
                            h_q, h_d = hist[res][ci, di, :, 0], hist[res][ci, di, :, 1]
                            P = torch.zeros(NE, n, dtype=torch.bool, device=dev)
                            U = torch.zeros_like(P)
                            Fu = torch.zeros_like(P)
                            blk = max(1, args.chunk // (NE * n))
                            for r0 in range(0, G * nq, blk):
                                sl = slice(r0, r0 + blk)
                                Fu |= forward(wq[sl], whq[sl], wxq[sl], r, Z, P, cfg, delta, a_q, h_q, U)
                            end_forward(U, a_q, h_q, n)
                            if not args.no_persist:
                                P |= Fu
                            for j in range(steps):
                                U.zero_()
                                Fu = forward(*wd[j], r, Z, P, cfg, delta, a_d, h_d, U)
                                end_forward(U, a_d, h_d, n)
                                if not args.no_persist:
                                    P |= Fu
                            # extra resident entries at the end of the answer: persisted ones the draw had dropped
                            persist[res][ci, di, :, lj] += (P & ~Z[None]).sum(-1).double()
                del wq, whq, wd
        print(f"  layer {blob['layers'][li]} ({lj + 1}/{len(lidx)}): {time.time() - t_l:.1f}s "
              f"(total {time.time() - T0:.0f}s)", flush=True)
    cpu = lambda d: {k: v.cpu() for k, v in d.items()}  # noqa: E731
    out = {"acc": cpu(acc), "hist": cpu(hist), "persist": cpu(persist), "metrics": METRICS, "hists": HISTS, "configs": CNAMES,
           "eps0": args.eps0, "delta": args.delta, "res": args.res, "kept": kept, "kept_sel": kept_sel, "layers": [blob["layers"][i]
           for i in lidx], "task": blob["task"], "row": blob["row"], "n": n, "nq": nq, "steps": steps,
           "n_kv": nkv, "G": G, "draws": args.draws, "weights": args.weights, "window": args.window, "sinks": args.sinks,
           "persist_on": not args.no_persist}
    torch.save(out, OUT / f"{tag}.pt")
    print(f"{tag}: done ({time.time() - T0:.0f}s)", flush=True)
    del blob
