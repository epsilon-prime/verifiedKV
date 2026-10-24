"""vkv1_diag: where does Vkv1's fetch volume come from, and how far is each lever from needle fetch? Exact weights
(perfect sketch), offline on the algo_improvements extracts, same residency as vkv1_sim (algo.residency, 64 sinks +
64-token window), one actual draw per (layer, residency), persistence per method (each method keeps its own fetched set
resident until the end of the answer). Existing Phase 4 code is imported read-only.

Methods (all draw-free plans unless stated; T* by the MSE rule; "grad" = eps-gradient fetch order):
    bern_simple_needle  the draft: simple Bernstein, needle order (per decision)
    bern_tight_grad     tight check, grad order (per decision; the best vkv1_sim config)
    chern_grad          exact two-point Chernoff bound, grad order (per decision)
    mc_grad             smallest k whose Monte Carlo P(fail) <= delta (fresh draws), grad order: the best any
                        per-decision draw-free rule on this order could do (an estimate, not a bound)
    avg_tight / avg_chern / avg_mc
                        Remark 2: per (forward, layer), k per row on a ladder minimising the expected transfer
                        subject to mean_rows P(k) <= delta (tight / Chernoff curves are bounds; MC is the ideal)
    shared_tight        one fetch set per (forward, KV head): shared order by max over its rows of w / B, smallest
                        prefix that makes every row pass the tight check
    needle              needle fetch's flag rule on exact weights: tail entries with w / sum_ctx w >= theta (0.003
                        question, 0.1 decode) for any query head of the position; no truncation
    oracle              smallest k (grad order) meeting eps0 for the ACTUAL draw (draw-dependent: a lower bound, not
                        an algorithm)

    python -u diag.py <xdir> --out out/<run> [--layers 0,18] [--max-steps 8] [--shard i/k]
"""

import argparse
import glob
import os
import sys
import time
import zlib
from pathlib import Path

import torch

HOME = os.path.expanduser("~")
sys.path.insert(0, os.path.join(HOME, "phase4"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vkv1 import algo  # noqa: E402  (read-only)
import diag_lib as dl  # noqa: E402

ALL = ("bern_simple_needle", "bern_tight_grad", "chern_grad", "mc_grad", "avg_tight", "avg_chern", "avg_mc",
       "shared_tight", "needle", "oracle")
ap = argparse.ArgumentParser()
ap.add_argument("xdir")
ap.add_argument("--out", required=True)
ap.add_argument("--res", nargs="+", default=["g16"])
ap.add_argument("--eps0", type=float, nargs="+", default=[0.3, 0.5])
ap.add_argument("--delta", type=float, default=0.05)
ap.add_argument("--methods", nargs="+", default=list(ALL))
ap.add_argument("--mc-draws", type=int, default=100)
ap.add_argument("--theta", type=float, nargs=2, default=[0.003, 0.1], help="needle thresholds: question, decode")
ap.add_argument("--sinks", type=int, default=64)
ap.add_argument("--window", type=int, default=64)
ap.add_argument("--layers", default="all")
ap.add_argument("--max-steps", type=int, default=0)
ap.add_argument("--chunk", type=int, default=3_000_000, help="max KV heads x rows x n elements per block")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--shard", default="0/1")
args = ap.parse_args()
assert all(m in ALL for m in args.methods), args.methods
OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"
F64 = torch.float64
METRICS = ("n", "fetch", "k", "xfer", "fail", "fail0", "giveup", "fwd", "fwd_any", "fwd_xfer", "fwd_ctx")
MI = {m: i for i, m in enumerate(METRICS)}
TRUNC = {m for m in ALL if m != "needle"}
AVG = {"avg_tight": "tight", "avg_chern": "chern", "avg_mc": "mc"}


def weights(q, keys, kx, tpos):
    """q [H, R, d], keys [H, n, d], kx [H, E, d], tpos [R] -> w [H, R, n], wx [H, R] (shared shift per row)."""
    s = q @ keys.transpose(-1, -2)
    se = (q @ kx.transpose(-1, -2)).masked_fill(torch.arange(kx.shape[1], device=dev)[None, None] > tpos[None, :, None],
                                               -float("inf"))
    m = torch.maximum(s.amax(-1), se.amax(-1))[..., None]
    return torch.exp((s - m).double()), torch.exp((se - m).double()).sum(-1)


def rank_of(order):
    rank = torch.empty_like(order)
    rank.scatter_(-1, order, torch.arange(order.shape[-1], device=order.device).expand_as(order))
    return rank


def forward(w, wx, r, tail, Z, q_len, methods, eps0s, theta, gen):
    """One forward over the cache for one layer: w [H, R, n] (row = g * q_len + t), r / tail / Z [H, n].
    Returns {(method, eps0): (F [H, R, n] bool, k [H, R], giveup [H, R] bool)}, T* [H, R, n], sure, B."""
    H, R, n = w.shape
    delta = args.delta
    r3, tail3, Z3 = r[:, None], tail[:, None], Z[:, None]
    zero = torch.zeros((), dtype=F64, device=dev)
    sure = torch.where(tail3, zero, w).sum(-1) + wx
    B = w.sum(-1) + wx
    need = set(methods)
    lad = dl.ladder(n, dev)
    out = {}
    T_full = torch.zeros(H, R, n, dtype=torch.bool, device=dev)
    keep_order = bool(need & set(AVG))
    order_g = torch.empty(H, R, n, dtype=torch.int32, device=dev) if keep_order else None
    # candidates per row: the ladder plus the row's own per-decision k* (last column), per (curve kind, eps0)
    P_lad = {(AVG[m], e): torch.zeros(H, R, len(lad) + 1, dtype=F64, device=dev) for m in need & set(AVG) for e in eps0s}
    cost_lad = {key: torch.zeros_like(v) for key, v in P_lad.items()}
    k_lad = {key: torch.zeros(H, R, len(lad) + 1, dtype=torch.long, device=dev) for key in P_lad}
    kshare = {e: torch.zeros(H, R, dtype=torch.long, device=dev) for e in eps0s} if "shared_tight" in need else None
    if "shared_tight" in need:
        score = torch.where(tail3, w / B[..., None], torch.full_like(w, -1.0)).amax(1)  # [H, n]
        shared_idx = score.argsort(-1, descending=True)
    for m in need & {"bern_simple_needle", "bern_tight_grad", "chern_grad", "mc_grad", "oracle"}:
        for e in eps0s:
            out[(m, e)] = [torch.zeros(H, R, n, dtype=torch.bool, device=dev),
                           torch.zeros(H, R, dtype=torch.long, device=dev), torch.zeros(H, R, dtype=torch.bool, device=dev)]
    rc = max(1, args.chunk // (H * n))
    for a in range(0, R, rc):
        sl = slice(a, a + rc)
        wc, sc, Bc = w[:, sl], sure[:, sl], B[:, sl]
        pg = algo.prepare(wc, r3, tail3, sc, delta, trunc=True, order="grad")
        T_full[:, sl] = pg.T
        if keep_order:
            order_g[:, sl] = pg.order.int()
            cost_c = dl.expected_cost(pg, r3, tail3)

        def cand(kind, e, Pc):
            key = (kind, e)
            P_lad[key][:, sl], cost_lad[key][:, sl], k_lad[key][:, sl] = dl.with_row_k(
                Pc, cost_c, lad, dl.first_pass(Pc, delta, pg.n_tail))

        def put(m, e, plan, k):
            k = torch.minimum(k, plan.n_tail)
            out[(m, e)][0][:, sl] = algo.fetch_mask(plan, k) & tail3
            out[(m, e)][1][:, sl] = k

        if "bern_simple_needle" in need:
            pn = algo.prepare(wc, r3, tail3, sc, delta, trunc=True, order="needle")
            for e in eps0s:
                put("bern_simple_needle", e, pn, algo.kstar(pn, e, "simple")[0])
            del pn
        if need & {"bern_tight_grad", "avg_tight"}:
            for e in eps0s:
                Pc = pg.p_fail_tight(e)
                if "bern_tight_grad" in need:
                    put("bern_tight_grad", e, pg, dl.first_pass(Pc, delta, pg.n_tail))
                if "avg_tight" in need:
                    cand("tight", e, Pc)
        if need & {"chern_grad", "avg_chern"}:
            for e, Pc in zip(eps0s, dl.chernoff_curves(pg, wc, r3, tail3, eps0s)):
                if "chern_grad" in need:
                    put("chern_grad", e, pg, dl.first_pass(Pc, delta, pg.n_tail))
                if "avg_chern" in need:
                    cand("chern", e, Pc)
        if need & {"mc_grad", "avg_mc"}:
            for e, Pc in zip(eps0s, dl.mc_curves(pg, wc, r3, tail3, sc, Bc, eps0s, args.mc_draws, gen)):
                if "mc_grad" in need:
                    put("mc_grad", e, pg, dl.first_pass(Pc, delta, pg.n_tail))
                if "avg_mc" in need:
                    cand("mc", e, Pc)
        if "shared_tight" in need:
            ps = dl.prepare_order(wc, r3, tail3, sc, delta, shared_idx[:, None, :].expand(-1, wc.shape[1], -1))
            for e in eps0s:
                kshare[e][:, sl] = dl.first_pass(ps.p_fail_tight(e), delta, ps.n_tail)
            del ps
        if "oracle" in need:
            no = torch.zeros_like(pg.T)
            for e in eps0s:
                err0 = algo.estimate(wc, r3, tail3, sc, Z3, pg.T, no) - Bc
                curve = algo.error_curve(pg, wc, r3, tail3, Z3, err0)
                put("oracle", e, pg, algo.oracle_k(curve, Bc, e, pg.n_tail))
    n_tail = tail.sum(-1)[:, None]  # [H, 1]
    for m in need & set(AVG):  # Remark 2 allocation over every row of this (forward, layer)
        for e in eps0s:
            key = (AVG[m], e)
            P, Lk = P_lad[key], P_lad[key].shape[-1]
            idx = dl.allocate(P.reshape(-1, Lk), cost_lad[key].reshape(-1, Lk), delta).view(H, R)
            k = torch.minimum(k_lad[key].gather(-1, idx[..., None])[..., 0], n_tail)
            give = P.gather(-1, idx[..., None])[..., 0] > delta
            F = torch.zeros(H, R, n, dtype=torch.bool, device=dev)
            for a in range(0, R, rc):
                sl = slice(a, a + rc)
                F[:, sl] = (rank_of(order_g[:, sl].long()) < k[:, sl, None]) & tail3
            out[(m, e)] = [F, k, give]
    if "shared_tight" in need:
        rk = rank_of(shared_idx)  # [H, n]
        for e in eps0s:
            kh = torch.minimum(kshare[e].amax(1), n_tail[:, 0])  # [H]
            F = ((rk < kh[:, None]) & tail)[:, None].expand(H, R, n).clone()
            out[("shared_tight", e)] = [F, kh[:, None].expand(H, R).clone(), torch.zeros(H, R, dtype=torch.bool, device=dev)]
    if "needle" in need:
        G = R // q_len
        frac = w / w.sum(-1, keepdim=True)
        flag = (tail3 & (frac >= theta)).view(H, G, q_len, n).any(1, keepdim=True).expand(H, G, q_len, n)
        F = flag.reshape(H, R, n).clone()
        for e in eps0s:
            out[("needle", e)] = [F, F.sum(-1), torch.zeros(H, R, dtype=torch.bool, device=dev)]
    return out, T_full, sure, B


def evaluate(out, T_full, sure, B, w, r, tail, Z, P, acc, phase, lj):
    """Realised errors and transfers for every (method, eps0); persist the union of F into P[(m, e)]."""
    r3, tail3, Z3 = r[:, None], tail[:, None], Z[:, None]
    n = w.shape[-1]
    no = torch.zeros_like(T_full)
    r0 = tail3 & (r3 <= 0)
    e_no = {}
    for (m, e), (F, k, give) in out.items():
        T = T_full if m in TRUNC else r0.expand_as(T_full)
        key = m in TRUNC
        if key not in e_no:
            e_no[key] = (algo.estimate(w, r3, tail3, sure, Z3, T, no) - B).abs() / B
        err = (algo.estimate(w, r3, tail3, sure, Z3, T, F) - B).abs() / B
        xf = F & ~Z3
        u = xf.any(1).sum(-1).double()  # per KV head: new entries read from the host for this forward
        a = acc[f"{m}|{e}"][phase, lj]
        a[MI["n"]] += err.numel()
        a[MI["fetch"]] += float((k > 0).sum())
        a[MI["k"]] += float(k.sum())
        a[MI["xfer"]] += float(xf.sum())
        a[MI["fail"]] += float((err > e).sum())
        a[MI["fail0"]] += float((e_no[key] > e).sum())
        a[MI["giveup"]] += float(give.sum())
        a[MI["fwd"]] += u.numel()
        a[MI["fwd_any"]] += float((u > 0).sum())
        a[MI["fwd_xfer"]] += float(u.sum())
        a[MI["fwd_ctx"]] += u.numel() * n
        P[f"{m}|{e}"] |= F.any(1)


si, sk = (int(x) for x in args.shard.split("/"))
files = sorted(glob.glob(f"{args.xdir}/*.pt"))[: args.limit or None][si::sk]
T0 = time.time()
for f in files:
    tag = Path(f).stem
    if (OUT / f"{tag}.pt").exists():
        print(f"{tag}: exists, skipping", flush=True)
        continue
    blob = torch.load(f, map_location="cpu")
    n, nq, H = blob["n"], blob["nq"], blob["n_kv"]
    G = blob["n_heads"] // H
    lidx = list(range(len(blob["layers"]))) if args.layers == "all" else [int(x) for x in args.layers.split(",")]
    steps_total = 0 if blob["q_decode"] is None else blob["q_decode"].shape[2]
    steps = steps_total if not args.max_steps else min(args.max_steps, steps_total)
    r_all = {res: algo.residency(blob["tau_kvzip"].double(), 1 / float(res[1:]), args.sinks, args.window)
             for res in args.res}
    keys_s = [f"{m}|{e}" for m in args.methods for e in args.eps0]
    acc = {res: {k: torch.zeros(2, len(lidx), len(METRICS), dtype=F64) for k in keys_s} for res in args.res}
    persist = {res: {k: torch.zeros(len(lidx), dtype=F64) for k in keys_s} for res in args.res}
    print(f"{tag}: n={n} nq={nq} steps {steps}/{steps_total} layers {len(lidx)} methods {args.methods}", flush=True)
    for lj, li in enumerate(lidx):
        t_l = time.time()
        keys = blob["keys"][li].to(dev).float()  # [H, n, d]
        kx = blob["keys_extra"][li].to(dev).float()
        d = keys.shape[-1]
        qq = blob["q_question"][li].to(dev).float().view(H, G, nq, d).reshape(H, G * nq, d)
        tq = torch.arange(nq, device=dev).repeat(G)
        qd = None if steps == 0 else blob["q_decode"][li, :, :steps].to(dev).float().view(H, G, steps, d)
        wq, wxq = weights(qq, keys, kx, tq)
        for res in args.res:
            r = r_all[res][li].to(dev)  # [H, n]
            seed = zlib.crc32(f"{tag}|{li}|{res}".encode())
            gz = torch.Generator(device=dev).manual_seed(seed)
            gmc = torch.Generator(device=dev).manual_seed(seed ^ 0x5EED)
            Z = torch.rand(r.shape, generator=gz, dtype=F64, device=dev) < r
            tail0 = r < 1
            P = {k: torch.zeros(H, n, dtype=torch.bool, device=dev) for k in keys_s}
            out, T_full, sure, B = forward(wq, wxq, r, tail0, Z, nq, args.methods, args.eps0, args.theta[0], gmc)
            evaluate(out, T_full, sure, B, wq, r, tail0, Z, P, acc[res], 0, lj)
            del out, T_full
            for j in range(steps):
                wd, wxd = weights(qd[:, :, j], keys, kx, torch.full((G,), nq + j, device=dev))
                for m in args.methods:
                    for e in args.eps0:
                        tail = tail0 & ~P[f"{m}|{e}"]
                        o, T_f, su, Bd = forward(wd, wxd, r, tail, Z, 1, [m], [e], args.theta[1], gmc)
                        evaluate(o, T_f, su, Bd, wd, r, tail, Z, P, acc[res], 1, lj)
            for k in keys_s:  # extra resident entries at the end of the answer (persisted ones the draw had dropped)
                persist[res][k][lj] += float((P[k] & ~Z).sum()) / (H * n)
        del wq, wxq
        print(f"  layer {blob['layers'][li]} ({lj + 1}/{len(lidx)}): {time.time() - t_l:.1f}s "
              f"(total {time.time() - T0:.0f}s)", flush=True)
    torch.save({"acc": acc, "persist": persist, "metrics": METRICS, "methods": args.methods, "eps0": args.eps0,
                "delta": args.delta, "res": args.res, "layers": [blob["layers"][i] for i in lidx], "task": blob["task"],
                "row": blob["row"], "n": n, "nq": nq, "steps": steps, "n_kv": H, "G": G, "mc_draws": args.mc_draws,
                "theta": args.theta, "sinks": args.sinks, "window": args.window}, OUT / f"{tag}.pt")
    print(f"{tag}: done ({time.time() - T0:.0f}s)", flush=True)
    del blob
