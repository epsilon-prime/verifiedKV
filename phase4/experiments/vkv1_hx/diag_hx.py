"""vkv1_hx: options (1) hx bound and (2) capped draw-free store (theory: phase4/theory/hx_store_note.md). Built from a
copy of vkv1_nopersist/diag_np.py (no persistence unless a store is configured); existing code imported read-only.
Exact weights (perfect sketch), offline on the algo_improvements extracts, 64 sinks + 64-token window.

Methods (draw-free per-decision plans, grad order, T* by the MSE rule unless stated):
    chern_grad   first k with the exact two-point Chernoff bound <= delta (as in vkv1_diag)
    hx_grad      first k with the hx bound <= delta (heavy part exact by DP, light part Chernoff; ladder + refine)
    mc_grad      first k with Monte Carlo P(fail) <= delta (the ideal per-decision draw-free rule; not a bound)
    needle       needle fetch's flag rule on exact weights (theta 0.003 question, 0.1 decode), no truncation
    oracle       smallest k meeting eps0 for the ACTUAL draw (a lower bound, not an algorithm)
Residencies (--res): "g16" = draw at 1/16, no store (== vkv1_nopersist); "g16s0.01" = draw at 1/16 - 0.01 plus a store
of 0.01 n entries per (layer, KV head), filled draw-free from this answer's fetch sets (hx_lib.store_update), emptied
per question. Memory is <= 1/16 n in both (plus one forward's fetch buffer).

    python -u diag_hx.py <xdir> --out out/<run> [--layers 0,18] [--max-steps 8] [--shard i/k]
"""

import argparse
import glob
import os
import re
import sys
import time
import zlib
from pathlib import Path

import torch

HOME = os.path.expanduser("~")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hx_lib as hx  # noqa: E402
from hx_lib import algo, dl  # noqa: E402  (read-only)

ALL = ("chern_grad", "hx_grad", "mc_grad", "needle", "oracle")
ap = argparse.ArgumentParser()
ap.add_argument("xdir")
ap.add_argument("--out", required=True)
ap.add_argument("--res", nargs="+", default=["g16", "g16s0.01"])
ap.add_argument("--eps0", type=float, nargs="+", default=[0.3, 0.5])
ap.add_argument("--delta", type=float, default=0.05)
ap.add_argument("--methods", nargs="+", default=list(ALL))
ap.add_argument("--mc-draws", type=int, default=100)
ap.add_argument("--hx-m", type=int, default=64, help="heavy terms handled exactly")
ap.add_argument("--hx-bins", type=int, default=512)
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
METRICS = ("n", "fetch", "k", "xfer", "fail", "fail0", "giveup", "fwd", "fwd_any", "fwd_xfer", "fwd_ctx", "fwd_peak",
           "bound", "store_occ")
MI = {m: i for i, m in enumerate(METRICS)}
TRUNC = {m for m in ALL if m != "needle"}


def parse_res(res):
    mt = re.fullmatch(r"g(\d+)(?:s([0-9.]+))?", res)
    assert mt, res
    return 1 / float(mt.group(1)), float(mt.group(2) or 0.0)


def weights(q, keys, kx, tpos):
    """q [H, R, d], keys [H, n, d], kx [H, E, d], tpos [R] -> w [H, R, n], wx [H, R] (shared shift per row)."""
    s = q @ keys.transpose(-1, -2)
    se = (q @ kx.transpose(-1, -2)).masked_fill(torch.arange(kx.shape[1], device=dev)[None, None] > tpos[None, :, None],
                                               -float("inf"))
    m = torch.maximum(s.amax(-1), se.amax(-1))[..., None]
    return torch.exp((s - m).double()), torch.exp((se - m).double()).sum(-1)


def forward(w, wx, r, tail, Z, q_len, methods, eps0s, theta, gen):
    """One forward over the cache for one layer: w [H, R, n] (row = g * q_len + t), r / tail / Z [H, n].
    Returns {(method, eps0): [F [H, R, n], k [H, R], bound at k [H, R]]}, T* [H, R, n], sure, B."""
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
    for m in need - {"needle"}:
        for e in eps0s:
            out[(m, e)] = [torch.zeros(H, R, n, dtype=torch.bool, device=dev),
                           torch.zeros(H, R, dtype=torch.long, device=dev), torch.zeros(H, R, dtype=F64, device=dev)]
    rc = max(1, args.chunk // (H * n))
    for a in range(0, R, rc):
        sl = slice(a, a + rc)
        wc, sc, Bc = w[:, sl], sure[:, sl], B[:, sl]
        pg = algo.prepare(wc, r3, tail3, sc, delta, trunc=True, order="grad")
        T_full[:, sl] = pg.T

        def put(m, e, k, p):
            k = torch.minimum(k, pg.n_tail)
            out[(m, e)][0][:, sl] = algo.fetch_mask(pg, k) & tail3
            out[(m, e)][1][:, sl] = k
            out[(m, e)][2][:, sl] = p

        if "chern_grad" in need:
            for e, Pc in zip(eps0s, dl.chernoff_curves(pg, wc, r3, tail3, eps0s)):
                k = dl.first_pass(Pc, delta, pg.n_tail)
                put("chern_grad", e, k, algo.curve_at(Pc, k))
        if "hx_grad" in need:
            ks, ps = hx.hx_kstar(pg, wc, r3, tail3, eps0s, delta, lad, args.hx_m, args.hx_bins)
            for e, k, p in zip(eps0s, ks, ps):
                put("hx_grad", e, k, p)
        if "mc_grad" in need:
            for e, Pc in zip(eps0s, dl.mc_curves(pg, wc, r3, tail3, sc, Bc, eps0s, args.mc_draws, gen)):
                k = dl.first_pass(Pc, delta, pg.n_tail)
                put("mc_grad", e, k, algo.curve_at(Pc, k))
        if "oracle" in need:
            no = torch.zeros_like(pg.T)
            err0 = algo.estimate(wc, r3, tail3, sc, Z3, pg.T, no) - Bc
            curve = algo.error_curve(pg, wc, r3, tail3, Z3, err0)
            for e in eps0s:
                put("oracle", e, algo.oracle_k(curve, Bc, e, pg.n_tail), torch.zeros_like(Bc))
    if "needle" in need:
        G = R // q_len
        frac = w / w.sum(-1, keepdim=True)
        flag = (tail3 & (frac >= theta)).view(H, G, q_len, n).any(1, keepdim=True).expand(H, G, q_len, n)
        F = flag.reshape(H, R, n).clone()
        for e in eps0s:
            out[("needle", e)] = [F, F.sum(-1), torch.zeros(H, R, dtype=F64, device=dev)]
    return out, T_full, sure, B


def evaluate(out, T_full, sure, B, w, r, tail, Z, acc, phase, lj, occ):
    """Realised errors and transfers for every (method, eps0). occ[(m, e)]: store entries the draw had dropped (the
    store's real memory use) as a fraction of n, max over KV heads."""
    r3, tail3, Z3 = r[:, None], tail[:, None], Z[:, None]
    n = w.shape[-1]
    no = torch.zeros_like(T_full)
    r0 = tail3 & (r3 <= 0)
    e_no = {}
    for (m, e), (F, k, p) in out.items():
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
        a[MI["fwd"]] += u.numel()
        a[MI["fwd_any"]] += float((u > 0).sum())
        a[MI["fwd_xfer"]] += float(u.sum())
        a[MI["fwd_ctx"]] += u.numel() * n
        a[MI["fwd_peak"]] = max(float(a[MI["fwd_peak"]]), float(u.max()) / n)
        a[MI["bound"]] += float(p.clamp(max=1).sum())  # the certified P(fail) at the chosen k (chern / hx / mc)
        a[MI["store_occ"]] = max(float(a[MI["store_occ"]]), occ.get((m, e), 0.0))


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
    cfg = {res: parse_res(res) for res in args.res}
    r_all = {res: algo.residency(blob["tau_kvzip"].double(), kap - rho, args.sinks, args.window)
             for res, (kap, rho) in cfg.items()}
    keys_s = [f"{m}|{e}" for m in args.methods for e in args.eps0]
    acc = {res: {k: torch.zeros(2, len(lidx), len(METRICS), dtype=F64) for k in keys_s} for res in args.res}
    print(f"{tag}: n={n} nq={nq} steps {steps}/{steps_total} layers {len(lidx)} methods {args.methods} res {cfg}",
          flush=True)
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
            kap, rho = cfg[res]
            cap = int(rho * n)
            r = r_all[res][li].to(dev)  # [H, n]
            seed = zlib.crc32(f"{tag}|{li}|{res}".encode())
            gz = torch.Generator(device=dev).manual_seed(seed)
            gmc = torch.Generator(device=dev).manual_seed(seed ^ 0x5EED)
            Z = torch.rand(r.shape, generator=gz, dtype=F64, device=dev) < r
            tail0 = r < 1
            # question pass: the store is empty, so every method shares one forward
            out, T_full, sure, B = forward(wq, wxq, r, tail0, Z, nq, args.methods, args.eps0, args.theta[0], gmc)
            evaluate(out, T_full, sure, B, wq, r, tail0, Z, acc[res], 0, lj, {})
            store = {}
            if cap > 0:
                for (m, e), (F, _, _) in out.items():
                    store[(m, e)] = hx.store_update(torch.zeros(H, n, dtype=torch.bool, device=dev),
                                                    torch.zeros(H, n, dtype=F64, device=dev), F, wq, B, tail0, cap)
            del out, T_full
            for j in range(steps):
                wd, wxd = weights(qd[:, :, j], keys, kx, torch.full((G,), nq + j, device=dev))
                if cap == 0:  # no store: one shared forward per step
                    o, T_f, su, Bd = forward(wd, wxd, r, tail0, Z, 1, args.methods, args.eps0, args.theta[1], gmc)
                    evaluate(o, T_f, su, Bd, wd, r, tail0, Z, acc[res], 1, lj, {})
                    continue
                for m in args.methods:
                    for e in args.eps0:
                        S, prio = store[(m, e)]
                        tail = tail0 & ~S
                        occ = {(m, e): float((S & ~Z).sum(-1).max()) / n}
                        o, T_f, su, Bd = forward(wd, wxd, r, tail, Z, 1, [m], [e], args.theta[1], gmc)
                        evaluate(o, T_f, su, Bd, wd, r, tail, Z, acc[res], 1, lj, occ)
                        store[(m, e)] = hx.store_update(S, prio, o[(m, e)][0], wd, Bd, tail0, cap)
        del wq, wxq
        print(f"  layer {blob['layers'][li]} ({lj + 1}/{len(lidx)}): {time.time() - t_l:.1f}s "
              f"(total {time.time() - T0:.0f}s)", flush=True)
    torch.save({"acc": acc, "metrics": METRICS, "methods": args.methods, "eps0": args.eps0, "delta": args.delta,
                "res": args.res, "res_cfg": cfg, "layers": [blob["layers"][i] for i in lidx], "task": blob["task"],
                "row": blob["row"], "n": n, "nq": nq, "steps": steps, "n_kv": H, "G": G, "mc_draws": args.mc_draws,
                "hx_m": args.hx_m, "hx_bins": args.hx_bins, "theta": args.theta, "sinks": args.sinks,
                "window": args.window}, OUT / f"{tag}.pt")
    print(f"{tag}: done ({time.time() - T0:.0f}s)", flush=True)
    del blob
