"""Where is the slack between the Chernoff bound and the true failure rate? One context, one layer, question pass.

At each row's chern k*, compare (per tail and summed): Chernoff, hx (exact heavy + Chernoff light), the true P by Monte
Carlo, the normal approximation with the same mean / variance, and the Berry-Esseen hybrid
    P(S_L > t) <= Q(t / sigma_L) + C rho_L / sigma_L^3,  C = 0.5600 (Shevtsova 2010, independent non-identical terms)
used in place of Chernoff for the light part. Also the k each rule would choose on the ladder.

    python -u slack.py <extract.pt> --layer 18 [--heads 0,1]
"""

import argparse
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hx_lib as hx  # noqa: E402
from hx_lib import algo, dl  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("x")
ap.add_argument("--layer", type=int, default=18)
ap.add_argument("--heads", default="0,1,2,3")
ap.add_argument("--eps0", type=float, default=0.3)
ap.add_argument("--delta", type=float, default=0.05)
ap.add_argument("--draws", type=int, default=2000)
args = ap.parse_args()
dev = "cuda"
F64 = torch.float64
blob = torch.load(args.x, map_location="cpu")
n, nq, H = blob["n"], blob["nq"], blob["n_kv"]
G = blob["n_heads"] // H
li = args.layer
hs = [int(h) for h in args.heads.split(",")]
r_all = algo.residency(blob["tau_kvzip"].double(), 1 / 16, 64, 64)
keys = blob["keys"][li].to(dev).float()[hs]
kx = blob["keys_extra"][li].to(dev).float()[hs]
d = keys.shape[-1]
qq = blob["q_question"][li].to(dev).float().view(H, G, nq, d).reshape(H, G * nq, d)[hs]
tq = torch.arange(nq, device=dev).repeat(G)
s = qq @ keys.transpose(-1, -2)
se = (qq @ kx.transpose(-1, -2)).masked_fill(torch.arange(kx.shape[1], device=dev)[None, None] > tq[None, :, None], -float("inf"))
m = torch.maximum(s.amax(-1), se.amax(-1))[..., None]
w, wx = torch.exp((s - m).double()), torch.exp((se - m).double()).sum(-1)
r = r_all[li].to(dev)[hs]
tail = r < 1
r3, t3 = r[:, None], tail[:, None]
sure = torch.where(t3, torch.zeros((), dtype=F64, device=dev), w).sum(-1) + wx
B = w.sum(-1) + wx
e, delta = args.eps0, args.delta
lad = dl.ladder(n, dev)
Hh, R = w.shape[:2]
res = {}
for a in range(0, R, 8):
    sl = slice(a, a + 8)
    wc, sc, Bc = w[:, sl], sure[:, sl], B[:, sl]
    pg = algo.prepare(wc, r3, t3, sc, delta, trunc=True, order="grad")
    ch = dl.chernoff_curves(pg, wc, r3, t3, [e])[0]
    kc = dl.first_pass(ch, delta, pg.n_tail)
    ks1 = torch.minimum(lad.expand(*pg.n_tail.shape, -1), pg.n_tail[..., None])
    P_ch_l = ch.gather(-1, ks1)
    P_hx_l = hx.hx_at(pg, wc, r3, t3, [e], ks1, 64, 512)[0]
    mc = dl.mc_curves(pg, wc, r3, t3, sc, Bc, [e], args.draws, torch.Generator(device=dev).manual_seed(a))[0]
    P_mc_l = mc.gather(-1, ks1)
    # normal approximation and Berry-Esseen on the whole live sum at each ladder k
    c_s, r_s = hx._sorted_terms(pg, wc, r3, t3)
    var = algo._suffix_sum(c_s ** 2 * r_s * (1 - r_s)).gather(-1, ks1)
    rho = algo._suffix_sum(c_s ** 3 * r_s * (1 - r_s) * ((1 - r_s) ** 2 + r_s ** 2)).gather(-1, ks1)
    beta = pg.beta.gather(-1, ks1)
    sig = var.sqrt().clamp_min(1e-300)
    u_up, u_dn = e * Bc[..., None] + beta, e * Bc[..., None] - beta
    Q = lambda x: 0.5 * torch.erfc(x / math.sqrt(2))  # noqa: E731
    P_norm_l = Q(u_up / sig) + torch.where(u_dn > 0, Q(u_dn / sig), torch.ones_like(u_dn))
    be = 0.56 * rho / sig ** 3
    P_be_l = torch.minimum(P_norm_l + 2 * be, torch.ones_like(be))
    for name, P in (("chern", P_ch_l), ("hx", P_hx_l), ("mc", P_mc_l), ("normal", P_norm_l), ("BE", P_be_l)):
        ok = P <= delta
        ok[..., -1] = True
        kk = ks1.gather(-1, ok.to(torch.int8).argmax(-1)[..., None])[..., 0]
        res.setdefault(f"k_{name}", []).append(kk.flatten())
    j = (ks1 <= kc[..., None]).sum(-1) - 1  # ladder index just at / below chern's k*
    for name, P in (("chern", P_ch_l), ("hx", P_hx_l), ("mc", P_mc_l), ("normal", P_norm_l), ("BE", P_be_l)):
        res.setdefault(f"P_{name}@kc", []).append(P.gather(-1, j[..., None])[..., 0].flatten())
    res.setdefault("be_term@kc", []).append(be.gather(-1, j[..., None])[..., 0].flatten())
res = {k: torch.cat(v).double() for k, v in res.items()}
fetch = res["k_chern"] > 0
print(f"{args.x} layer {li} heads {hs}: rows {res['k_chern'].numel()}, eps0 {e}, delta {delta}, MC draws {args.draws}")
print(f"rows where chern fetches: {int(fetch.sum())}")
for name in ("chern", "hx", "mc", "normal", "BE"):
    k = res[f"k_{name}"]
    print(f"  k*_{name:6}: mean {float(k.mean()):8.1f}   mean over chern-fetching rows {float(k[fetch].mean()):8.1f}   "
          f"share k>0 {float((k > 0).double().mean()):.3f}")
print("at chern's k* (ladder point at/below it), over chern-fetching rows: median / mean")
for name in ("chern", "hx", "mc", "normal", "BE"):
    p = res[f"P_{name}@kc"][fetch]
    print(f"  P_{name:6} {float(p.median()):.4f} / {float(p.mean()):.4f}")
p = res["be_term@kc"][fetch]
print(f"  Berry-Esseen term C rho/sigma^3 (one tail): median {float(p.median()):.4f}, mean {float(p.mean()):.4f}")
