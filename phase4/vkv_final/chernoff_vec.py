"""chernoff.py (branch vkv_final_algo) with the 32-point lambda grid vectorised: the same arithmetic, computed for a
block of lambdas at once ([G, N, h] tensors) instead of one lambda per loop iteration. The profile showed decode is
launch-bound (2,279 CUDA kernels per decode-step layer, most of them the per-lambda loop); this cuts the launches by
the block size. Every element is computed by the same operations in the same dtype, suffix sums run along the last
dimension exactly as before, and the minimum over lambda is exact, so k* is meant to be bit-identical
(test_chernoff_vec.py checks it). COPY, 2026-10-27; chernoff.py itself is unchanged.
"""

import os
import sys

import torch

REPO_PHASE4 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # phase4/ of this repository
sys.path.insert(0, REPO_PHASE4)
sys.path.insert(0, os.path.join(REPO_PHASE4, "vkv_final"))
from vkv1 import algo  # noqa: E402  (read-only)
import chernoff  # noqa: E402  (read-only: token_terms, first_pass, GRID)

F64 = torch.float64
GRID = chernoff.GRID
BLOCK = int(os.environ.get("VKV_LAMBDA_BLOCK", "8"))  # lambdas per batched step (memory: BLOCK x N x h)


def _curves(lr, l1r, r_s, c_s, live, Bt, u_up, u_dn, extra_up=None, extra_dn=None, grid=GRID, dtype=torch.float32):
    """min over the lambda grid of the two-sided Chernoff bound, batched over lambda blocks.
    lr / l1r / r_s / c_s / live [N, h]; Bt [N, 1] f64; u_up / u_dn [N, h+1] f64; extra_*(lam) -> [G, N, 1] f64 or None."""
    best_up = torch.ones_like(u_up)
    best_dn = torch.ones_like(u_dn)
    zero = torch.zeros((), dtype=dtype, device=c_s.device)
    g_all = grid.to(c_s.device)
    for b0 in range(0, g_all.numel(), BLOCK):
        lam = g_all[b0: b0 + BLOCK][:, None, None] / Bt[None]  # [G, N, 1] f64, = g / Bt per lambda
        a = lam.to(dtype) * c_s[None]
        k_up = torch.where(live[None], torch.logaddexp(lr[None] + a * (1 - r_s[None]), l1r[None] - a * r_s[None]), zero)
        k_dn = torch.where(live[None], torch.logaddexp(lr[None] - a * (1 - r_s[None]), l1r[None] + a * r_s[None]), zero)
        K_up, K_dn = algo._suffix_sum(k_up.to(F64)), algo._suffix_sum(k_dn.to(F64))
        if extra_up is not None:
            K_up = K_up + extra_up(lam)
            K_dn = K_dn + extra_dn(lam)
        t_up = torch.exp((K_up - lam * u_up[None]).clamp(max=0.0)).amin(0)
        t_dn = torch.where(u_dn[None] > 0, torch.exp((K_dn - lam * u_dn[None]).clamp(max=0.0)),
                           torch.ones_like(K_dn)).amin(0)
        best_up = torch.minimum(best_up, t_up)
        best_dn = torch.minimum(best_dn, t_dn)
    return best_up + best_dn


def _prep(T, order, wh, r, tail, dtype, h=None):
    _, _, _, _, c = chernoff.token_terms(torch.where(tail, wh, torch.zeros((), dtype=wh.dtype, device=wh.device)),
                                         r, tail, T)
    o = order if h is None else order[:, :h]
    c_s = c.gather(-1, o).to(dtype)
    r_s = r.gather(-1, o)
    live = c_s > 0
    lr = torch.log(r_s.clamp(1e-300, 1.0)).to(dtype)
    l1r = torch.log1p(-r_s.clamp(0.0, 1 - 1e-16)).to(dtype)
    return lr, l1r, r_s.to(dtype), c_s, live


def chern_curve_rows(T, order, Bt, beta, wh, r, tail, eps0, grid=GRID, dtype=torch.float32):
    lr, l1r, r_s, c_s, live = _prep(T, order, wh, r, tail, dtype)
    Bt = Bt[:, None]
    return _curves(lr, l1r, r_s, c_s, live, Bt, eps0 * Bt + beta, eps0 * Bt - beta, grid=grid, dtype=dtype)


def chern_curve_head(T, order, Bt, beta, V, up, dn, wh, r, tail, eps0, head, grid=GRID, dtype=torch.float32):
    n = wh.shape[-1]
    h = min(head, n)
    lr, l1r, r_s, c_s, live = _prep(T, order, wh, r, tail, dtype, h)
    Bt = Bt[:, None]
    Vr, Mu, Md = V[:, h:h + 1], up[:, h:h + 1].clamp_min(1e-300), dn[:, h:h + 1].clamp_min(1e-300)
    bh = beta[:, : h + 1]

    def bennett(M):
        def f(lam):
            x = (lam * M[None]).clamp(max=700.0)
            return Vr[None] * torch.expm1(x).sub(x) / (M[None] * M[None])
        return f

    return _curves(lr, l1r, r_s, c_s, live, Bt, eps0 * Bt + bh, eps0 * Bt - bh, bennett(Mu), bennett(Md),
                   grid=grid, dtype=dtype)


def kstar_fast(plan: algo.Plan, wh, r, tail, eps0: float, delta: float, margin: float = 1e-3, screen: bool = True,
               dtype=torch.float32, head: int = 16384):
    """chernoff.kstar_fast with the lambda-vectorised curves (same screen, head, fallback and margin)."""
    shape = plan.n_tail.shape
    n = wh.shape[-1]
    k = torch.zeros(shape, dtype=torch.long, device=wh.device)
    need = torch.ones(shape, dtype=torch.bool, device=wh.device)
    if screen:
        need = plan.p_fail_tight(eps0)[..., 0] > delta
    idx = need.reshape(-1).nonzero().squeeze(-1)
    if idx.numel():
        flat = lambda x: x.expand(*shape, *x.shape[len(shape):]).reshape(-1, *x.shape[len(shape):])[idx]  # noqa: E731
        args = (flat(plan.T), flat(plan.order), flat(plan.Bt[..., None])[:, 0], flat(plan.beta))
        rows = (flat(wh), flat(r.expand(*shape, n)), flat(tail.expand(*shape, n)))
        nt = plan.n_tail.reshape(-1)[idx]
        thr = delta * (1 - margin)
        if head and head < n:
            P = chern_curve_head(*args, flat(plan.V), flat(plan.up), flat(plan.dn), *rows, eps0, head, dtype=dtype)
            ok = P <= thr
            found = ok.any(-1)
            kk = torch.minimum(ok.to(torch.int8).argmax(-1), nt)
            miss = (~found).nonzero().squeeze(-1)
            if miss.numel():
                sub = lambda x: x[miss]  # noqa: E731
                Pm = chern_curve_rows(*map(sub, args), *map(sub, rows), eps0, dtype=dtype)
                kk[miss] = chernoff.first_pass(Pm, thr, nt[miss])
        else:
            P = chern_curve_rows(*args, *rows, eps0, dtype=dtype)
            kk = chernoff.first_pass(P, thr, nt)
        k.view(-1)[idx] = kk
    return k, int((~need).sum()), int(idx.numel())
