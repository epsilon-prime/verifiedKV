"""Diagnostic building blocks for vkv1_diag (where does Vkv1's fetch volume come from?). Uses vkv1/algo.py read-only;
everything new lives here. Shapes: leading dims are rows (decisions), last dim n is the context. float64.

Per-row failure curves P(k), k = 0..n (fetch the first k entries of a draw-free order, T* frozen):
    tight  : the draft's Part III one-sided Bernstein (algo.Plan.p_fail_tight)
    chern  : exact Chernoff on the two-point terms X_i = c_i (Z_i - r_i), c_i = w on T*, w / r on the HT part;
             P <= min_l exp(K_up(l) - l u+) + min_l exp(K_dn(l) - l u-), u+- = eps0 B~ +- beta(k), l on a grid
    mc     : Monte Carlo estimate of the true P(|B hat_k - B| > eps0 B) over fresh draws (not a bound: the best any
             per-decision draw-free rule could do, up to MC error)
Allocation (Remark 2): pick k_r per row on a ladder of k values to minimise the expected transfer sum_r E|F_r \\ S|
(= prefix sums of 1 - r, draw-free) subject to mean_r P_r(k_r) <= delta, by a Lagrangian bisection.
"""

import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.expanduser("~"), "phase4"))
from vkv1 import algo  # noqa: E402  (read-only)

F64 = torch.float64


def ladder(n: int, device) -> torch.Tensor:
    """k values 0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 13, 16, ... (factor 2^(1/4)) and n."""
    ks = {0, n}
    j = 0
    while True:
        k = int(round(2 ** (j / 4)))
        if k >= n:
            break
        ks.add(k)
        j += 1
    return torch.tensor(sorted(ks), device=device)


def token_terms(wh, r, tail, T):
    """Per-entry (b, v, up, dn, c): bias, variance, moves, Chernoff scale (0 off the random part)."""
    zero = torch.zeros((), dtype=wh.dtype, device=wh.device)
    pos = tail & (r > 0)
    rr = r.clamp_min(1e-300)
    Tp, Hp = T & pos, pos & ~T
    b = torch.where(T & tail, wh * (1 - r), zero)
    v = torch.where(Tp, wh * wh * r * (1 - r), torch.where(Hp, wh * wh * (1 - r) / rr, zero))
    up = torch.where(Tp, wh * (1 - r), torch.where(Hp, wh * (1 - r) / rr, zero))
    dn = torch.where(Tp, wh * r, torch.where(Hp, wh, zero))
    c = torch.where(Tp, wh, torch.where(Hp, wh / rr, zero))
    c = torch.where(r < 1, c, zero)
    return b, v, up, dn, c


def prepare_order(wh, r, tail, sure, delta, idx, trunc=True) -> algo.Plan:
    """algo.prepare with an externally given fetch order idx [..., n] (draw-free; non-tail entries should come last)."""
    zero = torch.zeros((), dtype=wh.dtype, device=wh.device)
    r = r.expand_as(wh) if r.shape != wh.shape else r
    tail = tail.expand_as(wh) if tail.shape != wh.shape else tail
    wh = torch.where(tail, wh, zero)
    L = float(math.log(2.0 / delta))
    T = algo.truncation(wh, r, tail, trunc)
    b, v, up, dn, _ = token_terms(wh, r, tail, T)
    pos = tail & (r > 0)
    idx = idx.expand_as(wh)
    return algo.Plan(T=T, order=idx, n_tail=tail.sum(-1), beta=algo._suffix_sum(b.gather(-1, idx)),
                     V=algo._suffix_sum(v.gather(-1, idx)), up=algo._suffix_max(up.gather(-1, idx)),
                     dn=algo._suffix_max(dn.gather(-1, idx)), Bt=sure + wh.sum(-1), L=L, delta=delta,
                     b_plus_before=torch.where(pos, wh * (1 - r) / r.clamp_min(1e-300), zero).amax(-1),
                     beta_T=b.sum(-1))


GRID = torch.logspace(-1, 4.5, 32, dtype=F64)  # lambda * B~


def chernoff_curves(plan: algo.Plan, wh, r, tail, eps0s, grid=GRID):
    """Exact-MGF Chernoff failure bound along the plan's order, one [..., n+1] curve per eps0."""
    r = r.expand_as(wh) if r.shape != wh.shape else r
    tail = tail.expand_as(wh) if tail.shape != wh.shape else tail
    wh = torch.where(tail, wh, torch.zeros((), dtype=wh.dtype, device=wh.device))
    _, _, _, _, c = token_terms(wh, r, tail, plan.T)
    c_s, r_s = c.gather(-1, plan.order), r.gather(-1, plan.order)
    live = c_s > 0
    lr = torch.log(r_s.clamp(1e-300, 1.0))
    l1r = torch.log1p(-r_s.clamp(0.0, 1 - 1e-16))
    Bt = plan.Bt[..., None]
    best_up = [torch.ones_like(plan.beta) for _ in eps0s]
    best_dn = [torch.ones_like(plan.beta) for _ in eps0s]
    for g in grid.tolist():
        lam = g / Bt
        a = lam * c_s
        k_up = torch.where(live, torch.logaddexp(lr + a * (1 - r_s), l1r - a * r_s), 0.0)
        k_dn = torch.where(live, torch.logaddexp(lr - a * (1 - r_s), l1r + a * r_s), 0.0)
        K_up, K_dn = algo._suffix_sum(k_up), algo._suffix_sum(k_dn)
        for i, e in enumerate(eps0s):
            u_up = e * Bt + plan.beta
            u_dn = e * Bt - plan.beta
            best_up[i] = torch.minimum(best_up[i], torch.exp((K_up - lam * u_up).clamp(max=0.0)))
            t_dn = torch.where(u_dn > 0, torch.exp((K_dn - lam * u_dn).clamp(max=0.0)), torch.ones_like(u_dn))
            best_dn[i] = torch.minimum(best_dn[i], t_dn)
    return [bu + bd for bu, bd in zip(best_up, best_dn)]


def mc_curves(plan: algo.Plan, w, r, tail, sure, B, eps0s, draws: int, gen: torch.Generator):
    """Monte Carlo P(|B hat_k - B| > eps0 B) along the plan's order over ``draws`` fresh draws (one draw per entry,
    shared by all rows that broadcast over it, as one KV head's draw is shared by its query heads)."""
    zshape = torch.broadcast_shapes(r.shape, tail.shape)
    no = torch.zeros_like(plan.T)
    fails = [torch.zeros_like(plan.beta, dtype=torch.int32) for _ in eps0s]
    for _ in range(draws):
        Z = torch.rand(zshape, generator=gen, dtype=F64, device=w.device) < r
        err0 = algo.estimate(w, r, tail, sure, Z, plan.T, no) - B
        curve = algo.error_curve(plan, w, r, tail, Z, err0).abs()
        for i, e in enumerate(eps0s):
            fails[i] += (curve > e * B[..., None]).int()
    return [f.double() / draws for f in fails]


def first_pass(curve, thr, n_tail):
    ok = curve <= thr
    return torch.minimum(ok.to(torch.int8).argmax(-1), n_tail)


def expected_cost(plan: algo.Plan, r, tail):
    """Expected transfer E|F_k \\ S| = sum of (1 - r) over the first k entries of the order, [..., n+1] (draw-free)."""
    r = r.expand_as(plan.T) if r.shape != plan.T.shape else r
    tail = tail.expand_as(plan.T) if tail.shape != plan.T.shape else tail
    x = torch.where(tail, 1 - r, torch.zeros((), dtype=r.dtype, device=r.device)).gather(-1, plan.order)
    return torch.cat([torch.zeros_like(x[..., :1]), x.cumsum(-1)], -1)


def with_row_k(curve, cost, lad, k_row):
    """Ladder samples of a curve and its cost plus one extra candidate per row at k_row (its own per-decision k*), so
    the allocation can always fall back to the per-decision solution. Returns P [..., Lk + 1], cost, k candidates."""
    kk = torch.cat([lad.expand(*k_row.shape, -1), k_row[..., None]], -1)
    return curve.gather(-1, kk), cost.gather(-1, kk), kk


def allocate(P_lad: torch.Tensor, cost_lad: torch.Tensor, delta: float, iters: int = 60) -> torch.Tensor:
    """Remark 2: candidate index per row minimising sum cost subject to mean P <= delta (Lagrangian bisection on mu).
    P_lad, cost_lad [N, Lk]; some candidate per row must have P = 0 (fetch the whole tail)."""
    P = P_lad.clamp(max=1.0)

    def choose(mu):
        return (cost_lad + mu * P).argmin(-1)

    def meanP(idx):
        return float(P.gather(-1, idx[:, None]).mean())

    i0 = torch.zeros(P.shape[0], dtype=torch.long, device=P.device)
    if meanP(i0) <= delta:
        return i0
    hi = 1.0
    while meanP(choose(hi)) > delta and hi < 1e15:
        hi *= 4
    lo = 0.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if meanP(choose(mid)) > delta else (lo, mid)
    return choose(hi)
