# Verbatim copy of verifiedkv/chernoff.py.
# the certificate of vkv_final_algo); only the import of algo.py is adapted. Ground truth for
# offload_package.vkv; do not "optimize" this file.

"""The certificate of vkv_final_algo: the smallest number k of entries to fetch, along the plan's fetch order, such that
P(|B_hat - B| > eps0 B) <= delta. B is the softmax denominator of one attention row; B_hat is its estimate from the
resident sample, the truncated set T* and the fetched entries (see ../vkv1/algo.py and ../VKV_FINAL_ALGO.md).

Two functions are used by the press:

kstar_fast  (default) the same certificate, computed faster (validated: 0 invalid decisions on real extracts):
    1. Screen: rows whose one-sided Bernstein bound (algo.Plan.p_fail_tight, a valid bound) already passes at k = 0
       get k = 0; no Chernoff curve is computed for them.
    2. Head + Bennett: exact two-point terms only for the first ``head`` (16384) entries of the fetch order; the rest
       enter through Bennett's bound on their log-MGF, which bounds the exact one from above, so the curve is still
       a valid (slightly larger) bound. Rows that do not pass within the head get the full curve.
    3. Per-token log-MGF terms in float32 (float64 is slow on consumer GPUs); suffix sums, exponents and the minimum
       over lambda stay in float64, and the pass test uses delta (1 - 1e-3) to cover the rounding.
chernoff_curves  the exact curve in float64 (used with ``fast=False`` and by the tests as the reference).

The bound itself (per row, entries in fetch order, the first k fetched): the estimate's error is a sum of independent
two-point variables X_i = c_i (Z_i - r_i) (Z_i ~ Bernoulli(r_i) the draw; c_i = w_i on T*, w_i / r_i elsewhere) plus a
known bias beta(k). Chernoff: P(sum X > u) <= min_lambda exp(K(lambda) - lambda u), K the exact log-MGF
sum_i log(r_i e^{lambda c_i (1 - r_i)} + (1 - r_i) e^{-lambda c_i r_i}); the two sides use u = eps0 B~ +- beta(k), and
lambda runs over a 32-point grid with lambda B~ in [1e-1, 1e4.5]. Any k chosen from these curves depends only on the
plan weights and the keep probabilities, never on the draw, which is what makes the per-decision guarantee hold.

"""


import torch

from . import reference as algo

F64 = torch.float64
GRID = torch.logspace(-1, 4.5, 32, dtype=F64)  # lambda * B~


def token_terms(wh, r, tail, T):
    """Per-entry (b, v, up, dn, c): bias, variance, largest up / down moves, Chernoff scale (0 off the random part)."""
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


def first_pass(curve, thr, n_tail):
    """First k with curve[k] <= thr (k = n_tail always passes: nothing random is left)."""
    ok = curve <= thr
    return torch.minimum(ok.to(torch.int8).argmax(-1), n_tail)


def chernoff_curves(plan: algo.Plan, wh, r, tail, eps0s, grid=GRID):
    """Exact-MGF Chernoff failure bound along the plan's order, one [..., n+1] curve per eps0 (float64)."""
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


def chern_curve_rows(T, order, Bt, beta, wh, r, tail, eps0, grid=GRID, dtype=torch.float32):
    """Chernoff failure curve [N, n+1] for N independent rows (all tensors already row-selected, leading dim N)."""
    _, _, _, _, c = token_terms(torch.where(tail, wh, torch.zeros((), dtype=wh.dtype, device=wh.device)), r, tail, T)
    c_s = c.gather(-1, order).to(dtype)
    r_s = r.gather(-1, order)
    live = c_s > 0
    lr = torch.log(r_s.clamp(1e-300, 1.0)).to(dtype)
    l1r = torch.log1p(-r_s.clamp(0.0, 1 - 1e-16)).to(dtype)
    r_s = r_s.to(dtype)
    Bt = Bt[:, None]
    u_up = eps0 * Bt + beta
    u_dn = eps0 * Bt - beta
    best_up = torch.ones_like(beta)
    best_dn = torch.ones_like(beta)
    zero = torch.zeros((), dtype=dtype, device=wh.device)
    for g in grid.tolist():
        lam = g / Bt  # float64 [N, 1]
        a = lam.to(dtype) * c_s
        k_up = torch.where(live, torch.logaddexp(lr + a * (1 - r_s), l1r - a * r_s), zero)
        k_dn = torch.where(live, torch.logaddexp(lr - a * (1 - r_s), l1r + a * r_s), zero)
        K_up, K_dn = algo._suffix_sum(k_up.to(F64)), algo._suffix_sum(k_dn.to(F64))
        best_up = torch.minimum(best_up, torch.exp((K_up - lam * u_up).clamp(max=0.0)))
        t_dn = torch.where(u_dn > 0, torch.exp((K_dn - lam * u_dn).clamp(max=0.0)), torch.ones_like(u_dn))
        best_dn = torch.minimum(best_dn, t_dn)
    return best_up + best_dn


def chern_curve_head(T, order, Bt, beta, V, up, dn, wh, r, tail, eps0, head, grid=GRID, dtype=torch.float32):
    """Chernoff curve for k = 0..head only [N, head+1]: exact two-point terms for the first ``head`` entries of the
    order, Bennett for the rest: with V_rest and the up / down maximal moves M of those entries,
    log E exp(+-lam X_rest) <= V_rest (exp(lam M) - 1 - lam M) / M^2, an upper bound on the true log-MGF, so the curve
    is a valid (slightly larger) failure bound. V / up / dn are the plan's suffix curves [N, n+1]."""
    n = wh.shape[-1]
    h = min(head, n)
    _, _, _, _, c = token_terms(torch.where(tail, wh, torch.zeros((), dtype=wh.dtype, device=wh.device)), r, tail, T)
    oh = order[:, :h]
    c_s = c.gather(-1, oh).to(dtype)
    r_s = r.gather(-1, oh)
    live = c_s > 0
    lr = torch.log(r_s.clamp(1e-300, 1.0)).to(dtype)
    l1r = torch.log1p(-r_s.clamp(0.0, 1 - 1e-16)).to(dtype)
    r_s = r_s.to(dtype)
    Bt = Bt[:, None]
    Vr, Mu, Md = V[:, h:h + 1], up[:, h:h + 1].clamp_min(1e-300), dn[:, h:h + 1].clamp_min(1e-300)
    bh = beta[:, : h + 1]
    u_up = eps0 * Bt + bh
    u_dn = eps0 * Bt - bh
    best_up = torch.ones_like(bh)
    best_dn = torch.ones_like(bh)
    zero = torch.zeros((), dtype=dtype, device=wh.device)

    def bennett(lam, M):
        x = (lam * M).clamp(max=700.0)
        return Vr * torch.expm1(x).sub(x) / (M * M)

    for g in grid.tolist():
        lam = g / Bt  # float64 [N, 1]
        a = lam.to(dtype) * c_s
        k_up = torch.where(live, torch.logaddexp(lr + a * (1 - r_s), l1r - a * r_s), zero)
        k_dn = torch.where(live, torch.logaddexp(lr - a * (1 - r_s), l1r + a * r_s), zero)
        K_up = algo._suffix_sum(k_up.to(F64)) + bennett(lam, Mu)
        K_dn = algo._suffix_sum(k_dn.to(F64)) + bennett(lam, Md)
        best_up = torch.minimum(best_up, torch.exp((K_up - lam * u_up).clamp(max=0.0)))
        t_dn = torch.where(u_dn > 0, torch.exp((K_dn - lam * u_dn).clamp(max=0.0)), torch.ones_like(u_dn))
        best_dn = torch.minimum(best_dn, t_dn)
    return best_up + best_dn


def kstar_fast(plan: algo.Plan, wh, r, tail, eps0: float, delta: float, margin: float = 1e-3, screen: bool = True,
               dtype=torch.float32, head: int = 16384):
    """k* per row [...] for chern-grad: Bernstein screen at k = 0, then the float32-term Chernoff curve on the rest.
    wh [..., n]; r / tail broadcast to it. Returns (k, n_screened, n_curved)."""
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
        if head and head < n:  # exact terms on the first `head` entries of the order, Bennett on the rest
            P = chern_curve_head(*args, flat(plan.V), flat(plan.up), flat(plan.dn), *rows, eps0, head, dtype=dtype)
            ok = P <= thr
            found = ok.any(-1)
            kk = torch.minimum(ok.to(torch.int8).argmax(-1), nt)
            miss = (~found).nonzero().squeeze(-1)  # no pass within the head: the full curve for these rows
            if miss.numel():
                sub = lambda x: x[miss]  # noqa: E731
                Pm = chern_curve_rows(*map(sub, args), *map(sub, rows), eps0, dtype=dtype)
                kk[miss] = first_pass(Pm, thr, nt[miss])
        else:
            P = chern_curve_rows(*args, *rows, eps0, dtype=dtype)
            kk = first_pass(P, thr, nt)
        k.view(-1)[idx] = kk
    return k, int((~need).sum()), int(idx.numel())
