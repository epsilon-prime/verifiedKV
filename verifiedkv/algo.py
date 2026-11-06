"""Vkv1 core: truncated Horvitz-Thompson estimation of the attention denominator with a Bernstein certificate and a
certified fetch sweep (the paper, Algorithm 2). Pure torch, no kvpress.

Everything is vectorised over leading "row" dimensions: a row is one decision (layer, query head, query position).
Entry tensors are [..., n] over the context; ``r`` and ``tail`` broadcast against them. Use float64.

Notation (draft): tail L = entries with r < 1 that are not exact for another reason (persisted fetches); the plan
weights ``wh`` (sketch, or exact weights for a perfect sketch) and ``r`` decide everything; the draw Z is only used by
``estimate`` / ``attention_bias`` (draft section 2, design rule).

Entries with r = 0 (KVzip scores that underflow to zero) are never sampled, so every estimator counts them as 0:
they sit in T* permanently with bias w, no variance and no moves (the draft assumes r > 0; t4_mse_fetch.py does the
same). They come first in the truncation order and can still be fetched.

    plan = prepare(wh, r, tail, sure, delta, trunc=True, order="needle")
    k = kstar(plan, eps0, check="simple")      # or "tight" (draft Part III)
    F = fetch_mask(plan, k)
    bias = attention_bias(r, tail, Z, plan.T, F)
"""

import math
from dataclasses import dataclass

import torch

ORDERS = ("needle", "grad")
CHECKS = ("simple", "tight")


def _keep_prob(scores: torch.Tensor, c: float) -> torch.Tensor:
    """r = min(1, c s) (BernoulliPress._keep_prob)."""
    return (c * scores).clamp(max=1.0)


def _solve_c(scores: torch.Tensor, budget: float, iters: int = 100) -> float:
    """Smallest c with sum min(1, c s) >= budget, by bisection in log c (BernoulliPress._solve_c, verbatim)."""
    positive = scores[scores > 0]
    budget = min(float(budget), float(positive.numel()))
    assert budget > 0, "the budget must be positive"
    lo, hi = 1e-12 / positive.max().item(), 1.0 / positive.min().item()
    if _keep_prob(scores, lo).sum().item() >= budget:
        return lo
    for _ in range(iters):
        mid = math.sqrt(lo * hi)
        if _keep_prob(scores, mid).sum().item() >= budget:
            hi = mid
        else:
            lo = mid
    return hi


def residency(tau: torch.Tensor, kept: float, n_sink: int = 64, window: int = 64) -> torch.Tensor:
    """Algorithm 1, line 2-3: keep probabilities r [..., n] with mean ``kept`` over ALL entries (one global c over every
    leading index, i.e. all layers and KV heads), r = 1 on the first ``n_sink`` and last ``window`` entries; c is
    lowered until the forced entries are paid from the same budget. With window = 0 and KVzip's 4 sinks this is
    BernoulliPress.compress_post's r exactly. tau: KVzip scores in [0, 1], float64."""
    flat = tau.reshape(-1)
    budget = kept * flat.numel()

    def at(c):
        r = _keep_prob(flat, c).view(tau.shape).clone()
        r[..., :n_sink] = 1.0
        if window:
            r[..., -window:] = 1.0
        return r

    c = _solve_c(flat, budget)
    r = at(c)
    if r.sum().item() > budget:
        lo, hi = 0.0, c
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            lo, hi = (lo, mid) if at(mid).sum().item() > budget else (mid, hi)
        r = at(hi)
    return r


def _suffix_sum(x: torch.Tensor) -> torch.Tensor:
    """[..., n] -> [..., n + 1]: entry k = sum of x[k:] (0 at k = n). Computed from the end, so it is exactly
    non-increasing in k for non-negative x (no cancellation from total - prefix)."""
    s = x.flip(-1).cumsum(-1).flip(-1)
    return torch.cat([s, torch.zeros_like(s[..., :1])], -1)


def _suffix_max(x: torch.Tensor) -> torch.Tensor:
    """[..., n] -> [..., n + 1]: entry k = max of x[k:] (0 at k = n); x >= 0."""
    s = x.flip(-1).cummax(-1).values.flip(-1)
    return torch.cat([s, torch.zeros_like(s[..., :1])], -1)


def bernstein_width(V: torch.Tensor, b: torch.Tensor, L: float) -> torch.Tensor:
    """t with 2 exp(-t^2 / (2 (V + b t / 3))) = delta, L = ln(2 / delta) (draft section 6)."""
    a = b * L / 3
    return a + (a * a + 2 * V * L).sqrt()


def _tail_term(u: torch.Tensor, V: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """One-sided Bernstein exp(-u^2 / (2 (V + b u / 3))) for u > 0; 1 for u <= 0."""
    den = 2 * (V + b * u / 3)
    pos = u > 0
    e = torch.where(den > 0, -(u * u) / den.clamp_min(1e-300), torch.full_like(u, -float("inf")))
    return torch.where(pos, e.exp(), torch.ones_like(u))


@dataclass
class Plan:
    """Draw-free quantities of one or many decisions. Curves are [..., n + 1] over k = 0..n fetched entries."""

    T: torch.Tensor  # [..., n] bool, truncated set T* (includes r = 0 tail entries)
    order: torch.Tensor  # [..., n] fetch order (entry indices); non-tail entries last
    n_tail: torch.Tensor  # [...] tail size
    beta: torch.Tensor  # [..., n+1] known bias beta(k)
    V: torch.Tensor  # [..., n+1] variance V(k)
    up: torch.Tensor  # [..., n+1] largest upward move b+(k)
    dn: torch.Tensor  # [..., n+1] largest downward move b-(k)
    Bt: torch.Tensor  # [...] B tilde = sure + sum_L wh
    L: float  # ln(2 / delta)
    delta: float
    b_plus_before: torch.Tensor  # [...] largest HT upward move with no truncation (max jump)
    beta_T: torch.Tensor  # [...] beta at k = 0 (total truncation bias, Lemma 1's beta_T*)

    def eps_simple(self) -> torch.Tensor:
        """eps(k) of the draft's certificate (section 6), [..., n+1]."""
        t = bernstein_width(self.V, torch.maximum(self.up, self.dn), self.L)
        return (self.beta + t) / self.Bt[..., None]

    def p_fail_tight(self, eps0) -> torch.Tensor:
        """P_fail(k) of the tight check (draft Part III), [..., n+1]. eps0: float or a tensor broadcasting to [...]."""
        e0 = _per_row(eps0, self.Bt)
        u_up = e0 * self.Bt[..., None] + self.beta
        u_lo = e0 * self.Bt[..., None] - self.beta
        return _tail_term(u_up, self.V, self.up) + _tail_term(u_lo, self.V, self.dn)


def _per_row(eps0, like: torch.Tensor) -> torch.Tensor:
    """eps0 (float or tensor broadcasting to the row shape) -> [..., 1] against curves [..., n+1]."""
    return torch.as_tensor(eps0, dtype=like.dtype, device=like.device)[..., None]


def truncation(wh: torch.Tensor, r: torch.Tensor, tail: torch.Tensor, trunc: bool = True) -> torch.Tensor:
    """T* by the MSE rule (draft section 3.3): walk the tail by decreasing jump J = wh (1 - r) / r and stop at the first
    entry with J <= 2 beta_T (beta_T = bias collected so far, starting with the r = 0 entries). Returns a bool mask."""
    zero = torch.zeros((), dtype=wh.dtype, device=wh.device)
    r = r.expand_as(wh) if r.shape != wh.shape else r
    tail = tail.expand_as(wh) if tail.shape != wh.shape else tail
    r0 = tail & (r <= 0)
    if not trunc:
        return r0.clone()
    pos = tail & (r > 0)
    rr = r.clamp_min(1e-300)
    jump = torch.where(pos, wh * (1 - r) / rr, torch.full_like(wh, -1.0))
    order = jump.argsort(dim=-1, descending=True)
    j_s = jump.gather(-1, order)
    a_s = torch.where(pos, wh * (1 - r), zero).gather(-1, order)
    b0 = torch.where(r0, wh, zero).sum(-1, keepdim=True)
    beta_before = b0 + a_s.cumsum(-1) - a_s  # bias before adding the entry
    stop = (j_s <= 2 * beta_before) | (j_s < 0)
    stop = stop.to(torch.int8).cummax(-1).values.bool()  # the prefix ends at the first stop
    T = torch.zeros_like(pos)
    T.scatter_(-1, order, ~stop)
    return T | r0


def prepare(wh: torch.Tensor, r: torch.Tensor, tail: torch.Tensor, sure: torch.Tensor, delta: float,
            trunc: bool = True, order: str = "needle") -> Plan:
    """Algorithm 2 steps (a)-(c): T*, per-entry bias / variance / moves, and the sweep curves along the fetch order.

    wh   [..., n]  plan weights (exp(logit - m)); only the tail entries are read
    r    [..., n]  keep probabilities
    tail [..., n]  bool, the sampled part (r < 1, not persisted)
    sure [...]     exact mass outside the tail: sum over r = 1 / persisted entries + w^x (question / generated keys)
    order          "needle": decreasing wh (the draft); "grad": decreasing first-order drop of eps(0) when the entry is
                   fetched, b_i + v_i dt/dV at k = 0 (any draw-free order keeps Theorem 4)
    """
    assert order in ORDERS, order
    zero = torch.zeros((), dtype=wh.dtype, device=wh.device)
    r = r.expand_as(wh) if r.shape != wh.shape else r
    tail = tail.expand_as(wh) if tail.shape != wh.shape else tail
    wh = torch.where(tail, wh, zero)
    L = float(torch.log(torch.tensor(2.0 / delta, dtype=torch.float64)))
    T = truncation(wh, r, tail, trunc)
    pos = tail & (r > 0)
    rr = r.clamp_min(1e-300)
    Tp = T & pos
    Hp = pos & ~T
    b = torch.where(T, wh * (1 - r), zero)  # r = 0: wh
    v = torch.where(Tp, wh * wh * r * (1 - r), torch.where(Hp, wh * wh * (1 - r) / rr, zero))
    up = torch.where(Tp, wh * (1 - r), torch.where(Hp, wh * (1 - r) / rr, zero))
    dn = torch.where(Tp, wh * r, torch.where(Hp, wh, zero))
    if order == "needle":
        key = torch.where(tail, wh, torch.full_like(wh, -1.0))
    else:
        V0, m0 = v.sum(-1, keepdim=True), torch.maximum(up, dn).amax(-1, keepdim=True)
        dtdV = L / ((m0 * L / 3) ** 2 + 2 * V0 * L).sqrt().clamp_min(1e-300)
        key = torch.where(tail, b + v * dtdV, torch.full_like(wh, -1.0))
    idx = key.argsort(dim=-1, descending=True)
    return Plan(
        T=T, order=idx, n_tail=tail.sum(-1),
        beta=_suffix_sum(b.gather(-1, idx)), V=_suffix_sum(v.gather(-1, idx)),
        up=_suffix_max(up.gather(-1, idx)), dn=_suffix_max(dn.gather(-1, idx)),
        Bt=sure + wh.sum(-1), L=L, delta=delta,
        b_plus_before=torch.where(pos, wh * (1 - r) / rr, zero).amax(-1),
        beta_T=b.sum(-1),
    )


def kstar(plan: Plan, eps0, check: str = "simple") -> tuple[torch.Tensor, torch.Tensor]:
    """Smallest k whose certificate passes (draft section 7 / Part III), and per row whether the pass curve is
    non-monotone (passes at some k, fails at k + 1). k = n_tail always passes (beta = V = b = 0)."""
    assert check in CHECKS, check
    if check == "simple":
        ok = plan.eps_simple() <= _per_row(eps0, plan.Bt)
    else:
        ok = plan.p_fail_tight(eps0) <= plan.delta
    k = ok.to(torch.int8).argmax(-1)
    nonmono = (ok[..., :-1] & ~ok[..., 1:]).any(-1)
    return torch.minimum(k, plan.n_tail), nonmono


def fetch_mask(plan: Plan, k: torch.Tensor) -> torch.Tensor:
    """F_k: the first k entries of the fetch order, as a bool mask [..., n]."""
    rank = torch.empty_like(plan.order)
    rank.scatter_(-1, plan.order, torch.arange(plan.order.shape[-1], device=k.device).expand_as(plan.order))
    return rank < k[..., None]


def curve_at(curve: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    return curve.gather(-1, k[..., None])[..., 0]


def estimate(w: torch.Tensor, r: torch.Tensor, tail: torch.Tensor, sure: torch.Tensor, Z: torch.Tensor,
             T: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
    """B hat = sure + sum_F w + sum_{S cap T \\ F} w + sum_{S \\ (T cup F)} w / r, with exact weights w (draft Part II)."""
    zero = torch.zeros((), dtype=w.dtype, device=w.device)
    S = tail & Z
    c = torch.where(F & tail, w, torch.where(S & T, w, torch.where(S, w / r.clamp_min(1e-300), zero)))
    return sure + c.sum(-1)


def attention_bias(r: torch.Tensor, tail: torch.Tensor, Z: torch.Tensor, T: torch.Tensor,
                   F: torch.Tensor) -> torch.Tensor:
    """Algorithm 2 step (e) over the context: 0 on non-tail entries (r = 1, persisted) and on F and S cap T*;
    log(1 / r) on the other sampled entries; -inf on the rest. softmax(s + bias) then has denominator B hat."""
    S = tail & Z
    zero = torch.zeros((), dtype=r.dtype, device=r.device)
    ht = -torch.log(r.clamp_min(1e-300))
    out = torch.where(S & ~T, ht, zero)
    out = torch.where(tail & ~F & ~S, torch.full_like(out, -float("inf")), out)
    return torch.where(~tail | F, zero, out)


def error_curve(plan: Plan, w: torch.Tensor, r: torch.Tensor, tail: torch.Tensor, Z: torch.Tensor,
                err0: torch.Tensor) -> torch.Tensor:
    """Realised B hat_k - B along the fetch order for a given draw, [..., n+1] (err0 = B hat_0 - B, the no-fetch
    error with T*). Fetching entry j replaces its current contribution with w_j."""
    zero = torch.zeros((), dtype=w.dtype, device=w.device)
    S = tail & Z
    contrib = torch.where(S & plan.T, w, torch.where(S, w / r.clamp_min(1e-300), zero))
    d = torch.where(tail, w - contrib, zero).gather(-1, plan.order)
    return torch.cat([err0[..., None], err0[..., None] + d.cumsum(-1)], -1)


def oracle_k(err: torch.Tensor, B: torch.Tensor, eps0, n_tail: torch.Tensor) -> torch.Tensor:
    """Smallest k (same order) with realised |B hat_k - B| <= eps0 B: what a perfect certificate would fetch."""
    ok = err.abs() <= _per_row(eps0, B) * B[..., None]
    return torch.minimum(ok.to(torch.int8).argmax(-1), n_tail)
