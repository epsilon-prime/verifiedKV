"""Algorithm 2 (a)-(d) for many decisions at once, without full sorts.

``reference.prepare`` argsorts every row twice (truncation order and fetch order): at a 128k
context that is ~0.7 ms per sort per layer. Both sorts are only needed up to a short prefix, so
this computes the same result from two capped ``topk``s and complement reductions, and falls back
to the reference on the rows whose prefix does not fit the cap:

* T*: the truncation walk visits the tail by decreasing jump J and stops at the first entry with
  J <= 2 beta_before. Along that order J is non-increasing and beta_before non-decreasing, so the
  stop rule is a threshold and T* is a prefix of the J order. If the walk stops within the top
  ``t_max`` jumps, those are all it needs (beta_before only sums earlier entries).
* Fetch order: "needle" sorts by wh; "grad" by b + v dt/dV, with dt/dV from the row's totals
  after T* (plain reductions), so a top-``k_max`` of the key is the order's head.
* k*, Chernoff ("chern", ``reference_chernoff.kstar_fast`` with head = ``k_max``): the Bernstein
  screen at k = 0 (totals), then the exact two-point log-MGF on the first ``k_max`` entries of the
  order plus Bennett's bound for the rest, which needs only the rest's variance and largest moves:
  the curves at index ``k_max`` of the capped suffix sums / maxima. Those are exactly the inputs of
  the harness's ``chern_curve_head`` (it reads ``order[:, :head]`` and the curves at ``head``), so
  it is called on them directly. With ``k_max >= n`` the whole order is the head
  (``chern_curve_rows``, no Bennett), as in the harness.
* k*, Bernstein ("simple" / "tight"): the curves at k <= ``k_max`` are the top part from k on plus
  the complement's sum / max.

Rows whose walk or certificate does not finish within its cap (no k <= ``k_max`` passes) are
recomputed with the reference (full argsort; ``kstar_fast`` / ``kstar``), so the result equals the
reference up to the order of exactly tied keys (``topk.row_topk`` puts ties in position order,
``argsort`` in any; any draw-free order keeps the certificate). Inputs float64 like the reference.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from . import reference as ref
from . import reference_chernoff as chern
from .topk import row_topk

CHERN_MARGIN = 1e-3     # kstar_fast's pass test: P <= delta (1 - margin)


@dataclass
class Decision:
    T: torch.Tensor          # [R, n] bool, truncated set T* (incl. r = 0 tail entries)
    k: torch.Tensor          # [R] long, k* (entries fetched)
    F: torch.Tensor          # [R, WF] long, fetch-order positions; the first k[r] are fetched
    beta_k: torch.Tensor     # [R] known bias beta(k*)
    Bt: torch.Tensor         # [R] B tilde
    nonmono: torch.Tensor    # [R] bool, pass curve fails again after passing (within the cap)
    fallback: torch.Tensor   # [R] bool, row recomputed by the reference (cap hit)
    screened: torch.Tensor   # [R] bool, k = 0 by the Bernstein screen ("chern")

    @property
    def WF(self) -> int:
        return int(self.F.shape[-1])

    def fetch_valid(self) -> torch.Tensor:
        """[R, WF] bool: entry j of F is fetched (j < k*)."""
        return torch.arange(self.WF, device=self.F.device).view(1, -1) < self.k.view(-1, 1)


def _ln2_over(delta: float) -> float:
    # Same expression as reference.prepare, so L is bit-identical.
    return float(torch.log(torch.tensor(2.0 / delta, dtype=torch.float64)))


def screen(beta0, V0, up0, dn0, Bt, eps0: float, delta: float) -> torch.Tensor:
    """``kstar_fast``'s screen: the one-sided Bernstein bound (``Plan.p_fail_tight``) at k = 0
    passes; curves at k = 0 [R] are the row totals."""
    e0 = torch.as_tensor(eps0, dtype=Bt.dtype, device=Bt.device)
    p = (ref._tail_term(e0 * Bt + beta0, V0, up0) + ref._tail_term(e0 * Bt - beta0, V0, dn0))
    return p <= delta


def grad_dtdV(v, up, dn, L: float):
    """dt/dV of the Bernstein width at k = 0 (``reference.prepare``, order "grad") [R, 1]."""
    V0, m0 = v.sum(-1, keepdim=True), torch.maximum(up, dn).amax(-1, keepdim=True)
    return L / ((m0 * L / 3) ** 2 + 2 * V0 * L).sqrt().clamp_min(1e-300)


@torch.no_grad()
def decide(wh: torch.Tensor, r: torch.Tensor, tail: torch.Tensor, sure: torch.Tensor, *,
           eps0: float, delta: float, check: str = "chern", order: str = "grad",
           trunc: bool = True, t_max: int = 4096, k_max: int = 1024) -> Decision:
    """wh, r [R, n] float64, tail [R, n] bool, sure [R] float64 (see ``reference.prepare``)."""
    assert check in ("chern",) + ref.CHECKS and order in ref.ORDERS, (check, order)
    R, n = wh.shape
    dev = wh.device
    zero = torch.zeros((), dtype=wh.dtype, device=dev)
    wh = torch.where(tail, wh, zero)
    L = _ln2_over(delta)
    pos = tail & (r > 0)
    r0 = tail & (r <= 0)
    rr = r.clamp_min(1e-300)
    n_tail = tail.sum(-1)

    # ---- (a) T*: a prefix of the J order -------------------------------------------------------
    if trunc and n > 0:
        M = min(int(t_max), n)
        jump = torch.where(pos, wh * (1 - r) / rr, torch.full_like(wh, -1.0))
        j_s, j_i = row_topk(jump, M, values=True)
        a_s = torch.where(pos, wh * (1 - r), zero).gather(-1, j_i)
        b0 = torch.where(r0, wh, zero).sum(-1, keepdim=True)
        beta_before = b0 + a_s.cumsum(-1) - a_s
        stop = (j_s <= 2 * beta_before) | (j_s < 0)
        stop = stop.to(torch.int8).cummax(-1).values.bool()
        T = torch.zeros_like(pos).scatter_(-1, j_i, ~stop) | r0
        cap_T = ~stop[:, -1] & (pos.sum(-1) > M)
    else:
        T = r0.clone()
        cap_T = torch.zeros(R, dtype=torch.bool, device=dev)

    # ---- (b) per-entry bias / variance / moves, fetch-order key ----------------------------------
    Tp = T & pos
    Hp = pos & ~T
    b = torch.where(T, wh * (1 - r), zero)
    v = torch.where(Tp, wh * wh * r * (1 - r), torch.where(Hp, wh * wh * (1 - r) / rr, zero))
    up = torch.where(Tp, wh * (1 - r), torch.where(Hp, wh * (1 - r) / rr, zero))
    dn = torch.where(Tp, wh * r, torch.where(Hp, wh, zero))
    if order == "needle":
        key = torch.where(tail, wh, torch.full_like(wh, -1.0))
    else:
        key = torch.where(tail, b + v * grad_dtdV(v, up, dn, L), torch.full_like(wh, -1.0))

    # ---- (c) curves over the top-K of the fetch order + the complement ---------------------------
    K = min(int(k_max), n)
    o_i = row_topk(key, K) if K else torch.zeros((R, 0), dtype=torch.long, device=dev)
    rest = torch.ones_like(tail).scatter_(-1, o_i, False)
    Rb = torch.where(rest, b, zero).sum(-1, keepdim=True)
    RV = torch.where(rest, v, zero).sum(-1, keepdim=True)
    Rup = torch.where(rest, up, zero).amax(-1, keepdim=True) if n else Rb
    Rdn = torch.where(rest, dn, zero).amax(-1, keepdim=True) if n else Rb
    beta = ref._suffix_sum(b.gather(-1, o_i)) + Rb                         # [R, K + 1]
    Vc = ref._suffix_sum(v.gather(-1, o_i)) + RV
    upc = torch.maximum(ref._suffix_max(up.gather(-1, o_i)), Rup)
    dnc = torch.maximum(ref._suffix_max(dn.gather(-1, o_i)), Rdn)
    Bt = sure + wh.sum(-1)
    screened = torch.zeros(R, dtype=torch.bool, device=dev)
    if check == "chern":
        screened = screen(beta[:, 0], Vc[:, 0], upc[:, 0], dnc[:, 0], Bt, eps0, delta)
        k = torch.zeros(R, dtype=torch.long, device=dev)
        found = torch.ones(R, dtype=torch.bool, device=dev)
        nonmono = torch.zeros(R, dtype=torch.bool, device=dev)
        idx = (~screened).nonzero()[:, 0]
        if idx.numel():
            args = (T[idx], o_i[idx], Bt[idx], beta[idx])
            rows = (wh[idx], r[idx], tail[idx])
            if K < n:   # head + Bennett (the harness's ``head < n``)
                P = chern.chern_curve_head(*args, Vc[idx], upc[idx], dnc[idx], *rows, eps0, K)
            else:       # the head is the whole order
                P = chern.chern_curve_rows(*args, *rows, eps0)
            ok = P <= delta * (1 - CHERN_MARGIN)
            found[idx] = ok.any(-1)
            k[idx] = ok.to(torch.int8).argmax(-1)
            nonmono[idx] = (ok[:, :-1] & ~ok[:, 1:]).any(-1)
        cap_K = ~found
    else:
        e0 = torch.as_tensor(eps0, dtype=wh.dtype, device=dev)
        if check == "simple":
            t = ref.bernstein_width(Vc, torch.maximum(upc, dnc), L)
            ok = (beta + t) / Bt[:, None] <= e0
        else:
            u_up = e0 * Bt[:, None] + beta
            u_lo = e0 * Bt[:, None] - beta
            ok = ref._tail_term(u_up, Vc, upc) + ref._tail_term(u_lo, Vc, dnc) <= delta
        k = ok.to(torch.int8).argmax(-1)
        cap_K = ~ok.any(-1) & (n_tail > K)
        nonmono = (ok[:, :-1] & ~ok[:, 1:]).any(-1)
    k = torch.minimum(k, n_tail)
    beta_k = beta.gather(-1, k.clamp(max=K)[:, None])[:, 0]

    # ---- fallback: rows whose prefix exceeds a cap, by the reference -----------------------------
    fallback = cap_T | cap_K
    # The one host read of a decision: the fetch width and whether any row falls back.
    k_top, any_fb = (torch.stack([k.max(), fallback.any().long()]).tolist() if R else (0, 0))
    F = o_i[:, :k_top]
    if any_fb:
        rows = fallback.nonzero()[:, 0]
        plan, kf, nm = reference_rows(wh[rows], r[rows], tail[rows], sure[rows], eps0=eps0,
                                      delta=delta, check=check, order=order, trunc=trunc,
                                      k_max=k_max)
        T[rows] = plan.T
        k[rows] = kf
        nonmono[rows] = nm
        beta_k[rows] = ref.curve_at(plan.beta, kf)
        W = max(k_top, int(kf.max()))
        if W > F.shape[1]:
            F = torch.cat([F, F.new_zeros((R, W - F.shape[1]))], dim=1)
        F[rows] = plan.order[:, :W]
    return Decision(T=T, k=k, F=F, beta_k=beta_k, Bt=Bt, nonmono=nonmono, fallback=fallback,
                    screened=screened)


def reference_rows(wh, r, tail, sure, *, eps0, delta, check, order, trunc, k_max):
    """The reference decision for some rows: (plan, k*, nonmono). "chern" is the harness's
    ``kstar_fast`` with head = ``k_max`` (its rows that pass nowhere in the head get the full
    exact-term curve)."""
    plan = ref.prepare(wh, r, tail, sure, delta, trunc=trunc, order=order)
    if check == "chern":
        kf = chern.kstar_fast(plan, wh, r, tail, eps0, delta, head=int(k_max))[0]
        return plan, kf, torch.zeros_like(kf, dtype=torch.bool)
    kf, nm = ref.kstar(plan, eps0, check)
    return plan, kf, nm


def attention_weights(r: torch.Tensor, tail: torch.Tensor, sampled: torch.Tensor,
                      inT: torch.Tensor, inF: torch.Tensor) -> torch.Tensor:
    """exp(Algorithm 2 (e) bias) on attended entries: 1 on non-tail / F / sampled T*, 1/r on the
    other sampled tail entries, 0 (excluded) on unsampled, unfetched tail entries."""
    one = torch.ones((), dtype=torch.float32, device=r.device)
    w = torch.where(sampled & ~inT, 1.0 / r.float().clamp_min(1e-30), one)
    w = torch.where(tail & ~inF & ~sampled, torch.zeros_like(w), w)
    return torch.where(~tail | inF, one, w)
