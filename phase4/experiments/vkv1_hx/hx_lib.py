"""vkv1_hx building blocks (theory: phase4/theory/hx_store_note.md). Uses vkv1/algo.py and vkv1_diag/diag_lib.py
read-only. Shapes: leading dims are rows (decisions), last dim n is the context; ks [..., K] are candidate k per row.
float64.

hx bound at a given k (Section 1 of the note): with S = sum_live c_i Z_i, a = mu + eps B + beta, b = mu - eps B + beta,
    P(fail) = P(S > a) + P(S < b),
the m live entries with the largest c (the heavy set) are handled exactly by a DP over a grid of nb bins (jumps rounded
UP for the upper tail and DOWN for the lower tail: valid for every draw), and the rest by two-point Chernoff:
    P(S > a) <= sum_N P(N) min(1, inf_l exp(K_L(l) - l (a - h N))) + P(N in the overflow bin)
    P(S < b) <= sum_{N < b/h} P(N) min(1, inf_l exp(K_L(-l) + l (b - h N)))
m = 0 is exactly diag_lib.chernoff_curves (uncentred form, same lambda grid).

store (Section 2): at most cap entries per KV head, chosen by a draw-free priority (max over rows of w / B among the
entries some row fetched); the policy never reads Z.
"""

import math
import os
import sys

import torch

HOME = os.path.expanduser("~")
sys.path.insert(0, os.path.join(HOME, "phase4"))
sys.path.insert(0, os.path.join(HOME, "phase4", "experiments", "vkv1_diag"))
from vkv1 import algo  # noqa: E402  (read-only)
import diag_lib as dl  # noqa: E402  (read-only)

F64 = torch.float64


def _sorted_terms(plan: algo.Plan, wh, r, tail):
    r = r.expand_as(wh) if r.shape != wh.shape else r
    tail = tail.expand_as(wh) if tail.shape != wh.shape else tail
    wh = torch.where(tail, wh, torch.zeros((), dtype=wh.dtype, device=wh.device))
    _, _, _, _, c = dl.token_terms(wh, r, tail, plan.T)  # 0 off the random part (fetched-able, 0 < r < 1)
    return c.gather(-1, plan.order), r.gather(-1, plan.order)


def hx_at(plan: algo.Plan, wh, r, tail, eps0s, ks, m: int = 64, nb: int = 512, grid=dl.GRID):
    """hx failure bound at k = ks [..., K] (one candidate set per row), one [..., K] tensor per eps0."""
    c_s, r_s = _sorted_terms(plan, wh, r, tail)
    dev = c_s.device
    n = c_s.shape[-1]
    live = c_s > 0
    pos = torch.arange(n, device=dev)
    K = ks.shape[-1]
    Bt = plan.Bt[..., None]  # [..., 1]
    mu = algo._suffix_sum(c_s * r_s).gather(-1, ks)  # [..., K]
    beta = plan.beta.gather(-1, ks)
    # heavy set per candidate: the m largest c among the unfetched live entries (draw-free)
    mm = max(0, min(m, n))
    cH = torch.zeros(*ks.shape, mm, dtype=F64, device=dev)
    rH = torch.zeros_like(cH)
    if mm:
        for j in range(K):
            vals = torch.where(live & (pos >= ks[..., j:j + 1]), c_s, torch.full_like(c_s, -1.0))
            top, idx = vals.topk(mm, dim=-1)
            ok = top > 0
            cH[..., j, :] = torch.where(ok, top, 0.0)
            rH[..., j, :] = torch.where(ok, r_s.gather(-1, idx), 0.0)
    lr = torch.log(r_s.clamp(1e-300, 1.0))
    l1r = torch.log1p(-r_s.clamp(0.0, 1 - 1e-16))
    lrH = torch.log(rH.clamp(1e-300, 1.0))
    l1rH = torch.log1p(-rH.clamp(0.0, 1 - 1e-16))
    KL_up, KL_dn, lams = [], [], []
    for g in grid.tolist():  # uncentred light log-MGFs K_L(+l), K_L(-l) at each candidate k
        lam = g / Bt  # [..., 1]
        a = lam * c_s
        k_up = torch.where(live, torch.logaddexp(lr + a, l1r), 0.0)
        k_dn = torch.where(live, torch.logaddexp(lr - a, l1r), 0.0)
        aH = lam[..., None] * cH
        h_up = torch.where(rH > 0, torch.logaddexp(lrH + aH, l1rH), 0.0).sum(-1)
        h_dn = torch.where(rH > 0, torch.logaddexp(lrH - aH, l1rH), 0.0).sum(-1)
        KL_up.append(algo._suffix_sum(k_up).gather(-1, ks) - h_up)
        KL_dn.append(algo._suffix_sum(k_dn).gather(-1, ks) - h_dn)
        lams.append(lam)
    bins = torch.arange(nb, device=dev, dtype=F64)
    out = []
    for e in eps0s:
        a_up = mu + e * Bt + beta  # > 0
        b_dn = mu - e * Bt + beta
        # upper tail: jumps rounded up, overflow bin (N >= nb - 1) counts as failure
        h_a = a_up / (nb - 1)
        s_up = torch.ceil((cH / h_a[..., None]).clamp(max=nb)).long()
        pmf = _dp(s_up, rH, nb, overflow=True)
        t = a_up[..., None] - h_a[..., None] * bins  # t_N, > 0 for N <= nb - 2
        best = torch.ones_like(pmf)
        for lam, KL in zip(lams, KL_up):
            best = torch.minimum(best, torch.exp((KL[..., None] - lam[..., None] * t).clamp(max=0.0)))
        p_up = (pmf[..., :-1] * best[..., :-1]).sum(-1) + pmf[..., -1]
        # lower tail: jumps rounded down, bins with h N >= b contribute 0 (S_L >= 0); b <= 0 -> 0
        pos_b = b_dn > 0
        h_b = torch.where(pos_b, b_dn, torch.ones_like(b_dn)) / (nb - 1)
        s_dn = torch.floor((cH / h_b[..., None]).clamp(max=nb)).long()
        pmf = _dp(s_dn, rH, nb, overflow=False)
        t = b_dn[..., None] - h_b[..., None] * bins
        best = torch.ones_like(pmf)
        for lam, KL in zip(lams, KL_dn):
            best = torch.minimum(best, torch.exp((KL[..., None] + lam[..., None] * t).clamp(max=0.0)))
        p_dn = torch.where(pos_b, (pmf[..., :-1] * best[..., :-1]).sum(-1), torch.zeros_like(b_dn))
        out.append(p_up + p_dn)
    return out


def _dp(s, rH, nb: int, overflow: bool):
    """Distribution of N = sum_i s_i Z_i, Z_i ~ Bernoulli(rH_i), on bins 0..nb-1. s, rH [..., m].
    overflow=True: bin nb-1 collects N >= nb-1; False: mass at N >= nb-1 is dropped (callers only use N <= nb-2)."""
    pmf = torch.zeros(*s.shape[:-1], nb, dtype=F64, device=s.device)
    pmf[..., 0] = 1.0
    ar = torch.arange(nb, device=s.device)
    for i in range(s.shape[-1]):
        si, ri = s[..., i:i + 1], rH[..., i:i + 1]
        src = ar - si  # bin N receives from N - s_i
        sh = pmf.gather(-1, src.clamp(min=0)) * (src >= 0)
        if overflow:
            suf = pmf.flip(-1).cumsum(-1).flip(-1)  # suf[j] = sum_{N >= j} pmf[N]
            top = suf.gather(-1, (nb - 1 - si).clamp(min=0))  # mass landing at >= nb - 1
            sh = torch.cat([sh[..., :-1], top], -1)
        pmf = (1 - ri) * pmf + ri * sh
    return pmf


def hx_kstar(plan: algo.Plan, wh, r, tail, eps0s, delta: float, lad, m: int = 64, nb: int = 512, refine: int = 8):
    """Draw-free k per row and eps0: the first ladder k with hx <= delta, then the smallest passing k among `refine`
    evenly spaced points between the previous ladder point and it. Returns ([k [...]] per eps0, [bound at k])."""
    nt = plan.n_tail[..., None]
    ks1 = torch.minimum(lad.expand(*plan.n_tail.shape, -1), nt)
    P1 = hx_at(plan, wh, r, tail, eps0s, ks1, m, nb)
    ks_out, ps_out = [], []
    for i, e in enumerate(eps0s):
        ok = P1[i] <= delta
        ok[..., -1] = True  # k = n_tail fetches the whole tail (P = 0 up to the lambda grid); never fails to stop
        j = ok.to(torch.int8).argmax(-1)
        k_hi = ks1.gather(-1, j[..., None])[..., 0]
        p_hi = P1[i].gather(-1, j[..., None])[..., 0]
        k_lo = ks1.gather(-1, (j - 1).clamp(min=0)[..., None])[..., 0]
        frac = torch.arange(1, refine + 1, device=ks1.device, dtype=F64) / (refine + 1)
        ks2 = (k_lo[..., None] + torch.ceil((k_hi - k_lo)[..., None] * frac).long()).clamp(max=k_hi[..., None])
        P2 = hx_at(plan, wh, r, tail, [e], ks2, m, nb)[0]
        ok2 = (P2 <= delta) & (j[..., None] > 0)
        has = ok2.any(-1)
        j2 = ok2.to(torch.int8).argmax(-1)
        k = torch.where(has, ks2.gather(-1, j2[..., None])[..., 0], k_hi)
        p = torch.where(has, P2.gather(-1, j2[..., None])[..., 0], p_hi)
        ks_out.append(k)
        ps_out.append(p)
    return ks_out, ps_out


def store_update(S, prio, F, w, B, tail0, cap: int):
    """Draw-free store update after one forward. S [H, n] bool, prio [H, n] running max of w / B over rows that fetched
    the entry, F [H, R, n] this forward's fetch sets, w [H, R, n], B [H, R], tail0 [H, n]. Keeps the top-cap entries
    of S | F by prio (never reads Z). Returns (S, prio)."""
    p_new = torch.where(F, w / B[..., None], torch.zeros((), dtype=w.dtype, device=w.device)).amax(1)
    prio = torch.maximum(prio, p_new)
    cand = (S | F.any(1)) & tail0
    score = torch.where(cand, prio, torch.full_like(prio, -1.0))
    if cap <= 0:
        return torch.zeros_like(S), prio
    kk = min(cap, score.shape[-1])
    top, idx = score.topk(kk, dim=-1)
    S_new = torch.zeros_like(S)
    S_new.scatter_(-1, idx, top > 0)
    return S_new, prio


def ladder(n, device):
    return dl.ladder(n, device)


__all__ = ["hx_at", "hx_kstar", "store_update", "ladder", "algo", "dl", "math"]
