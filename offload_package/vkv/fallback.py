"""The capped rows' decision on the GPU, without host reads.

A row whose truncation walk or certificate does not finish within its cap (``t_max`` / ``k_max``)
gets the reference decision (``decision.reference_rows``: ``reference.prepare`` with full
argsorts, then ``kstar_fast`` with head = ``k_max``, or ``reference.kstar`` for the Bernstein
checks). Here one cooperative launch does it for every such row, read from a device list (any
number of rows; none costs one empty launch):

  FBR groups of S programs (FBR * S <= SMs: all resident, they wait on each other through a
  per-group counter); group g takes rows g, g + FBR, ... of the list. Per row:
  A  the jump J of every entry; a full sort by J (each program sorts its slice of positions,
     then places each entry by binary search in the other slices);
  B  the walk along the J order (a prefix scan over the slices) -> T* (Tm row), |T*|;
  C  the per-entry moves given T*, dt/dV ("grad") and the fetch-order key; a full sort by key;
  E  the curves along that order (suffix sums / maxima over the slices), then k*: the
     Bernstein screen, the Chernoff curve on the first k_max entries plus Bennett's bound for
     the rest, and where nothing passes there, the exact curve over all entries
     ("chern"); or the Bernstein curve over all entries ("simple" / "tight").

The row's T* (Tm), ranks (1 + place in the fetch order, every entry), k*, beta(k*), B~,
non-monotonicity, |T* cap tail| and the screen flag are overwritten in place. Same expressions
as the reference (float32 Chernoff terms, float64 elsewhere, torch's operation order for the
scalar formulas, launched without FMA contraction); sums and scans run in a different order, so
values can differ in the last bits.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from triton.language.extra import libdevice

from .kernels import _bennett, _entry, _lae32, _max2, _moves
from .topk import _barrier, _sort2_desc

_I64_MIN = tl.constexpr(-0x7FFFFFFFFFFFFFFF - 1)
_BIG = tl.constexpr(0x3FFFFFFF)
NPF = 128                 # float64 partials per program
_NPF = tl.constexpr(NPF)


@triton.jit
def _ov(v):
    """float64 -> int64 with the same order (-0.0 as +0.0)."""
    v = tl.where(v == 0.0, 0.0, v)
    b = v.to(tl.int64, bitcast=True)
    return b ^ ((b >> 63) & 0x7FFFFFFFFFFFFFFF)


@triton.jit
def _coop_sort(key, pos, sk, si, out_k, out_i, cnt, g, s, S, nb, SL: tl.constexpr):
    """Sort the group's S * SL (key, position) pairs, descending, ties by position: program s
    holds slice s (key, pos [SL]); the result is out_k / out_i [g, 0:S * SL]."""
    key, pos = _sort2_desc(key, pos)
    j = tl.arange(0, SL)
    gb = g.to(tl.int64) * S * SL
    tl.store(sk + gb + s * SL + j, key)
    tl.store(si + gb + s * SL + j, pos)
    nb = _barrier(cnt, g, nb, S)
    dest = s * SL + j
    for o in range(S):
        if o != s:
            c = tl.zeros([SL], tl.int32)
            step = SL
            while step > 0:
                t = c + step - 1
                okk = t < SL
                tk = tl.load(sk + gb + o * SL + t, mask=okk, other=_I64_MIN, cache_modifier=".cg")
                ti = tl.load(si + gb + o * SL + t, mask=okk, other=0, cache_modifier=".cg")
                above = okk & ((tk > key) | ((tk == key) & (ti < pos)))
                c = tl.where(above, c + step, c)
                step = step // 2
            dest += c
    dest -= s * SL
    tl.store(out_k + gb + dest, key)
    tl.store(out_i + gb + dest, pos)
    nb = _barrier(cnt, g, nb, S)
    return nb


@triton.jit
def _pf_all(pf, pg, k, S, SMAX: tl.constexpr):
    """Partial k of the group's S programs [SMAX] (0 past S)."""
    sv = tl.arange(0, SMAX)
    return tl.load(pf + pg + sv * _NPF + k, mask=sv < S, other=0.0, cache_modifier=".cg")


@triton.jit
def _tail1t(u, V, bm, THIRD):
    """``reference._tail_term`` with torch's operation order (b * u / 3 = (b * u) * (1/3))."""
    den = 2.0 * (V + (bm * u) * THIRD)
    e = tl.where(den > 0, -(u * u) / tl.maximum(den, 1e-300), float("-inf"))
    return tl.where(u > 0, libdevice.exp(e), 1.0)


@triton.jit
def _cterms(s_hat, r, Tm, m_row, rid, slot, h, pq, ib, rb, n, max_len, H: tl.constexpr):
    """Chernoff inputs of entries pq: c (float32 scale), live, log r, log(1 - r), r (float32)."""
    xq, rq, wq, tq, pmq, r0q = _entry(s_hat, r, m_row, rid, slot, h, pq, ib, n, max_len, H)
    Tq = (tl.load(Tm + rb + pq, mask=ib, other=0) != 0) | r0q
    c = tl.where(Tq & pmq, wq, tl.where(pmq & ~Tq, wq / tl.maximum(rq, 1e-300),
                                        0.0)).to(tl.float32)
    lr = tl.log(tl.minimum(tl.maximum(rq, 1e-300), 1.0)).to(tl.float32)
    l1r = libdevice.log1p(-tl.minimum(tl.maximum(rq, 0.0), 1 - 1e-16)).to(tl.float32)
    return c, c > 0, lr, l1r, rq.to(tl.float32)


@triton.jit(do_not_specialize=['n', 'Kc', 'max_len', 'R', 'S', 'FBR'])
def fallback_kernel(fbl, s_hat, r, slots, m, sure, grid, consts, cnt,
                    Tm, rank, kst, beta_out, Bt_out, nonmono_out, tsize_out, scr_out,
                    sk, si, ok_, oi_, curv, okb, pf,
                    n, Kc, max_len, R, S, FBR,
                    H: tl.constexpr, G: tl.constexpr, SL: tl.constexpr, CB: tl.constexpr,
                    NG: tl.constexpr, SMAX: tl.constexpr,
                    TRUNC: tl.constexpr, GRAD: tl.constexpr, CHECK: tl.constexpr):
    """CHECK: 0 = "chern", 1 = "simple", 2 = "tight". consts (float64): eps0, delta, thr
    (= delta (1 - margin)), L = ln(2 / delta), 1 / 3. Order slots BK = S * SL >= n + 1."""
    g = tl.program_id(0)
    s = tl.program_id(1)
    NSB: tl.constexpr = SL // CB
    BK = S * SL
    eps0 = tl.load(consts)
    delta = tl.load(consts + 1)
    thr = tl.load(consts + 2)
    L = tl.load(consts + 3)
    THIRD = tl.load(consts + 4)
    pg = (g * S).to(tl.int64) * _NPF            # the group's partials
    po = pg + s * _NPF                           # this program's
    gb = g.to(tl.int64) * BK
    FB = FBR.to(tl.int64) * BK
    svv = tl.arange(0, SMAX)
    later = (svv > s) & (svv < S)
    before = svv < s
    cb_ar = tl.arange(0, CB)
    ng = tl.arange(0, NG)
    nrows = tl.load(fbl + R)
    nb = tl.zeros([], tl.int32)
    for t in range(g, nrows, FBR):
        rid = tl.load(fbl + t)
        b = rid // (H * G)
        h = (rid // G) % H
        slot = tl.load(slots + b).to(tl.int64)
        m_row = tl.load(m + rid)
        rb = rid.to(tl.int64) * n
        p = s * SL + tl.arange(0, SL)
        inb = p < n
        # ---- row partials over the positions ---------------------------------------------------
        x, rd, wh, tail, posm, r0 = _entry(s_hat, r, m_row, rid, slot, h, p, inb, n, max_len, H)
        tl.store(pf + po + 0, tl.sum(tl.where(r0, wh, 0.0), 0))
        tl.store(pf + po + 1, tl.sum(r0.to(tl.float64), 0))
        tl.store(pf + po + 2, tl.sum(tail.to(tl.float64), 0))
        tl.store(pf + po + 3, tl.sum(wh, 0))
        nb = _barrier(cnt, g, nb, S)
        b0 = tl.sum(_pf_all(pf, pg, 0, S, SMAX), 0)
        n_r0 = tl.sum(_pf_all(pf, pg, 1, S, SMAX), 0)
        n_tail = tl.sum(_pf_all(pf, pg, 2, S, SMAX), 0).to(tl.int64)
        Bt = tl.load(sure + rid) + tl.sum(_pf_all(pf, pg, 3, S, SMAX), 0)
        t_walk = tl.zeros([], tl.int64)
        # ---- phase 0 (TRUNC): sort by the jump J, then walk; phase 1: sort by the key ----------
        for ph in range(0 if TRUNC else 1, 2):
            if ph == 1:
                if TRUNC:
                    # The walk: the first q with J_q <= 2 beta_before(q) along the J order.
                    asum = tl.zeros([], tl.float64)
                    for i in range(NSB):
                        q = s * SL + i * CB + cb_ar
                        pq = tl.load(oi_ + gb + q, cache_modifier=".cg")
                        ib = pq < n
                        xq, rq, wq, tq, pmq, r0q = _entry(s_hat, r, m_row, rid, slot, h, pq, ib,
                                                          n, max_len, H)
                        asum += tl.sum(tl.where(pmq, wq * (1 - rq), 0.0), 0)
                    tl.store(pf + po + 4, asum)
                    nb = _barrier(cnt, g, nb, S)
                    run = b0 + tl.sum(tl.where(before, _pf_all(pf, pg, 4, S, SMAX), 0.0), 0)
                    first = tl.zeros([], tl.int64) + _BIG
                    for i in range(NSB):
                        q = s * SL + i * CB + cb_ar
                        pq = tl.load(oi_ + gb + q, cache_modifier=".cg")
                        ib = pq < n
                        kq = tl.load(ok_ + gb + q, cache_modifier=".cg")
                        jq = (kq ^ ((kq >> 63) & 0x7FFFFFFFFFFFFFFF)).to(tl.float64, bitcast=True)
                        xq, rq, wq, tq, pmq, r0q = _entry(s_hat, r, m_row, rid, slot, h, pq, ib,
                                                          n, max_len, H)
                        a = tl.where(pmq, wq * (1 - rq), 0.0)
                        bbf = run + tl.cumsum(a, 0) - a
                        stop = (jq <= 2 * bbf) | (jq < 0) | ~ib
                        first = tl.minimum(first, tl.min(tl.where(stop, q, _BIG), 0).to(tl.int64))
                        run += tl.sum(a, 0)
                    tl.store(pf + po + 5, first.to(tl.float64))
                    nb = _barrier(cnt, g, nb, S)
                    t_walk = tl.min(tl.where(svv < S, _pf_all(pf, pg, 5, S, SMAX), 1e300),
                                    0).to(tl.int64)
                    for i in range(NSB):
                        q = s * SL + i * CB + cb_ar
                        pq = tl.load(oi_ + gb + q, cache_modifier=".cg")
                        tl.store(Tm + rb + pq, (q < t_walk).to(tl.int8), mask=pq < n)
                    nb = _barrier(cnt, g, nb, S)
                tmv = tl.load(Tm + rb + p, mask=inb, other=0)
                T, bb, v, up, dn = _moves(wh, rd, posm, r0, tmv)
                if GRAD:
                    tl.store(pf + po + 6, tl.sum(v, 0))
                    tl.store(pf + po + 7, tl.max(tl.maximum(up, dn), 0))
                    nb = _barrier(cnt, g, nb, S)
                    V0 = tl.sum(_pf_all(pf, pg, 6, S, SMAX), 0)
                    m0 = tl.max(_pf_all(pf, pg, 7, S, SMAX), 0)
                    a3 = (m0 * L) * THIRD
                    dtdv = (1.0 / tl.maximum(tl.sqrt(a3 * a3 + (2.0 * V0) * L), 1e-300)) * L
                    kv = tl.where(tail, bb + v * dtdv, -1.0)
                else:
                    kv = tl.where(tail, wh, -1.0)
            else:
                kv = tl.where(posm, wh * (1 - rd) / tl.maximum(rd, 1e-300), -1.0)
            kk = tl.where(inb, _ov(kv), _I64_MIN)
            nb = _coop_sort(kk, p, sk, si, ok_, oi_, cnt, g, s, S, nb, SL=SL)
        # ---- curves along the fetch order ------------------------------------------------------
        sb = tl.zeros([], tl.float64)
        sV = tl.zeros([], tl.float64)
        mu = tl.zeros([], tl.float64)
        md = tl.zeros([], tl.float64)
        hV = tl.zeros([], tl.float64)
        hu = tl.zeros([], tl.float64)
        hd = tl.zeros([], tl.float64)
        for i in range(NSB):
            q = s * SL + i * CB + cb_ar
            pq = tl.load(oi_ + gb + q, cache_modifier=".cg")
            ib = pq < n
            xq, rq, wq, tq, pmq, r0q = _entry(s_hat, r, m_row, rid, slot, h, pq, ib, n, max_len,
                                              H)
            Tq, bq, vq, uq, dq = _moves(wq, rq, pmq, r0q,
                                        tl.load(Tm + rb + pq, mask=ib, other=0))
            tl.store(rank + rb + pq, (q + 1).to(tl.int32), mask=ib)
            sb += tl.sum(bq, 0)
            sV += tl.sum(vq, 0)
            mu = tl.maximum(mu, tl.max(uq, 0))
            md = tl.maximum(md, tl.max(dq, 0))
            hq = q >= Kc
            hV += tl.sum(tl.where(hq, vq, 0.0), 0)
            hu = tl.maximum(hu, tl.max(tl.where(hq, uq, 0.0), 0))
            hd = tl.maximum(hd, tl.max(tl.where(hq, dq, 0.0), 0))
        tl.store(pf + po + 8, sb)
        tl.store(pf + po + 9, sV)
        tl.store(pf + po + 10, mu)
        tl.store(pf + po + 11, md)
        tl.store(pf + po + 12, hV)
        tl.store(pf + po + 13, hu)
        tl.store(pf + po + 14, hd)
        nb = _barrier(cnt, g, nb, S)
        beta0 = tl.sum(_pf_all(pf, pg, 8, S, SMAX), 0)
        Vt = tl.sum(_pf_all(pf, pg, 9, S, SMAX), 0)
        up0 = tl.max(_pf_all(pf, pg, 10, S, SMAX), 0)
        dn0 = tl.max(_pf_all(pf, pg, 11, S, SMAX), 0)
        RVh = tl.sum(_pf_all(pf, pg, 12, S, SMAX), 0)
        Mu = tl.maximum(tl.max(_pf_all(pf, pg, 13, S, SMAX), 0), 1e-300)
        Md = tl.maximum(tl.max(_pf_all(pf, pg, 14, S, SMAX), 0), 1e-300)
        cbb = tl.sum(tl.where(later, _pf_all(pf, pg, 8, S, SMAX), 0.0), 0)
        cvv = tl.sum(tl.where(later, _pf_all(pf, pg, 9, S, SMAX), 0.0), 0)
        cuu = tl.max(tl.where(later, _pf_all(pf, pg, 10, S, SMAX), 0.0), 0)
        cdd = tl.max(tl.where(later, _pf_all(pf, pg, 11, S, SMAX), 0.0), 0)
        for i in range(NSB):
            q = s * SL + (NSB - 1 - i) * CB + cb_ar
            pq = tl.load(oi_ + gb + q, cache_modifier=".cg")
            ib = pq < n
            xq, rq, wq, tq, pmq, r0q = _entry(s_hat, r, m_row, rid, slot, h, pq, ib, n, max_len,
                                              H)
            Tq, bq, vq, uq, dq = _moves(wq, rq, pmq, r0q,
                                        tl.load(Tm + rb + pq, mask=ib, other=0))
            tl.store(curv + gb + q, cbb + tl.cumsum(bq, 0, reverse=True))
            tl.store(curv + FB + gb + q, cvv + tl.cumsum(vq, 0, reverse=True))
            tl.store(curv + 2 * FB + gb + q,
                     tl.maximum(cuu, tl.associative_scan(uq, 0, _max2, reverse=True)))
            tl.store(curv + 3 * FB + gb + q,
                     tl.maximum(cdd, tl.associative_scan(dq, 0, _max2, reverse=True)))
            cbb += tl.sum(bq, 0)
            cvv += tl.sum(vq, 0)
            cuu = tl.maximum(cuu, tl.max(uq, 0))
            cdd = tl.maximum(cdd, tl.max(dq, 0))
        e0Bt = eps0 * Bt
        screened = tl.zeros([], tl.int1)
        kstar = tl.zeros([], tl.int64)
        nonmono = tl.zeros([], tl.int1)
        if CHECK == 0:
            screened = (_tail1t(e0Bt + beta0, Vt, up0, THIRD)
                        + _tail1t(e0Bt - beta0, Vt, dn0, THIRD)) <= delta
            found = screened
            for cp in range(2):                  # 0: head + Bennett, 1: the whole order
                head = cp == 0
                LIM = tl.where(head, Kc, n).to(tl.int64)
                go = (~found) & ((~head) | (Kc < n))
                if go:
                    benn = tl.where(head, 1.0, 0.0)
                    tu = tl.zeros([NG], tl.float64)
                    td = tl.zeros([NG], tl.float64)
                    for i in range(NSB):
                        q = s * SL + i * CB + cb_ar
                        pq = tl.load(oi_ + gb + q, cache_modifier=".cg")
                        ib = (pq < n) & (q < LIM)
                        c, live, lr, l1r, r32 = _cterms(s_hat, r, Tm, m_row, rid, slot, h, pq, ib,
                                                        rb, n, max_len, H)
                        for gi in range(NG):
                            lam = tl.load(grid + gi) / Bt
                            a = lam.to(tl.float32) * c
                            ku = tl.where(live, _lae32(lr + a * (1 - r32), l1r - a * r32), 0.0)
                            kd = tl.where(live, _lae32(lr - a * (1 - r32), l1r + a * r32), 0.0)
                            tu += tl.where(ng == gi, tl.sum(ku.to(tl.float64), 0), 0.0)
                            td += tl.where(ng == gi, tl.sum(kd.to(tl.float64), 0), 0.0)
                    tl.store(pf + po + 16 + ng, tu)
                    tl.store(pf + po + 16 + NG + ng, td)
                    nb = _barrier(cnt, g, nb, S)
                    pm = later[:, None]
                    pa = pf + pg + svv[:, None] * _NPF + 16 + ng[None, :]
                    lu = tl.sum(tl.where(pm, tl.load(pa, mask=pm, other=0.0,
                                                     cache_modifier=".cg"), 0.0), 0)
                    ld = tl.sum(tl.where(pm, tl.load(pa + NG, mask=pm, other=0.0,
                                                     cache_modifier=".cg"), 0.0), 0)
                    first = tl.zeros([], tl.int64) + _BIG
                    for i in range(NSB):
                        q = s * SL + (NSB - 1 - i) * CB + cb_ar
                        pq = tl.load(oi_ + gb + q, cache_modifier=".cg")
                        ib = (pq < n) & (q < LIM)
                        c, live, lr, l1r, r32 = _cterms(s_hat, r, Tm, m_row, rid, slot, h, pq, ib,
                                                        rb, n, max_len, H)
                        beta = tl.load(curv + gb + q)
                        u_up = e0Bt + beta
                        u_dn = e0Bt - beta
                        best_up = tl.full([CB], 1.0, tl.float64)
                        best_dn = tl.full([CB], 1.0, tl.float64)
                        for gi in range(NG):
                            lam = tl.load(grid + gi) / Bt
                            a = lam.to(tl.float32) * c
                            ku = tl.where(live, _lae32(lr + a * (1 - r32), l1r - a * r32), 0.0)
                            kd = tl.where(live, _lae32(lr - a * (1 - r32), l1r + a * r32), 0.0)
                            sel = ng == gi
                            Ku = tl.cumsum(ku.to(tl.float64), 0, reverse=True) + \
                                tl.sum(tl.where(sel, lu, 0.0), 0)
                            Kd = tl.cumsum(kd.to(tl.float64), 0, reverse=True) + \
                                tl.sum(tl.where(sel, ld, 0.0), 0)
                            ex_up = _bennett(lam, RVh, Mu) * benn
                            ex_dn = _bennett(lam, RVh, Md) * benn
                            best_up = tl.minimum(best_up, libdevice.exp(tl.minimum(
                                Ku + ex_up - lam * u_up, 0.0)))
                            best_dn = tl.minimum(best_dn, tl.where(u_dn > 0, libdevice.exp(
                                tl.minimum(Kd + ex_dn - lam * u_dn, 0.0)), 1.0))
                            lu += tl.where(sel, tl.sum(ku.to(tl.float64), 0), 0.0)
                            ld += tl.where(sel, tl.sum(kd.to(tl.float64), 0), 0.0)
                        okq = ((best_up + best_dn) <= thr) & (q <= LIM)
                        first = tl.minimum(first, tl.min(tl.where(okq, q, _BIG), 0).to(tl.int64))
                    tl.store(pf + po + 15, first.to(tl.float64))
                    nb = _barrier(cnt, g, nb, S)
                    gf = tl.min(tl.where(svv < S, _pf_all(pf, pg, 15, S, SMAX), 1e300),
                                0).to(tl.int64)
                    hit = gf <= LIM
                    kstar = tl.where(head, tl.where(hit, tl.minimum(gf, n_tail), kstar),
                                     tl.minimum(tl.where(hit, gf, 0), n_tail))
                    found = hit | (~head)
        else:
            # Bernstein over the whole order: ok(q) into okb, then the first pass and the
            # non-monotone check (q, q + 1 may sit in different slices).
            first = tl.zeros([], tl.int64) + _BIG
            for i in range(NSB):
                q = s * SL + i * CB + cb_ar
                beta = tl.load(curv + gb + q)
                Vq = tl.load(curv + FB + gb + q)
                uq = tl.load(curv + 2 * FB + gb + q)
                dq = tl.load(curv + 3 * FB + gb + q)
                if CHECK == 1:
                    aw = (tl.maximum(uq, dq) * L) * THIRD
                    tw = aw + tl.sqrt(aw * aw + (2.0 * Vq) * L)
                    okq = ((beta + tw) / Bt) <= eps0
                else:
                    okq = (_tail1t(e0Bt + beta, Vq, uq, THIRD)
                           + _tail1t(e0Bt - beta, Vq, dq, THIRD)) <= delta
                okq = okq & (q <= n)
                tl.store(okb + gb + q, okq.to(tl.int8))
                first = tl.minimum(first, tl.min(tl.where(okq, q, _BIG), 0).to(tl.int64))
            tl.store(pf + po + 15, first.to(tl.float64))
            nb = _barrier(cnt, g, nb, S)
            gf = tl.min(tl.where(svv < S, _pf_all(pf, pg, 15, S, SMAX), 1e300), 0).to(tl.int64)
            kstar = tl.minimum(tl.where(gf <= n, gf, 0), n_tail)
            nm = tl.zeros([], tl.int32)
            for i in range(NSB):
                q = s * SL + i * CB + cb_ar
                a_ = tl.load(okb + gb + q, cache_modifier=".cg") != 0
                b_ = tl.load(okb + gb + q + 1, mask=q + 1 < BK, other=1,
                             cache_modifier=".cg") != 0
                nm = tl.maximum(nm, tl.max((a_ & ~b_ & (q < n)).to(tl.int32), 0))
            tl.store(pf + po + 16, nm.to(tl.float64))
            nb = _barrier(cnt, g, nb, S)
            nonmono = tl.max(_pf_all(pf, pg, 16, S, SMAX), 0) > 0
        # ---- outputs -------------------------------------------------------------------------------
        if s == 0:
            tl.store(kst + rid, kstar)
            tl.store(beta_out + rid, tl.load(curv + gb + kstar, cache_modifier=".cg"))
            tl.store(Bt_out + rid, Bt)
            tl.store(nonmono_out + rid, nonmono.to(tl.int8))
            tl.store(tsize_out + rid, t_walk.to(tl.float64) + n_r0)
            tl.store(scr_out + rid, screened.to(tl.int8))
        nb = _barrier(cnt, g, nb, S)                    # scratch reused by the next row


_SMS = {}
_CHECK = {"chern": 0, "simple": 1, "tight": 2}


SL = 2048                # order slots per program
CB = 1024                # entries per step within a slice
SMAX = 256               # most programs per row


def plan(n: int, dev) -> tuple[int, int]:
    """(S, FBR): S programs per row with SL order slots each (S * SL >= n + 1, k = 0..n), FBR
    row groups with FBR * S <= SMs (all resident)."""
    sms = _SMS.get(dev)
    if sms is None:
        sms = _SMS[dev] = torch.cuda.get_device_properties(dev).multi_processor_count
    S = -(-(n + 1) // SL)
    if S > min(sms, SMAX):
        raise ValueError(f"vkv fallback: a {n}-entry row needs {S} co-resident programs "
                         f"(> {min(sms, SMAX)})")
    return S, max(1, sms // S)


def run(fbl, s_hat, r_l, slots_d, m, sure, grid, consts, Tm, rank, kst, beta_k, Bt, nonmono,
        tsize, scr, *, n, Kc, max_len, H, G, trunc, order, check):
    """Recompute the rows listed in ``fbl`` ([R] row ids, count at [R]) by the reference rule."""
    R = int(kst.numel())
    dev = s_hat.device
    S, FBR = plan(n, dev)
    BK = S * SL
    cnt = torch.zeros((FBR,), dtype=torch.int32, device=dev)
    sk = torch.empty((FBR, BK), dtype=torch.int64, device=dev)
    si = torch.empty((FBR, BK), dtype=torch.int32, device=dev)
    ok_ = torch.empty((FBR, BK), dtype=torch.int64, device=dev)
    oi_ = torch.empty((FBR, BK), dtype=torch.int32, device=dev)
    curv = torch.empty((4, FBR, BK), dtype=torch.float64, device=dev)
    okb = torch.empty((FBR, BK), dtype=torch.int8, device=dev)
    pf = torch.empty((FBR, S, NPF), dtype=torch.float64, device=dev)
    fallback_kernel[(FBR, S)](
        fbl, s_hat, r_l, slots_d, m, sure, grid, consts, cnt, Tm, rank, kst, beta_k, Bt,
        nonmono, tsize, scr, sk, si, ok_, oi_, curv, okb, pf, n, Kc, max_len, R, S, FBR,
        H=H, G=G, SL=SL, CB=CB, NG=int(grid.numel()), SMAX=SMAX, TRUNC=bool(trunc),
        GRAD=order == "grad", CHECK=_CHECK[check], num_warps=8, enable_fp_fusion=False)
