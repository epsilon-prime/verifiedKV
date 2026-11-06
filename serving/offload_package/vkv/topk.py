"""Row top-K in one launch: ``torch.topk`` on a [R, n] float64 row is ~34 kernels (eight radix
passes of three, then a sort) and ~110 us of host time per call; the decision makes three per
layer. Here the S programs of a row (S = SMs // R, so all programs are resident at once) split the
row, synchronise through a per-row counter, and run

* the radix select of the K-th largest key (8-bit digits from the top of the float64 bit pattern,
  each pass's histogram summed over the row's programs; stops once the threshold digit's bin is
  taken whole),
* the compaction of the K winners (everything above the threshold, then the threshold's equals by
  position),
* a bitonic sort of each program's slice of the winners and a merge by binary search in the other
  slices.

Order: value descending, ties by position ascending (a total order, so the result is unique;
``torch.topk``'s order of exact ties is unspecified). -0.0 counts as +0.0. NaN unsupported.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.standard import _indicator, _log2

_INT64_MIN = tl.constexpr(-0x7FFFFFFFFFFFFFFF - 1)


@triton.jit
def _cas2(x, p, flip, i: tl.constexpr):
    """``tl.sort``'s compare-and-swap on (key, position) pairs."""
    n_dims: tl.constexpr = _log2(x.numel)
    y = x ^ tl.xor_sum(x, n_dims - 1 - i, True)
    q = p ^ tl.xor_sum(p, n_dims - 1 - i, True)
    is_right = _indicator(n_dims, i)
    gt = (x > y) | ((x == y) & (p < q))
    sw = gt != (flip ^ is_right)
    return tl.where(sw, y, x), tl.where(sw, q, p)


@triton.jit
def _merge2(x, p, stage: tl.constexpr, order: tl.constexpr):
    if order == 2:
        flip = _indicator(_log2(x.numel), stage)
    else:
        flip = order
    for i in tl.static_range(stage):
        x, p = _cas2(x, p, flip, stage - 1 - i)
    return x, p


@triton.jit
def _sort2_desc(x, p):
    n_dims: tl.constexpr = _log2(x.numel)
    h = tl.reshape(x, [2] * n_dims)
    hp = tl.reshape(p, [2] * n_dims)
    for i in tl.static_range(1, n_dims + 1):
        h, hp = _merge2(h, hp, i, 2 if i < n_dims else 1)
    return tl.reshape(h, x.shape), tl.reshape(hp, p.shape)


@triton.jit
def _okey(x, off, pos, inb):
    """The float64 value as a signed int64 with the same order."""
    v = tl.load(x + off + pos, mask=inb, other=0.0).to(tl.float64)
    v = tl.where(v == 0.0, 0.0, v)
    b = v.to(tl.int64, bitcast=True)
    return b ^ ((b >> 63) & 0x7FFFFFFFFFFFFFFF)


@triton.jit
def _barrier(cnt, row, k, S):
    """The k-th (from 0) barrier of row ``row``'s S programs (counter zeroed per call)."""
    tl.debug_barrier()
    c = tl.atomic_add(cnt + row, 1, sem="acq_rel", scope="gpu")
    target = S * (k + 1)
    while c < target:
        c = tl.atomic_add(cnt + row, 0, sem="acq_rel", scope="gpu")
    tl.debug_barrier()
    return k + 1


@triton.jit(do_not_specialize=['n', 'K'])
def row_topk_kernel(x, n, K, cnt, hist, counts, cand_k, cand_i, out_i, out_v, rank,
                    S: tl.constexpr, BLK: tl.constexpr, BK: tl.constexpr, HAS_V: tl.constexpr,
                    HAS_RANK: tl.constexpr):
    row = tl.program_id(0)
    s = tl.program_id(1)
    off = row.to(tl.int64) * n
    CH = tl.cdiv(tl.cdiv(n, S), BLK) * BLK
    lo = s * CH
    hi = tl.minimum(lo + CH, n)
    bins = tl.arange(0, 256)
    sv = tl.arange(0, S)
    hb = row.to(tl.int64) * S * 256
    k_rem = K + tl.zeros([], tl.int32)
    pref = tl.zeros([], tl.int64)
    sh = tl.zeros([], tl.int64) + 56
    done = tl.zeros([], tl.int32)
    nb = tl.zeros([], tl.int32)
    # ---- radix select ---------------------------------------------------------------------------
    for p in tl.static_range(8):
        if done == 0:
            h = tl.zeros([256], tl.int32)
            for a in range(lo, hi, BLK):
                pos = a + tl.arange(0, BLK)
                inb = pos < hi
                key = _okey(x, off, pos, inb)
                if p == 0:
                    h += tl.histogram(((key >> 56) + 128).to(tl.int32), 256, mask=inb)
                else:
                    m = inb & ((key >> (64 - 8 * p)) == pref)
                    h += tl.histogram(((key >> (56 - 8 * p)) & 255).to(tl.int32), 256, mask=m)
            tl.store(hist + hb + s * 256 + bins, h)
            nb = _barrier(cnt, row, nb, S)
            h = tl.sum(tl.load(hist + hb + sv[:, None] * 256 + bins[None, :],
                               cache_modifier=".cg"), 0)
            ge = tl.sum(h, 0) - tl.cumsum(h, 0) + h          # entries with digit >= d
            dsel = tl.max(tl.where(ge >= k_rem, bins, -1), 0)
            cd = tl.sum(tl.where(bins == dsel, h, 0), 0)
            k_rem = k_rem - (tl.sum(tl.where(bins == dsel, ge, 0), 0) - cd)
            if p == 0:
                pref = (dsel - 128).to(tl.int64)
            else:
                pref = pref * 256 + dsel
            sh = sh * 0 + (56 - 8 * p)
            done = (cd == k_rem).to(tl.int32)
            nb = _barrier(cnt, row, nb, S)                   # the hist slots are reused
    # ---- compaction: above the threshold, then its equals by position (k_rem of them) -----------
    G = K - k_rem
    ng = tl.zeros([], tl.int32)
    ne = tl.zeros([], tl.int32)
    for a in range(lo, hi, BLK):
        pos = a + tl.arange(0, BLK)
        inb = pos < hi
        v = _okey(x, off, pos, inb) >> sh
        ng += tl.sum((inb & (v > pref)).to(tl.int32), 0)
        ne += tl.sum((inb & (v == pref)).to(tl.int32), 0)
    cb = row.to(tl.int64) * S * 2
    tl.store(counts + cb + s * 2, ng)
    tl.store(counts + cb + s * 2 + 1, ne)
    nb = _barrier(cnt, row, nb, S)
    before = sv < s
    cg = tl.sum(tl.where(before, tl.load(counts + cb + sv * 2, cache_modifier=".cg"), 0), 0)
    ce = G + tl.sum(tl.where(before, tl.load(counts + cb + sv * 2 + 1, cache_modifier=".cg"), 0),
                    0)
    kb = row.to(tl.int64) * BK
    for a in range(lo, hi, BLK):
        pos = a + tl.arange(0, BLK)
        inb = pos < hi
        key = _okey(x, off, pos, inb)
        v = key >> sh
        mg = inb & (v > pref)
        me = inb & (v == pref)
        ig = cg + tl.cumsum(mg.to(tl.int32), 0) - 1
        ie = ce + tl.cumsum(me.to(tl.int32), 0) - 1
        me = me & (ie < K)
        tl.store(cand_k + kb + ig, key, mask=mg)
        tl.store(cand_i + kb + ig, pos, mask=mg)
        tl.store(cand_k + kb + ie, key, mask=me)
        tl.store(cand_i + kb + ie, pos, mask=me)
        cg += tl.sum(mg.to(tl.int32), 0)
        ce += tl.sum(me.to(tl.int32), 0)
    nb = _barrier(cnt, row, nb, S)
    # ---- sort: own slice in registers, then its place among the other slices ---------------------
    SL: tl.constexpr = BK // S
    j = s * SL + tl.arange(0, SL)
    live = j < K
    ck = tl.load(cand_k + kb + j, mask=live, other=_INT64_MIN, cache_modifier=".cg")
    ci = tl.where(live, tl.load(cand_i + kb + j, mask=live, other=0, cache_modifier=".cg"), n + j)
    ck, ci = _sort2_desc(ck, ci)
    tl.store(cand_k + kb + j, ck)
    tl.store(cand_i + kb + j, ci)
    nb = _barrier(cnt, row, nb, S)
    dest = s * 0 + tl.arange(0, SL)
    for o in tl.static_range(S):
        if o != s:
            c = tl.zeros([SL], tl.int32)
            step = SL
            while step > 0:
                t = c + step - 1
                ok = t < SL
                tk = tl.load(cand_k + kb + o * SL + t, mask=ok, other=_INT64_MIN,
                             cache_modifier=".cg")
                ti = tl.load(cand_i + kb + o * SL + t, mask=ok, other=0, cache_modifier=".cg")
                above = ok & ((tk > ck) | ((tk == ck) & (ti < ci)))
                c = tl.where(above, c + step, c)
                step = step // 2
            dest += c
    keep = dest < K
    ob = row.to(tl.int64) * K
    tl.store(out_i + ob + dest, ci.to(tl.int64), mask=keep)
    if HAS_V:
        b = ck ^ ((ck >> 63) & 0x7FFFFFFFFFFFFFFF)
        tl.store(out_v + ob + dest, b.to(tl.float64, bitcast=True), mask=keep)
    if HAS_RANK:
        tl.store(rank + off + ci, dest + 1, mask=keep)


_NUM_SMS = {}


def _splits(R: int, BK: int, dev) -> int:
    """Programs per row: a power of two with R * S <= SMs (all resident: they wait on each
    other), at most 8 and at most BK / 16 (slices of >= 16 winners)."""
    sms = _NUM_SMS.get(dev)
    if sms is None:
        sms = _NUM_SMS[dev] = torch.cuda.get_device_properties(dev).multi_processor_count
    S = 1
    while S < 8 and R * S * 2 <= sms and BK // (S * 2) >= 16:
        S *= 2
    return S


def row_topk(x: torch.Tensor, K: int, *, values: bool = False, rank: torch.Tensor | None = None,
             cnt: torch.Tensor | None = None, BLK: int = 4096, num_warps: int = 8):
    """Positions [R, K] (int64) of each row's K largest entries of ``x`` [R, n] (float32/64,
    contiguous), largest first, ties by position; with ``values`` also the values [R, K] (float64).
    ``rank`` [R, n] int32 (zeroed by the caller) gets 1 + place at each winner's position.
    ``cnt``: a zeroed int32 [R] for the programs' barrier (allocated if None)."""
    R, n = x.shape
    if not 1 <= K <= n:
        raise ValueError(f"row_topk: K={K} outside [1, {n}]")
    if not x.is_cuda:
        # The same order: a stable descending sort (-0.0 as +0.0).
        xd = x.double()
        v, i = torch.sort(torch.where(xd == 0, 0.0, xd), dim=-1, descending=True, stable=True)
        v, i = v[:, :K], i[:, :K]
        if rank is not None:
            rank.scatter_(-1, i, torch.arange(1, K + 1, dtype=rank.dtype).expand(R, K).contiguous())
        return (v, i) if values else i
    x = x.contiguous()
    dev = x.device
    BK = max(2, triton.next_power_of_2(K))       # (a sort needs >= 2 entries)
    S = _splits(R, BK, dev)
    if cnt is None:
        cnt = torch.zeros((R,), dtype=torch.int32, device=dev)
    hist = torch.empty((R, S, 256), dtype=torch.int32, device=dev)
    counts = torch.empty((R, S, 2), dtype=torch.int32, device=dev)
    cand_k = torch.empty((R, BK), dtype=torch.int64, device=dev)
    cand_i = torch.empty((R, BK), dtype=torch.int32, device=dev)
    out_i = torch.empty((R, K), dtype=torch.int64, device=dev)
    out_v = torch.empty((R, K), dtype=torch.float64, device=dev) if values else out_i
    row_topk_kernel[(R, S)](x, n, K, cnt, hist, counts, cand_k, cand_i, out_i, out_v,
                            rank if rank is not None else out_i, S=S, BLK=BLK, BK=BK,
                            HAS_V=values, HAS_RANK=rank is not None, num_warps=num_warps)
    return (out_v, out_i) if values else out_i
