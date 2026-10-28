"""Triton kernels of the fused Vkv decode step (``decode.VkvDecoder.layer``).

The per-entry math is ``reference.prepare`` / ``reference_chernoff.kstar_fast`` (or the Bernstein
``kstar``) / ``attention_bias`` evaluated in the same precision (plan weights float64 from float32
logits, r float64 from the stored float32; the Chernoff per-entry terms float32 through libdevice,
their sums float64, as ``kstar_fast``), only grouped differently: each kernel recomputes ``wh = exp(f64(s_hat - m))`` and r from its inputs
instead of materialising the reference's [rows, n] float64 intermediates. Reductions go through
per-block partials (deterministic). Layouts (one layer):

  rows       R = B * H * G, row = (b * H + h) * G + g  (q head h * G + g of decode row b)
  r, cell    [S, max_len, H] (VkvState.r[layer], engine.cell[layer])
  alist      [S, H, max_len + 1] int32 attended positions, acnt [S, H] int32 their counts
  s_hat, key [R, n] float32 (n = longest sampled context of the batch), jump [R, n] float64
  Tm         [R, n] int8 truncated set T* (r = 0 entries are added on the fly)
  rank       [R, n] int32: 1 + position of the entry in the top-K fetch order, 0 = not in it
  key64      [R, n] float64 "grad" fetch-order key b + v dt/dV on the tail (-1 elsewhere)
"""
from __future__ import annotations

import triton
import triton.language as tl
from triton.language.extra import libdevice

NEG_INF = tl.constexpr(float("-inf"))


@triton.jit(do_not_specialize=['W', 'NBLK', 'max_len'])
def exact_scores_kernel(q, alist, acnt, cell, r, kv, slots, s_out, phys_out, r_out, pos_out, pmax,
                        W, NBLK, max_len,
                        HP: tl.constexpr, C2: tl.constexpr,
                        H: tl.constexpr, G: tl.constexpr, D: tl.constexpr, BLK: tl.constexpr):
    """Exact logits over the attended list A of (row b, KV head h) (its resident cells). Writes
    s [B, H, G, W], cell / r / position [B, H, W], and per-block maxima over A's non-tail (r = 1)
    entries [B, H, G, NBLK]."""
    bh = tl.program_id(0)
    blk = tl.program_id(1)
    b = bh // H
    h = bh % H
    slot = tl.load(slots + b).to(tl.int64)
    cnt = tl.load(acnt + slot * H + h)
    offs = blk * BLK + tl.arange(0, BLK)
    live = offs < cnt
    pos = tl.load(alist + (slot * H + h) * max_len + offs, mask=live, other=0).to(tl.int64)
    mi = (slot * max_len + pos) * H + h
    phys = tl.load(cell + mi, mask=live, other=0).to(tl.int64)
    rv = tl.load(r + mi, mask=live, other=1.0)
    d = tl.arange(0, D)
    k = tl.load(kv + phys[:, None] * C2 + d[None, :], mask=live[:, None], other=0.0).to(tl.float32)
    nt = live & (rv >= 1.0)
    obase = (b * H + h).to(tl.int64) * W
    tl.store(phys_out + obase + offs, phys, mask=live)
    tl.store(r_out + obase + offs, rv, mask=live)
    tl.store(pos_out + obase + offs, pos.to(tl.int32), mask=live)
    for g in tl.static_range(G):
        row = (b * H + h) * G + g
        qv = tl.load(q + row * D + d)
        s = tl.sum(k * qv[None, :], axis=1)
        tl.store(s_out + row.to(tl.int64) * W + offs, s, mask=live)
        tl.store(pmax + row * NBLK + blk, tl.max(tl.where(nt, s, NEG_INF), axis=0))


@triton.jit(do_not_specialize=['n', 'M', 'max_len'])
def shortlist_kernel(q, sl_pos, sl_ok, cell, cpu_row, kv, pool, slots, s_hat, keys_cnt,
                     n, M, max_len,
                     C2: tl.constexpr, PC2: tl.constexpr,
                     H: tl.constexpr, G: tl.constexpr, D: tl.constexpr, BLK: tl.constexpr):
    """The shortlist of (row b, KV head h): exact logits of its M tail positions replace their
    sketch logits in s_hat [R, n]. Keys of sampled entries come from their resident cells, the
    others straight from the (device-mapped, pinned) CPU pool: key-only host reads, counted."""
    bh = tl.program_id(0)
    blk = tl.program_id(1)
    b = bh // H
    h = bh % H
    slot = tl.load(slots + b).to(tl.int64)
    j = blk * BLK + tl.arange(0, BLK)
    ok = (j < M) & (tl.load(sl_ok + bh * M + j, mask=j < M, other=0) != 0)
    pos = tl.load(sl_pos + bh * M + j, mask=ok, other=0).to(tl.int64)
    mi = (slot * max_len + pos) * H + h
    c = tl.load(cell + mi, mask=ok, other=-1).to(tl.int64)
    res = ok & (c >= 0)
    host = ok & (c < 0)
    cr = tl.load(cpu_row + mi, mask=host, other=0).to(tl.int64)
    d = tl.arange(0, D)
    kr = tl.load(kv + c[:, None] * C2 + d[None, :], mask=res[:, None], other=0.0)
    kh = tl.load(pool + cr[:, None] * PC2 + d[None, :], mask=host[:, None], other=0.0)
    k = tl.where(res[:, None], kr, kh).to(tl.float32)
    for g in tl.static_range(G):
        row = (b * H + h) * G + g
        qv = tl.load(q + row * D + d)
        s = tl.sum(k * qv[None, :], axis=1)
        tl.store(s_hat + row.to(tl.int64) * n + pos, s, mask=ok)
    tl.atomic_add(keys_cnt, tl.sum(host.to(tl.int32), axis=0))


@triton.jit(do_not_specialize=['n', 'max_len'])
def group_best_kernel(s_hat, r, slots, best, n, max_len,
                      H: tl.constexpr, G: tl.constexpr, BLK: tl.constexpr):
    """The shortlist's ranking [B * H, n]: the largest sketch logit over the KV head's G query
    heads on the tail (r < 1), -inf elsewhere."""
    bh = tl.program_id(0)
    blk = tl.program_id(1)
    b = bh // H
    h = bh % H
    slot = tl.load(slots + b).to(tl.int64)
    pos = blk * BLK + tl.arange(0, BLK)
    inb = pos < n
    rv = tl.load(r + (slot * max_len + pos) * H + h, mask=inb, other=1.0)
    m = tl.full([BLK], float("-inf"), tl.float32)
    for g in tl.static_range(G):
        m = tl.maximum(m, tl.load(s_hat + (bh * G + g).to(tl.int64) * n + pos, mask=inb,
                                  other=float("-inf")))
    tl.store(best + bh.to(tl.int64) * n + pos, tl.where(rv < 1.0, m, float("-inf")), mask=inb)


@triton.jit(do_not_specialize=['n', 'NBLK', 'max_len'])
def tail_max_kernel(s_hat, r, slots, pmax, n, NBLK, max_len,
                    H: tl.constexpr, G: tl.constexpr, BLK: tl.constexpr):
    """Per-block maxima of the sketch logits over tail entries (r < 1) [R, NBLK]."""
    row = tl.program_id(0)
    blk = tl.program_id(1)
    b = row // (H * G)
    h = (row // G) % H
    slot = tl.load(slots + b).to(tl.int64)
    pos = blk * BLK + tl.arange(0, BLK)
    inb = pos < n
    rv = tl.load(r + (slot * max_len + pos) * H + h, mask=inb, other=1.0)
    x = tl.load(s_hat + row.to(tl.int64) * n + pos, mask=inb, other=NEG_INF)
    tl.store(pmax + row * NBLK + blk, tl.max(tl.where(inb & (rv < 1.0), x, NEG_INF), axis=0))


@triton.jit(do_not_specialize=['W', 'NBLK', 'NBT'])
def sure_kernel(s, r_a, acnt, slots, pmax_a, pmax_t, m, psum, W, NBLK, NBT,
                H: tl.constexpr, G: tl.constexpr, BLK: tl.constexpr, BP: tl.constexpr):
    """The stabiliser m = max(block maxima over A, over the tail) [R] (stored by block 0), and
    per-block sums of exp(f64(s - m)) over A's non-tail entries [R, NBLK] (``sure``)."""
    row = tl.program_id(0)
    blk = tl.program_id(1)
    b = row // (H * G)
    h = (row // G) % H
    mx = tl.full([BP], NEG_INF, tl.float32)
    for a in range(0, NBLK, BP):
        i = a + tl.arange(0, BP)
        mx = tl.maximum(mx, tl.load(pmax_a + row * NBLK + i, mask=i < NBLK, other=NEG_INF))
    for a in range(0, NBT, BP):
        i = a + tl.arange(0, BP)
        mx = tl.maximum(mx, tl.load(pmax_t + row * NBT + i, mask=i < NBT, other=NEG_INF))
    m_row = tl.max(mx, axis=0)
    if blk == 0:
        tl.store(m + row, m_row)
    slot = tl.load(slots + b).to(tl.int64)
    cnt = tl.load(acnt + slot * H + h)
    offs = blk * BLK + tl.arange(0, BLK)
    live = offs < cnt
    rv = tl.load(r_a + (b * H + h).to(tl.int64) * W + offs, mask=live, other=0.0)
    x = tl.load(s + row.to(tl.int64) * W + offs, mask=live, other=0.0) - m_row
    w = tl.where(live & (rv >= 1.0), tl.exp(x.to(tl.float64)), 0.0)
    tl.store(psum + row * NBLK + blk, tl.sum(w, axis=0))


@triton.jit
def dtdv_kernel(tot, V0, dtdv, consts, NBLK, R, BR: tl.constexpr, BP: tl.constexpr):
    """The "grad" key's dt/dV = L / max(sqrt((m0 L / 3)^2 + 2 V0 L), 1e-300) [R], m0 = the max
    of the block maxima of up / dn in ``tot`` [R, NBLK, 3]; the same IEEE operations, in the same
    order, as the torch expression it replaces (launch with enable_fp_fusion=False): torch
    divides by a host scalar as a multiply by its reciprocal, squares by x * x, and ``L / t`` is
    ``t.reciprocal() * L``. ``consts`` = (L, 1 / 3) float64 (a float argument would be fp32)."""
    L = tl.load(consts)
    THIRD = tl.load(consts + 1)
    rows = tl.program_id(0) * BR + tl.arange(0, BR)
    ok = rows < R
    mx = tl.zeros([BR, BP], tl.float64)
    for a in range(0, NBLK, BP):
        i = a + tl.arange(0, BP)
        mk = ok[:, None] & (i < NBLK)[None, :]
        p = tot + (rows[:, None] * NBLK + i[None, :]) * 3
        mx = tl.maximum(mx, tl.maximum(tl.load(p + 1, mask=mk, other=0.0),
                                       tl.load(p + 2, mask=mk, other=0.0)))
    m0 = tl.max(mx, axis=1)
    a = (m0 * L) * THIRD
    t = a * a + (2.0 * tl.load(V0 + rows, mask=ok, other=0.0)) * L
    one = tl.full([BR], 1.0, tl.float64)
    tl.store(dtdv + rows, (one / tl.maximum(tl.sqrt(t), 1e-300)) * L, mask=ok)


@triton.jit
def flags_kernel(cap_t, cap_k, fb, fbl, R, BR: tl.constexpr):
    """fb = walk or certificate capped [R]; fbl = those rows in order, their count at fbl[R]
    (``fallback.run``'s list: read on the device, never by the host)."""
    i = tl.arange(0, BR)
    ok = i < R
    f = ok & ((tl.load(cap_t + i, mask=ok, other=0) != 0) | (tl.load(cap_k + i, mask=ok, other=0)
                                                            != 0))
    tl.store(fb + i, f, mask=ok)
    tl.store(fbl + tl.cumsum(f.to(tl.int32), 0) - 1, i, mask=f)
    tl.store(fbl + R, tl.sum(f.to(tl.int32), 0))


@triton.jit(do_not_specialize=['n', 'max_len'])
def cold_kernel(rank, kst, cell, slots, cold, posc, n, max_len,
                H: tl.constexpr, G: tl.constexpr, BLK: tl.constexpr):
    """F (the entries with 1 <= rank <= k*) in fetch order: column rank - 1 of cold [R, n] marks
    a non-resident entry and posc [R, n] holds its position (both zeroed by the caller): what
    the staging copies in, in the order the list was always built."""
    row = tl.program_id(0)
    p = tl.program_id(1) * BLK + tl.arange(0, BLK)
    inb = p < n
    b = row // (H * G)
    h = (row // G) % H
    slot = tl.load(slots + b).to(tl.int64)
    rb = row.to(tl.int64) * n
    rk = tl.load(rank + rb + p, mask=inb, other=0)
    fv = inb & (rk > 0) & (rk <= tl.load(kst + row))
    c = tl.load(cell + (slot * max_len + p) * H + h, mask=fv, other=0)
    cd = fv & (c < 0)
    tl.store(cold + rb + rk - 1, cd, mask=cd)
    tl.store(posc + rb + rk - 1, p.to(tl.int64), mask=cd)


@triton.jit(do_not_specialize=['WD', 'CS', 'W1', 'W2'])
def place_kernel(cold, CS, base, phys, idx, w, slen, idx2, w2, slen2, WD, W1, W2,
                 HAS_BASE: tl.constexpr, BLK: tl.constexpr):
    """One fetch pass, one row: its cold entries' staged rows after the row's live prefix
    (``base``, or 0), weight 1: into the main list while it has room (W1 columns), the rest
    into the overflow list (W2). slen / slen2 = the lists' live prefixes."""
    row = tl.program_id(0)
    if HAS_BASE:
        off = tl.load(base + row).to(tl.int64)
    else:
        off = tl.zeros([], tl.int64)
    start = off
    rb = row.to(tl.int64)
    for a in range(0, WD, BLK):
        j = a + tl.arange(0, BLK)
        cd = (j < WD) & (tl.load(cold + rb * CS + j, mask=j < WD, other=0) != 0)
        dst = off + tl.cumsum(cd.to(tl.int64), 0) - 1
        p = tl.load(phys + rb * WD + j, mask=cd, other=0).to(tl.int64)
        m1 = cd & (dst < W1)
        m2 = cd & (dst >= W1)
        tl.store(idx + rb * W1 + dst, p, mask=m1)
        tl.store(w + rb * W1 + dst, 1.0, mask=m1)
        tl.store(idx2 + rb * W2 + dst - W1, p, mask=m2)
        tl.store(w2 + rb * W2 + dst - W1, 1.0, mask=m2)
        off += tl.sum(cd.to(tl.int64), 0)
    tl.store(slen + row, tl.minimum(off, W1).to(tl.int32))
    tl.store(slen2 + row, tl.maximum(off - W1, 0).to(tl.int32))


@triton.jit
def lse_merge_kernel(oa, la, ob, lb, out, lse, D: tl.constexpr):
    """``decode.lse2_merge`` per row (log2 LSEs; a part with lse = -inf adds nothing, and an
    empty second part returns the first exactly)."""
    row = tl.program_id(0)
    d = tl.arange(0, D)
    a = tl.load(la + row)
    b = tl.load(lb + row)
    mx = tl.maximum(a, b)
    mx = tl.where((mx == float("inf")) | (mx == float("-inf")) | (mx != mx), 0.0, mx)
    sa = tl.exp2(a - mx)
    sb = tl.exp2(b - mx)
    ssum = sa + sb
    scale = tl.where(ssum > 0, 1.0 / ssum, 0.0)
    pa = tl.where(sa > 0, tl.load(oa + row * D + d).to(tl.float32) * sa, 0.0)
    pb = tl.where(sb > 0, tl.load(ob + row * D + d).to(tl.float32) * sb, 0.0)
    tl.store(out + row * D + d, ((pa + pb) * scale).to(out.dtype.element_ty))
    tl.store(lse + row, tl.where(ssum > 0, mx + tl.log2(ssum), a))


@triton.jit
def _entry(s_hat, r, m_row, row, slot, h, pos, inb, n, max_len, H: tl.constexpr):
    """Per-entry quantities of ``reference.prepare`` before truncation."""
    rv = tl.load(r + (slot * max_len + pos) * H + h, mask=inb, other=1.0)
    x = tl.load(s_hat + row.to(tl.int64) * n + pos, mask=inb, other=0.0) - m_row
    tail = inb & (rv < 1.0)
    wh = tl.where(tail, tl.exp(x.to(tl.float64)), 0.0)
    rd = rv.to(tl.float64)
    posm = tail & (rv > 0.0)
    r0 = tail & (rv <= 0.0)
    return x, rd, wh, tail, posm, r0


@triton.jit(do_not_specialize=['n', 'NBLK', 'max_len'])
def prep_kernel(s_hat, r, slots, m, jump, key, part, n, NBLK, max_len,
                H: tl.constexpr, G: tl.constexpr, BLK: tl.constexpr):
    """jump J = wh (1 - r) / r on positive tail entries (-1 elsewhere, float64), the fetch-order
    key (the logit s_hat - m on the tail, -inf elsewhere: same order as wh, ties included), and
    per-block partials [R, NBLK, 4] = (sum_{r=0} wh, sum_tail wh, #pos, #tail)."""
    row = tl.program_id(0)
    blk = tl.program_id(1)
    b = row // (H * G)
    h = (row // G) % H
    slot = tl.load(slots + b).to(tl.int64)
    pos = blk * BLK + tl.arange(0, BLK)
    inb = pos < n
    x, rd, wh, tail, posm, r0 = _entry(s_hat, r, tl.load(m + row), row, slot, h, pos, inb, n,
                                       max_len, H)
    j = tl.where(posm, wh * (1 - rd) / tl.maximum(rd, 1e-300), -1.0)
    base = row.to(tl.int64) * n
    tl.store(jump + base + pos, j, mask=inb)
    tl.store(key + base + pos, tl.where(tail, x, NEG_INF), mask=inb)
    p = part + (row * NBLK + blk) * 4
    tl.store(p, tl.sum(tl.where(r0, wh, 0.0), axis=0))
    tl.store(p + 1, tl.sum(wh, axis=0))
    tl.store(p + 2, tl.sum(posm.to(tl.float64), axis=0))
    tl.store(p + 3, tl.sum(tail.to(tl.float64), axis=0))


@triton.jit(do_not_specialize=['n', 'M', 'NBLK', 'max_len'])
def walk_kernel(j_s, j_i, s_hat, r, slots, m, part, Tm, cap_out, n, M, NBLK, max_len,
                H: tl.constexpr, G: tl.constexpr, BM: tl.constexpr, BP: tl.constexpr):
    """One row: the truncation walk over the top-M jumps (T* = the prefix before the first
    J <= 2 beta_before, beta starting at the r = 0 mass), T* into Tm; cap = the walk did not stop
    within the top M although more positive entries exist."""
    row = tl.program_id(0)
    b = row // (H * G)
    h = (row // G) % H
    slot = tl.load(slots + b).to(tl.int64)
    acc = tl.zeros([BP], dtype=tl.float64)
    acc_pos = tl.zeros([BP], dtype=tl.float64)
    for a in range(0, NBLK, BP):
        i = a + tl.arange(0, BP)
        ok = i < NBLK
        acc += tl.load(part + (row * NBLK + i) * 4, mask=ok, other=0.0)
        acc_pos += tl.load(part + (row * NBLK + i) * 4 + 2, mask=ok, other=0.0)
    b0 = tl.sum(acc, axis=0)
    npos = tl.sum(acc_pos, axis=0)
    base = row.to(tl.int64) * n
    i = tl.arange(0, BM)
    inm = i < M
    js = tl.load(j_s + row * M + i, mask=inm, other=-1.0)
    ji = tl.load(j_i + row * M + i, mask=inm, other=0)
    x, rd, wh, tail, posm, r0 = _entry(s_hat, r, tl.load(m + row), row, slot, h, ji, inm,
                                       n, max_len, H)
    a_s = tl.where(posm, wh * (1 - rd), 0.0)
    beta_before = b0 + tl.cumsum(a_s, 0) - a_s
    stop = (js <= 2 * beta_before) | (js < 0) | (i >= M)
    ts = tl.min(tl.where(stop, i, BM), axis=0)
    tl.store(Tm + base + ji, (i < ts).to(tl.int8), mask=inm)
    tl.store(cap_out + row, ((ts >= M) & (npos > M)).to(tl.int8))


@triton.jit
def _moves(wh, rd, posm, r0, Tm_v):
    """b, v, up, dn of ``reference.prepare`` given T* membership (``Tm_v`` | r = 0)."""
    T = (Tm_v != 0) | r0
    Tp = T & posm
    Hp = posm & ~T
    rr = tl.maximum(rd, 1e-300)
    b = tl.where(T, wh * (1 - rd), 0.0)
    v = tl.where(Tp, wh * wh * rd * (1 - rd), tl.where(Hp, wh * wh * (1 - rd) / rr, 0.0))
    up = tl.where(Tp, wh * (1 - rd), tl.where(Hp, wh * (1 - rd) / rr, 0.0))
    dn = tl.where(Tp, wh * rd, tl.where(Hp, wh, 0.0))
    return T, b, v, up, dn


@triton.jit(do_not_specialize=['n', 'NBLK', 'max_len'])
def totals_kernel(s_hat, r, slots, m, Tm, part, n, NBLK, max_len,
                  H: tl.constexpr, G: tl.constexpr, BLK: tl.constexpr):
    """Per-block partials over the whole row after T*: (sum v, max up, max dn) [R, NBLK, 3], for
    the "grad" key's dt/dV (``reference.prepare``: V0 = sum v, m0 = max(up, dn))."""
    row = tl.program_id(0)
    blk = tl.program_id(1)
    b = row // (H * G)
    h = (row // G) % H
    slot = tl.load(slots + b).to(tl.int64)
    pos = blk * BLK + tl.arange(0, BLK)
    inb = pos < n
    x, rd, wh, tail, posm, r0 = _entry(s_hat, r, tl.load(m + row), row, slot, h, pos, inb, n,
                                       max_len, H)
    tmv = tl.load(Tm + row.to(tl.int64) * n + pos, mask=inb, other=0)
    T, bb, v, up, dn = _moves(wh, rd, posm, r0, tmv)
    p = part + (row * NBLK + blk) * 3
    tl.store(p, tl.sum(v, axis=0))
    tl.store(p + 1, tl.max(up, axis=0))
    tl.store(p + 2, tl.max(dn, axis=0))


@triton.jit(do_not_specialize=['n', 'NBLK', 'max_len'])
def gradkey_kernel(s_hat, r, slots, m, Tm, dtdv, key, n, NBLK, max_len,
                   H: tl.constexpr, G: tl.constexpr, BLK: tl.constexpr):
    """The "grad" fetch-order key b + v dt/dV on the tail, -1 elsewhere (float64) [R, n]."""
    row = tl.program_id(0)
    blk = tl.program_id(1)
    b = row // (H * G)
    h = (row // G) % H
    slot = tl.load(slots + b).to(tl.int64)
    pos = blk * BLK + tl.arange(0, BLK)
    inb = pos < n
    x, rd, wh, tail, posm, r0 = _entry(s_hat, r, tl.load(m + row), row, slot, h, pos, inb, n,
                                       max_len, H)
    base = row.to(tl.int64) * n
    tmv = tl.load(Tm + base + pos, mask=inb, other=0)
    T, bb, v, up, dn = _moves(wh, rd, posm, r0, tmv)
    tl.store(key + base + pos, tl.where(tail, bb + v * tl.load(dtdv + row), -1.0), mask=inb)


@triton.jit(do_not_specialize=['n', 'NBLK', 'max_len'])
def rest_kernel(s_hat, r, slots, m, Tm, rank, part, n, NBLK, max_len,
                H: tl.constexpr, G: tl.constexpr, BLK: tl.constexpr):
    """Per-block partials over the entries NOT in the top-K fetch order (the complement):
    (sum b, sum v, max up, max dn, #T* entries on the tail) [R, NBLK, 5]."""
    row = tl.program_id(0)
    blk = tl.program_id(1)
    b = row // (H * G)
    h = (row // G) % H
    slot = tl.load(slots + b).to(tl.int64)
    pos = blk * BLK + tl.arange(0, BLK)
    inb = pos < n
    x, rd, wh, tail, posm, r0 = _entry(s_hat, r, tl.load(m + row), row, slot, h, pos, inb, n,
                                       max_len, H)
    base = row.to(tl.int64) * n
    tmv = tl.load(Tm + base + pos, mask=inb, other=0)
    T, bb, v, up, dn = _moves(wh, rd, posm, r0, tmv)
    rest = tl.load(rank + base + pos, mask=inb, other=1) == 0
    p = part + (row * NBLK + blk) * 5
    tl.store(p, tl.sum(tl.where(rest, bb, 0.0), axis=0))
    tl.store(p + 1, tl.sum(tl.where(rest, v, 0.0), axis=0))
    tl.store(p + 2, tl.max(tl.where(rest, up, 0.0), axis=0))
    tl.store(p + 3, tl.max(tl.where(rest, dn, 0.0), axis=0))
    tl.store(p + 4, tl.sum((T & tail & rest).to(tl.float64), axis=0))


@triton.jit
def _max2(a, b):
    return tl.maximum(a, b)


@triton.jit(do_not_specialize=['n', 'K', 'NBLK', 'max_len'])
def curve_kernel(o_i, s_hat, r, slots, m, Tm, part_p, part_r, sure, k_out, cap_out, beta_out,
                 Bt_out, nonmono_out, tsize_out, eps0, delta, Lc, n, K, NBLK, max_len,
                 H: tl.constexpr, G: tl.constexpr, BK: tl.constexpr, BP: tl.constexpr,
                 TIGHT: tl.constexpr):
    """One row: the certificate curves over k = 0..K of the fetch order (the top-K part's suffix
    sums / maxima plus the complement's), k* = the first passing k (``reference.kstar``)."""
    row = tl.program_id(0)
    b = row // (H * G)
    h = (row // G) % H
    slot = tl.load(slots + b).to(tl.int64)
    s_b = tl.zeros([BP], dtype=tl.float64)
    s_v = tl.zeros([BP], dtype=tl.float64)
    m_up = tl.zeros([BP], dtype=tl.float64)
    m_dn = tl.zeros([BP], dtype=tl.float64)
    s_wh = tl.zeros([BP], dtype=tl.float64)
    s_tl = tl.zeros([BP], dtype=tl.float64)
    s_T = tl.zeros([BP], dtype=tl.float64)
    for a in range(0, NBLK, BP):
        i = a + tl.arange(0, BP)
        ok = i < NBLK
        pr = part_r + (row * NBLK + i) * 5
        s_b += tl.load(pr, mask=ok, other=0.0)
        s_v += tl.load(pr + 1, mask=ok, other=0.0)
        m_up = tl.maximum(m_up, tl.load(pr + 2, mask=ok, other=0.0))
        m_dn = tl.maximum(m_dn, tl.load(pr + 3, mask=ok, other=0.0))
        s_T += tl.load(pr + 4, mask=ok, other=0.0)
        pp = part_p + (row * NBLK + i) * 4
        s_wh += tl.load(pp + 1, mask=ok, other=0.0)
        s_tl += tl.load(pp + 3, mask=ok, other=0.0)
    Rb = tl.sum(s_b, axis=0)
    RV = tl.sum(s_v, axis=0)
    Rup = tl.max(m_up, axis=0)
    Rdn = tl.max(m_dn, axis=0)
    n_tail = tl.sum(s_tl, axis=0).to(tl.int64)
    Bt = tl.load(sure + row) + tl.sum(s_wh, axis=0)
    k = tl.arange(0, BK)
    ink = k < K
    oi = tl.load(o_i + row * K + k, mask=ink, other=0)
    x, rd, wh, tail, posm, r0 = _entry(s_hat, r, tl.load(m + row), row, slot, h, oi, ink, n,
                                       max_len, H)
    tmv = tl.load(Tm + row.to(tl.int64) * n + oi, mask=ink, other=0)
    T, bb, v, up, dn = _moves(wh, rd, posm, r0, tmv)
    # Curve at k (0 <= k <= K): the top-K entries from k on, plus the complement. Entry k of
    # these vectors is the curve at k; the curve at K is the complement alone.
    beta = tl.cumsum(bb, 0, reverse=True) + Rb
    V = tl.cumsum(v, 0, reverse=True) + RV
    upc = tl.maximum(tl.associative_scan(up, 0, _max2, reverse=True), Rup)
    dnc = tl.maximum(tl.associative_scan(dn, 0, _max2, reverse=True), Rdn)
    mv = tl.maximum(upc, dnc)
    if TIGHT:
        u_up = eps0 * Bt + beta
        u_lo = eps0 * Bt - beta
        den_u = 2 * (V + upc * u_up / 3)
        den_l = 2 * (V + dnc * u_lo / 3)
        e_u = tl.where(den_u > 0, -(u_up * u_up) / tl.maximum(den_u, 1e-300), NEG_INF)
        e_l = tl.where(den_l > 0, -(u_lo * u_lo) / tl.maximum(den_l, 1e-300), NEG_INF)
        p_u = tl.where(u_up > 0, tl.exp(e_u), 1.0)
        p_l = tl.where(u_lo > 0, tl.exp(e_l), 1.0)
        okk = (p_u + p_l) <= delta
        # the curve at K (complement only)
        uK = eps0 * Bt + Rb
        lK = eps0 * Bt - Rb
        dK_u = 2 * (RV + Rup * uK / 3)
        dK_l = 2 * (RV + Rdn * lK / 3)
        pK_u = tl.where(uK > 0, tl.exp(tl.where(dK_u > 0, -(uK * uK) / tl.maximum(dK_u, 1e-300),
                                                NEG_INF)), 1.0)
        pK_l = tl.where(lK > 0, tl.exp(tl.where(dK_l > 0, -(lK * lK) / tl.maximum(dK_l, 1e-300),
                                                NEG_INF)), 1.0)
        okK = (pK_u + pK_l) <= delta
    else:
        a3 = mv * Lc / 3
        t = a3 + tl.sqrt(a3 * a3 + 2 * V * Lc)
        okk = (beta + t) / Bt <= eps0
        aK = tl.maximum(Rup, Rdn) * Lc / 3
        tK = aK + tl.sqrt(aK * aK + 2 * RV * Lc)
        okK = (Rb + tK) / Bt <= eps0
    okk = okk & ink
    first = tl.min(tl.where(okk, k, BK), axis=0)
    found = first < BK
    kstar = tl.where(found, first, tl.where(okK, K, 0))
    kstar = tl.minimum(kstar.to(tl.int64), n_tail)
    cap = (~found) & (~okK) & (n_tail > K)
    # Non-monotone pass curve within 0..K (passes at some k, fails later): last fail > first pass.
    first_ok = tl.where(found, first, tl.where(okK, K, BK + 1))
    last_fail = tl.max(tl.where(ink & ~okk, k, -1), axis=0)
    last_fail = tl.where(okK, last_fail, K)
    tl.store(k_out + row, kstar)
    tl.store(cap_out + row, cap.to(tl.int8))
    beta_at = tl.sum(tl.where(k == kstar, beta, 0.0), axis=0)
    tl.store(beta_out + row, tl.where(kstar >= K, Rb, beta_at))
    tl.store(Bt_out + row, Bt)
    tl.store(nonmono_out + row, (last_fail > first_ok).to(tl.int8))
    tl.store(tsize_out + row, tl.sum(s_T, axis=0) + tl.sum((T & tail & ink).to(tl.float64),
                                                            axis=0))


@triton.jit
def _tail1(u, V, bm):
    """One-sided Bernstein exp(-u^2 / (2 (V + b u / 3))) for u > 0, 1 for u <= 0
    (``reference._tail_term``)."""
    den = 2 * (V + bm * u / 3)
    e = tl.where(den > 0, -(u * u) / tl.maximum(den, 1e-300), NEG_INF)
    return tl.where(u > 0, libdevice.exp(e), 1.0)


@triton.jit
def _lae32(x, y):
    """torch.logaddexp in float32: max + log1p(exp(-|x - y|))."""
    return tl.maximum(x, y) + libdevice.log1p(libdevice.exp(-tl.abs(x - y)))


@triton.jit
def _bennett(lam, Vr, M):
    """Bennett's bound on the rest's log-MGF: Vr (exp(lam M) - 1 - lam M) / M^2 (0 if Vr = 0,
    where the harness's closed form is 0 / 0 but the rest has nothing random)."""
    x = tl.minimum(lam * M, 700.0)
    return tl.where(Vr > 0, Vr * (libdevice.expm1(x) - x) / (M * M), 0.0)


@triton.jit
def _chern_lam(lam, c, live, lr, l1r, r32, u_up, u_dn, uK_up, uK_dn, RV, Mu, Md,
               BENNETT: tl.constexpr):
    """One lambda of the grid: the bound's terms exp(K(lam, k) - lam u(k)) at every k of the head
    (up / down) and at k = K."""
    a = lam.to(tl.float32) * c
    k_up = tl.where(live, _lae32(lr + a * (1 - r32), l1r - a * r32), 0.0)
    k_dn = tl.where(live, _lae32(lr - a * (1 - r32), l1r + a * r32), 0.0)
    Ku = tl.cumsum(k_up.to(tl.float64), 0, reverse=True)
    Kd = tl.cumsum(k_dn.to(tl.float64), 0, reverse=True)
    ex_up = tl.zeros([], dtype=tl.float64)
    ex_dn = tl.zeros([], dtype=tl.float64)
    if BENNETT:
        ex_up = _bennett(lam, RV, Mu)
        ex_dn = _bennett(lam, RV, Md)
    t_up = libdevice.exp(tl.minimum(Ku + ex_up - lam * u_up, 0.0))
    t_dn = tl.where(u_dn > 0, libdevice.exp(tl.minimum(Kd + ex_dn - lam * u_dn, 0.0)), 1.0)
    tK_up = libdevice.exp(tl.minimum(ex_up - lam * uK_up, 0.0))
    tK_dn = tl.where(uK_dn > 0, libdevice.exp(tl.minimum(ex_dn - lam * uK_dn, 0.0)), 1.0)
    return t_up, t_dn, tK_up, tK_dn


@triton.jit(do_not_specialize=['n', 'K', 'NBLK', 'max_len'])
def chern_kernel(o_i, s_hat, r, slots, m, Tm, part_p, part_r, sure, grid, k_out, cap_out,
                 beta_out, Bt_out, nonmono_out, tsize_out, scr_out, eps0, delta, thr,
                 n, K, NBLK, max_len,
                 H: tl.constexpr, G: tl.constexpr, BK: tl.constexpr, BP: tl.constexpr,
                 NG: tl.constexpr, BENNETT: tl.constexpr):
    """One row, ``reference_chernoff.kstar_fast`` with head = K: the Bernstein screen at k = 0;
    else the two-sided Chernoff bound min over the lambda grid of exp(K(lam, k) - lam u(k)) for
    k = 0..K, with the exact two-point log-MGF of the top-K entries of the fetch order from k on
    plus (BENNETT, K < n) Bennett's bound for the rest; k* = the first k with P <= thr. cap =
    unscreened and no k <= K passes (the caller recomputes those rows by the reference).

    One program per row; ``chern_elem_kernel`` .. ``chern_finish_kernel`` are the same
    computation split over many programs (bit-identical k*, and the decoder's default)."""
    row = tl.program_id(0)
    b = row // (H * G)
    h = (row // G) % H
    slot = tl.load(slots + b).to(tl.int64)
    s_b = tl.zeros([BP], dtype=tl.float64)
    s_v = tl.zeros([BP], dtype=tl.float64)
    m_up = tl.zeros([BP], dtype=tl.float64)
    m_dn = tl.zeros([BP], dtype=tl.float64)
    s_wh = tl.zeros([BP], dtype=tl.float64)
    s_tl = tl.zeros([BP], dtype=tl.float64)
    s_T = tl.zeros([BP], dtype=tl.float64)
    for a in range(0, NBLK, BP):
        i = a + tl.arange(0, BP)
        ok = i < NBLK
        pr = part_r + (row * NBLK + i) * 5
        s_b += tl.load(pr, mask=ok, other=0.0)
        s_v += tl.load(pr + 1, mask=ok, other=0.0)
        m_up = tl.maximum(m_up, tl.load(pr + 2, mask=ok, other=0.0))
        m_dn = tl.maximum(m_dn, tl.load(pr + 3, mask=ok, other=0.0))
        s_T += tl.load(pr + 4, mask=ok, other=0.0)
        pp = part_p + (row * NBLK + i) * 4
        s_wh += tl.load(pp + 1, mask=ok, other=0.0)
        s_tl += tl.load(pp + 3, mask=ok, other=0.0)
    Rb = tl.sum(s_b, axis=0)
    RV = tl.sum(s_v, axis=0)
    Rup = tl.max(m_up, axis=0)
    Rdn = tl.max(m_dn, axis=0)
    n_tail = tl.sum(s_tl, axis=0).to(tl.int64)
    Bt = tl.load(sure + row) + tl.sum(s_wh, axis=0)
    k = tl.arange(0, BK)
    ink = k < K
    oi = tl.load(o_i + row * K + k, mask=ink, other=0)
    x, rd, wh, tail, posm, r0 = _entry(s_hat, r, tl.load(m + row), row, slot, h, oi, ink, n,
                                       max_len, H)
    tmv = tl.load(Tm + row.to(tl.int64) * n + oi, mask=ink, other=0)
    T, bb, v, up, dn = _moves(wh, rd, posm, r0, tmv)
    beta = tl.cumsum(bb, 0, reverse=True) + Rb         # curve at k < K; at K: Rb
    e0Bt = eps0 * Bt
    # Screen (Plan.p_fail_tight at k = 0: the row totals).
    beta0 = tl.sum(bb, axis=0) + Rb
    V0 = tl.sum(v, axis=0) + RV
    up0 = tl.maximum(tl.max(up, axis=0), Rup)
    dn0 = tl.maximum(tl.max(dn, axis=0), Rdn)
    screened = (_tail1(e0Bt + beta0, V0, up0) + _tail1(e0Bt - beta0, V0, dn0)) <= delta
    # Two-point terms (token_terms' c; float32 like kstar_fast).
    Tp = T & posm
    Hp = posm & ~T
    c = tl.where(Tp, wh, tl.where(Hp, wh / tl.maximum(rd, 1e-300), 0.0)).to(tl.float32)
    live = c > 0
    lr = tl.log(tl.minimum(tl.maximum(rd, 1e-300), 1.0)).to(tl.float32)
    l1r = libdevice.log1p(-tl.minimum(tl.maximum(rd, 0.0), 1 - 1e-16)).to(tl.float32)
    r32 = rd.to(tl.float32)
    u_up = e0Bt + beta
    u_dn = e0Bt - beta
    uK_up = e0Bt + Rb
    uK_dn = e0Bt - Rb
    Mu = tl.maximum(Rup, 1e-300)
    Md = tl.maximum(Rdn, 1e-300)
    best_up = tl.full([BK], 1.0, tl.float64)
    best_dn = tl.full([BK], 1.0, tl.float64)
    bK_up = tl.full([], 1.0, tl.float64)
    bK_dn = tl.full([], 1.0, tl.float64)
    for gi in tl.static_range(NG):
        lam = tl.load(grid + gi) / Bt
        t_up, t_dn, tK_up, tK_dn = _chern_lam(lam, c, live, lr, l1r, r32, u_up, u_dn,
                                              uK_up, uK_dn, RV, Mu, Md, BENNETT)
        best_up = tl.minimum(best_up, t_up)
        best_dn = tl.minimum(best_dn, t_dn)
        bK_up = tl.minimum(bK_up, tK_up)
        bK_dn = tl.minimum(bK_dn, tK_dn)
    okk = ((best_up + best_dn) <= thr) & ink
    okK = (bK_up + bK_dn) <= thr
    first = tl.min(tl.where(okk, k, BK), axis=0)
    found = first < BK
    kstar = tl.where(found, first, tl.where(okK, K, 0))
    kstar = tl.where(screened, 0, kstar)
    kstar = tl.minimum(kstar.to(tl.int64), n_tail)
    cap = (~screened) & (~found) & (~okK)
    first_ok = tl.where(found, first, tl.where(okK, K, BK + 1))
    last_fail = tl.max(tl.where(ink & ~okk, k, -1), axis=0)
    last_fail = tl.where(okK, last_fail, K)
    tl.store(k_out + row, kstar)
    tl.store(cap_out + row, cap.to(tl.int8))
    tl.store(scr_out + row, screened.to(tl.int8))
    beta_at = tl.sum(tl.where(k == kstar, beta, 0.0), axis=0)
    tl.store(beta_out + row, tl.where(kstar >= K, Rb, beta_at))
    tl.store(Bt_out + row, Bt)
    tl.store(nonmono_out + row, ((~screened) & (last_fail > first_ok)).to(tl.int8))
    tl.store(tsize_out + row, tl.sum(s_T, axis=0) + tl.sum((T & tail & ink).to(tl.float64),
                                                            axis=0))


# ---- the certificate, split over many programs ------------------------------------------------
# chern_kernel's computation in four launches with the same expressions and, for every reduction
# and scan, the same [BK] block and num_warps, so k* and every output are bit-identical:
#   elem   (rows, BK / EB): per-entry terms (_entry, _moves, the two-point inputs);
#   row    (rows):          the NBLK partials, the screen, beta (scan), u(k), scalars;
#   lam    (rows, NS):      NG / NS lambdas each, per-part minima of the bound (a min is exact);
#   finish (rows):          min over the parts, k*, cap, the statistics.
# Row scalars (sc, 8 per row): Bt, Rb, RV, Rup, Rdn, screened, n_tail, tsize.

@triton.jit(do_not_specialize=['n', 'K', 'max_len'])
def chern_elem_kernel(o_i, s_hat, r, slots, m, Tm, f32_o, f64_o, n, K, max_len,
                      H: tl.constexpr, G: tl.constexpr, BK: tl.constexpr, EB: tl.constexpr):
    row = tl.program_id(0)
    b = row // (H * G)
    h = (row // G) % H
    slot = tl.load(slots + b).to(tl.int64)
    k = tl.program_id(1) * EB + tl.arange(0, EB)
    ink = k < K
    oi = tl.load(o_i + row * K + k, mask=ink, other=0)
    x, rd, wh, tail, posm, r0 = _entry(s_hat, r, tl.load(m + row), row, slot, h, oi, ink, n,
                                       max_len, H)
    tmv = tl.load(Tm + row.to(tl.int64) * n + oi, mask=ink, other=0)
    T, bb, v, up, dn = _moves(wh, rd, posm, r0, tmv)
    Tp = T & posm
    Hp = posm & ~T
    c = tl.where(Tp, wh, tl.where(Hp, wh / tl.maximum(rd, 1e-300), 0.0)).to(tl.float32)
    lr = tl.log(tl.minimum(tl.maximum(rd, 1e-300), 1.0)).to(tl.float32)
    l1r = libdevice.log1p(-tl.minimum(tl.maximum(rd, 0.0), 1 - 1e-16)).to(tl.float32)
    RB = tl.num_programs(0).to(tl.int64) * BK                     # one array's stride
    o = row.to(tl.int64) * BK + k
    tl.store(f32_o + o, c)
    tl.store(f32_o + RB + o, lr)
    tl.store(f32_o + 2 * RB + o, l1r)
    tl.store(f32_o + 3 * RB + o, rd.to(tl.float32))
    tl.store(f64_o + o, bb)
    tl.store(f64_o + RB + o, v)
    tl.store(f64_o + 2 * RB + o, up)
    tl.store(f64_o + 3 * RB + o, dn)
    tl.store(f64_o + 4 * RB + o, (T & tail & ink).to(tl.float64))


@triton.jit(do_not_specialize=['NBLK'])
def chern_row_kernel(part_p, part_r, sure, f64_o, sc, eps0, delta, NBLK,
                     BK: tl.constexpr, BP: tl.constexpr):
    row = tl.program_id(0)
    s_b = tl.zeros([BP], dtype=tl.float64)
    s_v = tl.zeros([BP], dtype=tl.float64)
    m_up = tl.zeros([BP], dtype=tl.float64)
    m_dn = tl.zeros([BP], dtype=tl.float64)
    s_wh = tl.zeros([BP], dtype=tl.float64)
    s_tl = tl.zeros([BP], dtype=tl.float64)
    s_T = tl.zeros([BP], dtype=tl.float64)
    for a in range(0, NBLK, BP):
        i = a + tl.arange(0, BP)
        ok = i < NBLK
        pr = part_r + (row * NBLK + i) * 5
        s_b += tl.load(pr, mask=ok, other=0.0)
        s_v += tl.load(pr + 1, mask=ok, other=0.0)
        m_up = tl.maximum(m_up, tl.load(pr + 2, mask=ok, other=0.0))
        m_dn = tl.maximum(m_dn, tl.load(pr + 3, mask=ok, other=0.0))
        s_T += tl.load(pr + 4, mask=ok, other=0.0)
        pp = part_p + (row * NBLK + i) * 4
        s_wh += tl.load(pp + 1, mask=ok, other=0.0)
        s_tl += tl.load(pp + 3, mask=ok, other=0.0)
    Rb = tl.sum(s_b, axis=0)
    RV = tl.sum(s_v, axis=0)
    Rup = tl.max(m_up, axis=0)
    Rdn = tl.max(m_dn, axis=0)
    n_tail = tl.sum(s_tl, axis=0).to(tl.int64)
    Bt = tl.load(sure + row) + tl.sum(s_wh, axis=0)
    RB = tl.num_programs(0).to(tl.int64) * BK
    k = tl.arange(0, BK)
    o = row.to(tl.int64) * BK + k
    bb = tl.load(f64_o + o)
    v = tl.load(f64_o + RB + o)
    up = tl.load(f64_o + 2 * RB + o)
    dn = tl.load(f64_o + 3 * RB + o)
    tt = tl.load(f64_o + 4 * RB + o)
    beta = tl.cumsum(bb, 0, reverse=True) + Rb
    e0Bt = eps0 * Bt
    beta0 = tl.sum(bb, axis=0) + Rb
    V0 = tl.sum(v, axis=0) + RV
    up0 = tl.maximum(tl.max(up, axis=0), Rup)
    dn0 = tl.maximum(tl.max(dn, axis=0), Rdn)
    screened = (_tail1(e0Bt + beta0, V0, up0) + _tail1(e0Bt - beta0, V0, dn0)) <= delta
    tl.store(f64_o + 5 * RB + o, beta)
    tl.store(f64_o + 6 * RB + o, e0Bt + beta)
    tl.store(f64_o + 7 * RB + o, e0Bt - beta)
    so = sc + row * 8
    tl.store(so, Bt)
    tl.store(so + 1, Rb)
    tl.store(so + 2, RV)
    tl.store(so + 3, Rup)
    tl.store(so + 4, Rdn)
    tl.store(so + 5, screened.to(tl.float64))
    tl.store(so + 6, n_tail.to(tl.float64))
    tl.store(so + 7, tl.sum(s_T, axis=0) + tl.sum(tt, axis=0))


@triton.jit
def chern_lam_kernel(f32_o, f64_o, sc, grid, best_buf, bK_buf, eps0,
                     BK: tl.constexpr, NG: tl.constexpr, NS: tl.constexpr,
                     BENNETT: tl.constexpr):
    row = tl.program_id(0)
    sp = tl.program_id(1)
    RB = tl.num_programs(0).to(tl.int64) * BK
    k = tl.arange(0, BK)
    o = row.to(tl.int64) * BK + k
    so = sc + row * 8
    Bt = tl.load(so)
    Rb = tl.load(so + 1)
    RV = tl.load(so + 2)
    Mu = tl.maximum(tl.load(so + 3), 1e-300)
    Md = tl.maximum(tl.load(so + 4), 1e-300)
    c = tl.load(f32_o + o)
    lr = tl.load(f32_o + RB + o)
    l1r = tl.load(f32_o + 2 * RB + o)
    r32 = tl.load(f32_o + 3 * RB + o)
    u_up = tl.load(f64_o + 6 * RB + o)
    u_dn = tl.load(f64_o + 7 * RB + o)
    live = c > 0
    e0Bt = eps0 * Bt
    uK_up = e0Bt + Rb
    uK_dn = e0Bt - Rb
    best_up = tl.full([BK], 1.0, tl.float64)
    best_dn = tl.full([BK], 1.0, tl.float64)
    bK_up = tl.full([], 1.0, tl.float64)
    bK_dn = tl.full([], 1.0, tl.float64)
    for j in tl.static_range(NG // NS):
        lam = tl.load(grid + sp * (NG // NS) + j) / Bt
        t_up, t_dn, tK_up, tK_dn = _chern_lam(lam, c, live, lr, l1r, r32, u_up, u_dn,
                                              uK_up, uK_dn, RV, Mu, Md, BENNETT)
        best_up = tl.minimum(best_up, t_up)
        best_dn = tl.minimum(best_dn, t_dn)
        bK_up = tl.minimum(bK_up, tK_up)
        bK_dn = tl.minimum(bK_dn, tK_dn)
    base = (row.to(tl.int64) * NS + sp) * 2
    tl.store(best_buf + base * BK + k, best_up)
    tl.store(best_buf + (base + 1) * BK + k, best_dn)
    tl.store(bK_buf + base, bK_up)
    tl.store(bK_buf + base + 1, bK_dn)


@triton.jit(do_not_specialize=['K'])
def chern_finish_kernel(f64_o, sc, best_buf, bK_buf, k_out, cap_out, beta_out, Bt_out,
                        nonmono_out, tsize_out, scr_out, thr, K,
                        BK: tl.constexpr, NS: tl.constexpr):
    row = tl.program_id(0)
    RB = tl.num_programs(0).to(tl.int64) * BK
    k = tl.arange(0, BK)
    ink = k < K
    so = sc + row * 8
    Bt = tl.load(so)
    Rb = tl.load(so + 1)
    screened = tl.load(so + 5) != 0
    n_tail = tl.load(so + 6).to(tl.int64)
    beta = tl.load(f64_o + 5 * RB + row.to(tl.int64) * BK + k)
    best_up = tl.full([BK], 1.0, tl.float64)
    best_dn = tl.full([BK], 1.0, tl.float64)
    bK_up = tl.full([], 1.0, tl.float64)
    bK_dn = tl.full([], 1.0, tl.float64)
    for sp in tl.static_range(NS):
        base = (row.to(tl.int64) * NS + sp) * 2
        best_up = tl.minimum(best_up, tl.load(best_buf + base * BK + k))
        best_dn = tl.minimum(best_dn, tl.load(best_buf + (base + 1) * BK + k))
        bK_up = tl.minimum(bK_up, tl.load(bK_buf + base))
        bK_dn = tl.minimum(bK_dn, tl.load(bK_buf + base + 1))
    okk = ((best_up + best_dn) <= thr) & ink
    okK = (bK_up + bK_dn) <= thr
    first = tl.min(tl.where(okk, k, BK), axis=0)
    found = first < BK
    kstar = tl.where(found, first, tl.where(okK, K, 0))
    kstar = tl.where(screened, 0, kstar)
    kstar = tl.minimum(kstar.to(tl.int64), n_tail)
    cap = (~screened) & (~found) & (~okK)
    first_ok = tl.where(found, first, tl.where(okK, K, BK + 1))
    last_fail = tl.max(tl.where(ink & ~okk, k, -1), axis=0)
    last_fail = tl.where(okK, last_fail, K)
    tl.store(k_out + row, kstar)
    tl.store(cap_out + row, cap.to(tl.int8))
    tl.store(scr_out + row, screened.to(tl.int8))
    beta_at = tl.sum(tl.where(k == kstar, beta, 0.0), axis=0)
    tl.store(beta_out + row, tl.where(kstar >= K, Rb, beta_at))
    tl.store(Bt_out + row, Bt)
    tl.store(nonmono_out + row, ((~screened) & (last_fail > first_ok)).to(tl.int8))
    tl.store(tsize_out + row, tl.load(so + 7))


@triton.jit(do_not_specialize=['n', 'W', 'Wtot'])
def weights_kernel(pos_a, r_a, phys_a, acnt, slots, Tm, rank, k_star, idx_out, w_out, slen_out,
                   n, W, Wtot,
                   H: tl.constexpr, G: tl.constexpr, BLK: tl.constexpr, HAS_TAIL: tl.constexpr):
    """Attention list entries for A (Algorithm 2 (e) as weights): 1 on non-tail / F / sampled
    T*, 1/r on the other sampled tail entries. [B, Hq, Wtot] index / weight, A first."""
    row = tl.program_id(0)
    blk = tl.program_id(1)
    b = row // (H * G)
    h = (row // G) % H
    slot = tl.load(slots + b).to(tl.int64)
    cnt = tl.load(acnt + slot * H + h)
    offs = blk * BLK + tl.arange(0, BLK)
    live = offs < cnt
    abase = (b * H + h).to(tl.int64) * W
    pos = tl.load(pos_a + abase + offs, mask=live, other=0).to(tl.int64)
    rv = tl.load(r_a + abase + offs, mask=live, other=1.0)
    ph = tl.load(phys_a + abase + offs, mask=live, other=0)
    w = tl.full([BLK], 1.0, tl.float32)
    if HAS_TAIL:
        inctx = live & (pos < n)
        base = row.to(tl.int64) * n
        inT = tl.load(Tm + base + pos, mask=inctx, other=0) != 0
        rk = tl.load(rank + base + pos, mask=inctx, other=0)
        inF = (rk > 0) & (rk <= tl.load(k_star + row))
        samp = live & (rv < 1.0)
        w = tl.where(samp & ~inT & ~inF, 1.0 / tl.maximum(rv, 1e-30), w)
    obase = row.to(tl.int64) * Wtot
    tl.store(idx_out + obase + offs, ph, mask=live)
    tl.store(w_out + obase + offs, w, mask=live)
    if blk == 0:
        tl.store(slen_out + row, cnt)


@triton.jit(do_not_specialize=['R', 'Hq'])
def stats_kernel(stats, ctx_d, k, tsize, beta, Bt, nonmono, fb, scr, slen, xfer, keys, R, Hq,
                 chunked, BR: tl.constexpr):
    """Accumulate one layer call's statistics (``state.STATS`` order) in one launch."""
    i = tl.arange(0, BR)
    ok = i < R
    has = ok & (tl.load(ctx_d + i // Hq, mask=ok, other=0) > 0)
    kk = tl.where(has, tl.load(k + i, mask=ok, other=0), 0).to(tl.float64)
    vals0 = tl.sum(has.to(tl.float64), axis=0)
    vals1 = tl.sum((kk > 0).to(tl.float64), axis=0)
    vals2 = tl.sum(kk, axis=0)
    vals4 = tl.sum(tl.where(has, tl.load(tsize + i, mask=ok, other=0.0), 0.0), axis=0)
    bt = tl.load(Bt + i, mask=ok, other=1.0)
    vals5 = tl.sum(tl.where(has, tl.load(beta + i, mask=ok, other=0.0) / bt, 0.0), axis=0)
    vals6 = tl.sum((has & (tl.load(nonmono + i, mask=ok, other=0) != 0)).to(tl.float64), axis=0)
    vals7 = tl.sum((has & (tl.load(fb + i, mask=ok, other=0) != 0)).to(tl.float64), axis=0)
    vals8 = tl.sum((has & (tl.load(scr + i, mask=ok, other=0) != 0)).to(tl.float64), axis=0)
    att = tl.sum(tl.load(slen + i, mask=ok, other=0).to(tl.float64), axis=0)
    xf = tl.load(xfer).to(tl.float64)
    ky = tl.load(keys).to(tl.float64)
    tl.store(stats + 0, tl.load(stats + 0) + vals0)
    tl.store(stats + 1, tl.load(stats + 1) + vals1)
    tl.store(stats + 2, tl.load(stats + 2) + vals2)
    tl.store(stats + 3, tl.load(stats + 3) + xf)
    tl.store(stats + 4, tl.load(stats + 4) + vals4)
    tl.store(stats + 5, tl.load(stats + 5) + vals5)
    tl.store(stats + 6, tl.load(stats + 6) + vals6)
    tl.store(stats + 7, tl.load(stats + 7) + vals7)
    tl.store(stats + 8, tl.load(stats + 8) + vals8)
    tl.store(stats + 9, tl.load(stats + 9) + 1.0)
    tl.store(stats + 10, tl.load(stats + 10) + chunked)
    tl.store(stats + 11, tl.load(stats + 11) + ky)
    tl.store(stats + 12, tl.load(stats + 12) + att + xf)


@triton.jit(do_not_specialize=['W', 'max_len'])
def fa_table_kernel(alist, acnt, cell, slots, bt, seq, W, max_len, home,
                    H: tl.constexpr, BLK: tl.constexpr, HAS_HOME: tl.constexpr = False):
    """FlashAttention page table (page size 1) of (row b, KV head h): the cells of its attended
    list A in list order, bt [B * H, W] int32 (0 past the count), seq [B * H] int32 = the count
    (0 for a padding row, slot < 0)."""
    bh = tl.program_id(0)
    blk = tl.program_id(1)
    b = bh // H
    h = bh % H
    slot = tl.load(slots + b).to(tl.int64)
    cnt = tl.load(acnt + tl.maximum(slot, 0) * H + h)
    cnt = tl.where(slot >= 0, tl.minimum(cnt, W), 0)
    slot = tl.maximum(slot, 0)
    offs = blk * BLK + tl.arange(0, BLK)
    live = offs < cnt
    pos = tl.load(alist + (slot * H + h) * max_len + offs, mask=live, other=0).to(tl.int64)
    phys = tl.load(cell + (slot * max_len + pos) * H + h, mask=live, other=0)
    if HAS_HOME:
        # recall: an offloaded entry is attended in its home cell (copied back this step).
        hm = tl.load(home + (slot * max_len + pos) * H + h, mask=live & (phys < 0), other=0)
        phys = tl.where(phys < 0, hm, phys)
    tl.store(bt + bh.to(tl.int64) * W + offs, tl.where(live, phys, 0), mask=offs < W)
    if blk == 0:
        tl.store(seq + bh, cnt)


@triton.jit(do_not_specialize=['WF', 'W1', 'W2', 'max_len'])
def restore_place_kernel(cold, posc, kst, base, slots, home, restored, cpu_row, pool, kv,
                         idx, w, slen, idx2, w2, slen2, WF, W1, W2, max_len,
                         H: tl.constexpr, G: tl.constexpr, C2: tl.constexpr, PC2: tl.constexpr,
                         BLK: tl.constexpr):
    """No compaction: F's cold entries of one row (columns < k* of cold / posc [R, WF], fetch
    order) are copied from the pinned CPU pool back into their own cells (``home``), once per
    entry -- the first row to flip ``restored`` copies, the query heads sharing the KV head and
    later steps find it there -- and listed with weight 1 after the row's A prefix (``base``):
    the main list while it has room (W1 columns), the rest in the overflow list (W2)."""
    row = tl.program_id(0)
    b = row // (H * G)
    h = (row // G) % H
    slot = tl.load(slots + b).to(tl.int64)
    off = tl.load(base + row).to(tl.int64)
    kk = tl.minimum(tl.load(kst + row), WF)
    rb = row.to(tl.int64) * WF
    d = tl.arange(0, C2)
    for a in range(0, kk, BLK):
        j = a + tl.arange(0, BLK)
        cd = (j < kk) & (tl.load(cold + rb + j, mask=j < kk, other=0) != 0)
        pos = tl.load(posc + rb + j, mask=cd, other=0)
        mi = (slot * max_len + pos) * H + h
        hc = tl.load(home + mi, mask=cd, other=0).to(tl.int64)
        old = tl.atomic_xchg(restored + mi, 1, mask=cd)
        win = cd & (old == 0)
        cr = tl.load(cpu_row + mi, mask=win, other=0).to(tl.int64)
        val = tl.load(pool + cr[:, None] * PC2 + d[None, :], mask=win[:, None], other=0.0)
        tl.store(kv + hc[:, None] * C2 + d[None, :], val, mask=win[:, None])
        dst = off + tl.cumsum(cd.to(tl.int64), 0) - 1
        m1 = cd & (dst < W1)
        m2 = cd & (dst >= W1)
        tl.store(idx + row.to(tl.int64) * W1 + dst, hc, mask=m1)
        tl.store(w + row.to(tl.int64) * W1 + dst, 1.0, mask=m1)
        tl.store(idx2 + row.to(tl.int64) * W2 + dst - W1, hc, mask=m2)
        tl.store(w2 + row.to(tl.int64) * W2 + dst - W1, 1.0, mask=m2)
        off += tl.sum(cd.to(tl.int64), 0)
    tl.store(slen + row, tl.minimum(off, W1).to(tl.int32))
    tl.store(slen2 + row, tl.maximum(off - W1, 0).to(tl.int32))


@triton.jit(do_not_specialize=['E'])
def recall_copy_kernel(src, dst, cnt, pool, kv, E, C2: tl.constexpr, PC2: tl.constexpr,
                       BLK: tl.constexpr):
    """``recall``: copy one layer's offloaded entries (src CPU rows / dst home cells [E] int64,
    the first ``cnt`` live) from the pinned CPU pool back to the GPU cells."""
    offs = tl.program_id(0) * BLK + tl.arange(0, BLK)
    live = offs < tl.load(cnt)
    cr = tl.load(src + offs, mask=live, other=0)
    hc = tl.load(dst + offs, mask=live, other=0)
    d = tl.arange(0, C2)
    val = tl.load(pool + cr[:, None] * PC2 + d[None, :], mask=live[:, None], other=0.0)
    tl.store(kv + hc[:, None] * C2 + d[None, :], val, mask=live[:, None])
