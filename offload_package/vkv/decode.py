"""Vkv decode step for one layer: Algorithm 2 on the per-head token engine, then sparse attention.

Per decode row b, query head h (KV head g = h // group), as the harness's ``vkv_final_bias`` does
per row (``phase4/vkv_final/press_final.py``):

  attended A   every resident cell of (b, g): context entries with r = 1, the sampled tail S,
               generated tokens;
  tail         context entries with r < 1 (sampled S on the GPU, dropped ones on the CPU);
  shortlist    per (b, g): the ``shortlist`` tail positions of largest sketch logit over the
               group's query heads; their exact logits (keys from the resident cell, or read
               straight from the pinned CPU pool) replace the sketch logits;
  plan weights wh = exp(plan logit - m) on the tail (sketch, exact on the shortlist); exact
               w = exp(s - m) on A \\ tail (sure); m = max over tail plan logits and exact
               non-tail logits;
  decision     T*, k*, F (``decision.decide`` / the fused kernels): "grad" fetch order and the
               Chernoff certificate by default;
  attention    A with weight 1 (non-tail, F, S cap T*) or 1/r (S \\ T* \\ F), plus F \\ S fetched
               from the CPU into staging with weight 1: the kernel adds log2(weight) to the logit,
               which is the harness's additive bias.

Nothing persists: fetched entries serve this forward only (the staging ring may keep them, so a
later fetch of the same entry can hit, but r, the tail and A never change).

``layer`` runs the fused kernels (``kernels.py``, ``topk.py``, ``fallback.py``) with no host read:
capped rows get the reference decision on the GPU, and the fetch lists have fixed widths (A plus
k_max per row, an overflow list for capped rows past that). With VKV_CUDA=1 the decision runs
in two CUDA launches instead (``_layer_cuda``, ``csrc/vkv_decide.cu``) where ``_cuda_ok`` holds.
``layer_torch`` is the same step in plain torch over ``decision.decide`` (the cross-check, and
k_max > ``FUSED_K_MAX``). ``layer_graphed`` replays ``layer`` from per-layer CUDA graphs.
"""
from __future__ import annotations

import functools
import math
import os

import torch
import triton

from . import kernels as K
from . import reference as ref
from . import reference_chernoff as chern
from .decision import CHERN_MARGIN, attention_weights, decide, reference_rows
from .state import STATS
from . import fallback as FB
from .topk import row_topk

STATS_COPIED = STATS.index("copied")

_NEG_INF = float("-inf")
_LOG2E = 1.0 / math.log(2.0)
try:
    from vllm.vllm_flash_attn import flash_attn_varlen_func as _fa_varlen
except Exception:  # pragma: no cover - FA3 build absent
    _fa_varlen = None
_BLK = 128              # positions per program of the tail kernels
_BLK_A = 32             # attended entries per program
_BLK_S = 32             # shortlist entries per program
_BLK_T = 1024           # positions per program of the shortlist's group max
_BLK_F = 1024           # fetched entries per program (cold / place kernels)
_BP = 256               # partials per step of the per-row reductions
FUSED_K_MAX = 4096
# Attended-list width bucket of the per-layer CUDA graphs.
GRAPH_WA_BUCKET = int(os.environ.get("VKV_GRAPH_WA_BUCKET", "512"))
GRAPH_MAX = int(os.environ.get("VKV_GRAPH_MAX", "256"))      # graphs kept (LRU)
# Programs per row for the certificate's lambda grid (1 = the single-program kernel).
CHERN_SPLIT = int(os.environ.get("OFFLOAD_PACKAGE_VKV_CHERN_SPLIT", "32"))
_CHERN_EB = 512          # entries per program of the certificate's per-entry kernel
CUDA_CLUSTER = 4         # most blocks per row of the CUDA decide kernel (it picks <= this)


def lse2_merge(out_a, lse_a, out_b, lse_b):
    """Merge two log2-domain partial softmaxes over disjoint key sets (exact). A part with no
    keys (lse = -inf) contributes nothing, whatever its output holds."""
    m = torch.maximum(lse_a, lse_b)
    m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
    sa = torch.exp2(lse_a - m)
    sb = torch.exp2(lse_b - m)
    s = sa + sb
    scale = torch.where(s > 0, s.reciprocal(), torch.zeros_like(s)).unsqueeze(-1)
    pa = torch.where(sa.unsqueeze(-1) > 0, out_a.float() * sa.unsqueeze(-1), 0.0)
    pb = torch.where(sb.unsqueeze(-1) > 0, out_b.float() * sb.unsqueeze(-1), 0.0)
    lse = torch.where(s > 0, m + torch.log2(s), lse_a)
    return (pa + pb) * scale, lse


def _zeros(dev, *specs):
    """Zeroed tensors (shape, dtype) carved from one buffer: one memset for all of them."""
    sizes = [math.prod(shape) * dt.itemsize for shape, dt in specs]
    offs, o = [], 0
    for sz in sizes:
        offs.append(o)
        o += (sz + 15) // 16 * 16
    buf = torch.zeros((max(o, 16),), dtype=torch.uint8, device=dev)
    return [buf[a:a + sz].view(dt).view(shape)
            for a, sz, (shape, dt) in zip(offs, sizes, specs)]


def _buffers_sig(*objs):
    """(address, shape) of every device tensor the objects hold (directly or in a list): a
    captured layer is valid while none of them is reallocated."""
    out = []
    for o in objs:
        for v in vars(o).values():
            if isinstance(v, torch.Tensor):
                if v.is_cuda or v.is_pinned():
                    out.append((v.data_ptr(), tuple(v.shape)))
            elif isinstance(v, (list, tuple)) and v and isinstance(v[0], torch.Tensor):
                out.extend((t.data_ptr(), tuple(t.shape)) for t in v)
    return tuple(out)


@functools.lru_cache(maxsize=None)
def _np2(x: int) -> int:
    return triton.next_power_of_2(max(1, int(x)))


@functools.lru_cache(maxsize=None)
def _ln2_over(delta: float) -> float:
    # Same expression as reference.prepare.
    return float(torch.log(torch.tensor(2.0 / delta, dtype=torch.float64)))


class VkvDecoder:
    def __init__(self, engine, store, sparse_ops) -> None:
        self.eng = engine
        self.store = store
        self.st = engine.vkv
        self.ops = sparse_ops
        self._ident = {}
        self._k_split = {}
        self.rows = None                  # per step: see begin_step
        self.torch_path = os.environ.get("VKV_DECODE", "fused") == "torch"
        self._grid_d = None             # chern.GRID on the device (lazy: the decoder may be built
        self._zero_d = None             # inside a graph capture, where H2D copies are illegal)
        self._consts = {}
        # The shortlist kernel reads cold keys straight from the CPU pool (device-mapped pinned
        # memory); otherwise the torch path gathers them on the host.
        self._pool_mapped = bool(engine.cpu_pool.is_pinned())
        self._row_t = {}
        # Attend-only rows (n = 0: before the round, KVzip eviction) through FlashAttention
        # (VKV_FA=0: the sparse decode kernel).
        self.fa = os.environ.get("VKV_FA", "1") == "1" and _fa_varlen is not None
        self._cu = {}
        self._fa_cache = None
        # VKV_CUDA=1: the decision in CUDA (cuda_decide: one decide + one emit launch per layer)
        # where it applies (see _cuda_ok); else the Triton kernels.
        self.cuda = os.environ.get("VKV_CUDA", "0") == "1" and not self.torch_path
        self._sms = None
        # vLLM FULL decode graphs (set by the wrapper): layer_full only, rows must be attend-only.
        self.full = False
        self._full_sig = None
        # Per-layer CUDA graphs (VKV_GRAPHS=1): see layer_graphed.
        self.graphs = os.environ.get("VKV_GRAPHS", "0") == "1" and not self.torch_path
        self._graphs = {}                 # key -> (graph, q in, out, lse out)
        self._warm = set()
        self._gpool = None
        self._gio = {}
        self.captures = 0

    @property
    def _grid(self):
        if self._grid_d is None:
            self._grid_d = chern.GRID.to(self.eng.dev)
        return self._grid_d

    @property
    def _zero_i32(self):
        if self._zero_d is None:
            self._zero_d = torch.zeros(1, dtype=torch.int32, device=self.eng.dev)
        return self._zero_d

    # ---- per step ------------------------------------------------------------------------------

    def begin_step(self, req_ids, hq: int | None = None) -> None:
        """Host view of this step's decode rows (once per step, before layer 0): append the new
        positions to the attended lists and read their widest list (the one per-step host read).
        ``hq``: query heads (sizes the graphs' miss list; default 8 per KV head)."""
        eng, st = self.eng, self.st
        ids = [str(r) for r in req_ids]
        slots = [eng.slot_of(r) for r in ids]
        if not ids or min(slots) < 0:
            raise RuntimeError(f"vkv decode: rows {ids} are not all tracked by the engine")
        st.grow(eng.S)
        lens = [int(eng._reqs[r].mapped) for r in ids]
        for s_, n_ in zip(slots, lens):
            st.append_positions(s_, n_)
        ctx = [int(st.ctx[s]) for s in slots]
        # Persistent per (slots, ctx): a captured layer reads them at fixed addresses.
        rk = (tuple(slots), tuple(ctx))
        rt = self._row_t.get(rk)
        if rt is None:
            if len(self._row_t) > 64:
                self._row_t.clear()
            rt = self._row_t[rk] = (torch.tensor(slots, dtype=torch.long, device=eng.dev),
                                    torch.tensor(ctx, dtype=torch.long, device=eng.dev))
        slots_d, ctx_d = rt
        WA = int(st.acnt[:, slots_d].max())
        if self.graphs:
            # Padded to a bucket (the kernels mask by each row's count), so a captured layer
            # serves every step until the widest list crosses the bucket.
            WA = min(-(-max(WA, 1) // GRAPH_WA_BUCKET) * GRAPH_WA_BUCKET, st.max_len)
        self.rows = {
            "ids": ids, "slots": slots, "lens": lens, "ctx": ctx, "slots_d": slots_d,
            "ctx_d": ctx_d, "Lmax": max(lens), "n": max(ctx), "WA": WA,
        }
        if self.full:
            if any(ctx):
                raise RuntimeError("vkv FULL decode graphs serve attend-only rows (evict=True); "
                                   f"rows {ids} have sampled contexts {ctx}")
            sig = tuple(t.data_ptr() for t in (st.alist, st.acnt, eng.cell, eng.kv_rows,
                                               eng.decode_rows_d))
            if self._full_sig is None:
                self._full_sig = sig
            elif sig != self._full_sig:
                raise RuntimeError("vkv FULL decode graphs: a captured buffer was reallocated")
        if self.graphs:
            if self._gpool is None:
                # Every buffer a captured fetch reads, allocated now and never again (the miss
                # list for every query head's whole context).
                self.store.prepare_graphs(
                    max(1, eng.decode_rows_d.numel()) * (hq or 8 * eng.H) * st.max_len)
                self._gpool = torch.cuda.graph_pool_handle()
            self.rows["sig"] = (rk, WA, _buffers_sig(eng, st, st.sketch, self.store))

    # ---- helpers -------------------------------------------------------------------------------

    def _fb_consts(self, cfg, dev):
        """(eps0, delta, thr, L, 1 / 3) float64 on the device for ``fallback.run``."""
        key = ("fb", cfg.eps0, cfg.delta)
        c = self._consts.get(key)
        if c is None:
            c = self._consts[key] = torch.tensor(
                [cfg.eps0, cfg.delta, cfg.delta * (1 - CHERN_MARGIN), _ln2_over(cfg.delta),
                 1.0 / 3.0], dtype=torch.float64, device=dev)
        return c

    def _dtdv_consts(self, delta: float, dev):
        """(L, 1 / 3) float64 on the device for ``dtdv_kernel``."""
        c = self._consts.get(delta)
        if c is None:
            c = self._consts[delta] = torch.tensor([_ln2_over(delta), 1.0 / 3.0],
                                                   dtype=torch.float64, device=dev)
        return c

    def _identity_tables(self, B: int):
        t = self._ident.get(B)
        if t is None:
            dev = self.eng.dev
            t = self._ident[B] = (torch.zeros(B + 1, dtype=torch.int32, device=dev),
                                  torch.arange(self.eng.n_units, dtype=torch.int32, device=dev))
        return t

    def _decode(self, q, idx, w, slen, sm_scale, soft_cap):
        eng = self.eng
        B, Hq, W = idx.shape
        ks = self._k_split.get((B, Hq, W))
        if ks is None:
            ks = self._k_split[(B, Hq, W)] = int(self.ops.pick_k_split(B, Hq, W, 64))
        indptr, indices = self._identity_tables(B)
        return self.ops.sparse_decode_run(
            q, eng.kv_rows, eng.kv_rows_v, eng.HP * eng.C2, eng.C2, 0, 0,
            indptr, indices, slen, idx, w, int(eng.H), eng.HP,
            float(sm_scale), float(soft_cap or 0.0), int(ks), True)

    def _shortlist(self, layer, s_hat, B, H, G, n):
        """[B, H, M] positions and validity of each (row, KV head)'s shortlist: the tail entries
        with the largest sketch logit over the group's query heads (``vkv_final_bias`` (a))."""
        M = min(int(self.st.cfg.shortlist), n)
        tail = (self.st.r[layer][self.rows["slots_d"], :n] < 1).transpose(1, 2)   # [B, H, n]
        best = s_hat.view(B, H, G, n).amax(2).masked_fill(~tail, _NEG_INF)
        pos = row_topk(best.view(B * H, n), M).view(B, H, M)
        return pos, tail.gather(-1, pos)

    def _shortlist_fused(self, layer, s_hat, B, H, G, n, cnt=None):
        """``_shortlist`` in two launches: the group max over the tail, then its top M."""
        M = min(int(self.st.cfg.shortlist), n)
        best = torch.empty((B * H, n), dtype=torch.float32, device=s_hat.device)
        K.group_best_kernel[(B * H, triton.cdiv(n, _BLK_T))](
            s_hat, self.st.r[layer], self.rows["slots_d"], best, n, self.st.max_len, H=H, G=G,
            BLK=_BLK_T)
        v, pos = row_topk(best, M, values=True, cnt=cnt)
        return pos, v > _NEG_INF

    # ---- one layer: fused ---------------------------------------------------------------------------

    @torch.no_grad()
    def layer(self, layer: int, q: torch.Tensor, sm_scale: float, soft_cap: float = 0.0,
              need_lse: bool = True):
        """Attention output [B, Hq, D] and its log2 LSE [B, Hq] for this layer's decode rows
        (the LSE may be None when ``need_lse`` is False)."""
        eng, st, cfg, rows = self.eng, self.st, self.st.cfg, self.rows
        if (self.torch_path or cfg.k_max > FUSED_K_MAX
                or (cfg.shortlist and not self._pool_mapped)):
            return self.layer_torch(layer, q, sm_scale, soft_cap)
        B, Hq, D = q.shape
        H = eng.H
        G = Hq // H
        R = B * Hq
        dev = q.device
        if len(rows["slots"]) != B:
            raise RuntimeError(f"vkv decode: {B} query rows for {len(rows['slots'])} requests")
        slots_d, n, WA = rows["slots_d"], rows["n"], rows["WA"]
        r_l, acnt_l = st.r[layer], st.acnt[layer]
        max_len = st.max_len
        qs = (q.float() * float(sm_scale)).contiguous()                  # harness q_all

        if cfg.recall and self.fa:
            return self._recall(layer, q, sm_scale, soft_cap, need_lse)

        if n == 0 and self.fa:
            # Nothing sampled (before the round, or KVzip eviction): plain attention over A.
            return self._attend_fa(layer, q, sm_scale, soft_cap, need_lse)

        if n > 0 and self.cuda and self._cuda_ok(B, Hq, D):
            return self._layer_cuda(layer, q, sm_scale, soft_cap)

        # ---- exact logits over the attended lists A ------------------------------------------------
        NBA = triton.cdiv(WA, _BLK_A)
        s = torch.empty((R, WA), dtype=torch.float32, device=dev)
        physA = torch.empty((B * H, WA), dtype=torch.long, device=dev)
        rA = torch.empty((B * H, WA), dtype=torch.float32, device=dev)
        pA = torch.empty((B * H, WA), dtype=torch.int32, device=dev)
        pmax_a = torch.empty((R, NBA), dtype=torch.float32, device=dev)
        K.exact_scores_kernel[(B * H, NBA)](
            qs, st.alist[layer], acnt_l, eng.cell[layer], r_l, eng.kv_rows, slots_d, s, physA,
            rA, pA, pmax_a, WA, NBA, max_len, HP=eng.HP, C2=eng.C2, H=H, G=G, D=D, BLK=_BLK_A)
        slen = torch.empty((R,), dtype=torch.int32, device=dev)
        if n == 0:
            Wtot = WA
            idx = torch.empty((R, Wtot), dtype=torch.long, device=dev)
            w = torch.empty((R, Wtot), dtype=torch.float32, device=dev)
            # No request sampled yet (before its round): exact attention over A.
            K.weights_kernel[(R, NBA)](pA, rA, physA, acnt_l, slots_d, pA, pA, slen, idx, w, slen,
                                       1, WA, Wtot, H=H, G=G, BLK=_BLK_A, HAS_TAIL=False)
            return self._decode(q.contiguous(), idx.view(B, Hq, Wtot), w.view(B, Hq, Wtot),
                                slen.view(B, Hq, 1), sm_scale, soft_cap)

        # ---- plan logits: sketch over the tail, exact on the shortlist ------------------------------
        s_hat = st.sketch.score(layer, slots_d, qs, n).view(R, n)
        # Everything that starts at zero, in one memset: ranks, T*, flags, counters.
        posc, rank, Tm, coldF, cap_t, scr, keys_cnt, cnt_s, cnt_j, cnt_k = _zeros(
            dev, ((R, n), torch.int64), ((R, n), torch.int32), ((R, n), torch.int8),
            ((R, n), torch.bool), ((R,), torch.int8), ((R,), torch.int8), ((1,), torch.int32),
            ((B * H,), torch.int32), ((R,), torch.int32), ((R,), torch.int32))
        if cfg.shortlist:
            sl_pos, sl_ok = self._shortlist_fused(layer, s_hat, B, H, G, n, cnt_s)
            M = sl_pos.shape[-1]
            K.shortlist_kernel[(B * H, triton.cdiv(M, _BLK_S))](
                qs, sl_pos, sl_ok, eng.cell[layer], eng.cpu_row[layer], eng.kv_rows,
                eng.cpu_pool, slots_d, s_hat, keys_cnt, n, M, max_len, C2=eng.C2,
                PC2=eng.cpu_pool.stride(0), H=H, G=G, D=D, BLK=_BLK_S)

        # ---- the stabiliser m, sure -------------------------------------------------------------------
        NB = triton.cdiv(n, _BLK)
        pmax_t = torch.empty((R, NB), dtype=torch.float32, device=dev)
        K.tail_max_kernel[(R, NB)](s_hat, r_l, slots_d, pmax_t, n, NB, max_len,
                                   H=H, G=G, BLK=_BLK)
        m = torch.empty((R,), dtype=torch.float32, device=dev)
        psum = torch.empty((R, NBA), dtype=torch.float64, device=dev)
        K.sure_kernel[(R, NBA)](s, rA, acnt_l, slots_d, pmax_a, pmax_t, m, psum, WA, NBA, NB,
                                H=H, G=G, BLK=_BLK_A, BP=_BP)
        sure = psum.sum(-1)

        # ---- decision: T* --------------------------------------------------------------------------
        jump = torch.empty((R, n), dtype=torch.float64, device=dev)
        key = torch.empty((R, n), dtype=torch.float32, device=dev)
        part = torch.empty((R, NB, 4), dtype=torch.float64, device=dev)
        K.prep_kernel[(R, NB)](s_hat, r_l, slots_d, m, jump, key, part, n, NB, max_len,
                               H=H, G=G, BLK=_BLK)
        if cfg.trunc:
            M_t = min(int(cfg.t_max), n)
            j_s, j_i = row_topk(jump, M_t, values=True, cnt=cnt_j)
            K.walk_kernel[(R,)](j_s, j_i, s_hat, r_l, slots_d, m, part, Tm, cap_t, n, M_t, NB,
                                max_len, H=H, G=G, BM=_np2(M_t), BP=_BP, num_warps=8)
        # (r = 0 tail entries are in T* without a walk: _moves adds them on the fly.)

        # ---- decision: fetch order (top-K), complement, certificate ---------------------------------
        if cfg.order == "grad":
            tot = torch.empty((R, NB, 3), dtype=torch.float64, device=dev)
            K.totals_kernel[(R, NB)](s_hat, r_l, slots_d, m, Tm, tot, n, NB, max_len,
                                     H=H, G=G, BLK=_BLK)
            V0 = tot[..., 0].sum(-1)
            dtdv = torch.empty((R,), dtype=torch.float64, device=dev)
            K.dtdv_kernel[(triton.cdiv(R, 32),)](tot, V0, dtdv, self._dtdv_consts(cfg.delta, dev),
                                                 NB, R, BR=32, BP=128, enable_fp_fusion=False)
            key = torch.empty((R, n), dtype=torch.float64, device=dev)
            K.gradkey_kernel[(R, NB)](s_hat, r_l, slots_d, m, Tm, dtdv, key, n, NB, max_len,
                                      H=H, G=G, BLK=_BLK)
        Kc = min(int(cfg.k_max), n)
        o_i = row_topk(key, Kc, rank=rank, cnt=cnt_k)
        part_r = torch.empty((R, NB, 5), dtype=torch.float64, device=dev)
        K.rest_kernel[(R, NB)](s_hat, r_l, slots_d, m, Tm, rank, part_r, n, NB, max_len,
                               H=H, G=G, BLK=_BLK)
        kst = torch.empty((R,), dtype=torch.int64, device=dev)
        cap_k = torch.empty((R,), dtype=torch.int8, device=dev)
        beta_k = torch.empty((R,), dtype=torch.float64, device=dev)
        Bt = torch.empty((R,), dtype=torch.float64, device=dev)
        nonmono = torch.empty((R,), dtype=torch.int8, device=dev)
        tsize = torch.empty((R,), dtype=torch.float64, device=dev)
        if cfg.check == "chern":
            self._chern(o_i, s_hat, r_l, slots_d, m, Tm, part, part_r, sure, kst, cap_k, beta_k,
                        Bt, nonmono, tsize, scr, n, Kc, NB, max_len, H, G)
        else:
            K.curve_kernel[(R,)](o_i, s_hat, r_l, slots_d, m, Tm, part, part_r, sure, kst, cap_k,
                                 beta_k, Bt, nonmono, tsize, float(cfg.eps0), float(cfg.delta),
                                 _ln2_over(cfg.delta), n, Kc, NB, max_len, H=H, G=G,
                                 BK=_np2(Kc), BP=_BP, TIGHT=cfg.check == "tight", num_warps=8)
        # ---- capped rows: the reference decision, on the GPU (fallback.py) ---------------------------
        fb = torch.empty((R,), dtype=torch.bool, device=dev)
        fbl = torch.empty((R + 1,), dtype=torch.int32, device=dev)
        K.flags_kernel[(1,)](cap_t, cap_k, fb, fbl, R, BR=_np2(R))
        FB.run(fbl, s_hat, r_l, slots_d, m, sure, self._grid, self._fb_consts(cfg, dev), Tm,
               rank, kst, beta_k, Bt, nonmono, tsize, scr, n=n, Kc=Kc, max_len=max_len, H=H,
               G=G, trunc=cfg.trunc, order=cfg.order, check=cfg.check)

        # ---- attention list: A with Algorithm 2 (e) weights, then the fetched entries -------------
        # F = the entries with 1 <= rank <= k*; its cold part is staged in. Every row's list holds
        # A and k_max more; a capped row's k* can exceed k_max, and the rest goes to an overflow
        # list (its own decode, LSE-merged). No host read: the widths are fixed.
        W1 = WA + Kc
        idx = torch.empty((R, W1), dtype=torch.long, device=dev)
        w = torch.empty((R, W1), dtype=torch.float32, device=dev)
        K.weights_kernel[(R, NBA)](pA, rA, physA, acnt_l, slots_d, Tm, rank, kst, idx, w, slen,
                                   n, WA, W1, H=H, G=G, BLK=_BLK_A, HAS_TAIL=True)
        K.cold_kernel[(R, triton.cdiv(n, _BLK_F))](rank, kst, eng.cell[layer], slots_d, coldF,
                                                    posc, n, max_len, H=H, G=G, BLK=_BLK_F)
        if st.home is not None:
            out, lse = self._attend_restore(layer, q, sm_scale, soft_cap, idx, w, slen, coldF,
                                            posc, kst)
            passes = 1
        else:
            out, lse, passes = self._attend_fetch(layer, q, sm_scale, soft_cap, idx, w, slen,
                                                  coldF, posc)
        xfer = coldF.sum()
        st.add_stats_kernel(rows["ctx_d"], Hq, kst, tsize, beta_k, Bt, nonmono, fb, scr, slen,
                            xfer, keys_cnt, passes - 1)
        return out, lse

    @torch.no_grad()
    def layer_full(self, layer: int, q: torch.Tensor, sm_scale: float, soft_cap: float = 0.0,
                   need_lse: bool = True):
        """Attend-only decode at fixed shapes for vLLM's FULL decode graphs (KVzip eviction):
        the rows from the engine's persistent decode-row map, the page table max_len wide (the
        attention's work follows each list's own length). Reads only buffers at fixed addresses,
        so a captured call stays valid while ``begin_step`` updates them in place."""
        B = q.shape[0]
        return self._attend_fa(layer, q, sm_scale, soft_cap, need_lse,
                               self.eng.decode_rows_d[:B], self.st.max_len)

    def _attend_fa(self, layer, q, sm_scale, soft_cap, need_lse, slots_d=None, WA=None):
        """FlashAttention (vLLM's FA3) over each (row, KV head)'s attended cells: one varlen
        "sequence" per (b, h) with its G query heads and one KV head, page size 1 over the
        engine's cell rows, the page table built from A in list order (positions ascending: the
        context, then the generated tokens). Its work follows each list's own length."""
        eng, st = self.eng, self.st
        B, Hq, D = q.shape
        H = eng.H
        G = Hq // H
        BH = B * H
        if slots_d is None:
            slots_d, WA = self.rows["slots_d"], self.rows["WA"]
        dev = q.device
        bt = torch.empty((BH, WA), dtype=torch.int32, device=dev)
        seq = torch.empty((BH,), dtype=torch.int32, device=dev)
        has_home = st.cfg.recall and st.home is not None
        K.fa_table_kernel[(BH, triton.cdiv(WA, _BLK_F))](
            st.alist[layer], st.acnt[layer], eng.cell[layer], slots_d, bt, seq, WA,
            st.max_len, st.home[layer] if has_home else eng.cell[layer], H=H, BLK=_BLK_F,
            HAS_HOME=has_home)
        cu = self._cu.get(BH)
        if cu is None:
            cu = self._cu[BH] = torch.arange(BH + 1, dtype=torch.int32, device=dev)
        kc, vc = self._fa_kv()
        res = _fa_varlen(q.reshape(BH, G, D), kc, vc, 1, cu, WA, seqused_k=seq,
                         block_table=bt, softmax_scale=float(sm_scale),
                         softcap=float(soft_cap or 0.0), return_softmax_lse=need_lse,
                         fa_version=3)
        if not need_lse:
            return res.view(B, Hq, D), None
        o, lse = res                                   # lse [G, B * H], natural log
        return (o.view(B, Hq, D),
                (lse.view(G, B, H).permute(1, 2, 0) * _LOG2E).reshape(B, Hq))

    def _recall(self, layer, q, sm_scale, soft_cap, need_lse):
        """``cfg.recall`` baseline: copy every offloaded entry of the layer from the pinned CPU
        pool, then attend exactly. No compaction: back into their own cells, one attention over
        the whole context. Compaction: into a GPU staging buffer (KV head major), attended as a
        second FlashAttention part and LSE-merged with the resident cells."""
        eng, st = self.eng, self.st
        B, Hq, D = q.shape
        H = eng.H
        G = Hq // H
        recs = [st.rec.get(s_) for s_ in self.rows["slots"]]
        for rec in recs:
            if rec is not None:
                st.stats[STATS_COPIED] += rec[2][layer]
        if st.home is not None:
            for rec in recs:
                if rec is not None:
                    src, dst, cnt = rec[:3]
                    K.recall_copy_kernel[(triton.cdiv(src.shape[1], 32),)](
                        src[layer], dst[layer], cnt[layer:layer + 1], eng.cpu_pool, eng.kv_rows,
                        src.shape[1], C2=eng.C2, PC2=eng.cpu_pool.stride(0), BLK=32)
            return self._attend_fa(layer, q, sm_scale, soft_cap, need_lse)
        o, lse = self._attend_fa(layer, q, sm_scale, soft_cap, True)
        if all(r is None for r in recs):
            return o, (lse if need_lse else None)
        # Staging rows of every request, one after another; per (b, h) its rows' table.
        E = sum(r[0].shape[1] for r in recs if r is not None)
        buf = getattr(self, "_rec_buf", None)
        if buf is None or buf.shape[0] < E:
            buf = self._rec_buf = torch.empty((E, eng.C2), dtype=eng.kv_rows.dtype,
                                              device=q.device)
        Wh = max(r[3].shape[-1] for r in recs if r is not None)
        bt = torch.zeros((B, H, Wh), dtype=torch.int32, device=q.device)
        seq = torch.zeros((B, H), dtype=torch.int32, device=q.device)
        base = 0
        for b, rec in enumerate(recs):
            if rec is None:
                continue
            src, dst, cnt, tab, hcnt = rec
            K.recall_copy_kernel[(triton.cdiv(src.shape[1], 32),)](
                src[layer], dst[layer], cnt[layer:layer + 1], eng.cpu_pool, buf[base:],
                src.shape[1], C2=eng.C2, PC2=eng.cpu_pool.stride(0), BLK=32)
            bt[b, :, :tab.shape[-1]] = tab[layer] + base
            seq[b] = hcnt[layer]
            base += src.shape[1]
        n_, C2 = buf.shape
        kc = buf.as_strided((n_, 1, 1, D), (C2, C2, C2, 1))
        vc = buf[:, D:].as_strided((n_, 1, 1, D), (C2, C2, C2, 1))
        BH = B * H
        cu = self._cu.get(BH)
        if cu is None:
            cu = self._cu[BH] = torch.arange(BH + 1, dtype=torch.int32, device=q.device)
        o2, l2 = _fa_varlen(q.reshape(BH, G, D), kc, vc, 1, cu, Wh, seqused_k=seq.view(BH),
                            block_table=bt.view(BH, Wh), softmax_scale=float(sm_scale),
                            softcap=float(soft_cap or 0.0), return_softmax_lse=True,
                            fa_version=3)
        l2 = (l2.view(G, B, H).permute(1, 2, 0) * _LOG2E).reshape(B, Hq)
        o, lse = self._merge(o.reshape(B * Hq, D).contiguous(), lse.reshape(-1).contiguous(),
                             o2.reshape(B * Hq, D).contiguous(), l2.reshape(-1).contiguous())
        return o.view(B, Hq, D), (lse.view(B, Hq) if need_lse else None)

    def _fa_kv(self):
        """The cell rows as FlashAttention's paged K / V caches [cells, 1, 1, D] (page size 1,
        one KV head), viewing the K and V halves of each row in place."""
        eng = self.eng
        kv = self._fa_cache
        if kv is None or kv[0] is not eng.kv_rows:
            n, C2 = eng.kv_rows.shape
            D = C2 // 2
            kv = self._fa_cache = (
                eng.kv_rows,
                eng.kv_rows.as_strided((n, 1, 1, D), (C2, C2, C2, 1)),
                eng.kv_rows_v.as_strided((n, 1, 1, D), (C2, C2, C2, 1)))
        return kv[1], kv[2]

    def layer_graphed(self, layer: int, q: torch.Tensor, sm_scale: float, soft_cap: float = 0.0,
                      need_lse: bool = True):
        """``layer`` replayed from a CUDA graph per (layer, rows, widths, buffers): the first call
        of a key runs eagerly (compiles, fills the host-side caches), the second captures, later
        ones only copy q in and replay. ``layer`` has no host read and, with the padded A width,
        the same launches for every step of a key. The returned tensors are shared by all layers:
        copy them before the next call."""
        rows = self.rows
        if rows is None or "sig" not in rows or self.st.cfg.recall:
            return self.layer(layer, q, sm_scale, soft_cap, need_lse)
        B, Hq, D = q.shape
        key = (layer, B, Hq, D, q.dtype, rows["n"], float(sm_scale), float(soft_cap),
               need_lse, rows["sig"])
        g = self._graphs.get(key)
        if g is None:
            if key not in self._warm:
                if len(self._warm) > 4 * GRAPH_MAX:
                    self._warm.clear()
                self._warm.add(key)
                return self.layer(layer, q, sm_scale, soft_cap, need_lse)
            g = self._capture(key, layer, q, sm_scale, soft_cap, need_lse)
        else:
            self._graphs[key] = self._graphs.pop(key)       # most recent last (LRU)
        g[1].copy_(q)
        g[0].replay()
        return g[2], (g[3] if need_lse else None)

    def _capture(self, key, layer, q, sm_scale, soft_cap, need_lse=True):
        B, Hq, D = q.shape
        io = self._gio.get((B, Hq, D, q.dtype))
        if io is None:
            io = self._gio[(B, Hq, D, q.dtype)] = (
                torch.empty_like(q), torch.empty((B, Hq, D), dtype=q.dtype, device=q.device),
                torch.empty((B, Hq), dtype=torch.float32, device=q.device))
        q_in, o_out, l_out = io
        q_in.copy_(q)
        while len(self._graphs) >= GRAPH_MAX:
            self._graphs.pop(next(iter(self._graphs)))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self._gpool, capture_error_mode="thread_local"):
            o, lse = self.layer(layer, q_in, sm_scale, soft_cap, need_lse)
            o_out.copy_(o)
            if need_lse:
                l_out.copy_(lse)
        self.captures += 1
        g = self._graphs[key] = (graph, q_in, o_out, l_out)
        return g

    # ---- one layer: CUDA ---------------------------------------------------------------------------

    def _cuda_ok(self, B: int, Hq: int, D: int) -> bool:
        """The configurations the CUDA decision implements: the vkv_final defaults ("grad"
        order, Chernoff certificate, truncation), heads up to 4096, head_dim 64 / 128, every
        row's block resident. Fetched entries are restored in place (no compaction) or staged by
        the store (compaction: every position of every KV head must fit its ring at once)."""
        cfg, st = self.st.cfg, self.st
        if self._sms is None:
            self._sms = torch.cuda.get_device_properties(self.eng.dev).multi_processor_count
        return (cfg.order == "grad" and cfg.check == "chern" and cfg.trunc
                and cfg.k_max <= 4096 and cfg.t_max <= 4096 and cfg.shortlist <= 4096
                and self._pool_mapped and D in (64, 128)
                and (st.home is not None or B * self.eng.H * self.rows["n"] <= int(self.store.C))
                and st.sketch.nch <= 64 and B * Hq <= self._sms)

    def _layer_cuda(self, layer, q, sm_scale, soft_cap):
        """``layer`` with the decision in CUDA (``csrc/vkv_decide.cu``): decide (A logits ..
        k*), the capped rows' fallback (Triton, device list), emit (A weights; F restored in
        place and listed, or with compaction packed for ``store.stage_packed``, which stages it
        into the ring and lists it), attention over the list (+ overflow), statistics."""
        from .cuda_decide import ext
        eng, st, cfg, rows = self.eng, self.st, self.st.cfg, self.rows
        B, Hq, D = q.shape
        H = eng.H
        G = Hq // H
        R = B * Hq
        dev = q.device
        slots_d, n, WA = rows["slots_d"], rows["n"], rows["WA"]
        max_len = st.max_len
        Kc = min(int(cfg.k_max), n)
        M_t = min(int(cfg.t_max), n)
        M_sl = min(int(cfg.shortlist), n)
        sk = st.sketch
        f32, f64, i32, i64 = torch.float32, torch.float64, torch.int32, torch.int64
        s_hat = torch.empty((R, n), dtype=f32, device=dev)
        jbuf = torch.empty((R, n), dtype=f64, device=dev)
        whb = torch.empty((R, n), dtype=f64, device=dev)
        Tm = torch.empty((R, n), dtype=torch.int8, device=dev)
        rank = torch.empty((R, n), dtype=i32, device=dev)
        sA = torch.empty((R, WA), dtype=f32, device=dev)
        rA = torch.empty((R, WA), dtype=f32, device=dev)
        pA = torch.empty((R, WA), dtype=i32, device=dev)
        physA = torch.empty((R, WA), dtype=i64, device=dev)
        o_i = torch.empty((R, Kc), dtype=i32, device=dev)
        m = torch.empty((R,), dtype=f32, device=dev)
        sure, beta_k, Bt, tsize = (torch.empty((R,), dtype=f64, device=dev) for _ in range(4))
        kst = torch.empty((R,), dtype=i64, device=dev)
        nonmono = torch.empty((R,), dtype=torch.int8, device=dev)
        scr = torch.empty((R,), dtype=torch.int8, device=dev)
        fb = torch.empty((R,), dtype=torch.bool, device=dev)
        fbl, keys_cnt, bar, xfer = _zeros(dev, ((R + 1,), i32), ((1,), i32), ((B * H,), i32),
                                          ((1,), i64))
        X = ext()
        X.decide(q.contiguous(), float(sm_scale), slots_d, st.r[layer], eng.cell[layer],
                 eng.cpu_row[layer], st.alist[layer], st.acnt[layer], eng.kv_rows, eng.cpu_pool,
                 sk.codes[layer], sk.zero[layer], sk.scale[layer], sk.channels[layer], self._grid,
                 [B, H, G, n, WA, max_len, M_sl, M_t, Kc, int(Kc < n), sk.nch, sk.per_byte,
                  sk.bits, sk.group, CUDA_CLUSTER],
                 [float(cfg.eps0), float(cfg.delta), float(cfg.delta * (1 - CHERN_MARGIN)),
                  _ln2_over(cfg.delta), 1.0 / 3.0],
                 s_hat, jbuf, whb, Tm, rank, sA, rA, pA, physA, o_i, m, sure, kst, beta_k, Bt,
                 nonmono, tsize, scr, fb, fbl, keys_cnt, bar)
        FB.run(fbl, s_hat, st.r[layer], slots_d, m, sure, self._grid, self._fb_consts(cfg, dev),
               Tm, rank, kst, beta_k, Bt, nonmono, tsize, scr, n=n, Kc=Kc, max_len=max_len, H=H,
               G=G, trunc=cfg.trunc, order=cfg.order, check=cfg.check)
        # The list: A, then F's cold entries (W1 = A + k_max), the rest of a capped row's F in
        # the overflow list (W2 = n), the two attended separately and LSE-merged.
        W1, W2 = WA + Kc, n
        idx = torch.empty((R, W1), dtype=i64, device=dev)
        w = torch.empty((R, W1), dtype=f32, device=dev)
        slen = torch.empty((R,), dtype=i32, device=dev)
        idx2 = torch.empty((R, W2), dtype=i64, device=dev)
        w2 = torch.empty((R, W2), dtype=f32, device=dev)
        slen2 = torch.empty((R,), dtype=i32, device=dev)
        ints = [H, G, n, WA, Kc, max_len]
        if st.home is None:
            # Compaction: emit packs F's cold positions per row; the store stages them into its
            # ring (the staging tail of the cache) and lists them after A.
            posc = torch.empty((R, n), dtype=i64, device=dev)
            ncold = torch.empty((R,), dtype=i32, device=dev)
            X.emit(slots_d, st.acnt[layer], eng.cell[layer], eng.cpu_row[layer],
                   eng.cell[layer], eng.cell[layer], eng.cpu_pool, eng.kv_rows, rA, pA, physA,
                   Tm, rank, kst, fb, o_i, jbuf, idx, w, slen, idx, w, slen, xfer, ints,
                   posc, ncold)
            self.store.stage_packed(layer, posc, ncold, eng.decode_rows_d[:B], G, slen, idx, w,
                                    slen, idx2, w2, slen2)
        else:
            # No compaction: emit restores F's cold entries into their own cells and lists them.
            X.emit(slots_d, st.acnt[layer], eng.cell[layer], eng.cpu_row[layer], st.home[layer],
                   st.restored[layer], eng.cpu_pool, eng.kv_rows, rA, pA, physA, Tm, rank, kst,
                   fb, o_i, jbuf, idx, w, slen, idx2, w2, slen2, xfer, ints)
        qc = q.contiguous()
        o, l2 = self._decode(qc, idx.view(B, Hq, W1), w.view(B, Hq, W1), slen.view(B, Hq, 1),
                             sm_scale, soft_cap)
        o2, l22 = self._decode(qc, idx2.view(B, Hq, W2), w2.view(B, Hq, W2),
                               slen2.view(B, Hq, 1), sm_scale, soft_cap)
        out, lse = self._merge(o, l2, o2, l22)
        st.add_stats_kernel(rows["ctx_d"], Hq, kst, tsize, beta_k, Bt, nonmono, fb, scr, slen,
                            xfer, keys_cnt, 0)
        return out, lse

    def _chern(self, o_i, s_hat, r_l, slots_d, m, Tm, part, part_r, sure, kst, cap_k, beta_k, Bt,
               nonmono, tsize, scr, n, Kc, NB, max_len, H, G):
        """k* by the Chernoff certificate: ``chern_kernel`` split over (row, entry block) and
        (row, lambda part) programs (one program per row cannot fill the GPU for a 4096-entry
        head); CHERN_SPLIT = 1 runs the single-program kernel. Same k* either way."""
        cfg, dev = self.st.cfg, s_hat.device
        R = int(o_i.shape[0])
        BK, NG = _np2(Kc), int(self._grid.numel())
        eps0, delta = float(cfg.eps0), float(cfg.delta)
        thr = float(cfg.delta * (1 - CHERN_MARGIN))
        NS = CHERN_SPLIT if CHERN_SPLIT > 1 and NG % CHERN_SPLIT == 0 else 1
        if NS == 1:
            K.chern_kernel[(R,)](o_i, s_hat, r_l, slots_d, m, Tm, part, part_r, sure, self._grid,
                                 kst, cap_k, beta_k, Bt, nonmono, tsize, scr, eps0, delta, thr,
                                 n, Kc, NB, max_len, H=H, G=G, BK=BK, BP=_BP, NG=NG,
                                 BENNETT=Kc < n, num_warps=8)
            return
        EB = min(_CHERN_EB, BK)
        f32 = torch.empty((4 * R * BK,), dtype=torch.float32, device=dev)
        f64 = torch.empty((8 * R * BK,), dtype=torch.float64, device=dev)
        sc = torch.empty((R * 8,), dtype=torch.float64, device=dev)
        best = torch.empty((R * NS * 2 * BK,), dtype=torch.float64, device=dev)
        bK = torch.empty((R * NS * 2,), dtype=torch.float64, device=dev)
        K.chern_elem_kernel[(R, BK // EB)](o_i, s_hat, r_l, slots_d, m, Tm, f32, f64, n, Kc,
                                           max_len, H=H, G=G, BK=BK, EB=EB)
        K.chern_row_kernel[(R,)](part, part_r, sure, f64, sc, eps0, delta, NB, BK=BK, BP=_BP,
                                 num_warps=8)
        K.chern_lam_kernel[(R, NS)](f32, f64, sc, self._grid, best, bK, eps0, BK=BK, NG=NG, NS=NS,
                                    BENNETT=Kc < n, num_warps=8)
        K.chern_finish_kernel[(R,)](f64, sc, best, bK, kst, cap_k, beta_k, Bt, nonmono, tsize,
                                    scr, thr, Kc, BK=BK, NS=NS, num_warps=8)

    # ---- fetch: F \ S into staging, in ring-sized passes ------------------------------------------

    def _pass_width(self, B: int, G: int, WF: int) -> int:
        """Fetch columns per pass: one pass's distinct cold keys must fit the staging ring.
        Columns are per query head (each has its own fetch order), so a pass of w columns needs
        at most B * Hq * w keys -- and never more than every position of every KV head."""
        C, H = int(self.store.C), self.eng.H
        if B * H * WF <= C:
            return WF
        return max(1, C // max(1, B * H * G))

    def _merge(self, oa, la, ob, lb):
        """``lse2_merge`` in one launch (output in ``oa``'s dtype)."""
        out = torch.empty_like(oa)
        lse = torch.empty_like(la)
        K.lse_merge_kernel[(la.numel(),)](oa, la, ob, lb, out, lse, D=oa.shape[-1])
        return out, lse

    def _attend_restore(self, layer, q, sm_scale, soft_cap, idx0, w0, cntA, cold, posc, kst):
        """No compaction: F's cold entries restored into their own cells (once each, from the
        pinned CPU pool) and listed after A, then one attention over the list (plus the overflow
        list of capped rows, LSE-merged). No staging ring, no passes."""
        eng, st = self.eng, self.st
        B, Hq = q.shape[0], q.shape[1]
        G = Hq // eng.H
        R = B * Hq
        W1, WF = idx0.shape[-1], cold.shape[-1]
        dev = q.device
        W2 = WF
        idx2 = torch.empty((R, W2), dtype=torch.long, device=dev)
        w2 = torch.empty((R, W2), dtype=torch.float32, device=dev)
        slen = torch.empty((R,), dtype=torch.int32, device=dev)
        slen2 = torch.empty((R,), dtype=torch.int32, device=dev)
        K.restore_place_kernel[(R,)](
            cold, posc, kst, cntA.reshape(R).contiguous(), self.rows["slots_d"], st.home[layer],
            st.restored[layer], eng.cpu_row[layer], eng.cpu_pool, eng.kv_rows, idx0, w0, slen,
            idx2, w2, slen2, WF, W1, W2, st.max_len, H=eng.H, G=G, C2=eng.C2,
            PC2=eng.cpu_pool.stride(0), BLK=32)
        qc = q.contiguous()
        o, l2 = self._decode(qc, idx0.view(B, Hq, W1), w0.view(B, Hq, W1), slen.view(B, Hq, 1),
                             sm_scale, soft_cap)
        o2, l22 = self._decode(qc, idx2.view(B, Hq, W2), w2.view(B, Hq, W2),
                               slen2.view(B, Hq, 1), sm_scale, soft_cap)
        return self._merge(o, l2, o2, l22)

    def _attend_fetch(self, layer, q, sm_scale, soft_cap, idx0, w0, cntA, cold, posc):
        """Attention over A plus the cold fetched entries, staged from the CPU. Per row
        (b, query head), in [R, ...]: ``idx0`` / ``w0`` [R, W1] hold A in the live prefix
        ``cntA`` [R] (any int dtype) and room after it; ``cold`` [R, WF] marks the entries to
        stage and ``posc`` [R, WF] holds their positions (0 elsewhere). Entries past W1 go to an
        overflow list. Ring-sized passes over the WF columns, LSE-merged. Returns
        (out, lse, passes)."""
        B, Hq = q.shape[0], q.shape[1]
        G = Hq // self.eng.H
        R = B * Hq
        W1 = idx0.shape[-1]
        WF = cold.shape[-1]
        dev = q.device
        qc = q.contiguous()
        base = cntA.reshape(R).contiguous()
        w_pass = self._pass_width(B, G, WF)
        passes = [(a, min(WF, a + w_pass)) for a in range(0, WF, w_pass)]
        out = lse = None
        for i, (a, b) in enumerate(passes):
            wd = b - a
            first = i == 0
            if first:
                idx, w, Wm = idx0, w0, W1
            else:
                Wm = wd
                idx = torch.empty((R, Wm), dtype=torch.long, device=dev)
                w = torch.empty((R, Wm), dtype=torch.float32, device=dev)
            W2 = wd if first else 1                 # later passes cannot overflow
            idx2 = torch.empty((R, W2), dtype=torch.long, device=dev)
            w2 = torch.empty((R, W2), dtype=torch.float32, device=dev)
            key, coldk, ppos, _ = self.store.stage_tokens(
                layer, posc[:, a:b].reshape(B, Hq * wd), self.eng.decode_rows_d[:B].view(B, 1),
                qheads=(wd, G))
            phys = self.store.phys_tokens(layer, key, coldk, ppos, None, None).reshape(R, wd)
            slen = torch.empty((R,), dtype=torch.int32, device=dev)
            slen2 = torch.empty((R,), dtype=torch.int32, device=dev)
            K.place_kernel[(R,)](cold[:, a:], WF, base, phys.contiguous(), idx, w, slen, idx2, w2,
                                 slen2, wd, Wm, W2, HAS_BASE=first, BLK=_BLK_F)
            o, l2 = self._decode(qc, idx.view(B, Hq, Wm), w.view(B, Hq, Wm),
                                 slen.view(B, Hq, 1), sm_scale, soft_cap)
            if first:
                o2, l22 = self._decode(qc, idx2.view(B, Hq, W2), w2.view(B, Hq, W2),
                                       slen2.view(B, Hq, 1), sm_scale, soft_cap)
                o, l2 = self._merge(o, l2, o2, l22)
            out, lse = (o, l2) if out is None else self._merge(out, lse, o, l2)
        return out, lse, len(passes)

    # ---- one layer: torch (cross-check) ------------------------------------------------------------

    @torch.no_grad()
    def layer_torch(self, layer: int, q: torch.Tensor, sm_scale: float, soft_cap: float = 0.0):
        eng, st, cfg, rows = self.eng, self.st, self.st.cfg, self.rows
        B, Hq, D = q.shape
        H = eng.H
        G = Hq // H
        dev = q.device
        slots_d, Lmax, n = rows["slots_d"], rows["Lmax"], rows["n"]
        if len(rows["slots"]) != B:
            raise RuntimeError(f"vkv decode: {B} query rows for {len(rows['slots'])} requests")

        # ---- attended set A: the resident cells, compact positions per (row, KV head) ----------
        cell = eng.cell[layer][slots_d, :Lmax].transpose(1, 2)           # [B, H, Lmax] int32
        rl = st.r[layer][slots_d, :Lmax].transpose(1, 2)                  # [B, H, Lmax] f32
        att = cell >= 0
        cnt = att.sum(-1)                                                 # [B, H]
        WA = int(cnt.max())
        col = torch.where(att, att.cumsum(-1) - 1, WA)
        Apos = torch.zeros((B, H, WA + 1), dtype=torch.long, device=dev)
        Apos.scatter_(-1, col, torch.arange(Lmax, device=dev).expand(B, H, Lmax))
        Apos = Apos[..., :WA]
        validA = torch.arange(WA, device=dev).view(1, 1, -1) < cnt.unsqueeze(-1)
        physA = cell.gather(-1, Apos).long().clamp(min=0)
        rA = rl.gather(-1, Apos)
        physA = physA.unsqueeze(2).expand(B, H, G, WA)
        cntA = cnt.unsqueeze(2).expand(B, H, G)

        qs = q.float() * float(sm_scale)                                  # harness q_all
        qg = qs.view(B, H, G, D)
        if n == 0:
            wA = validA.float().unsqueeze(2).expand(B, H, G, WA)
            slen = cntA.reshape(B, Hq, 1).to(torch.int32).contiguous()
            return self._decode(q.contiguous(), physA.reshape(B, Hq, WA).contiguous(),
                                wA.reshape(B, Hq, WA).contiguous(), slen, sm_scale, soft_cap)
        tail = rl[..., :n] < 1                                            # [B, H, n]

        # ---- scores: exact on A, sketch on the tail, exact on the shortlist -----------------------
        kA = eng.kv_rows[physA[:, :, 0].reshape(-1)][:, :D].view(B, H, WA, D)
        s = torch.matmul(qg, kA.float().transpose(-1, -2))               # [B, H, G, WA]
        s_hat = st.sketch.score(layer, slots_d, qs, n).view(B, H, G, n)   # [B, H, G, n]
        keys = torch.zeros((), dtype=torch.int64, device=dev)
        if cfg.shortlist:
            sl_pos, sl_ok = self._shortlist(layer, s_hat.reshape(B * Hq, n), B, H, G, n)
            sc = cell[..., :n].gather(-1, sl_pos)                         # [B, H, M]
            cr = eng.cpu_row[layer][slots_d, :n].transpose(1, 2).gather(-1, sl_pos)
            host = sl_ok & (sc < 0)
            ksl = eng.kv_rows[sc.long().clamp(min=0).reshape(-1), :D].view(*sc.shape, D).clone()
            if bool(host.any()):
                hr = cr[host].long().cpu()
                ksl[host] = eng.cpu_pool[hr][:, :D].to(dev)
            s_sl = torch.matmul(qg, ksl.float().transpose(-1, -2))       # [B, H, G, M]
            ex = torch.gather(s_hat, -1, sl_pos.unsqueeze(2).expand(B, H, G, sl_pos.shape[-1]))
            s_hat = s_hat.scatter(-1, sl_pos.unsqueeze(2).expand_as(ex),
                                  torch.where(sl_ok.unsqueeze(2), s_sl, ex))
            keys = host.sum()
        tq = tail.unsqueeze(2)
        ntA = (validA & (rA >= 1)).unsqueeze(2)                           # exact, non-tail
        m = torch.maximum(torch.where(tq, s_hat, _NEG_INF).amax(-1),
                          torch.where(ntA, s, _NEG_INF).amax(-1))         # [B, H, G]
        zero = torch.zeros((), dtype=torch.float64, device=dev)
        sure = torch.where(ntA, torch.exp((s - m.unsqueeze(-1)).double()), zero).sum(-1)
        wh = torch.exp((s_hat - m.unsqueeze(-1)).double())
        R = B * Hq
        tail_r = tq.expand(B, H, G, n).reshape(R, n)
        r_r = rl[..., :n].unsqueeze(2).expand(B, H, G, n).reshape(R, n).double()
        dec = decide(wh.view(R, n), r_r, tail_r, sure.view(R), eps0=cfg.eps0, delta=cfg.delta,
                     check=cfg.check, order=cfg.order, trunc=cfg.trunc, t_max=cfg.t_max,
                     k_max=cfg.k_max)
        WF = dec.WF

        # ---- weights on A (Algorithm 2 (e)) --------------------------------------------------------
        Fm = torch.zeros((R, n), dtype=torch.int8, device=dev)
        fv = dec.fetch_valid()
        if WF:
            Fm.scatter_reduce_(-1, dec.F, fv.to(torch.int8), "amax")
        inctx = (Apos < n).unsqueeze(2)
        Ac = Apos.clamp(max=n - 1).unsqueeze(2).expand(B, H, G, WA)
        T_A = dec.T.view(B, H, G, n).gather(-1, Ac) & inctx
        F_A = Fm.view(B, H, G, n).gather(-1, Ac).bool() & inctx
        tailA = (validA & (rA < 1)).unsqueeze(2)                          # = S on A
        wA = attention_weights(rA.unsqueeze(2), tailA, tailA, T_A, F_A)
        wA = torch.where(validA.unsqueeze(2), wA, torch.zeros_like(wA))   # [B, H, G, WA]

        if WF == 0:
            out, lse = self._decode(q.contiguous(), physA.reshape(B, Hq, WA).contiguous(),
                                    wA.reshape(B, Hq, WA).contiguous(),
                                    cntA.reshape(B, Hq, 1).to(torch.int32).contiguous(),
                                    sm_scale, soft_cap)
            xfer, passes = self._zero_i32, 1
        else:
            Fpos = dec.F.view(B, H, G, WF)
            fvq = fv.view(B, H, G, WF)
            cF = cell.gather(-1, Fpos.reshape(B, H, G * WF)).view(B, H, G, WF)
            coldF = fvq & (cF < 0)
            W0 = WA + WF
            idx0 = torch.zeros((B, H, G, W0), dtype=torch.long, device=dev)
            w0 = torch.zeros((B, H, G, W0), dtype=torch.float32, device=dev)
            idx0[..., :WA] = physA
            w0[..., :WA] = wA
            posc = torch.where(coldF, Fpos, torch.zeros_like(Fpos))
            out, lse, passes = self._attend_fetch(
                layer, q, sm_scale, soft_cap, idx0.view(B * Hq, W0), w0.view(B * Hq, W0), cntA,
                coldF.reshape(B * Hq, WF), posc.reshape(B * Hq, WF))
            xfer = coldF.sum()
        st.add_stats_kernel(rows["ctx_d"], Hq, dec.k, (dec.T & tail_r).sum(-1).double(),
                            dec.beta_k, dec.Bt, dec.nonmono, dec.fallback, dec.screened,
                            cntA.reshape(-1).to(torch.int32), xfer, keys, passes - 1)
        return out, lse

