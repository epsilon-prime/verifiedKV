"""Per-engine Vkv state: residency r, the key sketch, statistics.

Lives on ``HeadTokenOffloadEngine.vkv``. The engine's one Vkv round per request (``prefill_round``,
one step after prefill, like the first round of the other token policies) computes Algorithm 1 for
the whole prompt and returns which cells to offload; decode reads everything else from here.

r [L, S, max_len, H] float32 holds the keep probability of every (layer, position, KV head):
  * 1.0 for positions outside a request's sampled context: before the round, generated tokens,
    unwritten positions, and the r = 1 entries (sinks, local window, c * tau >= 1);
  * min(r, 1 - 2^-24) for the tail (r < 1 in float64 stays < 1 in float32).
It never changes after the round: fetched entries serve one forward only (no persistence, as
``vkv_final``), so tail = (r < 1) for the whole request and the attended set is the resident cells.

alist [L, S, H, max_len] int32 / acnt [L, S, H] keep that resident set as a position list per
(layer, request, KV head), so decode never re-derives it: all written positions until the round,
rebuilt from the surviving cells at the round, then appended to with every new token.
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass

import numpy as np
import torch

from . import reference as ref
from .sketch import SketchStore

_BELOW_ONE = float(np.nextafter(np.float32(1.0), np.float32(0.0)))


@dataclass(frozen=True)
class VkvConfig:
    kappa: float = 0.0625         # memory budget, fraction of the full KV cache (1 - compression
                                  # ratio), over all layers / heads; the sketch is paid from it
    charge_sketch: bool = True    # the draw keeps kappa - sketch_frac (harness draw_frac)
    eps0: float = 0.3             # certified relative error of the softmax denominator
    delta: float = 0.05           # certificate failure probability
    check: str = "chern"          # "chern" (Chernoff, vkv_final), "simple" or "tight" (Bernstein)
    order: str = "grad"           # fetch order: "grad" or "needle" (reference.ORDERS)
    trunc: bool = True            # MSE truncation T*
    shortlist: int = 128          # tail keys per KV head scored exactly every forward (0 = off)
    n_sink: int = 64              # first context entries at r = 1
    window: int = 64              # last context entries at r = 1
    bits: int = 2                 # sketch bits per channel
    n_pairs: int = 16             # RoPE pairs sketched (per 128 dims if pairs_per_128_dims)
    pairs_per_128_dims: bool = True
    group: int = 128              # sketch quantisation group (context tokens)
    seed: int = 0                 # draw seed = (crc32(prompt) ^ (seed * 7919)) & 0x7FFFFFFF
    t_max: int = 4096             # truncation-walk cap (exact fallback beyond)
    k_max: int = 4096             # fetch-order head: exact terms (Chernoff) / curve (Bernstein)
                                  # for the first k_max entries, the rest in closed form; rows
                                  # that pass nowhere in the head fall back to the reference
    evict: bool = False           # baseline: plain KVzip at the same budget (kvpress KVzipPress
                                  # with compression 1 - kappa): keep the top kappa of the
                                  # context cells by score, attend to those only; no draw,
                                  # sketch or fetch
    compact: bool = True          # pack the round's survivors into the request's first blocks
                                  # (frees pages; per (layer, head) in position order); False
                                  # offloads in place: cells stay where vLLM wrote them
    recall: bool = False          # baseline: offload what the draw drops (as vkv) but every
                                  # decode step copies all of it back and attends to the whole
                                  # context exactly: to its own cells (compact=False) or, with
                                  # compaction, to a GPU staging buffer attended separately

    def __post_init__(self):
        if not 0.0 < self.kappa <= 1.0:
            raise ValueError("vkv kappa must be in (0, 1]")
        if self.check not in CHECKS or self.order not in ref.ORDERS:
            raise ValueError(f"vkv check/order must be in {CHECKS}/{ref.ORDERS}")
        if self.shortlist < 0 or self.k_max < 1 or self.t_max < 1:
            raise ValueError("vkv shortlist must be >= 0, k_max and t_max >= 1")
        if self.recall and self.evict:
            raise ValueError("vkv recall and evict are exclusive")

    @classmethod
    def from_extra(cls, extra: dict) -> "VkvConfig":
        """Fields from ``kv_connector_extra_config`` keys ``vkv_<field>``."""
        kw = {}
        for f in cls.__dataclass_fields__.values():
            v = extra.get(f"vkv_{f.name}")
            if v is None:
                continue
            if isinstance(f.default, bool):
                kw[f.name] = (v if isinstance(v, bool)
                              else str(v).lower() in ("1", "true", "yes", "on"))
            else:
                kw[f.name] = type(f.default)(v)
        return cls(**kw)

    def pairs_for(self, head_dim: int) -> int:
        return self.n_pairs * head_dim // 128 if self.pairs_per_128_dims else self.n_pairs

    def sketch_frac(self, head_dim: int) -> float:
        """Sketch bytes per entry over the entry's fp16 K + V (kvpress ``sketch_bytes_per_entry`` /
        ``head_dim * KV_BYTES_PER_CHANNEL``): 9 / 512 for 16 pairs, 2 bits, groups of 128."""
        ch = 2 * min(self.pairs_for(head_dim), head_dim // 2)
        return (self.bits * ch / 8 + 2 * ch * 2 / self.group) / (head_dim * 4)

    def draw_frac(self, head_dim: int) -> float:
        """Fraction of the cells the Bernoulli draw keeps (harness ``VkvFinalPress.draw_frac``)."""
        kept = self.kappa - (self.sketch_frac(head_dim) if self.charge_sketch else 0.0)
        if kept <= 0:
            raise ValueError(f"vkv: the sketch ({self.sketch_frac(head_dim):.4f}) leaves nothing "
                             f"of the budget kappa={self.kappa}")
        return kept


CHECKS = ("chern",) + ref.CHECKS


def prompt_crc(prompt_ids) -> int:
    """crc32 of the context token ids as int64 bytes (the harness keys its draw on this)."""
    return zlib.crc32(np.asarray(list(prompt_ids), dtype=np.int64).tobytes())


def draw_seed(crc: int, seed: int) -> int:
    """The harness's generator seed (``BernoulliPress._generator`` masks it to 31 bits)."""
    return (crc ^ (seed * 7919)) & 0x7FFFFFFF


class VkvState:
    def __init__(self, engine, cfg: VkvConfig) -> None:
        if not getattr(engine, "headwise", False):
            raise ValueError("vkv needs the per-head token engine")
        self.cfg = cfg
        self.eng = engine
        self.L, self.H, self.max_len = engine.L, engine.H, engine.max_len
        self.D = engine.C2 // 2
        self.dev = engine.dev
        self.sketch = SketchStore(L=self.L, H=self.H, D=self.D, max_len=self.max_len,
                                  n_pairs=cfg.pairs_for(self.D), bits=cfg.bits, group=cfg.group,
                                  device=self.dev)
        self.S = 0
        self.r = None
        self.alist = None               # [L, S, H, max_len] int32 attended positions
        self.acnt = None                # [L, S, H] int32
        # compact=False: each context entry's own cell ("home", [L, S, max_len, H] int32, kept
        # by the round), and whether it holds the entry again (``restored`` int32, 1 once a
        # fetch copied it back from the CPU): fetched entries are restored in place, once.
        self.home = None
        self.restored = None
        # recall: per slot, the offloaded entries per layer (``build_recall``).
        self.rec: dict[int, tuple] = {}
        self.ctx = []                   # per slot: context length of the round (0 = no round)
        self.a_upto = []                # per slot: positions [0, a_upto) are in the lists
        self._prompts: dict[str, tuple[int, int]] = {}    # req id -> (ctx, crc)
        self.stats = torch.zeros(len(STATS), dtype=torch.float64, device=self.dev)
        self.round_log: list[dict] = []
        self.grow(engine.S)
        engine.on_release.append(self.on_release)

    # ---- storage ---------------------------------------------------------------------------------

    def grow(self, S: int) -> None:
        if S <= self.S:
            return
        with torch.inference_mode(False):
            r = torch.ones((self.L, S, self.max_len, self.H), dtype=torch.float32, device=self.dev)
            alist = torch.zeros((self.L, S, self.H, self.max_len), dtype=torch.int32,
                                device=self.dev)
            acnt = torch.zeros((self.L, S, self.H), dtype=torch.int32, device=self.dev)
            home = restored = None
            if not self.cfg.compact:
                home = torch.full((self.L, S, self.max_len, self.H), -1, dtype=torch.int32,
                                  device=self.dev)
                restored = torch.zeros((self.L, S, self.max_len, self.H), dtype=torch.int32,
                                       device=self.dev)
        if self.S:
            r[:, :self.S].copy_(self.r)
            alist[:, :self.S].copy_(self.alist)
            acnt[:, :self.S].copy_(self.acnt)
            if home is not None:
                home[:, :self.S].copy_(self.home)
                restored[:, :self.S].copy_(self.restored)
        self.r, self.alist, self.acnt = r, alist, acnt
        self.home, self.restored = home, restored
        self.ctx.extend([0] * (S - self.S))
        self.a_upto.extend([0] * (S - self.S))
        self.sketch.grow(S)
        self.S = S

    def on_release(self, slot: int) -> None:
        if slot < self.S:
            self.r[:, slot].fill_(1.0)
            self.acnt[:, slot].zero_()
            if self.restored is not None:
                # restored-in-place entries = unique CPU->GPU copies of the request
                self.stats[STATS.index("copied")] += self.restored[:, slot].sum()
                self.restored[:, slot].zero_()
                self.home[:, slot].fill_(-1)
            self.ctx[slot] = 0
            self.a_upto[slot] = 0
        self.rec.pop(slot, None)

    def nbytes(self) -> int:
        return sum(int(t.numel()) * t.element_size()
                   for t in (self.r, self.alist, self.acnt, self.home, self.restored)
                   if t is not None) + self.sketch.nbytes()

    # ---- attended lists --------------------------------------------------------------------------

    def append_positions(self, slot: int, hi: int) -> None:
        """Add the written positions [a_upto, hi) of ``slot`` to every (layer, KV head) list."""
        lo = self.a_upto[slot]
        if hi <= lo:
            return
        k = hi - lo
        cnt = self.acnt[:, slot]                                          # [L, H]
        dst = cnt.unsqueeze(-1).long() + torch.arange(k, device=self.dev)   # [L, H, k]
        src = torch.arange(lo, hi, device=self.dev, dtype=torch.int32).expand(self.L, self.H, k)
        self.alist[:, slot].scatter_(-1, dst, src)
        cnt += k
        self.a_upto[slot] = hi

    def rebuild_alist(self, slot: int, Lg: int) -> None:
        """After the round: the lists are the positions < Lg whose cell survived."""
        res = self.eng.cell[:, slot, :Lg] >= 0
        if self.cfg.recall and self.home is not None:
            res = res | (self.home[:, slot, :Lg] >= 0)                     # recalled every step
        res = res.permute(0, 2, 1)                                        # [L, H, Lg]
        col = res.cumsum(-1) - 1
        l, h, p = res.nonzero(as_tuple=True)                              # once per request
        buf = self.alist[:, slot]
        buf.zero_()
        buf[l, h, col[l, h, p]] = p.to(torch.int32)
        self.acnt[:, slot] = res.sum(-1).to(torch.int32)
        self.a_upto[slot] = Lg

    def build_recall(self, slot: int, Lg: int) -> None:
        """``cfg.recall``, after the round offloaded: every offloaded entry of ``slot`` grouped by
        layer, as (src CPU rows [L, E] int64, dst [L, E] int64, count [L] int32, tab, hcnt).
        No compaction: dst is the entry's home cell (tab / hcnt None). Compaction: dst is its row
        in the layer's staging buffer, KV head major (position order within a head); tab
        [L, H, Wh] int32 lists each head's staging rows and hcnt [L, H] int32 counts them."""
        cr = self.eng.cpu_row[:, slot, :Lg].permute(0, 2, 1)               # [L, H, Lg]
        off = cr >= 0
        cnt = off.reshape(self.L, -1).sum(-1)
        E = max(int(cnt.max()), 1)
        l, h, p = off.nonzero(as_tuple=True)                               # (l, h, p) order
        col = off.reshape(self.L, -1).cumsum(-1).view(off.shape)[l, h, p] - 1
        src = torch.zeros((self.L, E), dtype=torch.long, device=self.dev)
        dst = torch.zeros((self.L, E), dtype=torch.long, device=self.dev)
        src[l, col] = cr[l, h, p].long()
        tab = hcnt = None
        if self.home is not None:
            dst[l, col] = self.home[:, slot, :Lg][l, p, h].long()
        else:
            dst[l, col] = col
            hc = off.sum(-1)                                               # [L, H]
            Wh = max(int(hc.max()), 1)
            start = hc.cumsum(-1) - hc
            j = torch.arange(Wh, device=self.dev)
            tab = torch.where(j < hc.unsqueeze(-1), start.unsqueeze(-1) + j, 0).to(torch.int32)
            hcnt = hc.to(torch.int32)
        self.rec[slot] = (src, dst, cnt.to(torch.int32), tab, hcnt)

    # ---- request info ---------------------------------------------------------------------------

    def set_prompt(self, req_id: str, prompt_ids) -> None:
        """The request's context (prompt) tokens: their count bounds the sampled context, their
        crc keys the Bernoulli draw."""
        ids = list(prompt_ids)
        self._prompts[str(req_id)] = (len(ids), prompt_crc(ids))

    def forget(self, req_id: str) -> None:
        self._prompts.pop(str(req_id), None)

    # ---- Algorithm 1 at the round -------------------------------------------------------------------

    @torch.no_grad()
    def prefill_round(self, req_id: str, slot: int, Lg: int) -> torch.Tensor:
        """Residency, draw and sketch for request ``slot`` from its scores and (all resident)
        context keys. Returns ``keep`` [L, ctx, H] bool (the cells the draw keeps on the GPU);
        the caller offloads the others. Generated tokens (positions >= ctx) stay resident."""
        info = self._prompts.get(str(req_id))
        if info is None:
            raise RuntimeError(f"vkv: no prompt registered for {req_id} (set_prompt)")
        ctx, crc = info
        ctx = min(int(ctx), int(Lg))
        eng, cfg = self.eng, self.cfg
        self.grow(eng.S)
        cells = eng.cell[:, slot, :ctx]                                     # [L, ctx, H]
        if not bool((cells >= 0).all()):
            raise RuntimeError(f"vkv: {req_id} has offloaded context cells before its round")
        tau = eng.score[:, slot, :ctx].permute(0, 2, 1).double()            # [L, H, ctx]
        # KVzip marks never-scored sinks +inf; they are at r = 1 anyway (n_sink).
        tau = torch.where(torch.isposinf(tau), torch.ones_like(tau), tau)
        if cfg.evict:
            return self._evict_round(req_id, slot, tau.contiguous())
        if cfg.recall:
            return self._recall_round(req_id, slot, ctx, crc)
        tau = ref.transform_scores_for_sampling(tau.reshape(-1)).view_as(tau)
        kept_frac = cfg.draw_frac(self.D)
        r = ref.residency(tau, kept_frac, cfg.n_sink, cfg.window)           # [L, H, ctx] f64
        gen = torch.Generator(device=r.device).manual_seed(draw_seed(crc, cfg.seed))
        keep = (torch.rand(r.reshape(-1).shape, generator=gen, dtype=torch.float64,
                           device=r.device) < r.reshape(-1)).view_as(r)
        r32 = r.float()
        r32 = torch.where((r < 1) & (r32 >= 1), torch.full_like(r32, _BELOW_ONE), r32)
        self.r[:, slot, :ctx] = r32.permute(0, 2, 1)
        self.r[:, slot, ctx:] = 1.0
        self.ctx[slot] = ctx
        # Sketch every context key (the harness sketches the whole context), layer by layer.
        C = self.D
        for l in range(self.L):
            k = eng.kv_rows[cells[l].reshape(-1).long(), :C].view(ctx, self.H, C)
            self.sketch.build(l, slot, k.permute(1, 0, 2))
        kept = float(keep.double().mean())
        self.round_log.append({"req": str(req_id), "ctx": ctx, "kept": kept,
                               "r_mean": float(r.mean()), "r_one": float((r >= 1).double().mean()),
                               "r_zero": float((r <= 0).double().mean())})
        if len(self.round_log) <= 4:
            print(f"[VKV] round {req_id}: ctx={ctx} kappa={cfg.kappa} draw={kept_frac:.4f} "
                  f"kept={kept:.4f} "
                  f"(r=1: {self.round_log[-1]['r_one']:.4f}, r=0: "
                  f"{self.round_log[-1]['r_zero']:.4f})", flush=True)
        return keep.permute(0, 2, 1)                                        # [L, ctx, H]

    def _recall_round(self, req_id: str, slot: int, ctx: int, crc: int) -> torch.Tensor:
        """``cfg.recall``: the same amount stays on the GPU as vkv's draw (``draw_frac``), but no
        scores: every decode step recalls all offloaded cells, so which cells does not matter.
        The sink and window stay; each other cell stays with the probability that makes the
        expected kept fraction ``draw_frac`` (seeded like the draw). ctx = 0: attend-only."""
        cfg = self.cfg
        pos = torch.arange(ctx, device=self.dev)
        fixed = (pos < cfg.n_sink) | (pos >= ctx - cfg.window)              # [ctx]
        nf = int(fixed.sum())
        p = min(max((cfg.draw_frac(self.D) * ctx - nf) / max(ctx - nf, 1), 0.0), 1.0)
        gen = torch.Generator(device=self.dev).manual_seed(draw_seed(crc, cfg.seed))
        keep = torch.rand((self.L, self.H, ctx), generator=gen, device=self.dev) < p
        keep |= fixed.view(1, 1, ctx)
        self.r[:, slot] = 1.0
        self.ctx[slot] = 0
        kept = float(keep.double().mean())
        self.round_log.append({"req": str(req_id), "ctx": ctx, "kept": kept, "r_mean": kept,
                               "r_one": kept, "r_zero": 1.0 - kept})
        if len(self.round_log) <= 4:
            print(f"[VKV] recall round {req_id}: ctx={ctx} kept={kept:.4f} (no scores)",
                  flush=True)
        return keep.permute(0, 2, 1)                                        # [L, ctx, H]

    def _evict_round(self, req_id: str, slot: int, score: torch.Tensor) -> torch.Tensor:
        """``cfg.evict``: kvpress ``KVzipPress.compress_post`` (not layerwise) at compression
        ratio 1 - kappa: the int(numel * ratio) lowest scores over (layer, KV head, token) give
        each layer its count, then each layer drops its own lowest that many. ctx = 0 makes the
        decoder attend to the resident cells only (exactly, weight 1)."""
        L, H, ctx = score.shape
        n_pr = int(score.numel() * (1.0 - self.cfg.kappa))
        keep = torch.ones_like(score, dtype=torch.bool)
        if n_pr > 0:
            pi = torch.topk(-score.reshape(-1), n_pr).indices
            per = torch.bincount(pi // (H * ctx), minlength=L).tolist()
            for l, k in enumerate(per):
                if k:
                    keep[l].view(-1)[torch.topk(-score[l].reshape(-1), k).indices] = False
        self.r[:, slot] = 1.0
        self.ctx[slot] = 0
        kept = float(keep.double().mean())
        self.round_log.append({"req": str(req_id), "ctx": ctx, "kept": kept, "r_mean": kept,
                               "r_one": kept, "r_zero": 1.0 - kept})
        if len(self.round_log) <= 4:
            print(f"[VKV] evict round {req_id}: ctx={ctx} kappa={self.cfg.kappa} kept={kept:.4f}",
                  flush=True)
        return keep.permute(0, 2, 1)

    # ---- statistics ----------------------------------------------------------------------------------

    def add_stats(self, **vals) -> None:
        for k, v in vals.items():
            i = STATS.index(k)
            self.stats[i] += v if isinstance(v, torch.Tensor) else float(v)

    def add_stats_kernel(self, ctx_d, Hq, k, tsize, beta, Bt, nonmono, fb, screened, slen,
                         xfer, keys, chunked) -> None:
        """One decode layer call's statistics (``kernels.stats_kernel``): per-row decision values
        [R], ``xfer`` / ``keys`` device scalars (cold fetched entries, shortlist host key reads)."""
        import triton
        from .kernels import stats_kernel
        R = int(k.numel())
        stats_kernel[(1,)](self.stats, ctx_d, k, tsize, beta, Bt, nonmono, fb, screened, slen,
                           xfer, keys, R, Hq, float(chunked),
                           BR=triton.next_power_of_2(max(R, 1)))

    def report(self) -> dict:
        s = dict(zip(STATS, self.stats.tolist()))
        d = max(s["decisions"], 1.0)
        s.update(fetch_any_frac=s["fetch_any"] / d, k_mean=s["k_sum"] / d,
                 xfer_mean=s["xfer_sum"] / d, fallback_frac=s["fallback"] / d,
                 screened_frac=s["screened"] / d,
                 keys_per_layer=s["keys_sum"] / max(s["layer_calls"], 1.0))
        return s


STATS = ("decisions", "fetch_any", "k_sum", "xfer_sum", "T_size_sum", "beta_rel_sum", "nonmono",
         "fallback", "screened", "layer_calls", "chunked", "keys_sum", "attended_sum", "copied")
