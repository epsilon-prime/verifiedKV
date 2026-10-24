"""COPY of phase4/vkv1/press.py (2026-10-24) with one addition: check="chern", the exact two-point Chernoff
certificate from phase4/experiments/vkv1_diag/diag_lib.py in place of Bernstein (same truncation, order, sweep,
persistence). Import this module INSTEAD of vkv1.press (both route kvpress's needle hook).

Vkv1Press: vkv_draft0.pdf (Algorithms 1 and 2) in a real generation run. Research harness on kvpress (imported
read-only, like phase3/experiments/algo_improvements/eps_press.py); not a kvpress press.

Residency (Algorithm 1, ``algo.residency``): KVzip scores tau (CachedKVzipPress), one global c with
sum min(1, c tau) = kept * size over all layers and KV heads, the first ``n_sink`` (default 64) and last ``window``
(default 64) context entries at r = 1 and paid from the same budget, one Bernoulli draw from BernoulliPress's generator
keyed on the context tokens. With n_sink = 4, window = 0 the residency and the draw are BernoulliPress's exactly.

Every forward over the cache (the question pass and each decode step), every layer, KV head, query head and position
(one row = one decision): Algorithm 2 on the plan weights, which are the exact weights (``sketch="perfect"``: the full
cache is on the GPU, so this is the perfect-sketch setting) or needle fetch's 2-bit 16-pair sketch (``sketch="needle"``):
    T* (MSE rule) -> certificate sweep in fetch order -> k* -> F; attention bias 0 on r = 1 / persisted / F / sampled T*,
    log(1/r) on other sampled tail entries, -inf elsewhere (additive mask through kvpress's needle hook, sdpa only).
After the forward, the union of F over its rows stays resident (exact) for the rest of the answer (``persist``); a new
question (a multi-token forward) starts from nothing. Nothing is freed or moved: dropped entries are masked, and
"fetching" is unmasking; transfers are counted (entries read from the host = F minus the sampled set).

The full cache is present, so every decision also records the realised error |B hat - B| / B of the denominator.

Import this module before running (it routes kvpress's needle hook for states carrying "vkv1").
"""

import zlib
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch

import kvpress.presses.needle_fetch_press as nfp
from kvpress.presses.bernoulli_press import BernoulliPress, transform_scores_for_sampling
from kvpress.presses.cached_kvzip_press import CachedKVzipPress
from kvpress.utils import get_cache_metadata

import os
import sys

sys.path.insert(0, os.path.join(os.path.expanduser("~"), "phase4"))
sys.path.insert(0, os.path.join(os.path.expanduser("~"), "phase4", "experiments", "vkv1_diag"))
from vkv1 import algo  # noqa: E402  (read-only)
import diag_lib as dl  # noqa: E402  (exact two-point Chernoff curves)

CHECKS = (*algo.CHECKS, "chern")

CHUNK = 6_000_000  # KV heads x rows x ctx elements per block of the float64 plan
KINDS = ("question", "decode")
STATS = ("decisions", "fetch_any", "k_sum", "xfer_sum", "fail", "fail_T0", "fail_HT0", "beta_rel_sum", "T_size_sum",
         "fwd", "fwd_any", "fwd_xfer", "fwd_ctx")


def new_stats() -> dict:
    st = {f"{k}_{m}": 0.0 for k in KINDS for m in STATS}
    st.update(decode_steps=0, decode_steps_any=0, persisted_extra=0.0, persisted_slots=0.0, _step_any=False)
    return st


def vkv1_bias(query: torch.Tensor, key: torch.Tensor, state: dict, scaling: float) -> torch.Tensor:
    """Per-row attention bias [bsz, num_heads, q_len, ctx] (Algorithm 2 (e)), statistics, persistence."""
    cfg, bias0, r, keep = state["vkv1"], state["bias"], state["r"], state["keep"]
    bsz, n_kv, ctx = bias0.shape
    assert bsz == 1, "batch size 1 only"
    num_heads, q_len, d = query.shape[1], query.shape[2], query.shape[3]
    G = num_heads // n_kv
    rows = G * q_len
    k_len = key.shape[2]
    dev = query.device
    q_all = query.float().reshape(n_kv, rows, d) * scaling  # row = g * q_len + t
    t_pos = torch.arange(rows, device=dev) % q_len
    promoted = state["promoted"][0]  # [n_kv, ctx]
    if q_len > 1:
        promoted.zero_()  # a new question: nothing carries over from the previous answer
    kind = "decode" if q_len == 1 else "question"
    st = state["stats"]
    if q_len == 1 and state["layer_idx"] == state["first_layer"]:
        st["decode_steps"] += 1
        st["_step_any"] = False
    r_h = r[0].double()  # [n_kv, ctx], float64 (a float32 r could round 1 - 1e-9 up to 1)
    Z = keep[0]  # sampled (bool) [n_kv, ctx]; r = 1 entries are always kept
    tail = (r_h < 1) & ~promoted  # [n_kv, ctx]
    tail3, r3, Z3 = tail[:, None], r_h[:, None], Z[:, None]
    allowed = torch.arange(k_len - ctx, device=dev)[None, :] <= (k_len - q_len - ctx + t_pos)[:, None]
    out = torch.empty(n_kv, rows, ctx, dtype=torch.float32, device=dev)
    union = torch.zeros(n_kv, ctx, dtype=torch.bool, device=dev)
    xfer_union = torch.zeros_like(union)
    zero = torch.zeros((), dtype=torch.float64, device=dev)
    blk = max(1, CHUNK // (n_kv * ctx))
    for a in range(0, rows, blk):
        sl = slice(a, a + blk)
        q = q_all[:, sl]
        s = torch.matmul(q, key[0, :, :ctx].float().transpose(-1, -2))  # [n_kv, R, ctx], exact
        se = torch.matmul(q, key[0, :, ctx:].float().transpose(-1, -2)).masked_fill(~allowed[sl], -float("inf"))
        if cfg["sketch"] == "perfect":
            m = torch.maximum(s.amax(-1), se.amax(-1))[..., None]
            w = torch.exp((s - m).double())
            wh = w
        else:
            k_hat, ch = state["k_hat"][0], state["channels"]
            q_s = q if ch is None else q.gather(-1, ch[0][:, None, :].expand(-1, q.shape[1], -1))
            s_hat = torch.matmul(q_s, k_hat.transpose(-1, -2))
            m = torch.maximum(torch.where(tail3, s_hat, s).amax(-1), se.amax(-1))[..., None]
            w = torch.exp((s - m).double())
            wh = torch.exp((s_hat - m).double())
        wx = torch.exp((se - m).double()).sum(-1)
        sure = torch.where(tail3, zero, w).sum(-1) + wx  # exact resident mass (r = 1, persisted, w^x)
        plan = algo.prepare(wh, r3, tail3, sure, cfg["delta"], trunc=cfg["trunc"], order=cfg["order"])
        if cfg["check"] == "chern":  # exact two-point Chernoff bound (vkv1_diag), first passing k
            k = dl.first_pass(dl.chernoff_curves(plan, wh, r3, tail3, [cfg["eps0"]])[0], cfg["delta"], plan.n_tail)
        else:
            k, _ = algo.kstar(plan, cfg["eps0"], cfg["check"])
        F = algo.fetch_mask(plan, k) & tail3
        out[:, sl] = algo.attention_bias(r3, tail3, Z3, plan.T, F).float()
        union |= F.any(1)
        xf = F & ~Z3
        xfer_union |= xf.any(1)
        # statistics, with the realised error against the full cache
        B = w.sum(-1) + wx
        no = torch.zeros_like(F)
        e = (algo.estimate(w, r3, tail3, sure, Z3, plan.T, F) - B).abs() / B
        e0 = (algo.estimate(w, r3, tail3, sure, Z3, plan.T, no) - B).abs() / B
        eh = (algo.estimate(w, r3, tail3, sure, Z3, tail3 & (r3 <= 0), no) - B).abs() / B
        eps0 = cfg["eps0"]
        st[f"{kind}_decisions"] += k.numel()
        st[f"{kind}_fetch_any"] += int((k > 0).sum())
        st[f"{kind}_k_sum"] += int(k.sum())
        st[f"{kind}_xfer_sum"] += int(xf.sum())
        st[f"{kind}_fail"] += int((e > eps0).sum())
        st[f"{kind}_fail_T0"] += int((e0 > eps0).sum())
        st[f"{kind}_fail_HT0"] += int((eh > eps0).sum())
        st[f"{kind}_beta_rel_sum"] += float((algo.curve_at(plan.beta, k) / plan.Bt).sum())
        st[f"{kind}_T_size_sum"] += int((plan.T & tail3).sum())
    u = xfer_union.sum(-1)
    st[f"{kind}_fwd"] += n_kv
    st[f"{kind}_fwd_any"] += int((u > 0).sum())
    st[f"{kind}_fwd_xfer"] += int(u.sum())
    st[f"{kind}_fwd_ctx"] += n_kv * ctx
    if q_len == 1 and not st["_step_any"] and bool((u > 0).any()):
        st["decode_steps_any"] += 1
        st["_step_any"] = True
    if cfg["persist"]:
        new = union & ~promoted
        st["persisted_extra"] += int((new & ~Z).sum())  # entries the draw had dropped that now stay resident
        promoted |= union
    return out.view(n_kv, G, q_len, ctx).reshape(1, num_heads, q_len, ctx)


_prev_bias = nfp.needle_decode_bias


def _route(query, key, state, scaling):
    if "vkv1" in state:
        return vkv1_bias(query, key, state, scaling)
    return _prev_bias(query, key, state, scaling)


nfp.needle_decode_bias = _route  # kvpress's attention patch imports it from the module at call time


@dataclass
class Vkv1Press(nfp.NeedleFetchPress):
    """See the module docstring. ``kept`` is the resident fraction kappa (sketch not charged; with the perfect sketch
    there is none)."""

    bern: BernoulliPress = field(default_factory=lambda: BernoulliPress(press=CachedKVzipPress()))
    kept: float = 0.0625
    eps0: float = 0.3
    delta: float = 0.05
    check: str = "simple"
    order: str = "needle"
    trunc: bool = True
    sketch: str = "perfect"  # or "needle" (2-bit, 16 RoPE pairs per 128 dims)
    n_sink: int = 64  # first entries at r = 1 (attention sinks)
    window: int = 64  # last context entries at r = 1 (local window)
    persist: bool = True
    bits: int = 2
    n_pairs: int | None = 16
    pairs_per_128_dims: bool = True

    def __post_init__(self):
        super().__post_init__()
        assert self.check in CHECKS and self.order in algo.ORDERS and self.sketch in ("perfect", "needle")
        # KVzipPress only scores when its ratio is > 0; NeedleFetchPress derives that ratio from _total. The value is
        # otherwise unused: ``kept`` sets the residency.
        self._total = self._total or 0.9375
        self.stats = new_stats()
        self.kept_log: list[float] = []

    def reset(self):
        self.stats.clear()
        self.stats.update(new_stats())
        self.kept_log.clear()

    @contextmanager
    def __call__(self, model):
        self.post_init_from_model(model)
        self.bern.compress_post = self._vkv1_compress_post  # BernoulliPress hands it to the scorer
        try:
            with self.bern(model):
                yield
        finally:
            del self.bern.compress_post

    def _vkv1_compress_post(self, model):
        sc = self.bern.press
        scores = sc.score_val  # [n_rows, 1, n_kv, ctx]
        n_rows, bsz, n_kv, ctx = scores.shape
        assert bsz == 1
        flat = transform_scores_for_sampling(scores.reshape(-1).double())
        r = algo.residency(flat.view(n_rows, 1, n_kv, ctx), self.kept, self.n_sink, self.window)
        crc = zlib.crc32(sc._context_ids.detach().cpu().numpy().tobytes())
        gen = self.bern._generator(r.device, crc ^ (self.bern.seed * 7919))
        keep = (torch.rand(r.reshape(-1).shape, generator=gen, dtype=torch.float64, device=r.device)
                < r.reshape(-1)).view_as(r)
        bias = torch.where(keep, -torch.log(r.clamp_min(1e-300)), torch.tensor(-float("inf"), device=r.device))
        self.kept_log.append(float(keep.double().mean()))
        metadata = get_cache_metadata(sc._cache)
        ab = metadata.setdefault("attention_bias", {})
        rows = sc._score_row_by_layer
        for module in self.bern._attention_modules(model):
            ab[int(module.layer_idx)] = bias[rows[int(module.layer_idx)]].float()
            module.masked_key_indices = None
        self._attach_sketch(model)  # moves attention_bias into metadata["needle"] with the sketch (k_hat, channels)
        cfg = {"eps0": self.eps0, "delta": self.delta, "check": self.check, "order": self.order, "trunc": self.trunc,
               "sketch": self.sketch, "persist": self.persist}
        for li, stt in metadata["needle"].items():
            row = rows[int(li)]
            stt.update({"vkv1": cfg, "r": r[row], "keep": keep[row], "stats": self.stats,
                        "promoted": torch.zeros_like(keep[row])})
        self.stats["persisted_slots"] += float(sum(int(s["keep"].numel()) for s in metadata["needle"].values()))
