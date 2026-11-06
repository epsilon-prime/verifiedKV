"""VerifiedKV attention press used in all quality experiments.

The attention bias is bit-identical to press_reference.py (tests/test_press.py). This version removes every
statistics-only computation. The certificate uses chernoff_vec.kstar_fast unless VKV_VEC=0.
"""

import os
import sys
import zlib
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch

import kvpress.presses.needle_fetch_press as nfp
from kvpress.presses.bernoulli_press import BernoulliPress, transform_scores_for_sampling
from kvpress.presses.cached_kvzip_press import CachedKVzipPress
from kvpress.utils import get_cache_metadata

from verifiedkv import algo
from verifiedkv import chernoff

if os.environ.get("VKV_VEC", "1") == "1":  # lambda-vectorised Chernoff (chernoff_vec.py), the default; VKV_VEC=0: loop version
    from verifiedkv import chernoff_vec
    chernoff.kstar_fast = chernoff_vec.kstar_fast

CHUNK = 6_000_000  # KV heads x rows x ctx elements per block of the float64 plan


def vkv_grid_bias(query: torch.Tensor, key: torch.Tensor, state: dict, scaling: float) -> torch.Tensor:
    """One forward of one layer: the attention bias [1, num_heads, q_len, ctx] (see press_reference.vkv_final_bias)."""
    cfg, r, keep = state["vkv_grid"], state["r"], state["keep"]
    bsz, n_kv, ctx = r.shape
    assert bsz == 1, "batch size 1 only"
    num_heads, q_len, d = query.shape[1], query.shape[2], query.shape[3]
    G = num_heads // n_kv
    rows = G * q_len
    k_len = key.shape[2]
    dev = query.device
    q_all = query.float().reshape(n_kv, rows, d) * scaling  # row = g * q_len + t
    t_pos = torch.arange(rows, device=dev) % q_len
    r_h = r[0].double()
    Z = keep[0]
    tail = r_h < 1
    tail3, r3, Z3 = tail[:, None], r_h[:, None], Z[:, None]
    allowed = torch.arange(k_len - ctx, device=dev)[None, :] <= (k_len - q_len - ctx + t_pos)[:, None]
    out = torch.empty(n_kv, rows, ctx, dtype=torch.float32, device=dev)
    blk = max(1, CHUNK // (n_kv * ctx))
    sketched = cfg["plan_weights"] == "sketch"
    k_hat, ch = (state["k_hat"][0], state["channels"]) if sketched else (None, None)

    def sketch_scores(q):
        q_s = q if ch is None else q.gather(-1, ch[0][:, None, :].expand(-1, q.shape[1], -1))
        return torch.matmul(q_s, k_hat.transpose(-1, -2))

    C_head = None
    if sketched and cfg["shortlist_head"]:  # one list of M tail tokens per KV head, by the best score over its rows
        best = torch.full((n_kv, ctx), -float("inf"), device=dev)
        for a in range(0, rows, blk):
            best = torch.maximum(best, sketch_scores(q_all[:, a: a + blk]).amax(1))
        best = torch.where(tail, best, torch.full_like(best, -float("inf")))
        idx = best.topk(min(int(cfg["shortlist_head"]), ctx), -1).indices
        C_head = torch.zeros(n_kv, ctx, dtype=torch.bool, device=dev).scatter_(-1, idx, True) & tail

    zero = torch.zeros((), dtype=torch.float64, device=dev)
    for a in range(0, rows, blk):
        sl = slice(a, a + blk)
        q = q_all[:, sl]
        s = torch.matmul(q, key[0, :, :ctx].float().transpose(-1, -2))  # exact scores (resident / shortlist use)
        se = torch.matmul(q, key[0, :, ctx:].float().transpose(-1, -2)).masked_fill(~allowed[sl], -float("inf"))
        if sketched:
            s_hat = sketch_scores(q)
            if C_head is not None:
                s_hat = torch.where(C_head[:, None], s, s_hat)
            s_plan = torch.where(tail3, s_hat, s)  # sketch on the tail, exact elsewhere
        else:
            s_plan = s
        m = torch.maximum(s_plan.amax(-1), se.amax(-1))[..., None]
        wh = torch.exp((s_plan - m).double())
        wx = torch.exp((se - m).double()).sum(-1)
        sure = torch.where(tail3, zero, wh).sum(-1) + wx
        plan = algo.prepare(wh, r3, tail3, sure, cfg["delta"], trunc=True, order="grad")
        if cfg["fast"]:
            k, _, _ = chernoff.kstar_fast(plan, wh, r3, tail3, cfg["eps0"], cfg["delta"])
        else:
            P = chernoff.chernoff_curves(plan, wh, r3, tail3, [cfg["eps0"]])[0]
            k = chernoff.first_pass(P, cfg["delta"], plan.n_tail)
        F = algo.fetch_mask(plan, k) & tail3
        out[:, sl] = algo.attention_bias(r3, tail3, Z3, plan.T, F).float()
    return out.view(n_kv, G, q_len, ctx).reshape(1, num_heads, q_len, ctx)


_prev_hook = nfp.needle_decode_bias


def _route(query, key, state, scaling):
    if "vkv_grid" in state:
        return vkv_grid_bias(query, key, state, scaling)
    return _prev_hook(query, key, state, scaling)


nfp.needle_decode_bias = _route


@dataclass
class VkvGridPress(nfp.NeedleFetchPress):
    """press_reference.VkvFinalPress without statistics (same fields and defaults)."""

    bern: BernoulliPress = field(default_factory=lambda: BernoulliPress(press=CachedKVzipPress()))
    kept: float = 0.0625
    eps0: float = 0.3
    delta: float = 0.05
    shortlist_head: int = 128
    plan_weights: str = "sketch"
    charge_sketch: bool = True
    fast: bool = True
    n_sink: int = 64
    window: int = 64
    bits: int = 2
    n_pairs: int | None = 16
    pairs_per_128_dims: bool = True

    def __post_init__(self):
        super().__post_init__()
        assert self.plan_weights in ("sketch", "perfect"), self.plan_weights
        self._total = self._total or 0.9375
        self.kept_draw = None
        self.kept_log: list[float] = []

    @property
    def draw_frac(self) -> float:
        paid = self.sketch_frac if (self.plan_weights == "sketch" and self.charge_sketch) else 0.0
        return self.kept - paid

    @contextmanager
    def __call__(self, model):
        self.post_init_from_model(model)
        self.bern.compress_post = self._compress_post
        try:
            with self.bern(model):
                yield
        finally:
            del self.bern.compress_post

    def _compress_post(self, model):
        sc = self.bern.press
        scores = sc.score_val
        n_rows, bsz, n_kv, ctx = scores.shape
        assert bsz == 1
        kept = self.draw_frac
        assert kept > 0, f"the sketch ({self.sketch_frac:.4f}) leaves nothing of the budget {self.kept}"
        self.kept_draw = kept
        flat = transform_scores_for_sampling(scores.reshape(-1).double())
        r = algo.residency(flat.view(n_rows, 1, n_kv, ctx), kept, self.n_sink, self.window)
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
        self._attach_sketch(model)
        cfg = {"eps0": self.eps0, "delta": self.delta, "shortlist_head": self.shortlist_head,
               "plan_weights": self.plan_weights, "fast": self.fast}
        for li, stt in metadata["needle"].items():
            row = rows[int(li)]
            stt.update({"vkv_grid": cfg, "r": r[row], "keep": keep[row]})


def vkv_grid(compression_ratio: float = 0.9375, eps0: float = 0.3, delta: float = 0.05, shortlist_head: int = 128,
             **kw) -> VkvGridPress:
    return VkvGridPress(kept=1 - compression_ratio, eps0=eps0, delta=delta, shortlist_head=shortlist_head, **kw)
