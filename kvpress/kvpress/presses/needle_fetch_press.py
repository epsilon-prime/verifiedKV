# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Generator

import torch
from torch import nn
from transformers import PreTrainedModel

from kvpress.presses.base_press import BasePress
from kvpress.presses.bernoulli_press import BernoulliPress
from kvpress.presses.kvzip_press import KVzipPress
from kvpress.adapters import get_adapter
from kvpress.utils import get_cache_metadata

logger = logging.getLogger(__name__)

KV_BYTES_PER_CHANNEL = 4  # fp16 K and V: 2 bytes each per channel


def one_bit_sketch(keys: torch.Tensor, group: int) -> torch.Tensor:
    """Dequantised 1-bit sketch of post-RoPE keys, per channel in token groups.

    Each (group, channel) stores two fp16 levels, the means of the values above and below the
    midpoint of its range, and one bit per entry. ``keys`` is ``[bsz, n_kv, n, d]``; returns the
    reconstruction with the same shape, in float32.
    """
    bsz, n_kv, n, d = keys.shape
    k = keys.float()
    n_groups = -(-n // group)
    pad = n_groups * group - n
    if pad:
        k = torch.cat([k, k[:, :, -1:].expand(bsz, n_kv, pad, d)], dim=2)
    k = k.view(bsz, n_kv, n_groups, group, d)
    mid = (k.amin(3, keepdim=True) + k.amax(3, keepdim=True)) / 2
    above = k > mid
    cnt_a = above.sum(3, keepdim=True).clamp_min(1)
    cnt_b = (~above).sum(3, keepdim=True).clamp_min(1)
    mu_a = ((k * above).sum(3, keepdim=True) / cnt_a).half().float()
    mu_b = ((k * ~above).sum(3, keepdim=True) / cnt_b).half().float()
    k_hat = torch.where(above, mu_a, mu_b)
    return k_hat.view(bsz, n_kv, n_groups * group, d)[:, :, :n]


def quant_sketch(keys: torch.Tensor, bits: int, group: int) -> torch.Tensor:
    """Dequantised ``bits``-bit sketch, per channel in token groups: two-means levels for 1 bit,
    uniform levels between the group's minimum and maximum (fp16 zero and scale) otherwise."""
    if bits == 1:
        return one_bit_sketch(keys, group)
    bsz, n_kv, n, d = keys.shape
    k = keys.float()
    n_groups = -(-n // group)
    pad = n_groups * group - n
    if pad:
        k = torch.cat([k, k[:, :, -1:].expand(bsz, n_kv, pad, d)], dim=2)
    k = k.view(bsz, n_kv, n_groups, group, d)
    levels = 2**bits - 1
    zero = k.amin(3, keepdim=True).half().float()
    scale = ((k.amax(3, keepdim=True) - zero) / levels).clamp_min(1e-6).half().float()
    k_hat = zero + ((k - zero) / scale).round().clamp(0, levels) * scale
    return k_hat.view(bsz, n_kv, n_groups * group, d)[:, :, :n]


def top_energy_pairs(keys: torch.Tensor, n_pairs: int, rotary_dim: int | None = None) -> torch.Tensor:
    """Channel indices of the ``n_pairs`` RoPE pairs with the largest mean key energy, per KV head.

    RoPE rotates within each pair (``j``, ``j + d/2``), so a pair's share of ``<q, k>`` does not
    depend on position, and the choice needs no queries. Returns ``[bsz, n_kv, 2 * n_pairs]``.

    With partial RoPE (``rotary_dim < d``, e.g. Qwen3.5: 64 of 256 dims) only the first ``rotary_dim``
    channels rotate, in pairs (``j``, ``j + rotary_dim/2``); every other channel is position-free on
    its own. The same ``2 * n_pairs`` channels are then chosen from rotary pairs (kept whole) and single
    channels, greedily by mean energy per channel.
    """
    d = keys.shape[-1]
    if rotary_dim is None or rotary_dim >= d:
        half = d // 2
        k = keys.float()
        energy = (k[..., :half] ** 2 + k[..., half:] ** 2).mean(2)  # [bsz, n_kv, half]
        pairs = energy.topk(min(n_pairs, half), dim=-1).indices
        return torch.cat([pairs, pairs + half], dim=-1)
    return _top_energy_channels_partial(keys, 2 * n_pairs, rotary_dim)


def _top_energy_channels_partial(keys: torch.Tensor, n_channels: int, rotary_dim: int) -> torch.Tensor:
    """Greedy channel choice under partial RoPE (see ``top_energy_pairs``); ``[bsz, n_kv, n_channels]``."""
    bsz, n_kv, _, d = keys.shape
    half_r = rotary_dim // 2
    energy = (keys.float() ** 2).mean(2)  # [bsz, n_kv, d]
    out = torch.empty(bsz, n_kv, n_channels, dtype=torch.long, device=keys.device)
    for b in range(bsz):
        for h in range(n_kv):
            e = energy[b, h].tolist()
            # units: (energy per channel, channels); a rotary pair is atomic, a pass-through channel is single
            units = [((e[j] + e[j + half_r]) / 2, (j, j + half_r)) for j in range(half_r)]
            units += [(e[c], (c,)) for c in range(rotary_dim, d)]
            units.sort(key=lambda u: -u[0])
            chosen: list[int] = []
            for _, chans in units:
                if len(chosen) + len(chans) <= n_channels:
                    chosen.extend(chans)
                if len(chosen) == n_channels:
                    break
            out[b, h] = torch.tensor(chosen, device=keys.device)
    return out


def sketch_bytes_per_entry(
    head_dim: int,
    group: int,
    bits: int = 1,
    n_pairs: int | None = None,
    sketch: str = "quant",
    pq_subspaces: int = 16,
    pq_bits: int = 8,
    ctx: int | None = None,
) -> float:
    """Sketch bytes per (token, KV head).

    "quant": packed bits plus two fp16 values per sketched channel (levels, or zero and scale), amortised
    over the group. The per-head list of pairs is negligible.

    "pq": one ``pq_bits`` code per subspace, plus the per-(layer, KV head) codebook (2**pq_bits fp16
    centroids per subspace, head_dim * 2**pq_bits values in all) amortised over the ``ctx`` context tokens.
    The codebook is left out while the context length is unknown.
    """
    if sketch == "pq":
        codebook = head_dim * 2**pq_bits * 2 / ctx if ctx else 0.0
        return pq_subspaces * pq_bits / 8 + codebook
    channels = head_dim if n_pairs is None else 2 * min(n_pairs, head_dim // 2)
    return bits * channels / 8 + 2 * channels * 2 / group


def _nearest_centroid(x: torch.Tensor, centroids: torch.Tensor, chunk: int = 8192) -> torch.Tensor:
    """Index of the nearest centroid for every row of ``x`` ([B, n, s] against [B, K, s]), in token chunks.
    |x|^2 is the same for every centroid, so |c|^2 - 2 x.c ranks them; no [B, n, K, s] tensor is built."""
    c_sq = (centroids * centroids).sum(-1)[:, None, :]
    return torch.cat([(c_sq - 2 * xc @ centroids.transpose(1, 2)).argmin(-1) for xc in x.split(chunk, dim=1)], 1)


def pq_sketch(
    keys: torch.Tensor,
    subspaces: int,
    bits: int,
    iters: int = 10,
    seed: int = 0,
    fit_sample: int | None = None,
    skip: int = 0,
) -> torch.Tensor:
    """Dequantised product-quantisation sketch of post-RoPE keys (PQCache; the skylight runtime's PQ).

    ``keys`` is ``[n_kv, n, d]``. Each key is split into ``subspaces`` contiguous sub-vectors, and every
    (KV head, subspace) gets a k-means codebook of 2**bits centroids (Euclidean, random-sample start,
    ``iters`` Lloyd steps, empty clusters refilled with random keys). The codebook is fit on ``fit_sample``
    random keys from positions ``>= skip`` (the sinks left out; all of them when ``fit_sample`` is None),
    then every key is encoded with its nearest centroid. Returns each key's centroids, ``[n_kv, n, d]`` in
    float32, so ``q . k_hat`` is the PQ score sum_s q_s . centroid_s[code_s]. The generator is local: the
    global random state is untouched.
    """
    n_kv, n, d = keys.shape
    assert d % subspaces == 0, f"head_dim {d} is not divisible by {subspaces} subspaces"
    sub = d // subspaces
    x = keys.float().reshape(n_kv, n, subspaces, sub).transpose(1, 2).reshape(n_kv * subspaces, n, sub)
    g = torch.Generator(device=x.device).manual_seed(seed)
    fit = x
    if fit_sample is not None or skip:
        pool = torch.arange(min(skip, n - 1), n, device=x.device)
        if fit_sample is not None and fit_sample < pool.numel():
            pool = pool[torch.randperm(pool.numel(), generator=g, device=x.device)[:fit_sample]]
        fit = x[:, pool]
    n_fit, k = fit.shape[1], 2**bits
    start = torch.stack([torch.randperm(n_fit, generator=g, device=x.device)[:k] for _ in range(x.shape[0])])
    centroids = fit.gather(1, start[:, :, None].expand(-1, -1, sub))
    for _ in range(iters):
        codes = _nearest_centroid(fit, centroids)
        sums = torch.zeros_like(centroids).scatter_add_(1, codes[:, :, None].expand(-1, -1, sub), fit)
        counts = torch.zeros(centroids.shape[:2], device=x.device).scatter_add_(
            1, codes, torch.ones_like(codes, dtype=x.dtype)
        )
        picks = torch.randint(0, n_fit, centroids.shape[:2], generator=g, device=x.device)
        refill = fit.gather(1, picks[:, :, None].expand(-1, -1, sub))
        centroids = torch.where(counts[:, :, None] > 0, sums / counts.clamp_min(1)[:, :, None], refill)
    codes = _nearest_centroid(x, centroids)
    k_hat = centroids.gather(1, codes[:, :, None].expand(-1, -1, sub))
    return k_hat.reshape(n_kv, subspaces, n, sub).transpose(1, 2).reshape(n_kv, n, d)


def needle_decode_bias(query: torch.Tensor, key: torch.Tensor, state: dict, scaling: float) -> torch.Tensor:
    """Per-query attention bias over the context: the static Bernoulli bias plus the fetched needles.

    For every query position and KV head, the sketch estimates each context logit (heavy hitters,
    ``r = 1``, use their exact logit, since they are resident). An entry with ``r < 1`` whose estimated
    attention weight reaches ``theta`` for any query head of the group joins the certainty set ``F``:
    its bias becomes 0, so it counts exactly. A kept entry in ``F`` just stops being upweighted; a
    dropped one is fetched. ``F`` depends only on the query and the sketch, never on the draw, so the
    rest remains a Horvitz-Thompson estimate.

    Returns ``[bsz, num_heads, q_len, ctx_len]``.
    """
    bias = state["bias"]  # [bsz, n_kv, ctx]
    k_hat = state["k_hat"]  # [bsz, n_kv, ctx, d]
    bsz, n_kv, ctx = bias.shape
    num_heads, q_len, d = query.shape[1], query.shape[2], query.shape[3]
    groups = num_heads // n_kv
    q = query.float().reshape(bsz, n_kv, groups * q_len, d) * scaling  # query arrives non-contiguous
    heavy = bias == 0  # kept with r = 1
    channels = state.get("channels")  # sketched channels per KV head, or None for all
    q_s = q if channels is None else q.gather(-1, channels[:, :, None, :].expand(-1, -1, q.shape[2], -1))
    s_hat = torch.matmul(q_s, k_hat.transpose(-1, -2))  # [bsz, n_kv, G*ql, ctx]
    s_exact = torch.matmul(q, key[:, :, :ctx].float().transpose(-1, -2))
    est = torch.where(heavy[:, :, None, :], s_exact, s_hat)
    log_w = s_hat - torch.logsumexp(est, dim=-1, keepdim=True)
    log_theta = state["log_theta"] if q_len == 1 else state.get("log_theta_question", state["log_theta"])
    flag = (log_w >= log_theta) & ~heavy[:, :, None, :]
    flag = flag.reshape(bsz, n_kv, groups, q_len, ctx).any(dim=2)  # [bsz, n_kv, ql, ctx]
    log_w = log_w.reshape(bsz, n_kv, groups * q_len, ctx)
    promoted = state.get("promoted")  # entries fetched earlier in this answer, still resident
    if promoted is not None and q_len > 1:
        # a multi-token forward over the cache reads a new question: fetches persist within one answer,
        # never into the next question on the same context (audit B2). A no-op for the first question.
        promoted.zero_()
    exact = flag if promoted is None else flag | promoted[:, :, None, :]
    new_bias = torch.where(exact, torch.zeros((), device=bias.device), bias[:, :, None, :])

    stats = state["stats"]
    dropped = torch.isinf(bias)[:, :, None, :]
    new = flag & dropped if promoted is None else flag & dropped & ~promoted[:, :, None, :]
    fetched = new.sum(-1)  # [bsz, n_kv, ql]
    kind = "decode" if q_len == 1 else "question"
    stats[f"{kind}_decisions"] += fetched.numel()
    stats[f"{kind}_fetch_any"] += int((fetched > 0).sum())
    stats[f"{kind}_fetched_entries"] += int(new.any(dim=2).sum()) if q_len > 1 else int(fetched.sum())
    stats[f"{kind}_ctx_entries"] += fetched.numel() * ctx
    if q_len == 1:  # per-layer decode breakdown: which layers drive the fetch rate
        # per generated token: a step starts at the first sketched layer; it counts as fetching if any
        # (layer, KV head) of it fetches, which is the unit a host round trip is paid in
        if state.get("layer_idx", 0) == state.get("first_layer", 0):
            stats["decode_steps"] = stats.get("decode_steps", 0) + 1
            stats["_step_any"] = False
        if not stats.get("_step_any", False) and bool((fetched > 0).any()):
            stats["decode_steps_any"] = stats.get("decode_steps_any", 0) + 1
            stats["_step_any"] = True
        per_layer = stats.setdefault("per_layer", {})
        rec = per_layer.setdefault(str(state.get("layer_idx", -1)), [0, 0, 0])
        rec[0] += fetched.numel()
        rec[1] += int((fetched > 0).sum())
        rec[2] += int(fetched.sum())
    if q_len > 1:  # one batched transfer per (layer, KV head) for the whole question
        per_head = new.any(dim=2).sum(-1)  # [bsz, n_kv]
        stats["question_heads"] = stats.get("question_heads", 0) + per_head.numel()
        stats["question_heads_any"] = stats.get("question_heads_any", 0) + int((per_head > 0).sum())
        frac = per_head.float() / ctx  # share of the context one (layer, KV head) transfers for the question
        stats["question_frac_sum"] = stats.get("question_frac_sum", 0.0) + float(frac.sum())
        stats["question_frac_max"] = max(stats.get("question_frac_max", 0.0), float(frac.max()))
    if promoted is not None:
        # after the call: a position never sees a later position's fetch. Persist at most ``cap``
        # entries per KV head (the memory reserve); past it, the heaviest new entries win and the
        # rest were fetched for this call only.
        cand = flag.any(dim=2) & ~promoted
        cap = state.get("cap")
        if cap is not None:
            room = (cap - promoted.sum(-1)).clamp_min(0)  # [bsz, n_kv]
            n_cand = cand.sum(-1)
            if bool((n_cand > room).any()):
                score = log_w.amax(dim=2).masked_fill(~cand, -float("inf"))
                rank = score.argsort(dim=-1, descending=True).argsort(dim=-1)
                cand &= rank < room[..., None]
        stats["promoted_dropped"] = stats.get("promoted_dropped", 0) + int((cand & torch.isinf(bias)).sum())
        promoted |= cand
    return new_bias.repeat_interleave(groups, dim=1)


@dataclass
class NeedleFetchPress(BasePress):
    """
    Bernoulli sampling plus a 1-bit key sketch that fetches dropped needles at decode (simulation).

    ``compression_ratio`` is the TOTAL budget, sketch included: at 0.875 the kept entries plus the
    sketch of every dropped entry occupy 1/8 of the full KV cache. The wrapped BernoulliPress
    therefore keeps ``k = (1/8 - s) / (1 - s)``, with ``s`` the sketch's bytes per entry against one
    entry's K and V.

    At every forward over the cache (question and each decode step) the sketch estimates each
    context entry's attention weight for the live query; entries the draw left out (or upweighted)
    whose estimate reaches ``theta`` are made exact (see ``needle_decode_bias``). With ``persist`` a
    fetched entry stays resident (exact) for the rest of the answer, and only new entries count as
    fetches; ``promoted_dropped`` counts the extra resident entries, which ``reserve`` (a fraction of
    the full cache) pays for out of the kept budget. ``theta_question`` is the threshold while the
    question is read (one batched transfer per head); ``theta`` applies at decode. With
    ``charge_overhead=False`` the sketch and reserve are not paid from the budget: the draw keeps the
    full ``1 - compression_ratio``, exactly as BernoulliPress at the same ratio (same draw), which
    isolates what fetching adds (an ablation; it uses more memory than the ratio). The full cache stays
    in GPU memory, so "fetching" is unmasking; fetch counts are recorded in ``fetch_stats``.
    Composition: the sampling is BernoulliPress's, unchanged.

    ``sketch="pq"`` replaces the low-bit channel sketch with product quantisation of whole keys
    (``pq_subspaces`` subspaces, 2**``pq_bits`` centroids each, codebook fit on ``pq_fit_sample`` keys past
    the first ``pq_skip``), as PQCache and the skylight runtime do. ``bits``, ``n_pairs`` and ``group`` then
    play no part. Its bytes, codebook included, are charged to the budget like the low-bit sketch's.
    """

    bern: BernoulliPress = field(default_factory=lambda: BernoulliPress(press=KVzipPress()))
    theta: float = 0.01
    theta_question: float | None = None
    persist: bool = False
    reserve: float = 0.0
    reserve_entries: int | None = None
    bits: int = 1
    n_pairs: int | None = None
    # n_pairs per 128 head dims: the same channel fraction on every model (Qwen3.5's 256-dim heads get
    # 2 * n_pairs, so its sketch costs the same share of the cache as on 128-dim heads)
    pairs_per_128_dims: bool = False
    group: int = 128
    head_dim: int = 128
    charge_overhead: bool = True
    sketch: str = "quant"
    pq_subspaces: int = 16
    pq_bits: int = 8
    pq_fit_sample: int | None = 4096
    pq_skip: int = 128
    pq_iters: int = 10
    _total: float = 0.0

    def __post_init__(self):
        assert self.sketch in ("quant", "pq"), f"unknown sketch {self.sketch!r}"
        self._ctx = None  # the context length, once known: the PQ codebook is charged per context
        self.reset_stats()

    def reset_stats(self):
        """Zero the counters in place: the per-layer decode state holds a reference to this dict."""
        if not hasattr(self, "fetch_stats"):
            self.fetch_stats = {}
        for k in ("decode", "question"):
            for m in ("decisions", "fetch_any", "fetched_entries", "ctx_entries"):
                self.fetch_stats[f"{k}_{m}"] = 0
        for k in ("per_layer", "promoted_dropped", "question_heads", "question_heads_any", "ctx_slots",
                  "decode_steps", "decode_steps_any", "_step_any", "question_frac_sum", "question_frac_max"):
            self.fetch_stats.pop(k, None)

    @property
    def pairs(self) -> int | None:
        """Sketched RoPE pairs (channel units) per KV head for this model's head_dim."""
        if self.n_pairs is None or not self.pairs_per_128_dims:
            return self.n_pairs
        return max(1, self.n_pairs * self.head_dim // 128)

    @property
    def sketch_frac(self) -> float:
        return sketch_bytes_per_entry(
            self.head_dim, self.group, self.bits, self.pairs, self.sketch, self.pq_subspaces, self.pq_bits,
            getattr(self, "_ctx", None)) / (self.head_dim * KV_BYTES_PER_CHANNEL)

    @property
    def compression_ratio(self):
        return self._total

    @compression_ratio.setter
    def compression_ratio(self, value):
        self._total = value
        self._set_kept(self.reserve)

    def _set_kept(self, reserve: float):
        total_kept = 1.0 - self._total
        s = self.sketch_frac
        if not self.charge_overhead:
            kept = total_kept  # sketch and reserve ride on top: the same draw as BernoulliPress
        else:
            kept = max((total_kept - s - reserve) / (1.0 - s), 0.0) if self._total > 0 else 1.0
        self.bern.compression_ratio = 1.0 - kept

    def _reserve_for(self, ctx: int) -> float:
        """The persistence reserve as a fraction of the cache: fixed, or ``reserve_entries`` per KV head
        (the persisted set grows with the question, not with the context)."""
        if self.reserve_entries is None:
            return self.reserve
        # entries per head, but never more than ``reserve`` of the cache (when set): on short
        # contexts a fixed count would eat the kept budget (128 entries is 6% of a 2k context)
        cap = self.reserve if self.reserve > 0 else 0.5 * (1.0 - self._total - self.sketch_frac)
        return min(self.reserve_entries / ctx, cap)

    def post_init_from_model(self, model):
        cfg = getattr(model.config, "text_config", model.config)  # VLM wrappers keep these in text_config
        self.head_dim = int(cfg.head_dim if getattr(cfg, "head_dim", None)
                            else cfg.hidden_size // cfg.num_attention_heads)
        rope = getattr(cfg, "rope_parameters", None) or {}
        factor = rope.get("partial_rotary_factor", getattr(cfg, "partial_rotary_factor", 1.0)) or 1.0
        self.rotary_dim = int(self.head_dim * factor)
        self.compression_ratio = self._total  # re-derive the kept fraction with the real head_dim
        self.bern.post_init_from_model(model)

    @contextmanager
    def __call__(self, model: PreTrainedModel) -> Generator:
        self.post_init_from_model(model)
        original = self.bern.compress_post

        def compress_post(m):
            ctx = int(self.bern.press.score_val.shape[-1])
            self._ctx = ctx  # before the budget: the PQ codebook's share depends on it
            self._ctx_reserve = self._reserve_for(ctx)
            self._set_kept(self._ctx_reserve)  # the kept budget for THIS context, reserve paid
            if self.charge_overhead and self._total > 0 and self.bern.compression_ratio >= 1.0:
                raise ValueError(
                    f"the sketch ({self.sketch_frac:.4f} of the cache) and reserve ({self._ctx_reserve:.4f}) "
                    f"leave nothing of the {1 - self._total:.4f} budget"
                )
            original(m)
            self._attach_sketch(m)

        # BernoulliPress hands self.compress_post to KVzip; an instance attribute takes precedence
        self.bern.compress_post = compress_post
        try:
            with self.bern(model):
                yield
        finally:
            del self.bern.compress_post

    def _attach_sketch(self, model: PreTrainedModel):
        cache = self.bern.press._cache
        metadata = get_cache_metadata(cache)
        biases = metadata.pop("attention_bias", {})
        needle = {}
        adapter = get_adapter(model)
        modules = {int(m.layer_idx): m for m in adapter.compressible_modules(model)}
        for layer_idx, bias in biases.items():
            keys, _ = adapter.get_keys_values(cache, modules[int(layer_idx)])
            ctx = bias.shape[-1]
            k_ctx = keys[:, :, :ctx]
            if self.sketch == "pq":
                assert k_ctx.shape[0] == 1, "the PQ sketch supports batch size 1 only"
                channels = None  # the sketch covers every channel
                k_hat = pq_sketch(k_ctx[0], self.pq_subspaces, self.pq_bits, self.pq_iters, seed=int(layer_idx),
                                  fit_sample=self.pq_fit_sample, skip=self.pq_skip)[None]
            else:
                channels = None if self.pairs is None else top_energy_pairs(
                    k_ctx, self.pairs, getattr(self, "rotary_dim", None))
                if channels is not None:
                    k_ctx = k_ctx.gather(-1, channels[:, :, None, :].expand(-1, -1, ctx, -1))
                k_hat = quant_sketch(k_ctx, self.bits, self.group)
            needle[layer_idx] = {
                "bias": bias,
                "k_hat": k_hat,
                "channels": channels,
                "log_theta": float(torch.log(torch.tensor(self.theta))),
                "log_theta_question": float(torch.log(torch.tensor(
                    self.theta if self.theta_question is None else self.theta_question))),
                "stats": self.fetch_stats,
                "layer_idx": int(layer_idx),
                "first_layer": int(min(biases)),
            }
            if self.persist:
                needle[layer_idx]["promoted"] = torch.zeros_like(bias, dtype=torch.bool)
                reserve = getattr(self, "_ctx_reserve", self.reserve)
                if reserve > 0:
                    # reserve is a fraction of the full cache; per KV head that is reserve * ctx entries
                    needle[layer_idx]["cap"] = int(reserve * ctx)
        metadata["needle"] = needle
        self.fetch_stats["ctx_slots"] = self.fetch_stats.get("ctx_slots", 0) + sum(
            int(b.numel()) for b in biases.values())  # (layer, KV head, entry) slots, for the promoted fraction
        logger.debug(f"NeedleFetchPress: sketched {len(needle)} layers, theta {self.theta}, "
                     f"sketch {self.sketch_frac:.4f} of KV")
