# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import math
import zlib
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Generator, Optional, Union

import torch
from torch import nn
from transformers import PreTrainedModel

from kvpress.adapters import get_adapter
from kvpress.presses.base_press import BasePress
from kvpress.presses.bernoulli_press import transform_scores_for_sampling
from kvpress.presses.compactor_press import CompactorPress
from kvpress.presses.kvzip_press import KVzipPress
from kvpress.utils import get_cache_metadata

logger = logging.getLogger(__name__)


def normal_quantile(p: float) -> float:
    """Standard normal quantile Φ^{-1}(p)."""
    return float(math.sqrt(2.0) * torch.erfinv(torch.tensor(2.0 * p - 1.0)).item())


def clt_sample_size(d_hat: float, m2_hat: float, epsilon: float, delta: float) -> int:
    """
    Required with-replacement sample size from the CLT relative-error bound

        m >= z_{1-δ/2}^2 / (ε^2 D̂^2) * (M̂_2 - D̂^2).
    """
    assert 0 < epsilon < 1, "epsilon must be in (0, 1)"
    assert 0 < delta < 1, "delta must be in (0, 1)"
    z = normal_quantile(1.0 - delta / 2.0)
    d_hat = max(float(d_hat), 1e-300)
    var = max(float(m2_hat) - d_hat * d_hat, 0.0)
    return int(math.ceil((z * z) / (epsilon * epsilon * d_hat * d_hat) * var))


def forced_keep_mask(scores: torch.Tensor, sink: int, local: int, topk_frac: float) -> torch.Tensor:
    """
    Boolean mask [..., n]: sink prefix, local suffix, and the top ``topk_frac * n`` scores are forced keeps.
    """
    n = scores.shape[-1]
    forced = torch.zeros(*scores.shape, dtype=torch.bool, device=scores.device)
    left = min(max(sink, 0), n)
    right = min(max(local, 0), max(0, n - left))
    forced[..., :left] = True
    if right:
        forced[..., n - right :] = True
    n_topk = min(n, max(0, int(topk_frac * n)))
    if n_topk > 0:
        forced.scatter_(-1, scores.topk(n_topk, dim=-1).indices, True)
    return forced


def residual_sampling_probs(pi: torch.Tensor, forced: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Normalised PPS distribution ``s`` on residual positions (0 on forced / zero-mass positions).

    Returns
    -------
    s : Tensor same shape as ``pi``
        Sampling probabilities (sum to 1 over residual, or all zeros if none).
    residual : BoolTensor
        Residual mask.
    """
    residual = ~forced
    s = torch.zeros_like(pi, dtype=torch.float64)
    if not bool(residual.any()):
        return s, residual
    pi_res = pi.masked_fill(~residual, 0.0).clamp_min(0.0).double()
    mass = pi_res.sum(dim=-1, keepdim=True)
    positive = mass > 0
    s = torch.where(positive, pi_res / mass.clamp_min(1e-300), s)
    return s, residual


def draw_base_sample(
    s: torch.Tensor,
    residual: torch.Tensor,
    base_m: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """
    Draw ``base_m`` absolute indices with replacement from the residual PPS distribution ``s``.

    Returns an empty int64 tensor when there is nothing to sample.
    """
    if base_m <= 0 or not bool(residual.any()) or float(s.detach().sum()) <= 0:
        return torch.empty(0, dtype=torch.long, device=s.device)
    # multinomial over the full length works because forced / empty positions have s = 0
    return torch.multinomial(s.double(), base_m, replacement=True, generator=generator)


def attention_magnitudes_from_query_key(
    query: torch.Tensor,
    key: torch.Tensor,
    ctx_len: int,
) -> torch.Tensor:
    """
    ``a_i = exp(<q, k_i> / √d)`` for context keys, using the last query position.

    query: [B, H_q, q_len, d], key: [B, H_kv, k_len, d] → a: [B, H_kv, ctx_len]
    """
    bsz, n_kv, _, head_dim = key.shape
    n_heads = query.shape[1]
    n_groups = n_heads // n_kv
    q = query[:, :, -1, :].float().view(bsz, n_kv, n_groups, head_dim).mean(dim=2)
    k = key[:, :, :ctx_len, :].float()
    logits = torch.einsum("bhd,bhnd->bhn", q, k) / math.sqrt(head_dim)
    logits = logits - logits.amax(dim=-1, keepdim=True)
    return logits.exp()


def importance_sample_bias_from_base(
    s: torch.Tensor,
    forced: torch.Tensor,
    a: torch.Tensor,
    base_idx: torch.Tensor,
    epsilon: float,
    delta: float,
    generator: torch.Generator,
    max_m: Optional[int] = None,
) -> tuple[torch.Tensor, int, float, float]:
    """
    Using a prefill base sample and decode-time magnitudes ``a``, choose ``m`` via CLT and build the bias.

    Forced positions get bias 0. Sampled residual positions get ``log(I_i / (m * s_i))``. Others get -inf.

    Returns
    -------
    bias, m, d_hat, d_true
        ``d_hat`` is the base-sample IS estimate of the residual denominator
        ``D = Σ_{s_i > 0} a_i``; ``d_true`` is that exact sum.
    """
    n = s.numel()
    device = s.device
    bias = torch.full((n,), -float("inf"), dtype=torch.float64, device=device)
    bias = bias.masked_fill(forced, 0.0)

    a = a.double()
    s = s.double()
    support = s > 0
    d_true = float(a[support].sum().item()) if bool(support.any()) else 0.0

    m0 = int(base_idx.numel())
    if m0 <= 0 or float(s.sum()) <= 0:
        return bias, 0, 0.0, d_true

    inv_s = 1.0 / s[base_idx].clamp_min(1e-300)
    weights = a[base_idx] * inv_s
    d_hat = weights.mean().item()
    m2_hat = (weights**2).mean().item()

    max_m = n if max_m is None else max(0, int(max_m))
    m = clt_sample_size(d_hat, m2_hat, epsilon, delta)
    m = min(max(m, m0), max_m)

    if m > m0:
        extra = torch.multinomial(s, m - m0, replacement=True, generator=generator)
        idx_all = torch.cat([base_idx, extra])
    else:
        idx_all = base_idx
        m = m0

    counts = torch.bincount(idx_all, minlength=n).double()
    kept = (~forced) & (counts > 0)
    bias[kept] = torch.log(counts[kept] / (m * s[kept].clamp_min(1e-300)))
    return bias, m, float(d_hat), d_true


def decode_attention_bias(query: torch.Tensor, key: torch.Tensor, state: dict[str, Any]) -> torch.Tensor:
    """
    Build a per-layer attention bias from stored ``π`` / base sample and the current query.

    ``state`` is the per-layer dict written at prefill (see ``VKvWRPress._store_layer_state``).
    When denominator micro-metrics are enabled, also logs mean ``|D̂ - D| / D`` over KV heads.
    """
    pi = state["pi"]  # [1, H_kv, ctx]
    forced = state["forced"]
    s = state["s"]
    base_idx = state["base_idx"]  # list of [m0_h] per KV head
    ctx_len = pi.shape[-1]
    a = attention_magnitudes_from_query_key(query, key, ctx_len)  # [1, H_kv, ctx]
    device = query.device
    n_kv = pi.shape[1]
    max_m = state["max_m"]
    bias = torch.empty_like(pi, dtype=torch.float64)

    from kvpress.metric_logging import maybe_log_denominator_error, want_denominator_error

    log_denom = want_denominator_error()
    d_hats: list[float] = []
    d_trues: list[float] = []

    for h in range(n_kv):
        generator = torch.Generator(device=device if device.type == "cuda" else "cpu")
        generator.manual_seed(int(state["seed"]) ^ (h * 104_729))
        idx_h = base_idx[h]
        bias[0, h], _, d_hat, d_true = importance_sample_bias_from_base(
            s[0, h],
            forced[0, h],
            a[0, h],
            idx_h,
            state["epsilon"],
            state["delta"],
            generator,
            max_m=max_m,
        )
        if log_denom:
            d_hats.append(d_hat)
            d_trues.append(d_true)

    if log_denom and d_hats:
        from kvpress.metric_logging import relative_denominator_error

        rels = [
            relative_denominator_error(d_hat, d_true)
            for d_hat, d_true in zip(d_hats, d_trues)
            if d_true > 0.0
        ]
        rels = [r for r in rels if math.isfinite(r)]
        d_hat_mean = sum(d_hats) / len(d_hats)
        d_true_mean = sum(d_trues) / len(d_trues)
        maybe_log_denominator_error(
            d_hat_mean,
            d_true_mean,
            layer_idx=int(state.get("layer_idx", -1)),
            q_len=int(query.shape[2]),
            k_len=int(key.shape[2]),
            source="vkvwr_bias",
            error=(sum(rels) / len(rels)) if rels else None,
            n_heads=len(rels) if rels else 0,
        )

    return bias.float()


@dataclass
class VKvWRPress(BasePress):
    """
    Variance-adaptive KV sampling with replacement (vKvWR).

    Prefill (no compression)
        Score with a wrapped ``KVzipPress`` / ``CompactorPress``, map scores to ``π``, force-keep sink /
        local / top-``topk_frac``, draw a base PPS sample on the residual, and store ``(π, forced, s,
        base sample)`` in cache metadata. All key/value pairs are kept.

    Decode
        With the real query, estimate ``D̂`` / ``M̂_2`` from the base sample, choose ``m`` via the CLT
        bound, draw any extra residual samples, and apply an IS attention bias
        ``log(I_i / (m * s_i))`` (forced tokens contribute with weight 1).

    By default ``topk_frac = base_sample_frac = (1 - compression_ratio) / 2.1``. Optional
    ``topk_frac_override`` / ``base_sample_frac_override`` replace those formulas when set.

    Limitations: ``attn_implementation="sdpa"``, batch size 1, pairs are masked rather than removed.
    """

    press: Union[KVzipPress, CompactorPress] = field(default_factory=KVzipPress)
    sink: int = 4
    local: int = 4
    epsilon: float = 0.1
    delta: float = 0.05
    max_sample_frac: float = 1.0
    seed: int = 0
    topk_frac_override: Optional[float] = None
    base_sample_frac_override: Optional[float] = None

    def __post_init__(self):
        assert isinstance(
            self.press, (KVzipPress, CompactorPress)
        ), "VKvWRPress requires a KVzipPress or a CompactorPress as input"
        assert 0 < self.epsilon < 1, "epsilon must be in (0, 1)"
        assert 0 < self.delta < 1, "delta must be in (0, 1)"
        assert 0 < self.max_sample_frac <= 1, "max_sample_frac must be in (0, 1]"
        assert self.sink >= 0 and self.local >= 0
        if self.topk_frac_override is not None:
            assert 0 <= self.topk_frac_override <= 1, "topk_frac_override must be in [0, 1]"
        if self.base_sample_frac_override is not None:
            assert 0 < self.base_sample_frac_override <= 1, "base_sample_frac_override must be in (0, 1]"
        if isinstance(self.press, KVzipPress) and self.press.layerwise:
            logger.warning(
                "VKvWRPress ignores KVzipPress.layerwise and samples independently per layer/head at decode."
            )
        self._context_crc = None

    def post_init_from_model(self, model):
        self.press.post_init_from_model(model)

    @property
    def compression_ratio(self):
        return self.press.compression_ratio

    @compression_ratio.setter
    def compression_ratio(self, value):
        self.press.compression_ratio = value

    @property
    def topk_frac(self) -> float:
        """Fraction of positions forced kept as top-k: override or ``(1 - compression_ratio) / 2.1``."""
        if self.topk_frac_override is not None:
            return float(self.topk_frac_override)
        return (1.0 - self.compression_ratio) / 2.1

    @property
    def base_sample_frac(self) -> float:
        """Pilot sample size as a fraction of length: override or ``(1 - compression_ratio) / 2.1``."""
        if self.base_sample_frac_override is not None:
            return float(self.base_sample_frac_override)
        return (1.0 - self.compression_ratio) / 2.1

    @staticmethod
    def _attention_modules(model: PreTrainedModel) -> list:
        return list(get_adapter(model).compressible_modules(model))

    @staticmethod
    def _check_sdpa(attn_implementation: str):
        if attn_implementation != "sdpa":
            raise ValueError(
                f"VKvWRPress requires attn_implementation='sdpa', got '{attn_implementation}'. "
                "The importance-sampling bias is applied through the attention mask."
            )

    @staticmethod
    def _generator(device: torch.device, seed: int) -> torch.Generator:
        generator = torch.Generator(device=device if device.type == "cuda" else "cpu")
        generator.manual_seed(seed & 0x7FFFFFFF)
        return generator

    @contextmanager
    def __call__(self, model: PreTrainedModel) -> Generator:
        """
        Run the wrapped press's scoring; store ``π`` and a base sample in cache metadata (no KV eviction).

        Callers must pass that metadata as ``kvpress_metadata`` on subsequent decode calls so the
        attention patch can build the query-dependent IS bias.
        """
        language_model = get_adapter(model).language_model(model)
        pre_hook = language_model.register_forward_pre_hook(self._capture_context, with_kwargs=True)
        try:
            if isinstance(self.press, KVzipPress):
                setattr(self.press, "compress_post", self.compress_post)
                try:
                    with self.press(model):
                        yield
                finally:
                    delattr(self.press, "compress_post")
            else:
                self.warn_unsupported_model(model)
                self.post_init_from_model(model)
                with self.hook_scope(model):
                    yield
        finally:
            pre_hook.remove()

    def _capture_context(self, module: nn.Module, args, kwargs):
        input_ids = kwargs.get("input_ids", args[0] if args else None)
        self._context_crc = zlib.crc32(input_ids.detach().cpu().numpy().tobytes())
        cache = kwargs.get("past_key_values")
        if cache is not None:
            metadata = get_cache_metadata(cache, kwargs.get("kvpress_metadata"))
            metadata.pop("attention_bias", None)
            metadata.pop("vkvwr", None)

    def _store_layer_state(
        self,
        metadata: dict,
        layer_idx: int,
        pi: torch.Tensor,
        device: torch.device,
        seed: int,
    ) -> None:
        """``pi`` has shape [1, H_kv, ctx]. Draw a per-head base sample and record decode state."""
        assert pi.shape[0] == 1
        _, n_kv, n = pi.shape
        forced = forced_keep_mask(pi, self.sink, self.local, self.topk_frac)
        s, residual = residual_sampling_probs(pi, forced)
        base_m = int(self.base_sample_frac * n)
        base_idx = []
        for h in range(n_kv):
            generator = self._generator(device, seed ^ (h * 104_729))
            base_idx.append(draw_base_sample(s[0, h], residual[0, h], base_m, generator).detach())

        metadata.setdefault("vkvwr", {})[layer_idx] = {
            "pi": pi.detach(),
            "forced": forced.detach(),
            "s": s.detach(),
            "base_idx": base_idx,  # list of [m0_h] long tensors, one per KV head
            "epsilon": self.epsilon,
            "delta": self.delta,
            "max_m": int(self.max_sample_frac * n),
            "seed": int(seed) & 0x7FFFFFFF,
            "layer_idx": int(layer_idx),
        }

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """CompactorPress path: score, store ``π`` + base sample; keep all KV pairs."""
        self._check_sdpa(module.config._attn_implementation)
        assert self._context_crc is not None, "the context tokens were not captured"
        assert hidden_states.shape[0] == 1, "VKvWRPress only supports batch size 1"
        layer_idx = int(module.layer_idx)
        device = hidden_states.device

        with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
            torch.manual_seed((self._context_crc ^ (layer_idx * 104729) ^ 0x5EED) & 0x7FFFFFFF)
            scores = self.press.score(module, hidden_states, keys, values, attentions, kwargs)

        pi = transform_scores_for_sampling(scores.double(), dim=-1)
        seed = self._context_crc ^ (self.seed * 7919) ^ (layer_idx * 104729)
        metadata = get_cache_metadata(kwargs["past_key_values"])
        self._store_layer_state(metadata, layer_idx, pi, device, seed)
        module.masked_key_indices = None
        return keys, values

    def compress_post(self, model: PreTrainedModel):
        """KVzipPress path: store per-layer ``π`` + base sample from global scores; keep all KV pairs."""
        self._check_sdpa(model.config._attn_implementation)

        scores = self.press.score_val  # [n_rows, bsz, n_kv_heads, ctx_len]
        assert scores.shape[1] == 1, "VKvWRPress only supports batch size 1"
        device = scores.device
        context_crc = zlib.crc32(self.press._context_ids.detach().cpu().numpy().tobytes())
        metadata = get_cache_metadata(self.press._cache)
        rows = self.press._score_row_by_layer

        for module in self._attention_modules(model):
            layer_idx = int(module.layer_idx)
            row = rows[layer_idx]
            pi = transform_scores_for_sampling(scores[row].double(), dim=-1)  # [1, H, ctx]
            seed = context_crc ^ (self.seed * 7919) ^ (layer_idx * 104729)
            self._store_layer_state(metadata, layer_idx, pi, device, seed)
            module.masked_key_indices = None

        logger.debug(f"VKvWRPress: stored base samples for {len(metadata.get('vkvwr', {}))} layers")
