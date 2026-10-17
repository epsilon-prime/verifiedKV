# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import math
import zlib
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Generator, Union

import torch
from torch import nn
from transformers import PreTrainedModel

from kvpress.adapters import get_adapter
from kvpress.presses.base_press import BasePress
from kvpress.presses.compactor_press import CompactorPress
from kvpress.presses.fastkvzip_press import FastKVzipScorer
from kvpress.presses.kvzip_press import KVzipPress
from kvpress.utils import compute_n_kept, get_cache_metadata

logger = logging.getLogger(__name__)


def transform_scores_for_sampling(scores: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Map scores to non-negative sampling weights.

    Values already in [0, 1] are left unchanged (KVzip-style masses). Otherwise they are treated as
    logits and softmax-normalised along ``dim`` (Compactor-style z-scores).
    """
    if scores.numel() > 0 and bool((scores >= 0).all()) and bool((scores <= 1).all()):
        return scores
    return torch.softmax(scores, dim=dim)


def _keep_prob(scores: torch.Tensor, c: float) -> torch.Tensor:
    """Keep probability r_i = min(1, c * s_i)."""
    return (c * scores).clamp(max=1.0)


def _solve_c(scores: torch.Tensor, budget: float, iters: int = 100) -> float:
    """
    Smallest c such that sum_i min(1, c * s_i) >= budget.

    The expected kept count is continuous and non-decreasing in c, so bisection finds it. The search runs
    in log c because c spans several orders of magnitude. At c = 1 / min(s) every token with a positive
    score is kept, which gives the upper bracket.
    """
    positive = scores[scores > 0]
    budget = min(float(budget), float(positive.numel()))
    assert budget > 0, "the budget must be positive"
    lo, hi = 1e-12 / positive.max().item(), 1.0 / positive.min().item()
    if _keep_prob(scores, lo).sum().item() >= budget:
        return lo
    for _ in range(iters):
        mid = math.sqrt(lo * hi)
        if _keep_prob(scores, mid).sum().item() >= budget:
            hi = mid
        else:
            lo = mid
    return hi


def _protected_mask(n: int, start: int, end: int, device) -> torch.Tensor:
    """The first ``start`` and last ``end`` positions, clipped as CompactorPress clips them."""
    left = min(start, n)
    right = min(end, max(0, n - left))
    mask = torch.zeros(n, dtype=torch.bool, device=device)
    mask[:left] = True
    if right:
        mask[n - right :] = True
    return mask


def _solve_c_per_head(t: torch.Tensor, k_free: torch.Tensor, iters: int = 100) -> torch.Tensor:
    """
    Vectorised `_solve_c`: for every row h of ``t`` ([H, n], non-negative), the smallest c_h with
    sum_i min(1, c_h * t_hi) >= k_free[h]. ``k_free`` must not exceed the number of positive entries.
    """
    positive = t > 0
    t_min = torch.where(positive, t, torch.full_like(t, torch.finfo(t.dtype).max)).amin(-1)
    lo = (1e-12 / t.amax(-1).clamp_min(1e-300)).log()
    hi = (1.0 / t_min.clamp_min(1e-300)).log()
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        ok = (mid.exp()[:, None] * t).clamp(max=1.0).sum(-1) >= k_free
        hi = torch.where(ok, mid, hi)
        lo = torch.where(ok, lo, mid)
    return hi.exp()


def _per_head_keep_prob(scores: torch.Tensor, compression_ratio: float, start: int, end: int) -> torch.Tensor:
    """
    Keep probabilities [bsz, H, n] for scores that are only comparable within a head (CompactorPress).

    Scores are mapped with ``transform_scores_for_sampling`` along the sequence (softmax when they are
    not already in [0, 1]). Every head then keeps compute_n_kept(n, compression_ratio) pairs in
    expectation, the same count as CompactorPress's top-k, with the protected positions at r = 1.
    """
    bsz, n_heads, n = scores.shape
    s = transform_scores_for_sampling(scores.double(), dim=-1)
    protected = _protected_mask(n, start, end, s.device)
    n_protected = int(protected.sum())
    n_kept = compute_n_kept(n, compression_ratio)
    t = s.masked_fill(protected, 0.0).reshape(bsz * n_heads, n)
    k_free = torch.full((bsz * n_heads,), float(max(n_kept - n_protected, 0)), dtype=torch.float64, device=s.device)
    k_free = torch.minimum(k_free, (t > 0).sum(-1).double())
    c = _solve_c_per_head(t, k_free)
    r = (c[:, None] * t).clamp(max=1.0)
    r = torch.where(k_free[:, None] > 0, r, torch.zeros_like(r))  # the protected positions use the whole budget
    return r.view(bsz, n_heads, n).masked_fill(protected, 1.0)


@dataclass
class BernoulliPress(BasePress):
    """
    Bernoulli KV cache sampling with a Horvitz-Thompson attention correction.

    Instead of keeping the top-k KV pairs, every KV pair i is kept independently with probability
    r_i = min(1, c * s_i), where s_i is its importance score and c is chosen so that the expected number of
    kept pairs matches the compression ratio. Pairs with c * s_i >= 1 are always kept; the rest are
    sampled. Each kept pair then receives an additive attention logit bias of log(1 / r_i) (dropped pairs
    receive -inf). Softmax is a ratio: that bias makes the unnormalised weight of every pair, and therefore
    both the numerator and the denominator of softmax, Horvitz-Thompson unbiased estimates of their full-cache
    counterparts. The ratio itself (the attention weights) is not unbiased.

    The scores come from the wrapped press, which is used unchanged. This press replaces only its
    selection step (top-k) with sampling. Three scorers are supported:

    - KVzipPress: its scores are comparable across layers and heads and already lie in [0, 1], so they are
      used as sampling weights with a single c shared by all of them, exactly like KVzip's global top-k.
      Its n_sink first tokens are always kept.
    - FastKVzipScorer: Fast KVzip's gate scores (trained to predict KVzip's), also in [0, 1] and ranked
      globally by Fast KVzip; handled exactly as KVzipPress's.
    - CompactorPress: its scores are z-scores and are softmax-normalised within each head before sampling.
      Each head gets its own c, keeping the same number of pairs per head as Compactor's top-k (in
      expectation). Its protected first and last tokens are always kept. Compactor's leverage sketch is
      seeded from the context and the layer, so a given (context, seed) pair is reproducible and the
      global random state is left untouched.

    Forced keeps are paid for out of the same budget. Raising any r_i above min(1, c * s_i) and weighting
    by 1 / r_i keeps the numerator and denominator estimates unbiased.

    Limitations:
    - Only the "sdpa" attention implementation is supported. The bias is added to the attention mask
      through kvpress's attention patch, which is bypassed by "eager" attention and cannot be expressed
      with flash attention.
    - Batch size 1 only.
    - As with the other head-wise presses in kvpress, dropped pairs are masked rather than removed, so
      peak memory is not reduced.

    Parameters
    ----------
    press : KVzipPress or CompactorPress
        Press providing the importance scores. Its compression_ratio sets the budget.
    seed : int, default=0
        Seed for the Bernoulli draws. Draws are also keyed on the context tokens (and, for CompactorPress,
        on the layer), so different contexts get independent draws and a given (context, seed) pair is
        reproducible.
    """

    press: Union[KVzipPress, FastKVzipScorer, CompactorPress] = field(default_factory=KVzipPress)
    seed: int = 0

    def __post_init__(self):
        assert isinstance(
            self.press, (KVzipPress, FastKVzipScorer, CompactorPress)
        ), "BernoulliPress requires a KVzipPress, a FastKVzipScorer or a CompactorPress as input"
        if isinstance(self.press, KVzipPress) and self.press.layerwise:
            logger.warning(
                "BernoulliPress ignores KVzipPress.layerwise and still draws from a single global keep-probability "
                "across layers (the same budget as KVzip's default global top-k)."
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

    @staticmethod
    def _attention_modules(model: PreTrainedModel) -> list:
        """The attention modules that hold a KV cache (all layers, except on hybrid stacks)."""
        return list(get_adapter(model).compressible_modules(model))

    @staticmethod
    def _check_sdpa(attn_implementation: str):
        if attn_implementation != "sdpa":
            raise ValueError(
                f"BernoulliPress requires attn_implementation='sdpa', got '{attn_implementation}'. "
                "The log(1/r) bias is applied through the attention mask."
            )

    @staticmethod
    def _generator(device: torch.device, seed: int) -> torch.Generator:
        generator = torch.Generator(device=device if device.type == "cuda" else "cpu")
        generator.manual_seed(seed & 0x7FFFFFFF)
        return generator

    @contextmanager
    def __call__(self, model: PreTrainedModel) -> Generator:
        """
        Run the wrapped press's scoring with this press's selection step in place of its top-k.

        The sampled attention bias is stored in metadata owned by the KV cache. Callers must pass that
        dictionary as ``kvpress_metadata`` on subsequent model calls using the cache.
        """
        language_model = get_adapter(model).language_model(model)
        pre_hook = language_model.register_forward_pre_hook(self._capture_context, with_kwargs=True)
        try:
            if isinstance(self.press, (KVzipPress, FastKVzipScorer)):
                # Instance attribute override: KVzipPress (and FastKVzipScorer) calls self.compress_post(model) once
                # its scores are ready.
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

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """CompactorPress path, called once per layer after prefill: sample and store the attention bias."""
        if self.compression_ratio <= 0:
            return keys, values
        self._check_sdpa(module.config._attn_implementation)
        assert self._context_crc is not None, "the context tokens were not captured"
        assert hidden_states.shape[0] == 1, "BernoulliPress only supports batch size 1"
        layer_idx = int(module.layer_idx)

        # Seed Compactor's random leverage sketch per (context, layer) without touching the global state
        device = hidden_states.device
        with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
            torch.manual_seed((self._context_crc ^ (layer_idx * 104729) ^ 0x5EED) & 0x7FFFFFFF)
            scores = self.press.score(module, hidden_states, keys, values, attentions, kwargs)  # [bsz, H, n]

        r = _per_head_keep_prob(scores, self.compression_ratio, self.press.sink_size_start, self.press.sink_size_end)
        generator = self._generator(device, self._context_crc ^ (self.seed * 7919) ^ (layer_idx * 104729))
        keep = torch.rand(r.shape, generator=generator, dtype=torch.float64, device=device) < r
        bias = torch.where(keep, -torch.log(r.clamp_min(1e-300)), torch.tensor(-float("inf"), device=device))

        # bias: [bsz, n_kv_heads, ctx_len]. Please refer to attention_patch.py for how it is used
        metadata = get_cache_metadata(kwargs["past_key_values"])
        metadata.setdefault("attention_bias", {})[layer_idx] = bias.float()
        module.masked_key_indices = None
        return keys, values  # dropped pairs are masked through the bias, not removed

    def compress_post(self, model: PreTrainedModel):
        """KVzipPress path: sample the kept KV pairs from its global scores and store the attention bias."""
        if self.compression_ratio <= 0:
            return
        self._check_sdpa(model.config._attn_implementation)

        # [n_rows, bsz, n_kv_heads, ctx_len], one row per compressible layer (see KVzipPress._score_row_by_layer)
        scores = self.press.score_val
        assert scores.shape[1] == 1, "BernoulliPress only supports batch size 1"
        n_layers, _, n_kv_heads, ctx_len = scores.shape
        n_sink = self.press.n_sink
        flat = transform_scores_for_sampling(scores.reshape(-1).double())
        budget = (1.0 - self.compression_ratio) * flat.numel()

        c = _solve_c(flat, budget)
        r = _keep_prob(flat, c).view(n_layers, 1, n_kv_heads, ctx_len)

        # KVzip marks sink tokens with a score of 1.0, but ordinary tokens can reach 1.0 too, so for c < 1 a
        # sink would only be kept with probability c. Top-k keeps sinks because they tie at the top. Force
        # r = 1 on sinks, then lower c so the forced keeps are paid for out of the same budget.
        r[..., :n_sink] = 1.0
        if r.sum().item() > budget:
            lo, hi = 0.0, c
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                r_mid = _keep_prob(flat, mid).view(n_layers, 1, n_kv_heads, ctx_len)
                r_mid[..., :n_sink] = 1.0
                if r_mid.sum().item() > budget:
                    hi = mid
                else:
                    lo = mid
            c = hi
            r = _keep_prob(flat, c).view(n_layers, 1, n_kv_heads, ctx_len)
            r[..., :n_sink] = 1.0
        r = r.reshape(-1)

        # Seed from the context tokens so that contexts of equal length do not share a keep pattern.
        device = r.device
        context_crc = zlib.crc32(self.press._context_ids.detach().cpu().numpy().tobytes())
        generator = self._generator(device, context_crc ^ (self.seed * 7919))
        keep = torch.rand(r.shape, generator=generator, dtype=torch.float64, device=device) < r

        keep = keep.view(n_layers, 1, n_kv_heads, ctx_len)
        r = r.view(n_layers, 1, n_kv_heads, ctx_len)
        bias = torch.where(keep, -torch.log(r.clamp_min(1e-300)), torch.tensor(-float("inf"), device=device))

        metadata = get_cache_metadata(self.press._cache)
        attention_bias = metadata.setdefault("attention_bias", {})
        rows = self.press._score_row_by_layer
        for module in self._attention_modules(model):
            row = rows[int(module.layer_idx)]
            # bias[row]: [bsz, n_kv_heads, ctx_len]. Please refer to attention_patch.py for how it is used
            attention_bias[int(module.layer_idx)] = bias[row].float()
            module.masked_key_indices = None

        logger.debug(f"BernoulliPress: c={c:.4g}, kept fraction {keep.double().mean().item():.4f}")
