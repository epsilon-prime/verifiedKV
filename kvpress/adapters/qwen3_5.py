# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adapter for the Qwen3.5 hybrid stack (Gated DeltaNet + periodic full attention).

Three things differ from a dense Llama-style stack and motivate a dedicated adapter:

1. Only a minority of layers hold a KV cache. The rest are ``linear_attention``
   (Gated DeltaNet) layers whose state is a fixed-size recurrent state, so there is
   nothing to compress there.
2. ``Qwen3_5DynamicCache`` is not a ``transformers.Cache``; it stores plain
   ``key_cache`` / ``value_cache`` lists with ``None`` entries for linear layers.
3. Queries come out of a *gated* ``q_proj`` (which projects to ``2 * head_dim`` per
   head), and RoPE is interleaved mRoPE applied to only part of the head dimension.
4. transformers' ``Qwen3_5GatedDeltaNet.forward`` reuses the cached conv and recurrent state only
   for single-token steps; a multi-token forward over a populated cache restarts the recurrence
   from zero and then overwrites the cached state. kvpress needs multi-token continuation
   (questions, KVzip's repeat passes), so importing this module installs a forward that continues
   from the cache on that path and is the stock code everywhere else.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5DynamicCache,
    Qwen3_5GatedDeltaNet,
    apply_mask_to_padding_states,
    rotate_half,
)

from kvpress.adapters.base import CacheLike, ModelAdapter, register_adapter


_STOCK_GATED_DELTANET_FORWARD = Qwen3_5GatedDeltaNet.forward


def _gated_deltanet_forward(self, hidden_states, cache_params=None, cache_position=None, attention_mask=None):
    """``Qwen3_5GatedDeltaNet.forward`` (transformers 5.2) with exact multi-token continuation.

    Identical to the stock forward for prefill (empty cache) and single-token decode. On a
    multi-token forward over a populated cache the stock code convolves without the cached inputs
    and runs the chunked delta rule from ``initial_state=None``; here the last ``kernel - 1`` cached
    conv inputs are prepended to the convolution and the cached recurrent state is the initial state,
    which is what a token-by-token feed computes.
    """
    hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
    batch_size, seq_len, _ = hidden_states.shape

    use_precomputed_states = (
        cache_params is not None and cache_params.has_previous_state and seq_len == 1 and cache_position is not None
    )
    conv_state = recurrent_state = None
    if cache_params is not None:
        conv_state = cache_params.conv_states[self.layer_idx]
        recurrent_state = cache_params.recurrent_states[self.layer_idx]
    continuing = not use_precomputed_states and conv_state is not None and recurrent_state is not None

    mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)
    z = self.in_proj_z(hidden_states).reshape(batch_size, seq_len, -1, self.head_v_dim)
    b = self.in_proj_b(hidden_states)
    a = self.in_proj_a(hidden_states)

    if use_precomputed_states:
        mixed_qkv = self.causal_conv1d_update(
            mixed_qkv, conv_state, self.conv1d.weight.squeeze(1), self.conv1d.bias, self.activation
        )
    else:
        k = self.conv_kernel_size
        # the conv window of the first new token reaches k - 1 inputs back into the cache
        prefix = conv_state[..., -(k - 1) :].to(mixed_qkv.dtype) if continuing else mixed_qkv[..., :0]
        conv_in = torch.cat([prefix, mixed_qkv], dim=-1)
        if cache_params is not None:
            cache_params.conv_states[self.layer_idx] = F.pad(conv_in, (k - conv_in.shape[-1], 0))
        n_prefix = prefix.shape[-1]
        if self.causal_conv1d_fn is not None:
            mixed_qkv = self.causal_conv1d_fn(
                x=conv_in, weight=self.conv1d.weight.squeeze(1), bias=self.conv1d.bias,
                activation=self.activation, seq_idx=None,
            )[..., n_prefix:]
        else:
            mixed_qkv = F.silu(self.conv1d(conv_in)[:, :, n_prefix : n_prefix + seq_len])

    mixed_qkv = mixed_qkv.transpose(1, 2)
    query, key, value = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
    query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
    key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
    value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

    beta = b.sigmoid()
    g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
    if self.num_v_heads // self.num_k_heads > 1:
        query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
        key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

    rule = self.recurrent_gated_delta_rule if use_precomputed_states else self.chunk_gated_delta_rule
    core_attn_out, last_recurrent_state = rule(
        query, key, value, g=g, beta=beta,
        initial_state=recurrent_state if (use_precomputed_states or continuing) else None,
        output_final_state=cache_params is not None,
        use_qk_l2norm_in_kernel=True,
    )
    if cache_params is not None:
        cache_params.recurrent_states[self.layer_idx] = last_recurrent_state

    core_attn_out = self.norm(core_attn_out.reshape(-1, self.head_v_dim), z.reshape(-1, self.head_v_dim))
    return self.out_proj(core_attn_out.reshape(batch_size, seq_len, -1))


Qwen3_5GatedDeltaNet.forward = _gated_deltanet_forward


@register_adapter("qwen3_5", "qwen3_5_text")
class Qwen3_5Adapter(ModelAdapter):
    def language_model(self, model: PreTrainedModel) -> nn.Module:
        if hasattr(model, "model") and hasattr(model.model, "language_model"):
            return model.model.language_model
        return model.model

    def compressible_modules(self, model: PreTrainedModel) -> list[nn.Module]:
        """Only ``full_attention`` layers own a KV cache; linear-attention layers are skipped."""
        return [
            layer.self_attn
            for layer in self.language_model(model).layers
            if getattr(layer, "layer_type", None) == "full_attention" and hasattr(layer, "self_attn")
        ]

    def prepare_module(self, model: PreTrainedModel, module: nn.Module) -> None:
        rotary_emb = getattr(self.language_model(model), "rotary_emb", None)
        if rotary_emb is not None:
            module.rotary_emb = rotary_emb

    def get_keys_values(self, cache: CacheLike, module: nn.Module) -> tuple[torch.Tensor, torch.Tensor]:
        return cache.key_cache[module.layer_idx], cache.value_cache[module.layer_idx]

    def set_keys_values(
        self, cache: CacheLike, module: nn.Module, keys: torch.Tensor, values: torch.Tensor
    ) -> None:
        cache.key_cache[module.layer_idx] = keys
        cache.value_cache[module.layer_idx] = values

    def make_cache(self, model: PreTrainedModel) -> CacheLike:
        config = model.config
        return Qwen3_5DynamicCache(getattr(config, "text_config", config))

    def supports_multi_token_continuation(self) -> bool:
        """True: this module installs a Gated DeltaNet forward that continues exactly from the cache
        (``_gated_deltanet_forward``), so questions can be fed in one forward pass."""
        return True

    def rewind_cache(self, cache: CacheLike, seq_lengths: list[int]) -> None:
        """Truncate the attention cache back to ``seq_lengths``.

        Linear-attention layers hold no per-token state to truncate; their recurrent and conv states
        are put back with ``restore_state`` from a ``snapshot_state`` taken at the same point.
        """
        for layer_idx, sequence_length in enumerate(seq_lengths):
            if cache.key_cache[layer_idx] is None:
                continue
            cache.key_cache[layer_idx] = cache.key_cache[layer_idx][:, :, :sequence_length]
            cache.value_cache[layer_idx] = cache.value_cache[layer_idx][:, :, :sequence_length]

    def snapshot_state(self, cache: CacheLike) -> dict:
        """Copies of every linear-attention layer's conv and recurrent state (about 1 MB per layer)."""
        return {
            i: (cache.conv_states[i].clone(), cache.recurrent_states[i].clone())
            for i in range(len(cache.conv_states))
            if cache.conv_states[i] is not None and cache.recurrent_states[i] is not None
        }

    def restore_state(self, cache: CacheLike, snapshot: dict) -> None:
        for i, (conv, recurrent) in snapshot.items():
            cache.conv_states[i] = conv.clone()
            cache.recurrent_states[i] = recurrent.clone()

    def prerope_queries(self, module: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
        """Split the gated ``q_proj`` output and drop the gate, keeping the query half."""
        input_shape = hidden_states.shape[:-1]
        query_states, _gate = torch.chunk(
            module.q_proj(hidden_states).view(*input_shape, -1, module.head_dim * 2), 2, dim=-1
        )
        return module.q_norm(query_states.view(*input_shape, -1, module.head_dim)).transpose(1, 2)

    def prerope_keys(self, module: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
        input_shape = hidden_states.shape[:-1]
        key_states = module.k_proj(hidden_states).view(*input_shape, -1, module.head_dim)
        return module.k_norm(key_states).transpose(1, 2)

    def apply_rope(
        self, module: nn.Module, states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        """Partial rotary embedding: only the leading ``cos.shape[-1]`` dims are rotated."""
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
        rotary_dim = cos.shape[-1]
        rotated, passthrough = states[..., :rotary_dim], states[..., rotary_dim:]
        rotated = (rotated * cos) + (rotate_half(rotated) * sin)
        return torch.cat([rotated, passthrough], dim=-1)
