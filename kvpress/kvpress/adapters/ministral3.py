# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adapter for Ministral 3 (``ministral3`` text model, also inside the ``mistral3`` VLM wrapper).

The stack is Llama-like except for one step: after RoPE, ``Ministral3Attention`` multiplies the
queries by a position-dependent factor ``1 + beta * log(1 + floor(pos / original_max))`` ("llama 4
scaling", 1.22 at 128k positions with beta 0.1 and original_max 16384). Presses that recompute queries
from hidden states must apply it too, or every score beyond ``original_max`` comes from a flatter
softmax than the one the model uses.
"""

from __future__ import annotations

import torch
from torch import nn
from transformers.models.ministral3.modeling_ministral3 import _get_llama_4_attn_scale

from kvpress.adapters.base import register_adapter
from kvpress.adapters.llama_like import LlamaLikeAdapter


@register_adapter("ministral3", "mistral3")
class Ministral3Adapter(LlamaLikeAdapter):
    def scale_queries(self, module: nn.Module, queries: torch.Tensor, kwargs: dict) -> torch.Tensor:
        rope = getattr(module.config, "rope_parameters", None) or {}
        beta = rope.get("llama_4_scaling_beta")
        cache_position = kwargs.get("cache_position")
        if beta is None or cache_position is None:
            return queries
        scale = _get_llama_4_attn_scale(cache_position, beta, rope.get("original_max_position_embeddings"))
        return queries * scale.to(queries.dtype)
