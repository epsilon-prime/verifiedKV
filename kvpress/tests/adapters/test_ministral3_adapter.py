# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from transformers.models.ministral3 import modeling_ministral3
from transformers.models.ministral3.configuration_ministral3 import Ministral3Config
from transformers.models.ministral3.modeling_ministral3 import Ministral3ForCausalLM

from kvpress.adapters import Ministral3Adapter, get_adapter
from kvpress.utils import apply_rope, get_prerope_query_states
from tests.fixtures import get_device

ORIGINAL_MAX = 8  # llama-4 scaling kicks in past this many positions


def tiny_ministral3_config(**overrides):
    kwargs = dict(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=256,
        rope_parameters={
            "rope_type": "yarn",
            "rope_theta": 10000.0,
            "factor": 4.0,
            "original_max_position_embeddings": ORIGINAL_MAX,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
            "llama_4_scaling_beta": 0.5,
        },
    )
    kwargs.update(overrides)
    return Ministral3Config(**kwargs)


@pytest.fixture(scope="module")
def tiny_ministral3_model():
    torch.manual_seed(0)
    config = tiny_ministral3_config()
    config._attn_implementation = "eager"
    return Ministral3ForCausalLM(config).eval().to(get_device())


def test_adapter_is_registered_for_ministral3(tiny_ministral3_model):
    assert isinstance(get_adapter(tiny_ministral3_model), Ministral3Adapter)


def test_recomputed_queries_match_what_the_model_attends_with(tiny_ministral3_model, monkeypatch):
    """KVzip rebuilds queries from hidden states (pre-RoPE projection, RoPE, then the adapter's
    ``scale_queries``). Those must equal the queries the attention kernel receives, including the
    position-dependent llama-4 factor, which is > 1 past ``original_max_position_embeddings``."""
    model = tiny_ministral3_model
    adapter = get_adapter(model)
    module = adapter.compressible_modules(model)[0]
    captured = {}

    real_eager = modeling_ministral3.eager_attention_forward

    def capturing_eager(mod, query, *args, **kwargs):
        if mod is module:
            captured["query"] = query.detach().clone()
        return real_eager(mod, query, *args, **kwargs)

    monkeypatch.setattr(modeling_ministral3, "eager_attention_forward", capturing_eager)

    def hook(mod, args, kwargs, output):
        captured["hidden_states"] = kwargs["hidden_states"].detach().clone()
        captured["kwargs"] = kwargs

    handle = module.register_forward_hook(hook, with_kwargs=True)
    try:
        with torch.no_grad():
            model(input_ids=torch.randint(0, 128, (1, 40), device=model.device))
    finally:
        handle.remove()

    kwargs = captured["kwargs"]
    with torch.no_grad():
        queries = get_prerope_query_states(module, captured["hidden_states"])
        cos, sin = kwargs["position_embeddings"]
        unscaled = apply_rope(module, queries, cos, sin)
        scaled = adapter.scale_queries(module, unscaled, kwargs)

    torch.testing.assert_close(scaled, captured["query"], rtol=1e-5, atol=1e-5)
    # the factor is real here: positions >= ORIGINAL_MAX are scaled, so the plain RoPE query is wrong there
    assert not torch.allclose(unscaled[..., ORIGINAL_MAX:, :], captured["query"][..., ORIGINAL_MAX:, :])
    torch.testing.assert_close(unscaled[..., :ORIGINAL_MAX, :], captured["query"][..., :ORIGINAL_MAX, :])


def test_scale_is_identity_without_llama4_beta():
    torch.manual_seed(0)
    rope = dict(tiny_ministral3_config().rope_parameters)
    rope.pop("llama_4_scaling_beta")
    model = Ministral3ForCausalLM(tiny_ministral3_config(rope_parameters=rope)).eval()
    module = get_adapter(model).compressible_modules(model)[0]
    q = torch.randn(1, 4, 20, 16)
    out = get_adapter(model).scale_queries(module, q, {"cache_position": torch.arange(20)})
    assert torch.equal(out, q)
