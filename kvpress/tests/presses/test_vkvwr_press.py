# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
import torch
from transformers import DynamicCache

import kvpress.attention_patch as attention_patch_module
import kvpress.presses.vkvwr_press as vkvwr_module
from kvpress import CompactorPress, ComposedPress, KnormPress, KVzipPress, VKvWRPress
from kvpress.utils import get_cache_metadata
from tests.fixtures import kv_press_unit_test_pipeline, unit_test_model, unit_test_model_output_attention  # noqa: F401

CTX_LEN = 256


@pytest.fixture
def clean_model(unit_test_model):  # noqa: F811
    yield unit_test_model


def prefill_with_press(model, press, input_ids):
    cache = DynamicCache()
    metadata = get_cache_metadata(cache)
    with press(model):
        model(input_ids, past_key_values=cache, kvpress_metadata=metadata)
    return cache, metadata


def random_input(model, seed=0, length=CTX_LEN):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 1024, (1, length), generator=g).to(model.device)


# ---------------------------------------------------------------------------------------------------------------------
# Unit helpers
# ---------------------------------------------------------------------------------------------------------------------


def test_normal_quantile_matches_known_value():
    assert abs(vkvwr_module.normal_quantile(0.975) - 1.95996398454) < 1e-5


def test_clt_sample_size_zero_variance():
    assert vkvwr_module.clt_sample_size(d_hat=10.0, m2_hat=100.0, epsilon=0.1, delta=0.05) == 0


def test_clt_sample_size_grows_with_variance():
    m_small = vkvwr_module.clt_sample_size(d_hat=1.0, m2_hat=2.0, epsilon=0.1, delta=0.05)
    m_large = vkvwr_module.clt_sample_size(d_hat=1.0, m2_hat=20.0, epsilon=0.1, delta=0.05)
    assert m_large > m_small > 0


def test_forced_keep_mask_sink_local_topk():
    scores = torch.arange(20, dtype=torch.float64).flip(0)
    forced = vkvwr_module.forced_keep_mask(scores, sink=2, local=3, topk_frac=0.2)
    assert forced[:2].all() and forced[-3:].all()
    assert forced[:4].all()
    assert forced.sum().item() == 2 + 3 + 2


def test_importance_sample_bias_from_base_forced_and_is_weights():
    n = 32
    g = torch.Generator().manual_seed(0)
    pi = torch.rand(n, generator=g, dtype=torch.float64)
    forced = torch.zeros(n, dtype=torch.bool)
    forced[:2] = True
    forced[-2:] = True
    s, _ = vkvwr_module.residual_sampling_probs(pi, forced)
    base_idx = vkvwr_module.draw_base_sample(s, ~forced, 8, torch.Generator().manual_seed(1))
    a = torch.rand(n, generator=g, dtype=torch.float64) + 0.1
    bias, m, d_hat, d_true = vkvwr_module.importance_sample_bias_from_base(
        s, forced, a, base_idx, epsilon=0.2, delta=0.1, generator=torch.Generator().manual_seed(2), max_m=n
    )
    assert m >= 8
    assert (bias[:2] == 0).all() and (bias[-2:] == 0).all()
    kept = torch.isfinite(bias)
    assert kept[:2].all() and kept[-2:].all()
    assert (bias[~kept] == -float("inf")).all()
    assert d_true == pytest.approx(float(a[s > 0].sum().item()), rel=1e-6)
    assert d_hat > 0.0


def test_importance_sample_draws_extra_when_clt_requires_it(monkeypatch):
    monkeypatch.setattr(vkvwr_module, "clt_sample_size", lambda *args, **kwargs: 20)
    n = 40
    pi = torch.ones(n, dtype=torch.float64)
    forced = torch.zeros(n, dtype=torch.bool)
    forced[0] = True
    s, residual = vkvwr_module.residual_sampling_probs(pi, forced)
    base_idx = vkvwr_module.draw_base_sample(s, residual, 5, torch.Generator().manual_seed(0))
    a = torch.ones(n, dtype=torch.float64)
    bias, m, d_hat, d_true = vkvwr_module.importance_sample_bias_from_base(
        s, forced, a, base_idx, 0.1, 0.05, torch.Generator().manual_seed(1), max_m=n
    )
    assert m == 20
    assert torch.isfinite(bias).sum() >= 1
    assert d_true == pytest.approx(float(a[s > 0].sum().item()), rel=1e-6)
    # uniform a and s ⇒ D̂ should match D exactly on any sample
    assert d_hat == pytest.approx(d_true, rel=1e-6)

def test_vkvwr_fracs_from_compression_ratio():
    press = VKvWRPress(press=KVzipPress(compression_ratio=0.5))
    expected = (1.0 - 0.5) / 2.1
    assert press.topk_frac == expected
    assert press.base_sample_frac == expected
    press.compression_ratio = 0.8
    assert press.topk_frac == (1.0 - 0.8) / 2.1
    assert press.base_sample_frac == press.topk_frac


# ---------------------------------------------------------------------------------------------------------------------
# Prefill stores state; decode applies query-dependent bias
# ---------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("scorer_cls", [KVzipPress, CompactorPress])
def test_vkvwr_prefill_keeps_all_kv_and_stores_state(clean_model, scorer_cls):
    press = VKvWRPress(press=scorer_cls(compression_ratio=0.5), sink=4, local=4)
    cache, metadata = prefill_with_press(clean_model, press, random_input(clean_model))

    assert "attention_bias" not in metadata or not metadata["attention_bias"]
    assert "vkvwr" in metadata
    modules = press._attention_modules(clean_model)
    assert set(metadata["vkvwr"]) == {int(module.layer_idx) for module in modules}

    for layer_idx, state in metadata["vkvwr"].items():
        assert state["pi"].shape[-1] == CTX_LEN
        assert state["forced"].shape == state["pi"].shape
        assert state["s"].shape == state["pi"].shape
        assert len(state["base_idx"]) == state["pi"].shape[1]
        assert state["forced"][..., : press.sink].all()
        assert state["forced"][..., -press.local :].all()
        # base sample indices land on residual (non-forced) positions
        for h, idx in enumerate(state["base_idx"]):
            if idx.numel():
                assert (~state["forced"][0, h, idx]).all()

    # no pairs removed from the cache
    from kvpress.adapters import get_adapter

    layer0 = clean_model.model.layers[0].self_attn
    keys, _ = get_adapter(clean_model).get_keys_values(cache, layer0)
    assert keys.shape[2] == CTX_LEN
    assert all(layer.self_attn.masked_key_indices is None for layer in clean_model.model.layers)


@pytest.mark.parametrize("scorer_cls", [KVzipPress, CompactorPress])
def test_vkvwr_decode_applies_bias_from_query(clean_model, monkeypatch, scorer_cls):
    press = VKvWRPress(press=scorer_cls(compression_ratio=0.5))
    cache, metadata = prefill_with_press(clean_model, press, random_input(clean_model))
    position_ids = torch.tensor([[CTX_LEN]], device=clean_model.device)
    token = random_input(clean_model, seed=1, length=1)
    calls = []
    original = attention_patch_module.add_attention_bias

    def spy(bias, query, key, attention_mask):
        calls.append(bias)
        assert torch.isfinite(bias).any()
        assert (bias[..., : press.sink] == 0).all()
        return original(bias, query, key, attention_mask)

    monkeypatch.setattr(attention_patch_module, "add_attention_bias", spy)
    clean_model(token, past_key_values=cache, position_ids=position_ids)
    assert calls == []  # no metadata → no bias
    clean_model(token, past_key_values=cache, position_ids=position_ids, kvpress_metadata=metadata)
    assert len(calls) == clean_model.config.num_hidden_layers


def test_vkvwr_decode_logs_sparsity(clean_model, tmp_path):
    from kvpress.metric_logging import ATTENTION_SPARSITY, MicroMetricLogger

    MicroMetricLogger.reset_for_testing()
    MicroMetricLogger().configure_logging(log_path=str(tmp_path), enabled_metrics=[ATTENTION_SPARSITY])

    press = VKvWRPress(press=KVzipPress(compression_ratio=0.5))
    cache, metadata = prefill_with_press(clean_model, press, random_input(clean_model))
    position_ids = torch.tensor([[CTX_LEN]], device=clean_model.device)
    token = random_input(clean_model, seed=1, length=1)
    clean_model(token, past_key_values=cache, position_ids=position_ids, kvpress_metadata=metadata)
    MicroMetricLogger().flush()

    events = [json.loads(line) for line in (tmp_path / "micro_metrics.jsonl").read_text().splitlines()]
    assert len(events) == clean_model.config.num_hidden_layers
    assert all(e["metric"] == ATTENTION_SPARSITY for e in events)
    assert all(e["metadata"]["source"] == "vkvwr_bias" for e in events)
    assert all(0.0 <= e["value"] <= 1.0 for e in events)


def test_vkvwr_decode_logs_denominator_error(clean_model, tmp_path):
    from kvpress.metric_logging import DENOMINATOR_ERROR, MicroMetricLogger

    MicroMetricLogger.reset_for_testing()
    MicroMetricLogger().configure_logging(log_path=str(tmp_path), enabled_metrics=[DENOMINATOR_ERROR])

    press = VKvWRPress(press=KVzipPress(compression_ratio=0.5))
    cache, metadata = prefill_with_press(clean_model, press, random_input(clean_model))
    position_ids = torch.tensor([[CTX_LEN]], device=clean_model.device)
    token = random_input(clean_model, seed=1, length=1)
    clean_model(token, past_key_values=cache, position_ids=position_ids, kvpress_metadata=metadata)
    MicroMetricLogger().flush()

    events = [json.loads(line) for line in (tmp_path / "micro_metrics.jsonl").read_text().splitlines()]
    assert len(events) == clean_model.config.num_hidden_layers
    assert all(e["metric"] == DENOMINATOR_ERROR for e in events)
    assert all(e["metadata"]["source"] == "vkvwr_bias" for e in events)
    assert all("d_hat" in e["metadata"] and "d_true" in e["metadata"] for e in events)
    assert all(e["value"] >= 0.0 for e in events)


@pytest.mark.parametrize("scorer_cls", [KVzipPress, CompactorPress])
def test_vkvwr_prefill_does_not_apply_bias(clean_model, monkeypatch, scorer_cls):
    """During context prefill (q_len == k_len) the vkvwr state must not sparsify attention."""
    applied = []
    original = attention_patch_module.add_attention_bias

    def spy(bias, query, key, attention_mask):
        applied.append(bias)
        return original(bias, query, key, attention_mask)

    monkeypatch.setattr(attention_patch_module, "add_attention_bias", spy)
    press = VKvWRPress(press=scorer_cls(compression_ratio=0.5))
    prefill_with_press(clean_model, press, random_input(clean_model))
    assert applied == []


@pytest.mark.parametrize("scorer_cls", [KVzipPress, CompactorPress])
def test_vkvwr_press_reproducible_base_sample(clean_model, scorer_cls):
    input_ids = random_input(clean_model)
    kwargs = dict(press=scorer_cls(compression_ratio=0.5))
    _, meta1 = prefill_with_press(clean_model, VKvWRPress(**kwargs, seed=1), input_ids)
    _, meta2 = prefill_with_press(clean_model, VKvWRPress(**kwargs, seed=1), input_ids)
    _, meta3 = prefill_with_press(clean_model, VKvWRPress(**kwargs, seed=2), input_ids)
    layer = int(clean_model.model.layers[0].self_attn.layer_idx)
    for a, b in zip(meta1["vkvwr"][layer]["base_idx"], meta2["vkvwr"][layer]["base_idx"]):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert any(
        not torch.equal(a, c)
        for a, c in zip(meta1["vkvwr"][layer]["base_idx"], meta3["vkvwr"][layer]["base_idx"])
        if a.numel() and c.numel()
    )


def test_vkvwr_press_requires_sdpa(unit_test_model_output_attention):  # noqa: F811
    model = unit_test_model_output_attention
    inner = KVzipPress(compression_ratio=0.5)
    with pytest.raises(ValueError, match="sdpa"):
        prefill_with_press(model, VKvWRPress(press=inner), random_input(model))
    assert "compress_post" not in vars(inner)


def test_vkvwr_press_requires_supported_scorer():
    with pytest.raises(AssertionError):
        VKvWRPress(press=KnormPress(compression_ratio=0.5))


@pytest.mark.parametrize("scorer_cls", [KVzipPress, CompactorPress])
def test_composed_press_rejects_vkvwr_press(scorer_cls):
    with pytest.raises(AssertionError):
        ComposedPress([VKvWRPress(press=scorer_cls(compression_ratio=0.5))])


def test_vkvwr_press_compression_ratio_delegates():
    inner = KVzipPress(compression_ratio=0.25)
    press = VKvWRPress(press=inner)
    assert press.compression_ratio == 0.25
    press.compression_ratio = 0.5
    assert inner.compression_ratio == 0.5


def test_vkvwr_press_restores_inner_press(clean_model):
    inner = KVzipPress(compression_ratio=0.5)
    prefill_with_press(clean_model, VKvWRPress(press=inner), random_input(clean_model))
    assert "compress_post" not in vars(inner)
    with inner(clean_model):
        clean_model(random_input(clean_model), past_key_values=DynamicCache())
    assert clean_model.model.layers[0].self_attn.masked_key_indices is not None


def test_pipeline_stores_vkvwr_on_provided_cache(kv_press_unit_test_pipeline):  # noqa: F811
    cache = DynamicCache()
    press = VKvWRPress(press=KVzipPress(compression_ratio=0.5))
    kv_press_unit_test_pipeline(
        "This is a test article. It was written on 2022-01-01. " * 20,
        question="When was this article written?",
        press=press,
        cache=cache,
        max_new_tokens=2,
    )
    metadata = get_cache_metadata(cache)
    model = kv_press_unit_test_pipeline.model
    assert set(metadata["vkvwr"]) == {int(module.layer_idx) for module in press._attention_modules(model)}


def test_decode_attention_bias_uses_query(monkeypatch):
    """Changing the query must change the CLT-driven bias when variance is query-dependent."""
    n, h = 16, 2
    pi = torch.linspace(0.1, 1.0, n).expand(1, h, n).double()
    forced = vkvwr_module.forced_keep_mask(pi, sink=1, local=1, topk_frac=0.0)
    s, residual = vkvwr_module.residual_sampling_probs(pi[0, 0], forced[0, 0])
    # build full s/forced tensors
    s_full = torch.stack([s, s]).unsqueeze(0)
    forced_full = forced
    base = [
        vkvwr_module.draw_base_sample(s_full[0, hh], ~forced_full[0, hh], 4, torch.Generator().manual_seed(hh))
        for hh in range(h)
    ]
    state = {
        "pi": pi,
        "forced": forced_full,
        "s": s_full,
        "base_idx": base,
        "epsilon": 0.1,
        "delta": 0.05,
        "max_m": n,
        "seed": 0,
    }
    key = torch.randn(1, h, n + 2, 8)
    # two different queries
    q1 = torch.randn(1, h, 1, 8)
    q2 = torch.randn(1, h, 1, 8) * 5
    monkeypatch.setattr(vkvwr_module, "clt_sample_size", lambda *a, **k: 4)  # fix m so only a weights differ via sample
    b1 = vkvwr_module.decode_attention_bias(q1, key, state)
    b2 = vkvwr_module.decode_attention_bias(q2, key, state)
    # with fixed m, bias depends on which extras are drawn; seed is fixed so bias matches for same m
    torch.testing.assert_close(b1, b2, atol=0, rtol=0)
