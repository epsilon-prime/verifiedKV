# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import DynamicCache
from transformers.integrations.sdpa_attention import sdpa_attention_forward

import kvpress.attention_patch as attention_patch_module
import kvpress.presses.bernoulli_press as bernoulli_module
from kvpress import BernoulliPress, CompactorPress, ComposedPress, KnormPress, KVzipPress
from kvpress.attention_patch import attention_patch
from kvpress.utils import compute_n_kept, get_cache_metadata
from tests.fixtures import kv_press_unit_test_pipeline, unit_test_model, unit_test_model_output_attention  # noqa: F401

# ---------------------------------------------------------------------------------------------------------------------
# attention_patch: the attention_bias path
# ---------------------------------------------------------------------------------------------------------------------

NUM_HEADS, NUM_KV_HEADS, HEAD_DIM = 4, 2, 8


class FakeAttention(nn.Module):
    """Minimal stand-in for a transformers attention module, as seen by the attention function."""

    def __init__(self, attn_implementation="sdpa"):
        super().__init__()
        self.config = SimpleNamespace(_attn_implementation=attn_implementation)
        self.layer_idx = 0
        self.num_key_value_groups = NUM_HEADS // NUM_KV_HEADS
        self.is_causal = True


def random_qkv(q_len, k_len, seed=0):
    g = torch.Generator().manual_seed(seed)
    query = torch.randn(1, NUM_HEADS, q_len, HEAD_DIM, generator=g)
    key = torch.randn(1, NUM_KV_HEADS, k_len, HEAD_DIM, generator=g)
    value = torch.randn(1, NUM_KV_HEADS, k_len, HEAD_DIM, generator=g)
    return query, key, value


def random_bias(ctx_len, seed=0):
    """log(1/r) on kept positions, -inf on dropped ones; the first position is always kept."""
    g = torch.Generator().manual_seed(seed)
    r = torch.rand(1, NUM_KV_HEADS, ctx_len, generator=g).clamp_min(0.05)
    keep = torch.rand(1, NUM_KV_HEADS, ctx_len, generator=g) < 0.6
    keep[..., 0] = True
    return torch.where(keep, -torch.log(r), torch.tensor(-float("inf")))


def bias_metadata(bias):
    return {"attention_bias": {0: bias}}


def reference_attention(query, key, value, bias, ctx_len):
    """Attention computed by hand: causal softmax(q.k / sqrt(d) + bias) v, with the bias on the context keys."""
    q_len, k_len = query.shape[2], key.shape[2]
    groups = NUM_HEADS // NUM_KV_HEADS
    key, value = key.repeat_interleave(groups, dim=1), value.repeat_interleave(groups, dim=1)
    logits = query @ key.transpose(-1, -2) / math.sqrt(HEAD_DIM)
    logits[..., :ctx_len] += bias.repeat_interleave(groups, dim=1)[:, :, None, :]
    future = torch.arange(k_len)[None, :] > (torch.arange(q_len)[:, None] + k_len - q_len)
    logits = logits.masked_fill(future, -float("inf"))
    return (logits.softmax(-1) @ value).transpose(1, 2)


def causal_masks(q_len, k_len):
    """The three mask conventions transformers may hand to SDPA."""
    allowed = torch.arange(k_len)[None, :] <= (torch.arange(q_len)[:, None] + k_len - q_len)
    additive = torch.zeros(q_len, k_len).masked_fill(~allowed, -float("inf"))
    return {"none": None, "bool": allowed[None, None], "additive": additive[None, None]}


def capture_call():
    """Attention function that records what the patch passes on."""
    calls = []

    def func(module, query, key, value, attention_mask, dropout, **kwargs):
        calls.append({"key": key.clone(), "attention_mask": attention_mask})
        return None, None

    return func, calls


@pytest.mark.parametrize("q_len", [1, 5])
@pytest.mark.parametrize("mask_kind", ["none", "bool", "additive"])
def test_attention_bias_matches_reference(q_len, mask_kind):
    k_len, ctx_len = 12, 9
    query, key, value = random_qkv(q_len, k_len)
    bias = random_bias(ctx_len)
    module = FakeAttention()

    # With mask None and q_len > 1 the patch must build the causal mask itself, since SDPA drops is_causal once a
    # mask is passed. The reference applies causal masking in every case.
    mask = causal_masks(q_len, k_len)[mask_kind]
    out, _ = attention_patch(sdpa_attention_forward)(
        module, query, key, value, mask, 0.0, kvpress_metadata=bias_metadata(bias)
    )
    expected = reference_attention(query, key, value, bias, ctx_len)
    torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("q_len", [1, 5])
@pytest.mark.parametrize("mask_kind", ["none", "bool", "additive"])
def test_zero_attention_bias_is_identity(q_len, mask_kind):
    """A bias of zeros must not change the output. This is the check that catches boolean masks being added to."""
    k_len, ctx_len = 12, 9
    query, key, value = random_qkv(q_len, k_len)
    mask = causal_masks(q_len, k_len)[mask_kind]

    if mask is None and q_len > 1:
        # SDPA's own is_causal mask is top-left aligned, which is wrong when there are fewer queries than keys, so the
        # unpatched call is not a valid reference here (transformers never passes None in that case). The patch builds
        # the bottom-right causal mask itself; compare against true causal attention instead.
        expected = reference_attention(query, key, value, torch.zeros(1, NUM_KV_HEADS, ctx_len), ctx_len)
    else:
        expected, _ = attention_patch(sdpa_attention_forward)(FakeAttention(), query, key, value, mask, 0.0)

    biased = FakeAttention()
    bias = torch.zeros(1, NUM_KV_HEADS, ctx_len)
    out, _ = attention_patch(sdpa_attention_forward)(
        biased, query, key, value, mask, 0.0, kvpress_metadata=bias_metadata(bias)
    )
    torch.testing.assert_close(out, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("q_len", [1, 5, 12])
def test_attention_patch_unchanged_without_bias(q_len):
    """Modules that never set attention_bias must see exactly the same arguments as before the patch existed."""
    k_len = 12
    query, key, value = random_qkv(q_len, k_len)
    for mask in causal_masks(q_len, k_len).values():
        func, calls = capture_call()
        module = FakeAttention()
        attention_patch(func)(module, query, key, value, mask, 0.0)
        assert calls[0]["attention_mask"] is mask
        torch.testing.assert_close(calls[0]["key"], key, atol=0, rtol=0)
        assert not hasattr(module, "attention_bias")


def test_attention_bias_takes_precedence_over_masked_key_indices():
    """BernoulliPress resets masked_key_indices; the patch must not also apply fake keys when a bias is set."""
    q_len, k_len, ctx_len = 1, 12, 9
    query, key, value = random_qkv(q_len, k_len)
    func, calls = capture_call()
    module = FakeAttention()
    bias = random_bias(ctx_len)
    module.masked_key_indices = (torch.tensor([0]), torch.tensor([0]), torch.tensor([3]))
    attention_patch(func)(module, query, key, value, None, 0.0, kvpress_metadata=bias_metadata(bias))
    torch.testing.assert_close(calls[0]["key"], key, atol=0, rtol=0)


def test_stale_attention_bias_raises():
    query, key, value = random_qkv(1, 8)
    module = FakeAttention()
    bias = random_bias(10)
    with pytest.raises(ValueError, match="stale"):
        attention_patch(sdpa_attention_forward)(
            module, query, key, value, None, 0.0, kvpress_metadata=bias_metadata(bias)
        )


@pytest.mark.parametrize("attn_implementation", ["flash_attention_2", "flex_attention"])
def test_attention_bias_requires_sdpa(attn_implementation):
    query, key, value = random_qkv(1, 12)
    func, _ = capture_call()
    module = FakeAttention(attn_implementation)
    bias = random_bias(9)
    with pytest.raises(ValueError, match="sdpa"):
        attention_patch(func)(module, query, key, value, None, 0.0, kvpress_metadata=bias_metadata(bias))


def test_attention_patch_ignores_stale_module_attribute():
    """Bias left on the module from the old API must not leak into attention."""
    query, key, value = random_qkv(1, 12)
    func, calls = capture_call()
    module = FakeAttention()
    module.attention_bias = random_bias(9)
    attention_patch(func)(module, query, key, value, None, 0.0)
    assert calls[0]["attention_mask"] is None


def test_attention_patch_ignores_bias_for_other_layers():
    query, key, value = random_qkv(1, 12)
    func, calls = capture_call()
    module = FakeAttention()
    module.layer_idx = 1
    attention_patch(func)(module, query, key, value, None, 0.0, kvpress_metadata=bias_metadata(random_bias(9)))
    assert calls[0]["attention_mask"] is None


def test_get_cache_metadata_binds_caller_dict():
    cache = DynamicCache()
    provided = {}
    assert get_cache_metadata(cache, provided) is provided
    assert cache.kvpress_metadata is provided
    assert get_cache_metadata(cache) is provided
    other = {"attention_bias": {}}
    assert get_cache_metadata(cache, other) is other
    assert cache.kvpress_metadata is other


# ---------------------------------------------------------------------------------------------------------------------
# Budget solver
# ---------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("keep_fraction", [0.5, 0.25, 0.125, 0.0625])
def test_solve_c_meets_budget(keep_fraction):
    scores = torch.rand(20_000, generator=torch.Generator().manual_seed(0), dtype=torch.float64) ** 4
    budget = keep_fraction * scores.numel()
    c = bernoulli_module._solve_c(scores, budget)
    kept = bernoulli_module._keep_prob(scores, c).sum().item()
    assert abs(kept - budget) / budget < 1e-6
    r = bernoulli_module._keep_prob(scores, c)
    assert r.max() <= 1.0 and r.min() >= 0.0


def test_solve_c_budget_above_positive_count():
    """With more budget than positive scores, every positive score is kept and zeros never are."""
    scores = torch.tensor([0.0, 0.0, 0.5, 0.1, 0.2], dtype=torch.float64)
    c = bernoulli_module._solve_c(scores, 4.5)
    r = bernoulli_module._keep_prob(scores, c)
    torch.testing.assert_close(r, torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0], dtype=torch.float64))


def test_transform_scores_for_sampling_keeps_unit_interval():
    scores = torch.tensor([[0.0, 0.3, 1.0], [0.2, 0.2, 0.5]])
    torch.testing.assert_close(bernoulli_module.transform_scores_for_sampling(scores), scores)


@pytest.mark.parametrize(
    "scores",
    [
        torch.tensor([-0.1, 0.3, 0.5]),
        torch.tensor([0.2, 1.5]),
        torch.randn(2, 8, generator=torch.Generator().manual_seed(0)),
    ],
)
def test_transform_scores_for_sampling_softmax_outside_unit_interval(scores):
    got = bernoulli_module.transform_scores_for_sampling(scores)
    torch.testing.assert_close(got, torch.softmax(scores, dim=-1))


# ---------------------------------------------------------------------------------------------------------------------
# BernoulliPress on a model
# ---------------------------------------------------------------------------------------------------------------------

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


def run_press(model, press, input_ids):
    _, metadata = prefill_with_press(model, press, input_ids)
    return [metadata.get("attention_bias", {}).get(int(module.layer_idx)) for module in press._attention_modules(model)]


def assert_no_module_attention_bias(model):
    for layer in model.model.layers:
        assert getattr(layer.self_attn, "attention_bias", None) is None


def random_input(model, seed=0, length=CTX_LEN):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 1024, (1, length), generator=g).to(model.device)


@pytest.mark.parametrize("compression_ratio", [0.5, 0.75, 0.875])
def test_bernoulli_press_bias(clean_model, compression_ratio):
    press = BernoulliPress(press=KVzipPress(compression_ratio=compression_ratio))
    biases = run_press(clean_model, press, random_input(clean_model))

    n_kv_heads = clean_model.config.num_key_value_heads
    for bias in biases:
        assert bias is not None and bias.shape == (1, n_kv_heads, CTX_LEN)
        kept = torch.isfinite(bias)
        # kept pairs carry log(1/r) >= 0; dropped pairs carry -inf
        assert (bias[kept] >= 0).all()
        assert (bias[~kept] == -float("inf")).all()
        # sinks are always kept, with r = 1
        assert kept[..., : press.press.n_sink].all()
        assert (bias[..., : press.press.n_sink] == 0).all()
    # the layers' masked_key_indices must not also be applied
    assert all(layer.self_attn.masked_key_indices is None for layer in clean_model.model.layers)


@pytest.mark.parametrize("compression_ratio", [0.5, 0.75, 0.875])
def test_bernoulli_press_expected_budget(clean_model, monkeypatch, compression_ratio):
    """The final keep probabilities, sinks included, must sum to the budget."""
    returned = []
    original = bernoulli_module._keep_prob

    def spy(scores, c):
        out = original(scores, c)
        returned.append(out)
        return out

    monkeypatch.setattr(bernoulli_module, "_keep_prob", spy)
    press = BernoulliPress(press=KVzipPress(compression_ratio=compression_ratio))
    run_press(clean_model, press, random_input(clean_model))

    r = returned[-1]  # the sink forcing writes into this tensor in place
    budget = (1 - compression_ratio) * r.numel()
    assert abs(r.sum().item() - budget) / budget < 1e-4
    r = r.view(clean_model.config.num_hidden_layers, 1, clean_model.config.num_key_value_heads, CTX_LEN)
    assert (r[..., : press.press.n_sink] == 1).all()


@pytest.mark.parametrize("compression_ratio", [0.5, 0.75])
def test_bernoulli_press_realised_budget(clean_model, compression_ratio):
    """The realised kept fraction concentrates around the target (4 standard deviations)."""
    press = BernoulliPress(press=KVzipPress(compression_ratio=compression_ratio))
    kept, total = 0, 0
    for seed in range(8):
        for bias in run_press(clean_model, press, random_input(clean_model, seed=seed)):
            kept += torch.isfinite(bias).sum().item()
            total += bias.numel()
    target = 1 - compression_ratio
    assert abs(kept / total - target) < 4 * math.sqrt(target * (1 - target) / total)


def test_bernoulli_press_reproducible(clean_model):
    input_ids = random_input(clean_model)
    first = run_press(clean_model, BernoulliPress(press=KVzipPress(compression_ratio=0.5), seed=1), input_ids)
    again = run_press(clean_model, BernoulliPress(press=KVzipPress(compression_ratio=0.5), seed=1), input_ids)
    other = run_press(clean_model, BernoulliPress(press=KVzipPress(compression_ratio=0.5), seed=2), input_ids)
    for a, b in zip(first, again):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert any(not torch.equal(torch.isfinite(a), torch.isfinite(c)) for a, c in zip(first, other))


def test_bernoulli_press_zero_compression_is_identity(clean_model):
    input_ids = random_input(clean_model)
    expected = clean_model(input_ids).logits
    press = BernoulliPress(press=KVzipPress(compression_ratio=0.0))
    with press(clean_model):
        logits = clean_model(input_ids, past_key_values=DynamicCache()).logits
    torch.testing.assert_close(logits, expected)


def test_bernoulli_press_compression_ratio_delegates():
    inner = KVzipPress(compression_ratio=0.25)
    press = BernoulliPress(press=inner)
    assert press.compression_ratio == 0.25
    press.compression_ratio = 0.5
    assert inner.compression_ratio == 0.5


def test_bernoulli_press_restores_inner_press(clean_model):
    inner = KVzipPress(compression_ratio=0.5)
    run_press(clean_model, BernoulliPress(press=inner), random_input(clean_model))
    assert "compress_post" not in vars(inner)
    # the inner press still works as a plain top-k KVzipPress afterwards
    with inner(clean_model):
        clean_model(random_input(clean_model), past_key_values=DynamicCache())
    assert clean_model.model.layers[0].self_attn.masked_key_indices is not None


@pytest.mark.parametrize("scorer_cls", [KVzipPress, CompactorPress])
def test_bernoulli_press_stores_bias_on_cache_not_modules(clean_model, scorer_cls):
    press = BernoulliPress(press=scorer_cls(compression_ratio=0.5))
    cache, metadata = prefill_with_press(clean_model, press, random_input(clean_model))
    assert_no_module_attention_bias(clean_model)
    assert metadata is cache.kvpress_metadata
    biases = metadata["attention_bias"]
    modules = press._attention_modules(clean_model)
    assert set(biases) == {int(module.layer_idx) for module in modules}
    for module in modules:
        bias = biases[int(module.layer_idx)]
        assert bias.shape == (1, clean_model.config.num_key_value_heads, CTX_LEN)
        assert torch.isfinite(bias).any()


@pytest.mark.parametrize("scorer_cls", [KVzipPress, CompactorPress])
def test_bernoulli_press_cache_metadata_is_isolated(clean_model, scorer_cls):
    input_ids = random_input(clean_model)
    cache_a, meta_a = prefill_with_press(
        clean_model, BernoulliPress(press=scorer_cls(compression_ratio=0.5), seed=1), input_ids
    )
    cache_b, meta_b = prefill_with_press(
        clean_model, BernoulliPress(press=scorer_cls(compression_ratio=0.5), seed=2), input_ids
    )
    assert meta_a is not meta_b
    assert cache_a.kvpress_metadata is meta_a
    assert cache_b.kvpress_metadata is meta_b
    layer_idx = int(clean_model.model.layers[0].self_attn.layer_idx)
    snapshot_b = meta_b["attention_bias"][layer_idx].clone()
    meta_a["attention_bias"][layer_idx].zero_()
    torch.testing.assert_close(meta_b["attention_bias"][layer_idx], snapshot_b, atol=0, rtol=0)


@pytest.mark.parametrize("scorer_cls", [KVzipPress, CompactorPress])
def test_bernoulli_press_decode_applies_bias_only_with_metadata(clean_model, monkeypatch, scorer_cls):
    press = BernoulliPress(press=scorer_cls(compression_ratio=0.5))
    cache, metadata = prefill_with_press(clean_model, press, random_input(clean_model))
    position_ids = torch.tensor([[CTX_LEN]], device=clean_model.device)
    token = random_input(clean_model, seed=1, length=1)
    calls = []
    original = attention_patch_module.add_attention_bias

    def spy(bias, query, key, attention_mask):
        calls.append(bias)
        return original(bias, query, key, attention_mask)

    monkeypatch.setattr(attention_patch_module, "add_attention_bias", spy)
    clean_model(token, past_key_values=cache, position_ids=position_ids)
    assert calls == []
    clean_model(token, past_key_values=cache, position_ids=position_ids, kvpress_metadata=metadata)
    n_layers = clean_model.config.num_hidden_layers
    assert len(calls) == n_layers
    for module, bias in zip(press._attention_modules(clean_model), calls):
        torch.testing.assert_close(bias, metadata["attention_bias"][int(module.layer_idx)], atol=0, rtol=0)


@pytest.mark.parametrize("scorer_cls", [KVzipPress, CompactorPress])
def test_bernoulli_press_prefill_drops_stale_cache_bias(clean_model, monkeypatch, scorer_cls):
    """A leftover bias on the cache metadata must not be applied during the next prefill."""
    cache = DynamicCache()
    metadata = get_cache_metadata(cache)
    n_kv = clean_model.config.num_key_value_heads
    stale = torch.full((1, n_kv, CTX_LEN), 7.0, device=clean_model.device)
    metadata["attention_bias"] = {int(layer.self_attn.layer_idx): stale.clone() for layer in clean_model.model.layers}
    applied = []
    original = attention_patch_module.add_attention_bias

    def spy(bias, query, key, attention_mask):
        applied.append(bias)
        return original(bias, query, key, attention_mask)

    monkeypatch.setattr(attention_patch_module, "add_attention_bias", spy)
    press = BernoulliPress(press=scorer_cls(compression_ratio=0.5))
    with press(clean_model):
        clean_model(random_input(clean_model), past_key_values=cache, kvpress_metadata=metadata)
    assert applied == []
    written = metadata["attention_bias"][int(clean_model.model.layers[0].self_attn.layer_idx)]
    assert not torch.equal(written, stale)


def test_pipeline_stores_bias_on_provided_cache(kv_press_unit_test_pipeline):  # noqa: F811
    cache = DynamicCache()
    press = BernoulliPress(press=KVzipPress(compression_ratio=0.5))
    kv_press_unit_test_pipeline(
        "This is a test article. It was written on 2022-01-01. " * 20,
        question="When was this article written?",
        press=press,
        cache=cache,
        max_new_tokens=2,
    )
    metadata = get_cache_metadata(cache)
    model = kv_press_unit_test_pipeline.model
    assert_no_module_attention_bias(model)
    assert set(metadata["attention_bias"]) == {int(module.layer_idx) for module in press._attention_modules(model)}


def test_bernoulli_press_requires_sdpa(unit_test_model_output_attention):  # noqa: F811
    model = unit_test_model_output_attention
    inner = KVzipPress(compression_ratio=0.5)
    with pytest.raises(ValueError, match="sdpa"):
        run_press(model, BernoulliPress(press=inner), random_input(model))
    assert "compress_post" not in vars(inner)


def test_bernoulli_press_requires_supported_scorer():
    with pytest.raises(AssertionError):
        BernoulliPress(press=KnormPress(compression_ratio=0.5))


def test_bernoulli_press_warns_on_layerwise_kvzip(caplog):
    with caplog.at_level("WARNING"):
        press = BernoulliPress(press=KVzipPress(compression_ratio=0.5, layerwise=True))
    assert any("layerwise" in record.message for record in caplog.records)
    assert press.press.layerwise is True


@pytest.mark.parametrize("scorer_cls", [KVzipPress, CompactorPress])
def test_composed_press_rejects_bernoulli_press(scorer_cls):
    with pytest.raises(AssertionError):
        ComposedPress([BernoulliPress(press=scorer_cls(compression_ratio=0.5))])


@pytest.mark.parametrize("scorer_cls", [KVzipPress, CompactorPress])
def test_bernoulli_press_bias_live_during_generation(
    kv_press_unit_test_pipeline, monkeypatch, scorer_cls  # noqa: F811
):
    """The bias must be applied while answering (question prefill and decoding) and never during prefill."""
    calls = []
    original = attention_patch_module.add_attention_bias

    def spy(bias, query, key, attention_mask):
        calls.append((query.shape[2], key.shape[2]))
        return original(bias, query, key, attention_mask)

    monkeypatch.setattr(attention_patch_module, "add_attention_bias", spy)
    press = BernoulliPress(press=scorer_cls(compression_ratio=0.5))
    context = "This is a test article. It was written on 2022-01-01. " * 20
    max_new_tokens = 5
    kv_press_unit_test_pipeline(
        context, question="When was this article written?", press=press, max_new_tokens=max_new_tokens
    )

    model = kv_press_unit_test_pipeline.model
    n_layers = model.config.num_hidden_layers
    assert all(q_len < k_len for q_len, k_len in calls)
    # one question prefill plus at least one decoding step, in every layer
    assert len(calls) >= 2 * n_layers


# ---------------------------------------------------------------------------------------------------------------------
# BernoulliPress on CompactorPress scores: per-head budget
# ---------------------------------------------------------------------------------------------------------------------

START, END = 8, 4  # CompactorPress's default protected positions


@pytest.mark.parametrize("n", [16, 256, 1024])
@pytest.mark.parametrize("compression_ratio", [0.5, 0.75, 0.875, 0.9375])
def test_per_head_keep_prob_meets_budget(n, compression_ratio):
    """Every head keeps compute_n_kept(n, ratio) pairs in expectation, protected positions included."""
    g = torch.Generator().manual_seed(n)
    scores = torch.randn(1, 3, n, generator=g)  # signed, like Compactor's z-scores
    r = bernoulli_module._per_head_keep_prob(scores, compression_ratio, START, END)
    protected = bernoulli_module._protected_mask(n, START, END, "cpu")
    n_kept, n_protected = compute_n_kept(n, compression_ratio), int(protected.sum())
    assert (r[..., protected] == 1).all()
    assert ((r >= 0) & (r <= 1)).all()
    if n_kept <= n_protected:
        # the protected positions use the whole budget
        assert (r.sum(-1) == n_protected).all()
        return
    # softmax maps every free position to a positive weight, so the full per-head budget is used
    torch.testing.assert_close(r.sum(-1), torch.full((1, 3), float(n_kept), dtype=torch.float64), atol=1e-6, rtol=0)


def test_per_head_keep_prob_is_monotone_in_score():
    """Within a head, a higher score never gets a lower keep probability (the ranking is kept)."""
    scores = torch.randn(1, 2, 64, generator=torch.Generator().manual_seed(0))
    r = bernoulli_module._per_head_keep_prob(scores, 0.75, 0, 0)
    order = scores.argsort(-1)
    assert (r.gather(-1, order).diff(dim=-1) >= -1e-12).all()


@pytest.mark.parametrize("compression_ratio", [0.5, 0.75, 0.875])
def test_bernoulli_compactor_bias(clean_model, compression_ratio):
    press = BernoulliPress(press=CompactorPress(compression_ratio=compression_ratio))
    biases = run_press(clean_model, press, random_input(clean_model))
    protected = bernoulli_module._protected_mask(CTX_LEN, START, END, clean_model.device)
    for bias in biases:
        assert bias is not None and bias.shape == (1, clean_model.config.num_key_value_heads, CTX_LEN)
        kept = torch.isfinite(bias)
        assert (bias[kept] >= 0).all()
        assert (bias[~kept] == -float("inf")).all()
        assert kept[..., protected].all()
        assert (bias[..., protected] == 0).all()
    assert all(layer.self_attn.masked_key_indices is None for layer in clean_model.model.layers)


def test_bernoulli_compactor_realised_budget_per_head(clean_model):
    """Per head, the realised kept count concentrates around compute_n_kept (4 standard deviations)."""
    compression_ratio = 0.75
    press = BernoulliPress(press=CompactorPress(compression_ratio=compression_ratio))
    kept, runs = 0, 0
    for seed in range(4):
        for bias in run_press(clean_model, press, random_input(clean_model, seed=seed)):
            kept += torch.isfinite(bias).sum(-1).double()
            runs += 1
    per_head = kept / runs
    target = compute_n_kept(CTX_LEN, compression_ratio)
    # the kept count of one head is a sum of CTX_LEN Bernoulli draws, whose variance is at most CTX_LEN / 4
    tol = 4 * math.sqrt(CTX_LEN / 4 / runs)
    assert ((per_head - target).abs() < tol).all()


def test_bernoulli_compactor_reproducible_and_rng_neutral(clean_model):
    input_ids = random_input(clean_model)
    state = torch.get_rng_state()
    first = run_press(clean_model, BernoulliPress(press=CompactorPress(compression_ratio=0.5), seed=1), input_ids)
    assert torch.equal(torch.get_rng_state(), state), "the seeded sketch must not advance the global generator"
    again = run_press(clean_model, BernoulliPress(press=CompactorPress(compression_ratio=0.5), seed=1), input_ids)
    other = run_press(clean_model, BernoulliPress(press=CompactorPress(compression_ratio=0.5), seed=2), input_ids)
    for a, b in zip(first, again):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert any(not torch.equal(torch.isfinite(a), torch.isfinite(c)) for a, c in zip(first, other))


def test_bernoulli_compactor_zero_compression_is_identity(clean_model):
    input_ids = random_input(clean_model)
    expected = clean_model(input_ids).logits
    with BernoulliPress(press=CompactorPress(compression_ratio=0.0))(clean_model):
        logits = clean_model(input_ids, past_key_values=DynamicCache()).logits
    torch.testing.assert_close(logits, expected)


def test_bernoulli_compactor_masking_matches_pruning(clean_model, monkeypatch):
    """Keeping exactly Compactor's top-k through the bias must give the same answer logits as Compactor's own
    pruning. blending=0 removes the random leverage term, so both presses see identical scores."""
    compression_ratio = 0.5

    def top_k_indicator(scores, ratio, start, end):
        idx = scores.topk(compute_n_kept(scores.shape[-1], ratio), dim=-1).indices
        return torch.zeros_like(scores, dtype=torch.float64).scatter_(-1, idx, 1.0)

    input_ids = random_input(clean_model)
    question = random_input(clean_model, seed=1, length=8)
    position_ids = torch.arange(CTX_LEN, CTX_LEN + 8, device=clean_model.device)[None]

    def answer_logits(press):
        cache = DynamicCache()
        metadata = get_cache_metadata(cache)
        with press(clean_model):
            clean_model.model(input_ids=input_ids, past_key_values=cache, kvpress_metadata=metadata)
        return clean_model(
            input_ids=question,
            past_key_values=cache,
            position_ids=position_ids,
            kvpress_metadata=metadata,
        ).logits

    pruned = answer_logits(CompactorPress(compression_ratio=compression_ratio, blending=0.0))
    monkeypatch.setattr(bernoulli_module, "_per_head_keep_prob", top_k_indicator)
    masked = answer_logits(BernoulliPress(press=CompactorPress(compression_ratio=compression_ratio, blending=0.0)))
    torch.testing.assert_close(masked, pruned, atol=1e-4, rtol=1e-4)


def test_bernoulli_compactor_requires_sdpa(unit_test_model_output_attention):  # noqa: F811
    model = unit_test_model_output_attention
    with pytest.raises(ValueError, match="sdpa"):
        run_press(model, BernoulliPress(press=CompactorPress(compression_ratio=0.5)), random_input(model))
    for layer in model.model.layers:
        assert len(layer.self_attn._forward_hooks) == 0
