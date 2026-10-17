# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DynamicCache, Qwen3_5ForCausalLM, apply_rotary_pos_emb

from kvpress import BernoulliPress, KnormPress, KVzipPress, SnapKVPress
from kvpress.adapters import Qwen3_5Adapter, get_adapter, get_adapter_from_module, has_adapter
from kvpress.utils import get_cache_metadata
from tests.fixtures import get_device

FULL_ATTENTION_LAYERS = [1, 3]


@pytest.fixture(scope="module")
def tiny_qwen3_5_model():
    """A randomly initialised Qwen3.5 stack: two linear-attention and two full-attention layers."""
    config = Qwen3_5TextConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        full_attention_interval=2,
        max_position_embeddings=512,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "mrope_section": [3, 3, 2],
            "mrope_interleaved": True,
            "partial_rotary_factor": 0.25,
        },
    )
    torch.manual_seed(0)
    model = Qwen3_5ForCausalLM(config).eval()
    return model.to(get_device())


def test_adapter_is_registered_for_qwen3_5(tiny_qwen3_5_model):
    assert has_adapter(tiny_qwen3_5_model)
    assert isinstance(get_adapter(tiny_qwen3_5_model), Qwen3_5Adapter)


def test_only_full_attention_layers_are_compressible(tiny_qwen3_5_model):
    """Linear-attention layers hold a recurrent state, not a KV cache, so they must be skipped."""
    adapter = get_adapter(tiny_qwen3_5_model)
    modules = adapter.compressible_modules(tiny_qwen3_5_model)
    assert [m.layer_idx for m in modules] == FULL_ATTENTION_LAYERS
    assert isinstance(get_adapter_from_module(modules[0]), Qwen3_5Adapter)


def test_hooks_only_land_on_full_attention_layers(tiny_qwen3_5_model):
    press = KnormPress(compression_ratio=0.5)
    with press(tiny_qwen3_5_model):
        for idx, layer in enumerate(tiny_qwen3_5_model.model.layers):
            if idx in FULL_ATTENTION_LAYERS:
                assert len(layer.self_attn._forward_hooks) == 1
            else:
                assert not hasattr(layer, "self_attn")
    for idx in FULL_ATTENTION_LAYERS:
        assert len(tiny_qwen3_5_model.model.layers[idx].self_attn._forward_hooks) == 0


def test_make_cache_returns_hybrid_cache(tiny_qwen3_5_model):
    cache = get_adapter(tiny_qwen3_5_model).make_cache(tiny_qwen3_5_model)
    assert isinstance(cache, Qwen3_5DynamicCache)
    assert len(cache) == tiny_qwen3_5_model.config.num_hidden_layers


@pytest.mark.parametrize("press_cls", [KnormPress, SnapKVPress])
def test_press_compresses_only_attention_layers(tiny_qwen3_5_model, press_cls):
    adapter = get_adapter(tiny_qwen3_5_model)
    seq_len = 128
    input_ids = torch.randint(0, 128, (1, seq_len), device=tiny_qwen3_5_model.device)

    cache = adapter.make_cache(tiny_qwen3_5_model)
    with press_cls(compression_ratio=0.5)(tiny_qwen3_5_model), torch.no_grad():
        tiny_qwen3_5_model.model(input_ids=input_ids, past_key_values=cache)

    kept = seq_len // 2
    assert cache.get_seq_length() == kept
    for layer_idx in range(tiny_qwen3_5_model.config.num_hidden_layers):
        if layer_idx in FULL_ATTENTION_LAYERS:
            assert cache.key_cache[layer_idx].shape[2] == kept
            assert cache.value_cache[layer_idx].shape[2] == kept
        else:
            assert cache.key_cache[layer_idx] is None


def test_prerope_states_match_the_model(tiny_qwen3_5_model):
    """The adapter must reproduce the gated q_proj split and the q/k norms exactly."""
    adapter = get_adapter(tiny_qwen3_5_model)
    module = adapter.compressible_modules(tiny_qwen3_5_model)[0]
    hidden_states = torch.randn(1, 8, tiny_qwen3_5_model.config.hidden_size, device=tiny_qwen3_5_model.device)

    with torch.no_grad():
        queries = adapter.prerope_queries(module, hidden_states)
        keys = adapter.prerope_keys(module, hidden_states)

        # Recompute the way Qwen3_5Attention.forward does.
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, module.head_dim)
        expected_q, _gate = torch.chunk(
            module.q_proj(hidden_states).view(*input_shape, -1, module.head_dim * 2), 2, dim=-1
        )
        expected_q = module.q_norm(expected_q.view(hidden_shape)).transpose(1, 2)
        expected_k = module.k_norm(module.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)

    assert queries.shape == (1, tiny_qwen3_5_model.config.num_attention_heads, 8, module.head_dim)
    assert keys.shape == (1, tiny_qwen3_5_model.config.num_key_value_heads, 8, module.head_dim)
    torch.testing.assert_close(queries, expected_q)
    torch.testing.assert_close(keys, expected_k)


def test_apply_rope_matches_model_partial_rotary(tiny_qwen3_5_model):
    """Qwen3.5 rotates only part of the head dim, so a Llama-style full rotation would be wrong."""
    adapter = get_adapter(tiny_qwen3_5_model)
    module = adapter.compressible_modules(tiny_qwen3_5_model)[0]
    bsz, seq_len = 1, 8
    shape = (bsz, tiny_qwen3_5_model.config.num_attention_heads, seq_len, module.head_dim)
    queries = torch.randn(shape, device=tiny_qwen3_5_model.device)

    position_ids = torch.arange(seq_len, device=tiny_qwen3_5_model.device).view(1, 1, -1).expand(3, bsz, -1)
    with torch.no_grad():
        cos, sin = tiny_qwen3_5_model.model.rotary_emb(queries, position_ids)
        expected, _ = apply_rotary_pos_emb(queries, queries, cos, sin)
        actual = adapter.apply_rope(module, queries, cos, sin)

    assert cos.shape[-1] < module.head_dim, "expected a partial rotary embedding"
    torch.testing.assert_close(actual, expected)


def test_rewind_cache_restores_prefill_lengths(tiny_qwen3_5_model):
    adapter = get_adapter(tiny_qwen3_5_model)
    input_ids = torch.randint(0, 128, (1, 32), device=tiny_qwen3_5_model.device)
    cache = adapter.make_cache(tiny_qwen3_5_model)
    with torch.no_grad():
        tiny_qwen3_5_model.model(input_ids=input_ids, past_key_values=cache)
    prefill_lengths = [cache.get_seq_length(i) for i in range(len(cache))]

    with torch.no_grad():
        tiny_qwen3_5_model.model(
            input_ids=torch.randint(0, 128, (1, 4), device=tiny_qwen3_5_model.device),
            past_key_values=cache,
        )
    assert cache.get_seq_length() == 36

    adapter.rewind_cache(cache, prefill_lengths)
    assert cache.get_seq_length() == 32
    for layer_idx in FULL_ATTENTION_LAYERS:
        assert cache.key_cache[layer_idx].shape[2] == 32


def test_bernoulli_bias_rows_follow_compressible_layers(tiny_qwen3_5_model, monkeypatch):
    """On a hybrid stack KVzip's score rows index the compressible layers only (layers 1 and 3 here), so
    BernoulliPress must write row i's bias to the i-th compressible module, not to model layer i.

    Row 0 (layer 1) scores 1.0 everywhere and row 1 (layer 3) scores ~0, with a 50% budget: layer 1 keeps
    almost everything with a near-zero bias, layer 3 keeps only its sink tokens. A row/layer mix-up swaps
    the two patterns (or fails on a missing row)."""
    model = tiny_qwen3_5_model
    monkeypatch.setattr(model.config, "_attn_implementation", "sdpa")
    modules = get_adapter(model).compressible_modules(model)
    n_ctx, n_kv = 32, model.config.num_key_value_heads

    press = BernoulliPress(press=KVzipPress(compression_ratio=0.5), seed=0)
    high, low = torch.ones(1, n_kv, n_ctx), torch.full((1, n_kv, n_ctx), 1e-9)
    press.press.score_val = torch.stack([high, low]).to(model.device)
    press.press._score_row_by_layer = {int(m.layer_idx): row for row, m in enumerate(modules)}
    press.press._context_ids = torch.arange(n_ctx)[None]
    press.press._cache = get_adapter(model).make_cache(model)
    try:
        press.compress_post(model)
        biases = get_cache_metadata(press.press._cache)["attention_bias"]
        full, linear_like = (biases[int(module.layer_idx)] for module in modules)
        assert full.shape == (1, n_kv, n_ctx)
        assert linear_like.shape == (1, n_kv, n_ctx)
        assert torch.isfinite(full).float().mean() > 0.6
        kept_low = torch.isfinite(linear_like)
        assert kept_low[..., : press.press.n_sink].all()
        assert not kept_low[..., press.press.n_sink :].any()
        for idx, layer in enumerate(model.model.layers):
            if idx not in FULL_ATTENTION_LAYERS:
                assert not hasattr(layer, "self_attn")
    finally:
        for m in modules:
            m.masked_key_indices = None


# --- multi-token continuation, state snapshots, KVzip and multi-question reuse (final_eval P1-P3) ---

QWEN3_5_TOKENIZER = "Qwen/Qwen3.5-9B"


def _tiny_config(vocab_size=128):
    return Qwen3_5TextConfig(
        vocab_size=vocab_size,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        full_attention_interval=2,
        max_position_embeddings=4096,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "mrope_section": [3, 3, 2],
            "mrope_interleaved": True,
            "partial_rotary_factor": 0.25,
        },
    )


def _long_memory(model):
    """Slow the DeltaNet decay so the recurrent state carries the prefix (random init forgets in a step
    or two, which would hide a dropped prefix): decay rate exp(A_log) * softplus(a + dt_bias) ~ 1e-3."""
    with torch.no_grad():
        for layer in model.model.layers:
            if hasattr(layer, "linear_attn"):
                layer.linear_attn.A_log.fill_(-5.0)
                layer.linear_attn.dt_bias.fill_(-2.0)
    return model


@pytest.fixture(scope="module")
def long_memory_qwen3_5_model():
    torch.manual_seed(1)
    return _long_memory(Qwen3_5ForCausalLM(_tiny_config()).eval()).to(get_device())


def _states_and_logits(model, chunks):
    """Feed ``chunks`` (a list of id tensors) into one cache; return the recurrent states and last logits."""
    cache = get_adapter(model).make_cache(model)
    with torch.no_grad():
        for ids in chunks:
            logits = model(input_ids=ids, past_key_values=cache).logits
    states = [s.clone() for s in cache.recurrent_states if s is not None]
    return states, logits[0, -1], cache


def test_multi_token_continuation_matches_a_single_pass(long_memory_qwen3_5_model):
    """Prefill 32 tokens, then continue with 8 in ONE forward: must equal a single 40-token pass and a
    token-by-token feed. This is what questions and KVzip's repeat passes do."""
    model = long_memory_qwen3_5_model
    assert get_adapter(model).supports_multi_token_continuation()
    ids = torch.randint(0, 128, (1, 40), device=model.device)
    ref_states, ref_logits, _ = _states_and_logits(model, [ids])
    multi_states, multi_logits, _ = _states_and_logits(model, [ids[:, :32], ids[:, 32:]])
    step_states, step_logits, _ = _states_and_logits(model, [ids[:, :32]] + [ids[:, i : i + 1] for i in range(32, 40)])
    for ref, multi, step in zip(ref_states, multi_states, step_states):
        torch.testing.assert_close(multi, ref, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(step, ref, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(multi_logits, ref_logits, rtol=1e-4, atol=1e-4)
    # token by token runs fla's fused recurrent kernel (not the chunked one) on GPU: kernel rounding ~1e-4
    torch.testing.assert_close(step_logits, ref_logits, rtol=1e-3, atol=1e-3)


def test_stock_forward_fails_the_same_check(long_memory_qwen3_5_model, monkeypatch):
    """The check above is sensitive: transformers' own forward restarts the recurrence on a multi-token
    continuation, and that visibly changes the state and the logits on this long-memory model."""
    from kvpress.adapters.qwen3_5 import _STOCK_GATED_DELTANET_FORWARD
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet

    model = long_memory_qwen3_5_model
    ids = torch.randint(0, 128, (1, 40), device=model.device)
    ref_states, ref_logits, _ = _states_and_logits(model, [ids])
    ours_states, ours_logits, _ = _states_and_logits(model, [ids[:, :32], ids[:, 32:]])
    monkeypatch.setattr(Qwen3_5GatedDeltaNet, "forward", _STOCK_GATED_DELTANET_FORWARD)
    stock_states, stock_logits, _ = _states_and_logits(model, [ids[:, :32], ids[:, 32:]])

    def err(a, b):
        return max((x - y).abs().max().item() for x, y in zip(a, b))

    ours, stock = err(ours_states, ref_states), err(stock_states, ref_states)
    assert stock > 1e-3 and stock > 50 * max(ours, 1e-6), (ours, stock)
    assert (stock_logits - ref_logits).abs().max() > 10 * (ours_logits - ref_logits).abs().max()


def test_snapshot_restore_round_trip(long_memory_qwen3_5_model):
    model = long_memory_qwen3_5_model
    adapter = get_adapter(model)
    ids = torch.randint(0, 128, (1, 24), device=model.device)
    _, _, cache = _states_and_logits(model, [ids])
    snap = adapter.snapshot_state(cache)
    assert sorted(snap) == [i for i in range(4) if i not in FULL_ATTENTION_LAYERS]
    before = {i: (c.clone(), r.clone()) for i, (c, r) in snap.items()}
    with torch.no_grad():
        model(input_ids=torch.randint(0, 128, (1, 5), device=model.device), past_key_values=cache)
    assert not torch.equal(cache.recurrent_states[0], before[0][1])
    adapter.restore_state(cache, snap)
    adapter.restore_state(cache, snap)  # the snapshot stays reusable
    for i, (conv, rec) in before.items():
        assert torch.equal(cache.conv_states[i], conv) and torch.equal(cache.recurrent_states[i], rec)


@pytest.fixture(scope="module")
def tokenized_qwen3_5():
    """A tiny long-memory Qwen3.5 with the real Qwen3.5 vocabulary, so KVzip and the pipeline can use the
    real tokenizer and chat template (needs the tokenizer files in the HF cache)."""
    from transformers import AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(QWEN3_5_TOKENIZER)
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"Qwen3.5 tokenizer not available: {exc}")
    torch.manual_seed(2)
    model = _long_memory(Qwen3_5ForCausalLM(_tiny_config(vocab_size=len(tokenizer))).eval()).to(get_device())
    model.config.name_or_path = QWEN3_5_TOKENIZER  # KVzip loads its tokenizer from here
    return model, tokenizer


def _context_ids(tokenizer, model, n_words=120):
    torch.manual_seed(3)
    words = " ".join(f"item{int(i)}" for i in torch.randint(0, 1000, (n_words,)))
    return tokenizer(words, return_tensors="pt").input_ids.to(model.device)


def test_kvzip_scoring_leaves_the_prefill_state(tokenized_qwen3_5):
    """KVzip re-feeds the context in repeat chunks through the same cache. Afterwards every DeltaNet layer
    must hold exactly the post-prefill state; before the fix it held the last repeat chunk's."""
    model, tokenizer = tokenized_qwen3_5
    adapter = get_adapter(model)
    ids = _context_ids(tokenizer, model)

    plain = adapter.make_cache(model)
    with torch.no_grad():
        model.model(input_ids=ids, past_key_values=plain)
    compressed = adapter.make_cache(model)
    press = KVzipPress(compression_ratio=0.5)
    modules = adapter.compressible_modules(model)
    try:
        with press(model), torch.no_grad():
            model.model(input_ids=ids, past_key_values=compressed)
        # kvpress KVzip masks (head-wise) instead of evicting: the repeat passes ran and pruned something
        assert all(getattr(m, "masked_key_indices", None) is not None for m in modules)
        assert compressed.get_seq_length() == ids.shape[1]  # repeat-chunk KV pairs were cut back off
        for i in range(len(plain.recurrent_states)):
            if plain.recurrent_states[i] is not None:
                # restored from a copy of this very state: equal up to float noise, relative to its ~1e-3 size
                torch.testing.assert_close(compressed.recurrent_states[i], plain.recurrent_states[i], rtol=1e-5, atol=1e-9)
                torch.testing.assert_close(compressed.conv_states[i], plain.conv_states[i], rtol=1e-5, atol=1e-9)
    finally:
        for m in modules:
            m.masked_key_indices = None


def test_kvzip_without_the_restore_would_leave_a_wrong_state(tokenized_qwen3_5, monkeypatch):
    """Sensitivity of the test above: with restore_state a no-op the cache ends on the last repeat chunk."""
    model, tokenizer = tokenized_qwen3_5
    adapter = get_adapter(model)
    ids = _context_ids(tokenizer, model)
    plain = adapter.make_cache(model)
    with torch.no_grad():
        model.model(input_ids=ids, past_key_values=plain)
    monkeypatch.setattr(type(adapter), "restore_state", lambda self, cache, snap: None)
    compressed = adapter.make_cache(model)
    modules = adapter.compressible_modules(model)
    try:
        with KVzipPress(compression_ratio=0.5)(model), torch.no_grad():
            model.model(input_ids=ids, past_key_values=compressed)
        i = next(i for i, s in enumerate(plain.recurrent_states) if s is not None)
        a, b = compressed.recurrent_states[i], plain.recurrent_states[i]
        assert ((a - b).norm() / b.norm()).item() > 1e-2  # states are ~1e-3 in size: compare relatively
    finally:
        for m in modules:
            m.masked_key_indices = None


def test_pipeline_questions_do_not_see_each_other(tokenized_qwen3_5):
    """One prefilled cache, several questions (LOFT asks 100): question B after A must answer exactly as
    B alone, which needs the recurrent state restored between questions, not only the KV cache cut."""
    from kvpress.pipeline import KVPressTextGenerationPipeline

    model, tokenizer = tokenized_qwen3_5
    model.config._attn_implementation = "sdpa"  # the Bernoulli bias path, as in the runs
    pipe = KVPressTextGenerationPipeline(model=model, tokenizer=tokenizer)
    context = " ".join(f"fact{i} is value{i * 7 % 13}." for i in range(40))
    q_a, q_b = "What is fact3?", "What is fact11?"
    for press in (None, BernoulliPress(press=KVzipPress(compression_ratio=0.5), seed=0)):
        alone = pipe(context, question=q_b, press=press, max_new_tokens=6)["answer"]
        both = pipe(context, questions=[q_a, q_b], press=press, max_new_tokens=6)["answers"]
        assert both[1] == alone, (type(press).__name__, both, alone)


def test_fastkvzip_gates_follow_compressible_layers(tiny_qwen3_5_model):
    """One gate per KV-holding layer: Fast KVzip scores and masks only the full-attention layers."""
    from kvpress.presses.fastkvzip_press import FastKVzipGate, FastKVzipPress

    model = tiny_qwen3_5_model
    cfg = model.config
    modules = get_adapter(model).compressible_modules(model)
    press = FastKVzipPress(compression_ratio=0.5)
    torch.manual_seed(0)
    press.gates = [
        FastKVzipGate(i, cfg.hidden_size, cfg.num_key_value_heads,
                      cfg.num_attention_heads // cfg.num_key_value_heads, torch.float32).to(model.device)
        for i in range(len(modules))
    ]
    seq_len = 128
    input_ids = torch.randint(0, 128, (1, seq_len), device=model.device)
    cache = get_adapter(model).make_cache(model)
    with press(model), torch.no_grad():
        model.model(input_ids=input_ids, past_key_values=cache)

    n_masked = 0
    for layer_idx, layer in enumerate(model.model.layers):
        attn = getattr(layer, "self_attn", None)
        if layer_idx in FULL_ATTENTION_LAYERS:
            n_masked += len(attn.masked_key_indices[0])
        else:
            assert attn is None or getattr(attn, "masked_key_indices", None) is None
    assert n_masked == int(len(modules) * cfg.num_key_value_heads * seq_len * 0.5)
    for m in modules:
        m.masked_key_indices = None


def test_fastkvzip_rejects_gates_for_another_layout(tiny_qwen3_5_model):
    from kvpress.presses.fastkvzip_press import FastKVzipGate, FastKVzipPress

    model = tiny_qwen3_5_model
    cfg = model.config
    press = FastKVzipPress(compression_ratio=0.5)
    press.gates = [FastKVzipGate(i, cfg.hidden_size, cfg.num_key_value_heads, 2, torch.float32)
                   for i in range(cfg.num_hidden_layers)]  # one per layer, as for a dense stack
    with pytest.raises(AssertionError, match="gates for"):
        with press(model):
            pass
