# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch
from transformers import DynamicCache
from transformers.integrations.sdpa_attention import sdpa_attention_forward

from kvpress import BernoulliPress, KVzipPress, NeedleFetchPress
from kvpress.attention_patch import attention_patch
from kvpress.presses.needle_fetch_press import (
    needle_decode_bias,
    one_bit_sketch,
    quant_sketch,
    sketch_bytes_per_entry,
    top_energy_pairs,
)
from kvpress.utils import get_cache_metadata
from tests.fixtures import kv_press_unit_test_pipeline  # noqa: F401
from tests.presses.test_bernoulli_press import (
    NUM_HEADS,
    NUM_KV_HEADS,
    FakeAttention,
    causal_masks,
    random_bias,
    random_qkv,
    reference_attention,
)


def fresh_stats():
    return NeedleFetchPress().fetch_stats


def needle_metadata(bias, key, log_theta, k_hat=None):
    ctx = bias.shape[-1]
    k_hat = key[:, :, :ctx].float() if k_hat is None else k_hat
    return {"needle": {0: {"bias": bias, "k_hat": k_hat, "log_theta": log_theta, "stats": fresh_stats()}}}


@pytest.mark.parametrize("q_len", [1, 5])
@pytest.mark.parametrize("mask_kind", ["none", "bool", "additive"])
def test_infinite_theta_is_bernoulli(q_len, mask_kind):
    """No entry can reach an infinite threshold, so the output is exactly the static Bernoulli one."""
    k_len, ctx_len = 12, 9
    query, key, value = random_qkv(q_len, k_len)
    bias = random_bias(ctx_len)
    mask = causal_masks(q_len, k_len)[mask_kind]
    meta = needle_metadata(bias, key, float("inf"))
    out, _ = attention_patch(sdpa_attention_forward)(FakeAttention(), query, key, value, mask, 0.0, kvpress_metadata=meta)
    torch.testing.assert_close(out, reference_attention(query, key, value, bias, ctx_len), atol=1e-5, rtol=1e-5)
    assert meta["needle"][0]["stats"]["decode_fetch_any" if q_len == 1 else "question_fetch_any"] == 0


@pytest.mark.parametrize("q_len", [1, 5])
@pytest.mark.parametrize("mask_kind", ["none", "bool", "additive"])
def test_zero_theta_is_full_attention(q_len, mask_kind):
    """Every non-heavy entry reaches a zero threshold: all biases become 0, which is the full cache."""
    k_len, ctx_len = 12, 9
    query, key, value = random_qkv(q_len, k_len)
    bias = random_bias(ctx_len)
    mask = causal_masks(q_len, k_len)[mask_kind]
    meta = needle_metadata(bias, key, -float("inf"))
    out, _ = attention_patch(sdpa_attention_forward)(FakeAttention(), query, key, value, mask, 0.0, kvpress_metadata=meta)
    zero = torch.zeros(1, NUM_KV_HEADS, ctx_len)
    torch.testing.assert_close(out, reference_attention(query, key, value, zero, ctx_len), atol=1e-5, rtol=1e-5)
    stats = meta["needle"][0]["stats"]
    kind = "decode" if q_len == 1 else "question"
    # decode counts per decision; a question prefill counts each entry once (one transfer for the batch)
    assert stats[f"{kind}_fetched_entries"] == int(torch.isinf(bias).sum())
    assert stats[f"{kind}_fetch_any"] == q_len * int(torch.isinf(bias).any(-1).sum())


def test_only_the_needle_is_fetched():
    """A dropped key that dominates the query is fetched; nothing else changes."""
    ctx_len, d = 9, 8
    query = torch.zeros(1, NUM_HEADS, 1, d)
    query[..., 0] = 1.0
    key = torch.zeros(1, NUM_KV_HEADS, ctx_len + 1, d)
    key[0, 0, 4, 0] = 40.0  # the needle, in KV head 0
    bias = torch.zeros(1, NUM_KV_HEADS, ctx_len)  # all heavy ...
    bias[0, 0, 3:6] = -float("inf")  # ... except three dropped entries in head 0
    bias[0, 0, 6] = math.log(4.0)  # and one kept entry with r = 1/4
    state = needle_metadata(bias, key, math.log(0.5))["needle"][0]
    out = needle_decode_bias(query, key, state, scaling=1.0)
    groups = NUM_HEADS // NUM_KV_HEADS
    expected = bias.clone()
    expected[0, 0, 4] = 0.0
    torch.testing.assert_close(out, expected.repeat_interleave(groups, dim=1)[:, :, None, :])
    assert state["stats"]["decode_fetched_entries"] == 1
    assert state["stats"]["decode_fetch_any"] == 1
    assert state["stats"]["decode_decisions"] == NUM_KV_HEADS


def test_one_bit_sketch_two_levels_per_group_and_channel():
    g = torch.Generator().manual_seed(0)
    keys = torch.randn(1, 2, 300, 8, generator=g)
    k_hat = one_bit_sketch(keys, group=128)
    assert k_hat.shape == keys.shape
    for start in (0, 128, 256):
        block = k_hat[:, :, start : start + 128]
        assert max(torch.unique(block[0, h, :, c]).numel() for h in range(2) for c in range(8)) <= 2
    # better than the per-group mean, i.e. zero bits
    gm = torch.cat([keys[:, :, s : s + 128].mean(2, keepdim=True).expand_as(keys[:, :, s : s + 128]) for s in (0, 128, 256)], 2)
    assert (k_hat - keys).pow(2).mean() < (gm - keys).pow(2).mean()


def test_one_bit_sketch_exact_on_two_valued_channels():
    keys = torch.tensor([1.0, -2.0, 1.0, 1.0, -2.0])[None, None, :, None].expand(1, 1, 5, 4).clone()
    torch.testing.assert_close(one_bit_sketch(keys, group=4), keys, atol=1e-3, rtol=0)


def test_total_budget_includes_the_sketch():
    press = NeedleFetchPress()
    press.compression_ratio = 0.875
    s = sketch_bytes_per_entry(128, 128) / 512
    kept = 1.0 - press.bern.compression_ratio
    assert press.compression_ratio == 0.875
    assert abs(kept + (1 - kept) * s - 0.125) < 1e-12
    assert 0.08 < kept < 0.10


def test_pipeline_attaches_sketch_and_counts_fetches(kv_press_unit_test_pipeline):  # noqa: F811
    cache = DynamicCache()
    press = NeedleFetchPress(bern=BernoulliPress(press=KVzipPress()), theta=0.01)
    press.compression_ratio = 0.5
    kv_press_unit_test_pipeline(
        "This is a test article. It was written on 2022-01-01. " * 20,
        question="When was this article written?",
        press=press,
        cache=cache,
        max_new_tokens=3,
    )
    metadata = get_cache_metadata(cache)
    assert "attention_bias" not in metadata
    model = kv_press_unit_test_pipeline.model
    assert set(metadata["needle"]) == {int(m.layer_idx) for m in BernoulliPress._attention_modules(model)}
    n_kv = model.config.num_key_value_heads
    assert press.fetch_stats["decode_decisions"] >= model.config.num_hidden_layers * n_kv
    assert press.fetch_stats["question_decisions"] > 0
    assert "compress_post" not in vars(press.bern)


def test_pipeline_infinite_theta_matches_bernoulli(kv_press_unit_test_pipeline):  # noqa: F811
    """With no fetching the answer must be exactly BernoulliPress's at the same kept fraction."""
    context = "This is a test article. It was written on 2022-01-01. " * 20
    needle = NeedleFetchPress(theta=float("inf"))
    needle.compression_ratio = 0.5
    bern = BernoulliPress(press=KVzipPress(compression_ratio=needle.bern.compression_ratio))
    a = kv_press_unit_test_pipeline(context, question="When was this article written?", press=needle, max_new_tokens=6)
    b = kv_press_unit_test_pipeline(context, question="When was this article written?", press=bern, max_new_tokens=6)
    assert a["answer"] == b["answer"]


@pytest.mark.parametrize("ratio", [0.5, 0.9375])
def test_uncharged_overhead_keeps_the_full_budget(ratio):
    """Without charging the sketch and reserve, the draw keeps exactly 1 - ratio, like BernoulliPress."""
    needle = NeedleFetchPress(charge_overhead=False, reserve=0.01, reserve_entries=128, bits=2, n_pairs=16)
    needle.compression_ratio = ratio
    assert needle.bern.compression_ratio == pytest.approx(ratio, abs=1e-12)
    needle._set_kept(0.01)  # the per-context reserve must not lower it either
    assert needle.bern.compression_ratio == pytest.approx(ratio, abs=1e-12)
    charged = NeedleFetchPress(reserve=0.01, reserve_entries=128, bits=2, n_pairs=16)
    charged.compression_ratio = ratio
    assert charged.bern.compression_ratio > ratio  # the default still pays for them


def test_pipeline_uncharged_infinite_theta_is_bernoulli_at_the_same_ratio(kv_press_unit_test_pipeline):  # noqa: F811
    """Uncharged and with no fetching, the answer equals BernoulliPress's at the SAME ratio (same draw)."""
    context = "This is a test article. It was written on 2022-01-01. " * 20
    needle = NeedleFetchPress(theta=float("inf"), charge_overhead=False, persist=True, reserve=0.01,
                              reserve_entries=128, bits=2, n_pairs=16)
    needle.compression_ratio = 0.5
    bern = BernoulliPress(press=KVzipPress(compression_ratio=0.5))
    a = kv_press_unit_test_pipeline(context, question="When was this article written?", press=needle, max_new_tokens=6)
    b = kv_press_unit_test_pipeline(context, question="When was this article written?", press=bern, max_new_tokens=6)
    assert a["answer"] == b["answer"]


def test_traffic_counters_per_step_and_question():
    """A decode step fetches if any layer does; the question-time share is per (layer, KV head)."""
    k_len, ctx_len = 12, 9
    _, key, _ = random_qkv(1, k_len)
    bias = random_bias(ctx_len)
    stats = fresh_stats()
    layers = {i: {"bias": bias, "k_hat": key[:, :, :ctx_len].float(), "stats": stats, "layer_idx": i, "first_layer": 0,
                  "log_theta": float("inf") if i == 0 else -float("inf")} for i in (0, 1)}
    for _ in range(3):  # three generated tokens: layer 0 never fetches, layer 1 always does
        for i in (0, 1):
            query, _, _ = random_qkv(1, k_len)
            needle_decode_bias(query, key, layers[i], 0.35)
    assert stats["decode_steps"] == 3
    assert stats["decode_steps_any"] == 3
    layers[1]["log_theta"] = float("inf")  # now nothing fetches
    for i in (0, 1):
        query, _, _ = random_qkv(1, k_len)
        needle_decode_bias(query, key, layers[i], 0.35)
    assert stats["decode_steps"] == 4 and stats["decode_steps_any"] == 3
    # question: a zero threshold transfers every dropped entry, so the share is the dropped share per head
    q_state = {"bias": bias, "k_hat": key[:, :, :ctx_len].float(), "stats": stats, "log_theta": -float("inf")}
    query, _, _ = random_qkv(5, k_len)
    needle_decode_bias(query, key, q_state, 0.35)
    dropped = torch.isinf(bias).float().sum(-1) / ctx_len  # [1, n_kv]
    assert stats["question_frac_sum"] == pytest.approx(float(dropped.sum()))
    assert stats["question_frac_max"] == pytest.approx(float(dropped.max()))
    NeedleFetchPress.reset_stats(type("P", (), {"fetch_stats": stats})())
    assert "decode_steps" not in stats and "question_frac_max" not in stats


def test_non_contiguous_query():
    """Attention modules hand over a transposed (non-contiguous) query; the bias must not depend on layout."""
    k_len, ctx_len, q_len = 12, 9, 5
    query, key, value = random_qkv(q_len, k_len)
    bias = random_bias(ctx_len)
    strided = query.transpose(1, 2).contiguous().transpose(1, 2)
    assert not strided.is_contiguous()
    a = needle_decode_bias(query, key, needle_metadata(bias, key, math.log(0.05))["needle"][0], 0.35)
    b = needle_decode_bias(strided, key, needle_metadata(bias, key, math.log(0.05))["needle"][0], 0.35)
    torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_persisted_needle_stays_exact_and_is_fetched_once():
    """With persistence a fetched entry stays exact for later queries that would not flag it."""
    ctx_len, d = 9, 8
    key = torch.zeros(1, NUM_KV_HEADS, ctx_len + 1, d)
    key[0, 0, 4, 0] = 40.0
    bias = torch.zeros(1, NUM_KV_HEADS, ctx_len)
    bias[0, 0, 3:6] = -float("inf")
    state = needle_metadata(bias, key, math.log(0.5))["needle"][0]
    state["promoted"] = torch.zeros_like(bias, dtype=torch.bool)
    q_hit = torch.zeros(1, NUM_HEADS, 1, d)
    q_hit[..., 0] = 1.0
    q_miss = torch.zeros(1, NUM_HEADS, 1, d)
    q_miss[..., 1] = 1.0
    needle_decode_bias(q_hit, key, state, 1.0)
    needle_decode_bias(q_hit, key, state, 1.0)
    out = needle_decode_bias(q_miss, key, state, 1.0)
    groups = NUM_HEADS // NUM_KV_HEADS
    assert out[0, 0, 0, 4] == 0 and torch.isinf(out[0, 0, 0, 3]) and torch.isinf(out[0, 0, 0, 5])
    assert out.shape == (1, NUM_KV_HEADS * groups, 1, ctx_len)
    assert state["stats"]["decode_fetched_entries"] == 1
    assert state["stats"]["decode_fetch_any"] == 1
    assert state["stats"]["promoted_dropped"] == 1


def test_question_prefill_positions_do_not_see_each_others_fetches():
    """Within one multi-token call, each position gets only its own flags (plus earlier calls')."""
    ctx_len, d = 9, 8
    key = torch.zeros(1, NUM_KV_HEADS, ctx_len + 2, d)
    key[0, 0, 4, 0] = 40.0
    bias = torch.zeros(1, NUM_KV_HEADS, ctx_len)
    bias[0, 0, 3:6] = -float("inf")
    state = needle_metadata(bias, key, math.log(0.5))["needle"][0]
    state["promoted"] = torch.zeros_like(bias, dtype=torch.bool)
    q = torch.zeros(1, NUM_HEADS, 2, d)
    q[:, :, 0, 1] = 1.0  # position 0 misses
    q[:, :, 1, 0] = 1.0  # position 1 hits
    out = needle_decode_bias(q, key, state, 1.0)
    assert torch.isinf(out[0, 0, 0, 4]) and out[0, 0, 1, 4] == 0
    assert bool(state["promoted"][0, 0, 4])


def test_reserve_comes_out_of_the_kept_budget():
    press = NeedleFetchPress(reserve=0.01)
    press.compression_ratio = 0.875
    s = sketch_bytes_per_entry(128, 128) / 512
    kept = 1.0 - press.bern.compression_ratio
    assert abs(kept + (1 - kept) * s + 0.01 - 0.125) < 1e-12


def test_question_threshold_applies_only_to_multi_token_calls():
    ctx_len, d = 9, 8
    key = torch.zeros(1, NUM_KV_HEADS, ctx_len + 2, d)
    key[0, 0, 4, 0] = 2.0  # a moderate entry: weight about e^2 / (e^2 + 8)
    bias = torch.zeros(1, NUM_KV_HEADS, ctx_len)
    bias[0, 0, 3:6] = -float("inf")
    q = torch.zeros(1, NUM_HEADS, 1, d)
    q[..., 0] = 1.0
    state = needle_metadata(bias, key, math.log(0.9))["needle"][0]
    state["log_theta_question"] = math.log(0.1)
    assert torch.isinf(needle_decode_bias(q, key, state, 1.0)[0, 0, 0, 4])  # decode: 0.48 < 0.9
    q2 = q.expand(1, NUM_HEADS, 2, d).clone()
    assert needle_decode_bias(q2, key, state, 1.0)[0, 0, 0, 4] == 0  # question: 0.48 >= 0.1


def test_promotion_cap_keeps_the_heaviest():
    """Past the reserve, only the heaviest new entries persist; all flagged ones are still exact now."""
    ctx_len, d = 9, 8
    key = torch.zeros(1, NUM_KV_HEADS, ctx_len + 1, d)
    key[0, 0, 3, 0] = 30.0
    key[0, 0, 4, 0] = 32.0  # the heaviest
    key[0, 0, 5, 0] = 31.0
    bias = torch.zeros(1, NUM_KV_HEADS, ctx_len)
    bias[0, 0, 3:6] = -float("inf")
    state = needle_metadata(bias, key, math.log(1e-3))["needle"][0]
    state["promoted"] = torch.zeros_like(bias, dtype=torch.bool)
    state["cap"] = 1
    q = torch.zeros(1, NUM_HEADS, 1, d)
    q[..., 0] = 1.0
    out = needle_decode_bias(q, key, state, 1.0)
    assert (out[0, 0, 0, 3:6] == 0).all()
    assert state["promoted"][0, 0].nonzero().flatten().tolist() == [4]
    assert state["stats"]["promoted_dropped"] == 1


def test_quant_sketch_levels_and_error_shrink_with_bits():
    g = torch.Generator().manual_seed(1)
    keys = torch.randn(1, 2, 256, 8, generator=g)
    errs = [(quant_sketch(keys, b, 128) - keys).pow(2).mean().item() for b in (2, 3, 4)]
    assert errs[0] > errs[1] > errs[2]
    block = quant_sketch(keys, 2, 128)[0, 0, :128, 0]
    assert torch.unique(block).numel() <= 4


def test_top_energy_pairs_selects_whole_rope_pairs():
    keys = torch.zeros(1, 1, 10, 8)
    keys[..., 1] = 5.0  # pair (1, 5) has the energy
    keys[..., 7] = 3.0  # then pair (3, 7)
    ch = top_energy_pairs(keys, 2)[0, 0].tolist()
    assert ch == [1, 3, 5, 7]


def test_channel_subset_sketch_in_decode_bias():
    """With a channel subset the estimate uses only those channels; an exact subset sketch of a
    needle living in those channels still finds it."""
    ctx_len, d = 9, 8
    key = torch.zeros(1, NUM_KV_HEADS, ctx_len + 1, d)
    key[0, 0, 4, 1] = 40.0
    bias = torch.zeros(1, NUM_KV_HEADS, ctx_len)
    bias[0, 0, 3:6] = -float("inf")
    channels = torch.tensor([1, 5]).expand(1, NUM_KV_HEADS, 2).contiguous()
    k_hat = key[:, :, :ctx_len].gather(-1, channels[:, :, None, :].expand(-1, -1, ctx_len, -1))
    state = needle_metadata(bias, key, math.log(0.5), k_hat=k_hat)["needle"][0]
    state["channels"] = channels
    q = torch.zeros(1, NUM_HEADS, 1, d)
    q[..., 1] = 1.0
    out = needle_decode_bias(q, key, state, 1.0)
    assert out[0, 0, 0, 4] == 0 and torch.isinf(out[0, 0, 0, 3])


def test_sketch_bytes():
    assert sketch_bytes_per_entry(128, 128) == 20.0
    assert sketch_bytes_per_entry(128, 128, bits=2, n_pairs=16) == 9.0
    press = NeedleFetchPress(bits=2, n_pairs=16)
    assert abs(press.sketch_frac - 9 / 512) < 1e-12


def test_reserve_entries_scales_with_context():
    press = NeedleFetchPress(reserve_entries=128, bits=2, n_pairs=16)
    press.compression_ratio = 0.875
    assert abs(press._reserve_for(32768) - 128 / 32768) < 1e-15
    assert press._reserve_for(100) == 0.5 * (0.125 - press.sketch_frac)  # capped for tiny contexts
    press._set_kept(press._reserve_for(32768))
    kept = 1.0 - press.bern.compression_ratio
    assert abs(kept + (1 - kept) * press.sketch_frac + 128 / 32768 - 0.125) < 1e-12


def test_pipeline_reserve_entries_caps_promotions(kv_press_unit_test_pipeline):  # noqa: F811
    press = NeedleFetchPress(theta=0.01, theta_question=1e-4, persist=True, reserve_entries=4, bits=2, n_pairs=4)
    press.compression_ratio = 0.5
    cache = DynamicCache()
    kv_press_unit_test_pipeline(
        "This is a test article. It was written on 2022-01-01. " * 20,
        question="When was this article written?", press=press, cache=cache, max_new_tokens=3,
    )
    for state in get_cache_metadata(cache)["needle"].values():
        assert int(state["promoted"].sum(-1).max()) <= state["cap"]


def test_reserve_entries_capped_by_reserve_fraction():
    press = NeedleFetchPress(reserve=0.01, reserve_entries=128, bits=2, n_pairs=32)
    press.compression_ratio = 0.875
    assert press._reserve_for(2048) == 0.01  # 128 / 2048 = 6% would eat the kept budget
    assert abs(press._reserve_for(32768) - 128 / 32768) < 1e-15


def test_cached_kvzip_scores_reproduce_uncached(kv_press_unit_test_pipeline, tmp_path, monkeypatch):  # noqa: F811
    """The second (cached) run must give the same answer as an uncached run, and must hit the cache."""
    from kvpress import CachedKVzipPress

    context = "This is a test article. It was written on 2022-01-01. " * 20
    q = "When was this article written?"
    ref = kv_press_unit_test_pipeline(context, question=q, press=BernoulliPress(press=KVzipPress(compression_ratio=0.5)),
                                      max_new_tokens=6)["answer"]
    for _ in range(2):  # first call writes the cache, second reads it
        press = BernoulliPress(press=CachedKVzipPress(compression_ratio=0.5, cache_dir=str(tmp_path)))
        out = kv_press_unit_test_pipeline(context, question=q, press=press, max_new_tokens=6)["answer"]
        assert out == ref
    assert len(list(tmp_path.glob("*.pt"))) == 1


# --- final_eval: per-question reset (P10), partial RoPE (P8), hybrid stacks (P7, P11) ---


def test_persisted_set_resets_at_each_new_question():
    """Fetches persist through one answer but never into the next question on the same context (audit
    B2): a multi-token forward is a new question and starts from an empty persisted set."""
    ctx_len, d = 9, 8
    key = torch.zeros(1, NUM_KV_HEADS, ctx_len + 2, d)
    key[0, 0, 4, 0] = 40.0
    bias = torch.zeros(1, NUM_KV_HEADS, ctx_len)
    bias[0, 0, 3:6] = -float("inf")
    state = needle_metadata(bias, key, math.log(0.5))["needle"][0]
    state["promoted"] = torch.zeros_like(bias, dtype=torch.bool)
    hit = torch.zeros(1, NUM_HEADS, 2, d)
    hit[..., 0] = 1.0
    miss2 = torch.zeros(1, NUM_HEADS, 2, d)
    miss2[..., 1] = 1.0
    miss1 = miss2[:, :, :1]

    needle_decode_bias(hit, key, state, 1.0)  # question 1 fetches the needle
    assert bool(state["promoted"][0, 0, 4])
    assert needle_decode_bias(miss1, key, state, 1.0)[0, 0, 0, 4] == 0  # its answer keeps it exact
    out = needle_decode_bias(miss2, key, state, 1.0)  # question 2 does not ask for it
    assert torch.isinf(out[0, 0, :, 4]).all()
    assert not state["promoted"].any()


def test_top_energy_pairs_full_rope_unchanged_by_rotary_dim():
    """``rotary_dim == head_dim`` (Llama, Qwen3, Ministral) takes the original code path exactly."""
    g = torch.Generator().manual_seed(0)
    keys = torch.randn(2, 3, 50, 16, generator=g)
    assert torch.equal(top_energy_pairs(keys, 4), top_energy_pairs(keys, 4, rotary_dim=16))


def test_top_energy_pairs_partial_rope_keeps_rotary_pairs_whole():
    """Partial RoPE (Qwen3.5 style): the first ``rotary_dim`` channels rotate in pairs (j, j + r/2); the
    rest are position-free singles. Pairs are taken whole, singles alone, by energy per channel."""
    d, r = 16, 4  # rotary pairs (0, 2) and (1, 3); singles 4..15
    keys = torch.zeros(1, 1, 10, d)
    keys[..., 0] = 3.0
    keys[..., 2] = 3.0  # pair (0, 2): energy 9 per channel
    keys[..., 9] = 4.0  # single: 16 per channel, the best unit
    keys[..., 1] = 5.0  # pair (1, 3): (25 + 0) / 2 = 12.5 per channel
    keys[..., 12] = 1.0  # single: 1
    ch = top_energy_pairs(keys, 2, rotary_dim=r)[0, 0].tolist()
    # greedy over 4 channels: single 9, then pair (1, 3); pair (0, 2) would overflow, so single 12 fills
    assert ch == [9, 1, 3, 12]
    for j in range(r // 2):  # a rotary pair is never split
        assert (j in ch) == (j + r // 2 in ch)


def test_needle_fetch_runs_on_a_hybrid_stack(monkeypatch):
    """Qwen3.5: the sketch reads K/V through the adapter (its cache has no ``layers``), only full-attention
    layers are sketched, channels follow partial RoPE, and a question plus decode steps go through."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    from kvpress.adapters import get_adapter
    from tests.adapters.test_qwen3_5_adapter import FULL_ATTENTION_LAYERS, _tiny_config
    from tests.fixtures import get_device

    torch.manual_seed(0)
    model = Qwen3_5ForCausalLM(_tiny_config()).eval().to(get_device())  # fla kernels need CUDA tensors on GPU
    model.config._attn_implementation = "sdpa"
    dev = model.device
    adapter = get_adapter(model)
    modules = adapter.compressible_modules(model)
    n_ctx, n_kv = 64, model.config.num_key_value_heads

    press = NeedleFetchPress(bern=BernoulliPress(press=KVzipPress(), seed=0), theta=1e-3, theta_question=1e-3,
                             persist=True, reserve_entries=8, bits=2, n_pairs=2)
    press.compression_ratio = 0.5
    press.post_init_from_model(model)
    assert press.head_dim == 16 and press.rotary_dim == 4

    cache = adapter.make_cache(model)
    with torch.no_grad():
        model.model(input_ids=torch.randint(0, 128, (1, n_ctx), device=dev), past_key_values=cache)
    inner = press.bern.press
    torch.manual_seed(1)
    inner.score_val = torch.rand(len(modules), 1, n_kv, n_ctx, device=dev)
    inner._score_row_by_layer = {int(m.layer_idx): row for row, m in enumerate(modules)}
    inner._context_ids = torch.arange(n_ctx, device=dev)[None]
    inner._cache = cache
    try:
        press.bern.compress_post(model)
        press._attach_sketch(model)
        meta = get_cache_metadata(cache)
        assert sorted(meta["needle"]) == FULL_ATTENTION_LAYERS
        for state in meta["needle"].values():
            assert state["channels"].shape == (1, n_kv, 4)
            for j in range(2):  # rotary pairs (j, j + 2) whole
                assert ((state["channels"] == j).any(-1) == (state["channels"] == j + 2).any(-1)).all()
        with torch.no_grad():
            model(input_ids=torch.randint(0, 128, (1, 5), device=dev), past_key_values=cache, kvpress_metadata=meta)
            for _ in range(3):
                model(input_ids=torch.randint(0, 128, (1, 1), device=dev), past_key_values=cache, kvpress_metadata=meta)
        assert press.fetch_stats["question_decisions"] > 0 and press.fetch_stats["decode_decisions"] > 0
    finally:
        for m in modules:
            m.masked_key_indices = None


def test_pairs_per_128_dims_keeps_the_sketch_share():
    """Qwen3.5 (head_dim 256) gets twice the pairs, so its sketch costs the same share of the cache."""
    small = NeedleFetchPress(n_pairs=16, bits=2, pairs_per_128_dims=True, head_dim=128)
    large = NeedleFetchPress(n_pairs=16, bits=2, pairs_per_128_dims=True, head_dim=256)
    fixed = NeedleFetchPress(n_pairs=16, bits=2, head_dim=256)
    assert (small.pairs, large.pairs, fixed.pairs) == (16, 32, 16)
    assert large.sketch_frac == pytest.approx(small.sketch_frac)
    assert fixed.sketch_frac == pytest.approx(small.sketch_frac / 2)
