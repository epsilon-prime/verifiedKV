# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from transformers import DynamicCache

from kvpress import BernoulliPress, KVzipPress, NeedleFetchPress
from kvpress.presses.needle_fetch_press import KV_BYTES_PER_CHANNEL, pq_sketch, sketch_bytes_per_entry
from kvpress.utils import get_cache_metadata
from tests.fixtures import kv_press_unit_test_pipeline  # noqa: F401

CONTEXT = "This is a test article. It was written on 2022-01-01. " * 20
QUESTION = "When was this article written?"


def test_pq_sketch_is_exact_with_a_centroid_per_key():
    keys = torch.randn(2, 64, 32)
    torch.testing.assert_close(pq_sketch(keys, 4, 6), keys)  # 64 centroids for 64 keys


def test_pq_sketch_has_at_most_2_to_the_bits_centroids_per_subspace():
    k_hat = pq_sketch(torch.randn(2, 500, 32), subspaces=4, bits=3)
    for h in range(2):
        for s in range(4):
            assert torch.unique(k_hat[h, :, 8 * s : 8 * (s + 1)], dim=0).shape[0] <= 8


def test_pq_sketch_error_shrinks_with_bits():
    keys = torch.randn(2, 2000, 32)
    errs = [float((pq_sketch(keys, 4, b) - keys).pow(2).mean()) for b in (2, 4, 6)]
    assert errs[0] > errs[1] > errs[2]


def test_pq_sketch_fit_leaves_out_the_skipped_tokens():
    keys = torch.randn(1, 600, 16)
    keys[:, :4] *= 50  # sink-like outliers
    k_hat = pq_sketch(keys, 2, 4, fit_sample=256, skip=8)
    # no centroid comes from the outliers, so the sketch does not reproduce them
    assert (k_hat[:, :4] - keys[:, :4]).norm() > 0.5 * keys[:, :4].norm()


def test_pq_sketch_is_seeded_and_leaves_the_global_rng_alone():
    keys = torch.randn(2, 300, 16)
    state = torch.random.get_rng_state()
    a = pq_sketch(keys, 2, 4, seed=3, fit_sample=128, skip=4)
    assert torch.equal(state, torch.random.get_rng_state())
    torch.testing.assert_close(a, pq_sketch(keys, 2, 4, seed=3, fit_sample=128, skip=4))


def test_pq_bytes_include_the_codebook():
    assert sketch_bytes_per_entry(128, 128, sketch="pq", pq_subspaces=16, pq_bits=8, ctx=65536) == 17.0
    assert sketch_bytes_per_entry(128, 128, sketch="pq", pq_subspaces=16, pq_bits=4) == 8.0  # context unknown
    assert sketch_bytes_per_entry(128, 128, bits=2, n_pairs=16) == 9.0  # the low-bit sketch is unchanged


def test_pq_budget_charges_the_codebook_for_the_context():
    press = NeedleFetchPress(sketch="pq", pq_subspaces=16, pq_bits=8, reserve=0.0)
    press._ctx = 32768
    press.compression_ratio = 0.875
    s = (16 + 128 * 256 * 2 / 32768) / (128 * KV_BYTES_PER_CHANNEL)
    assert press.sketch_frac == pytest.approx(s)
    assert 1 - press.bern.compression_ratio == pytest.approx((0.125 - s) / (1 - s))


def test_pipeline_pq_sketch_covers_every_channel(kv_press_unit_test_pipeline):  # noqa: F811
    # the unit-test model has head_dim 6: 3 subspaces of 2 dimensions
    cache = DynamicCache()
    press = NeedleFetchPress(
        bern=BernoulliPress(press=KVzipPress()), theta=0.01, sketch="pq", pq_subspaces=3, pq_bits=4,
        pq_fit_sample=None, pq_skip=4,
    )
    press.compression_ratio = 0.5
    kv_press_unit_test_pipeline(CONTEXT, question=QUESTION, press=press, cache=cache, max_new_tokens=3)
    needle = get_cache_metadata(cache)["needle"]
    head_dim = press.head_dim
    for state in needle.values():
        ctx = state["bias"].shape[-1]
        assert state["channels"] is None
        assert tuple(state["k_hat"].shape[-2:]) == (ctx, head_dim)
    assert press._ctx == ctx
    assert press.fetch_stats["decode_decisions"] > 0
    codes_only = sketch_bytes_per_entry(head_dim, 128, sketch="pq", pq_subspaces=3, pq_bits=4)
    assert press.sketch_frac > codes_only / (head_dim * KV_BYTES_PER_CHANNEL)  # the codebook is charged too


def test_pipeline_rejects_a_sketch_that_does_not_fit(kv_press_unit_test_pipeline):  # noqa: F811
    press = NeedleFetchPress(
        bern=BernoulliPress(press=KVzipPress()), sketch="pq", pq_subspaces=3, pq_bits=8, pq_fit_sample=None,
        pq_skip=4, reserve=0.0,
    )
    press.compression_ratio = 0.97
    with pytest.raises(ValueError, match="leave nothing"):
        kv_press_unit_test_pipeline(CONTEXT, question=QUESTION, press=press, max_new_tokens=3)
