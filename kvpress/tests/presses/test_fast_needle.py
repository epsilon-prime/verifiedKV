# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import types

import pytest
import torch
from transformers import DynamicCache

from kvpress import BernoulliPress, FastKVzipPress, FastKVzipScorer, NeedleFetchPress
from kvpress.adapters import get_adapter
from kvpress.presses.fastkvzip_press import FastKVzipGate
from kvpress.utils import get_cache_metadata
from tests.fixtures import kv_press_unit_test_pipeline  # noqa: F401

CONTEXT = "This is a test article. It was written on 2022-01-01. " * 20
QUESTION = "When was this article written?"


def random_gates(model, seed=0):
    """Randomly initialised gates of the right shapes: the unit-test model has no trained ones."""
    cfg = model.config
    n_layers = len(get_adapter(model).compressible_modules(model))
    torch.manual_seed(seed)
    gates = []
    for idx in range(n_layers):
        gate = FastKVzipGate(idx, cfg.hidden_size, cfg.num_key_value_heads,
                             cfg.num_attention_heads // cfg.num_key_value_heads, model.dtype)
        torch.nn.init.normal_(gate.b, std=1.0)  # spread the scores so the ranking is not flat
        gates.append(gate.to(model.device))
    return gates


def fast_needle(model, **kw):
    scorer = FastKVzipScorer()
    scorer.gates = random_gates(model)
    return NeedleFetchPress(bern=BernoulliPress(press=scorer, seed=kw.pop("seed", 0)), **kw)


@pytest.mark.parametrize("press_cls, expected", [(FastKVzipPress, 800), (FastKVzipScorer, 0)])
def test_scorer_has_no_window(press_cls, expected):
    """FastKVzipScorer protects only the sinks; the baseline keeps its 2% window."""

    class ZeroGate(torch.nn.Module):
        def forward(self, h):
            return torch.zeros(1, 2, h.shape[1])

    press = press_cls(window_size=None)
    press._row, press.gates, press.score_val = {0: 0}, [ZeroGate()], [None]
    press._score_fast(types.SimpleNamespace(layer_idx=0), torch.zeros(1, 40000, 8))
    scores = press.score_val[0][0, 0]
    assert (scores[: press.n_sink] == 1.0).all()
    assert (scores[press.n_sink:] == 1.0).sum().item() == expected


def test_bernoulli_on_fast_scores(kv_press_unit_test_pipeline):  # noqa: F811
    """BernoulliPress samples Fast KVzip's stacked scores with a global c, like KVzip's: a bias for every
    layer, sinks kept with r = 1, about the target kept, and the cache reference released afterwards."""
    model = kv_press_unit_test_pipeline.model
    scorer = FastKVzipScorer(compression_ratio=0.75)
    scorer.gates = random_gates(model)
    press = BernoulliPress(press=scorer)
    cache = DynamicCache()
    kv_press_unit_test_pipeline(CONTEXT, question=QUESTION, press=press, cache=cache, max_new_tokens=2)
    bias = get_cache_metadata(cache)["attention_bias"]
    modules = get_adapter(model).compressible_modules(model)
    assert set(bias) == {int(m.layer_idx) for m in modules}
    assert all(m.masked_key_indices is None for m in modules)
    kept, n = 0, 0
    for b in bias.values():
        assert (b[..., : scorer.n_sink] == 0).all()
        n += b.numel()
        kept += torch.isfinite(b).sum().item()
    assert scorer.score_val.shape == (len(modules), 1, model.config.num_key_value_heads, b.shape[-1])
    assert abs(kept / n - 0.25) < 0.1  # realised kept fraction near the target
    assert scorer._cache is None and scorer._context_ids is None
    assert "compress_post" not in vars(scorer)


def test_scorer_alone_is_global_topk(kv_press_unit_test_pipeline):  # noqa: F811
    """Used alone, FastKVzipScorer is Fast KVzip's global top-k (no window): it prunes the ratio exactly."""
    model = kv_press_unit_test_pipeline.model
    scorer = FastKVzipScorer(compression_ratio=0.5)
    scorer.gates = random_gates(model)
    kv_press_unit_test_pipeline(CONTEXT, question=QUESTION, press=scorer, max_new_tokens=2)
    modules = get_adapter(model).compressible_modules(model)
    pruned = sum(len(m.masked_key_indices[0]) for m in modules)
    assert pruned == int(scorer.score_val.numel() * 0.5)


def test_pipeline_fast_needle_attaches_sketch_and_counts_fetches(kv_press_unit_test_pipeline):  # noqa: F811
    model = kv_press_unit_test_pipeline.model
    press = fast_needle(model, theta=0.01)
    press.compression_ratio = 0.5
    cache = DynamicCache()
    kv_press_unit_test_pipeline(CONTEXT, question=QUESTION, press=press, cache=cache, max_new_tokens=3)
    metadata = get_cache_metadata(cache)
    assert "attention_bias" not in metadata
    assert set(metadata["needle"]) == {int(m.layer_idx) for m in BernoulliPress._attention_modules(model)}
    assert press.fetch_stats["decode_decisions"] > 0 and press.fetch_stats["question_decisions"] > 0
    assert "compress_post" not in vars(press.bern)


def test_pipeline_fast_needle_infinite_theta_is_fast_bernoulli(kv_press_unit_test_pipeline):  # noqa: F811
    """With no fetching the answer equals Bernoulli on the same Fast KVzip scores at the same kept fraction."""
    model = kv_press_unit_test_pipeline.model
    needle = fast_needle(model, theta=float("inf"))
    needle.compression_ratio = 0.5
    scorer = FastKVzipScorer(compression_ratio=needle.bern.compression_ratio)
    scorer.gates = random_gates(model)
    a = kv_press_unit_test_pipeline(CONTEXT, question=QUESTION, press=needle, max_new_tokens=6)
    b = kv_press_unit_test_pipeline(CONTEXT, question=QUESTION, press=BernoulliPress(press=scorer), max_new_tokens=6)
    assert a["answer"] == b["answer"]


def test_registry_fast_needle_matches_needle_final():
    """fast_needle_final differs from needle_final_c only in the scorer."""
    import sys
    import os

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "evaluation"))
    from evaluate_registry import PRESS_REGISTRY

    fast, ref = PRESS_REGISTRY["fast_needle_final"], PRESS_REGISTRY["needle_final_c"]
    fields = ("theta", "theta_question", "persist", "reserve", "reserve_entries", "bits", "n_pairs",
              "pairs_per_128_dims", "group", "charge_overhead")
    assert all(getattr(fast, f) == getattr(ref, f) for f in fields)
    assert isinstance(fast.bern.press, FastKVzipScorer) and fast.bern.press.window_ratio == 0
    for s in (1, 2):
        assert PRESS_REGISTRY[f"fast_needle_final_s{s}"].bern.seed == s
