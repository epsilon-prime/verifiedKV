# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from benchmarks.aime2024.calculate_metrics import calculate_metrics as aime2024_scorer
from benchmarks.aime2025.calculate_metrics import calculate_metrics as aime2025_scorer
from benchmarks.aime25.calculate_metrics import calculate_metrics as aime25_scorer
from benchmarks.infinite_bench.calculate_metrics import calculate_metrics as infinite_bench_scorer
from benchmarks.loft.calculate_metrics import calculate_metrics as loft_scorer
from benchmarks.longbench.calculate_metrics import calculate_metrics as longbench_scorer
from benchmarks.longbench.calculate_metrics import calculate_metrics_e as longbench_scorer_e
from benchmarks.longbenchv2.calculate_metrics import calculate_metrics as longbenchv2_scorer
from benchmarks.loogle.calculate_metrics import calculate_metrics as loogle_scorer
from benchmarks.math500.calculate_metrics import calculate_metrics as math500_scorer
from benchmarks.needle_in_haystack.calculate_metrics import calculate_metrics as needle_in_haystack_scorer
from benchmarks.ruler.calculate_metrics import calculate_metrics as ruler_scorer
from benchmarks.scbench.calculate_metrics import calculate_metrics as scbench_scorer
from benchmarks.zero_scrolls.calculate_metrics import calculate_metrics as zero_scrolls_scorer

from kvpress import (
    CachedKVzipPress,
    NeedleFetchPress,
    AdaKVPress,
    BernoulliPress,
    BlockPress,
    CAMPress,
    ChunkKVPress,
    CompactorPress,
    ComposedPress,
    CriticalAdaKVPress,
    CriticalKVPress,
    CURPress,
    DecodingPress,
    DMSPress,
    DuoAttentionPress,
    EntropyGatedChunkKVPress,
    ExpectedAttentionPress,
    FastKVzipPress,
    FastKVzipScorer,
    FinchPress,
    KeyDiffPress,
    KnormPress,
    KVComposePress,
    KVzapPress,
    KVzipPress,
    LagKVPress,
    LUKVPress,
    MergingPress,
    ObservedAttentionPress,
    PyramidKVPress,
    QFilterPress,
    RandomPress,
    RestoreKVPress,
    SnapKVPress,
    StreamingLLMPress,
    ThinKPress,
    TOVAPress,
    VKvWRPress,
)

# These dictionaries define the available datasets, scorers, and KVPress methods for evaluation.
DATASET_REGISTRY = {
    "loogle": "simonjegou/loogle",
    "ruler": "simonjegou/ruler",
    "ruler16k": "xAlg-AI/att-hub-ruler-16k",
    "ruler32k": "xAlg-AI/att-hub-ruler-32k",
    "ruler64k": "SaylorTwift/RULER-65536-llama-3.2-tokenizer",
    "ruler128k": "SaylorTwift/RULER-131072-llama-3.2-tokenizer",
    "loft": "f20180301/loft-rag",
    "zero_scrolls": "simonjegou/zero_scrolls",
    "infinitebench": "MaxJeblick/InfiniteBench",
    "longbench": "Xnhyacinth/LongBench",
    "longbench-e": "Xnhyacinth/LongBench",
    "longbench-v2": "simonjegou/LongBench-v2",
    "needle_in_haystack": "alessiodevoto/paul_graham_essays",
    "scbench": "microsoft/SCBench",
    # Datasets used to be used for decoding compression
    "aime25": "alessiodevoto/aime25",
    "aime2024": "xAlg-AI/att-hub-aime2024",
    "aime2025": "xAlg-AI/att-hub-aime2025",
    "math500": "alessiodevoto/math500",
}

SCORER_REGISTRY = {
    "loogle": loogle_scorer,
    "ruler": ruler_scorer,
    "ruler16k": ruler_scorer,
    "ruler32k": ruler_scorer,
    "ruler64k": ruler_scorer,
    "ruler128k": ruler_scorer,
    "loft": loft_scorer,
    "zero_scrolls": zero_scrolls_scorer,
    "infinitebench": infinite_bench_scorer,
    "longbench": longbench_scorer,
    "longbench-e": longbench_scorer_e,
    "longbench-v2": longbenchv2_scorer,
    "needle_in_haystack": needle_in_haystack_scorer,
    "scbench": scbench_scorer,
    "aime25": aime25_scorer,
    "aime2024": aime2024_scorer,
    "aime2025": aime2025_scorer,
    "math500": math500_scorer,
}


PRESS_REGISTRY = {
    "adakv_snapkv": AdaKVPress(SnapKVPress()),
    "bernoulli_compactor": BernoulliPress(press=CompactorPress()),
    "bernoulli_kvzip": BernoulliPress(press=KVzipPress()),
    "needle_kvzip_t001": NeedleFetchPress(theta=0.01),
    "needle_kvzip_t003": NeedleFetchPress(theta=0.03),
    "needle_kvzip_t01": NeedleFetchPress(theta=0.1),
    "needle_p_t001": NeedleFetchPress(theta=0.01, persist=True),
    "needle_p_t003": NeedleFetchPress(theta=0.03, persist=True),
    "needle_p_t01": NeedleFetchPress(theta=0.1, persist=True),
    "needle_q001_d01_r1": NeedleFetchPress(theta=0.1, theta_question=0.01, persist=True, reserve=0.01),
    "needle_q0003_d01_r2": NeedleFetchPress(theta=0.1, theta_question=0.003, persist=True, reserve=0.02),
    "needle_q001_d003_r1": NeedleFetchPress(theta=0.03, theta_question=0.01, persist=True, reserve=0.01),
    "needle_p_t003_r1": NeedleFetchPress(theta=0.03, persist=True, reserve=0.01),
    "needle_s2b16": NeedleFetchPress(theta=0.1, theta_question=0.01, persist=True, reserve=0.01, bits=2, n_pairs=16, pairs_per_128_dims=True),
    "needle_s2b16_q003": NeedleFetchPress(theta=0.1, theta_question=0.003, persist=True, reserve=0.01, bits=2, n_pairs=16, pairs_per_128_dims=True),
    "needle_s2b32": NeedleFetchPress(theta=0.1, theta_question=0.01, persist=True, reserve=0.01, bits=2, n_pairs=32),
    "needle_s2b16_e128": NeedleFetchPress(theta=0.1, theta_question=0.01, persist=True, reserve_entries=128, bits=2, n_pairs=16, pairs_per_128_dims=True),
    "needle_s2b32_e128": NeedleFetchPress(theta=0.1, theta_question=0.01, persist=True, reserve_entries=128, bits=2, n_pairs=32),
    "needle_s2b16_m": NeedleFetchPress(theta=0.1, theta_question=0.01, persist=True, reserve=0.01, reserve_entries=128, bits=2, n_pairs=16, pairs_per_128_dims=True),
    "needle_s2b32_m": NeedleFetchPress(theta=0.1, theta_question=0.01, persist=True, reserve=0.01, reserve_entries=128, bits=2, n_pairs=32),
    "needle_s2b16_mq003": NeedleFetchPress(theta=0.1, theta_question=0.003, persist=True, reserve=0.01, reserve_entries=128, bits=2, n_pairs=16, pairs_per_128_dims=True),
    # cached KVzip scores: identical results, one scoring pass per context shared by all arms
    "kvzip_c": CachedKVzipPress(),
    "bernoulli_kvzip_c": BernoulliPress(press=CachedKVzipPress()),
    "needle_final_c": NeedleFetchPress(bern=BernoulliPress(press=CachedKVzipPress()), theta=0.1, theta_question=0.003, persist=True, reserve=0.01, reserve_entries=128, bits=2, n_pairs=16, pairs_per_128_dims=True),
    # ablation: sketch and reserve NOT charged, so the draw equals bernoulli_kvzip_c's at the same ratio
    # (same seed): any difference from Bernoulli is the fetching alone, at the cost of the sketch on top
    "needle_uncharged_c": NeedleFetchPress(bern=BernoulliPress(press=CachedKVzipPress()), theta=0.1, theta_question=0.003, persist=True, reserve=0.01, reserve_entries=128, bits=2, n_pairs=16, pairs_per_128_dims=True, charge_overhead=False),
    # question-time fetch only: the decode trigger switched off (theta infinite), budget as needle_final_c
    "needle_qonly_c": NeedleFetchPress(bern=BernoulliPress(press=CachedKVzipPress()), theta=float("inf"), theta_question=0.003, persist=True, reserve=0.01, reserve_entries=128, bits=2, n_pairs=16, pairs_per_128_dims=True),
    # 16x ablation: half the sketch (8 RoPE pairs, 0.9% of K+V) to leave more kept budget
    "needle_p8_c": NeedleFetchPress(bern=BernoulliPress(press=CachedKVzipPress()), theta=0.1, theta_question=0.003, persist=True, reserve=0.01, reserve_entries=128, bits=2, n_pairs=8),
    # needle_final_c with a product-quantisation sketch in place of the 2-bit one (16 subspaces, codebook fit on
    # 4096 keys past the first 128, as in the skylight runtime), its codebook charged to the budget per context.
    # 4-bit codes cost about the 2-bit sketch's bytes (~1.6% of K+V at 45k tokens); 8-bit codes are the skylight
    # runtime's (~3.5%), which leaves nothing at 32x, so that arm runs at 8x and 16x only
    "needle_final_pq4_c": NeedleFetchPress(bern=BernoulliPress(press=CachedKVzipPress()), theta=0.1, theta_question=0.003, persist=True, reserve=0.01, reserve_entries=128, sketch="pq", pq_subspaces=16, pq_bits=4),
    "needle_final_pq8_c": NeedleFetchPress(bern=BernoulliPress(press=CachedKVzipPress()), theta=0.1, theta_question=0.003, persist=True, reserve=0.01, reserve_entries=128, sketch="pq", pq_subspaces=16, pq_bits=8),
    # final_eval: extra Bernoulli draws for LOFT, where every question of a task shares one context (one
    # draw per cell). Seed 0 is the arm above; s1/s2 differ only in the draw (same scores, same budget).
    **{f"bernoulli_kvzip_c_s{s}": BernoulliPress(press=CachedKVzipPress(), seed=s) for s in (1, 2)},
    **{f"needle_final_c_s{s}": NeedleFetchPress(bern=BernoulliPress(press=CachedKVzipPress(), seed=s), theta=0.1, theta_question=0.003, persist=True, reserve=0.01, reserve_entries=128, bits=2, n_pairs=16, pairs_per_128_dims=True) for s in (1, 2)},
    **{f"needle_uncharged_c_s{s}": NeedleFetchPress(bern=BernoulliPress(press=CachedKVzipPress(), seed=s), theta=0.1, theta_question=0.003, persist=True, reserve=0.01, reserve_entries=128, bits=2, n_pairs=16, pairs_per_128_dims=True, charge_overhead=False) for s in (1, 2)},
    # fast needle fetch: needle_final_c with Fast KVzip's gate scores in place of KVzip's (no 2% window, see
    # FastKVzipScorer); same budget, sketch, thresholds and reserve, and the same draw seeds
    "fast_needle_final": NeedleFetchPress(bern=BernoulliPress(press=FastKVzipScorer()), theta=0.1, theta_question=0.003, persist=True, reserve=0.01, reserve_entries=128, bits=2, n_pairs=16, pairs_per_128_dims=True),
    **{f"fast_needle_final_s{s}": NeedleFetchPress(bern=BernoulliPress(press=FastKVzipScorer(), seed=s), theta=0.1, theta_question=0.003, persist=True, reserve=0.01, reserve_entries=128, bits=2, n_pairs=16, pairs_per_128_dims=True) for s in (1, 2)},
    "vkvwr_compactor": VKvWRPress(press=CompactorPress()),
    "vkvwr_kvzip": VKvWRPress(press=KVzipPress()),
    "block_keydiff": BlockPress(press=KeyDiffPress(), block_size=128),
    "chunkkv": ChunkKVPress(press=SnapKVPress(), chunk_length=20),
    "critical_adakv_expected_attention": CriticalAdaKVPress(ExpectedAttentionPress(use_vnorm=False)),
    "critical_adakv_snapkv": CriticalAdaKVPress(SnapKVPress()),
    "critical_expected_attention": CriticalKVPress(ExpectedAttentionPress(use_vnorm=False)),
    "critical_snapkv": CriticalKVPress(SnapKVPress()),
    "cur": CURPress(),
    "duo_attention": DuoAttentionPress(),
    "duo_attention_on_the_fly": DuoAttentionPress(on_the_fly_scoring=True),
    "entropy_gated_chunkkv": EntropyGatedChunkKVPress(press=SnapKVPress()),
    "expected_attention": AdaKVPress(ExpectedAttentionPress(epsilon=1e-2)),
    # final_eval baseline: the authors' 2% window at every length (the stock 4096-token window fills the
    # whole budget at 32k / 8-32x and 128k / 32x); the stock press stays available as fastkvzip_w4096
    "fastkvzip": FastKVzipPress(window_size=None),
    "fastkvzip_w4096": FastKVzipPress(),
    "finch": FinchPress(),
    "keydiff": KeyDiffPress(),
    "kvcompose": KVComposePress(),
    "kvcompose_unstructured": KVComposePress(structured=False),
    "kvzip": KVzipPress(),
    "kvzip_plus": KVzipPress(kvzip_plus_normalization=True),
    "kvzap_linear": DMSPress(press=KVzapPress(model_type="linear")),
    "kvzap_mlp": DMSPress(press=KVzapPress(model_type="mlp")),
    "kvzap_mlp_head": KVzapPress(model_type="mlp"),
    "kvzap_mlp_layer": AdaKVPress(KVzapPress(model_type="mlp")),
    "lagkv": LagKVPress(),
    "lukv": LUKVPress(ExpectedAttentionPress(epsilon=2e-2), sink=4, window=1),
    "knorm": KnormPress(),
    "observed_attention": ObservedAttentionPress(),
    "pyramidkv": PyramidKVPress(),
    "qfilter": QFilterPress(),
    "random": RandomPress(),
    "RestoreKV": RestoreKVPress(),
    "RestoreKV_plus": RestoreKVPress(kvzip_plus_normalization=True),  # RestoreKV+ (KVzip+ scoring)
    "snap_think": ComposedPress([SnapKVPress(), ThinKPress()]),
    "snapkv": SnapKVPress(),
    "streaming_llm": StreamingLLMPress(),
    "think": ThinKPress(),
    "tova": TOVAPress(),
    "compactor": CompactorPress(),
    "adakv_compactor": AdaKVPress(CompactorPress()),
    "no_press": None,
    "cam_streaming_llm": CAMPress(base_press=StreamingLLMPress()),
    "cam_knorm": CAMPress(base_press=KnormPress()),
    "cam_adakv_snapkv": CAMPress(base_press=AdaKVPress(SnapKVPress())),
    "cam_tova": CAMPress(base_press=TOVAPress()),
    "decoding_knorm": DecodingPress(base_press=KnormPress()),
    "decoding_streaming_llm": DecodingPress(base_press=StreamingLLMPress()),
    "decoding_tova": DecodingPress(base_press=TOVAPress()),
    "decoding_qfilter": DecodingPress(base_press=QFilterPress()),
    "decoding_adakv_expected_attention_e2": DecodingPress(base_press=AdaKVPress(ExpectedAttentionPress(epsilon=1e-2))),
    "decoding_adakv_snapkv": DecodingPress(base_press=AdaKVPress(SnapKVPress())),
    "decoding_keydiff": DecodingPress(base_press=KeyDiffPress()),
    # MergingPress: merge-on-evict during prefill (values-only merge preserves RoPE keys)
    "merging_knorm": MergingPress(KnormPress()),
    "merging_snapkv": MergingPress(SnapKVPress()),
    "merging_expected_attention": MergingPress(ExpectedAttentionPress(epsilon=1e-2)),
    "merging_kvzap_mlp": MergingPress(KVzapPress(model_type="mlp")),
}
