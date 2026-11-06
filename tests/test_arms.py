"""Budgets, settings and registration of the experiment arms (CPU)."""
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "evaluation"))
sys.path.insert(0, str(HERE.parent / "experiments" / "quality"))
import arms  # noqa: E402
import evaluate  # noqa: E402
from verifiedkv import press as press_grid  # noqa: E402
from kvpress.presses.cached_kvzip_press import CachedKVzipPress  # noqa: E402
from kvpress.presses.fastkvzip_press import FastKVzipScorer  # noqa: E402

RATIOS = (0.875, 0.9375, 0.96875)


@pytest.mark.parametrize("ratio", RATIOS)
@pytest.mark.parametrize("head_dim", (128, 256))
def test_sketch_arms_draw_as_press(ratio, head_dim):
    """Same draw fraction as press_grid.vkv_grid(ratio) (kept = 1 - ratio, the 2-bit sketch paid), bit for bit."""
    ref = press_grid.vkv_grid(ratio)
    ref.head_dim = head_dim
    for scorer in (CachedKVzipPress(), FastKVzipScorer()):
        p = arms._make(scorer)
        p.compression_ratio = ratio  # what kvpress's EvaluationRunner does
        p.head_dim = head_dim
        p.compression_ratio = p._total  # what post_init_from_model does after reading head_dim
        assert p.draw_frac == ref.draw_frac
        assert abs(p.sketch_frac - 0.017578125) < 1e-12  # 32 pairs on 256-dim heads: the same share


@pytest.mark.parametrize("ratio", RATIOS)
def test_perfect_pays_two_percent(ratio):
    p = arms._make(CachedKVzipPress(), plan_weights="perfect", fixed_charge=arms.FIXED_CHARGE_PERFECT)
    p.compression_ratio = ratio
    assert p.draw_frac == (1.0 - ratio) - 0.02
    assert p.draw_frac > 0


def test_registered_through_the_harness():
    presses = arms.register()
    assert set(presses) == set(arms.ARMS)
    for name, p in presses.items():
        assert evaluate.PRESS_REGISTRY[name] is p
        assert (p.eps0, p.delta, p.shortlist_head) == (0.05, 0.1, 128)
        assert p.fast and p.n_sink == 64 and p.window == 64 and p.bits == 2
    assert isinstance(presses["vkv_final_fastkvzip"].bern.press, FastKVzipScorer)
    assert type(presses["vkv_final_kvzip"].bern.press) is CachedKVzipPress
    assert presses["vkv_final_perfect"].plan_weights == "perfect"
    # kvpress's runner sets the ratio on the registered object, the budget follows it
    cfg = evaluate.EvaluationConfig(dataset="ruler32k", data_dir="qa_1", press_name="vkv_final_kvzip",
                                    compression_ratio=0.96875)
    r = evaluate.EvaluationRunner(cfg)
    r._setup_press()
    assert r.press is presses["vkv_final_kvzip"] and r.press._total == 0.96875
