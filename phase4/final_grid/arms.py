"""The final grid's arms, registered in kvpress's PRESS_REGISTRY so the harness (run_grid.py) runs them like any
other press. The settings live HERE ONLY, so every person and every job runs the same arms.

    vkv_final_kvzip       vkv_final (press_grid.VkvGridPress) on KVzip scores (CachedKVzipPress), 2-bit sketch,
                          per-KV-head shortlist 128; the sketch (1.76% of the cache) is paid from the budget
    vkv_final_fastkvzip   the same on Fast KVzip gate scores (FastKVzipScorer: no local window, as phase 3's
                          fast_needle_final)
    vkv_final_perfect     plan on the exact attention weights (an ideal sketch); FIXED_CHARGE_PERFECT = 2% of the
                          cache is paid from the budget as a stand-in for that sketch's memory

Budget: the harness sets compression_ratio (0.875 / 0.9375 / 0.96875); the Bernoulli draw then keeps
    1 - compression_ratio - charge
of all KV entries, sinks (64) and window (64) included, where charge = the 2-bit sketch's share (1.76% for every
model here: Qwen3.5's 256-dim heads get 32 pairs) or 2% for the perfect arm. At 32x: 1.37% (2-bit) / 1.125% (perfect).
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "phase4" / "vkv_final"))

import evaluate_registry  # noqa: E402  (kvpress/evaluation on sys.path, set by run_grid.py)
from kvpress.presses.bernoulli_press import BernoulliPress  # noqa: E402
from kvpress.presses.cached_kvzip_press import CachedKVzipPress  # noqa: E402
from kvpress.presses.fastkvzip_press import FastKVzipScorer  # noqa: E402
import press_grid  # noqa: E402  (routes the attention hook; VKV_VEC=1 by default: the vectorised certificate)

EPS0 = 0.05            # relative error target of the certificate
DELTA = 0.1            # allowed failure probability per row
SHORTLIST_HEAD = 128   # tail tokens per KV head re-scored exactly
FIXED_CHARGE_PERFECT = 0.02
ARMS = ("vkv_final_kvzip", "vkv_final_fastkvzip", "vkv_final_perfect")


@dataclass
class VkvFinalGridPress(press_grid.VkvGridPress):
    """press_grid.VkvGridPress whose budget follows the harness's compression_ratio (kvpress sets it on the press;
    NeedleFetchPress stores it in _total and re-derives it in post_init_from_model). ``fixed_charge`` replaces the
    sketch's own size as the memory paid out of the budget (None = the real 2-bit sketch, 0 when planning on exact
    weights). Same bias as VkvGridPress with kept = 1 - compression_ratio (test_arms.py)."""

    fixed_charge: float | None = None

    @property
    def draw_frac(self) -> float:
        if self.fixed_charge is not None:
            paid = self.fixed_charge
        else:
            paid = self.sketch_frac if (self.plan_weights == "sketch" and self.charge_sketch) else 0.0
        return (1.0 - self._total) - paid


def _make(scorer, **kw) -> VkvFinalGridPress:
    p = VkvFinalGridPress(bern=BernoulliPress(press=scorer), eps0=EPS0, delta=DELTA, shortlist_head=SHORTLIST_HEAD,
                          **kw)
    p.kept = None  # unused: the budget comes from compression_ratio (draw_frac above)
    return p


def register() -> dict:
    presses = {
        "vkv_final_kvzip": _make(CachedKVzipPress()),
        "vkv_final_fastkvzip": _make(FastKVzipScorer()),
        "vkv_final_perfect": _make(CachedKVzipPress(), plan_weights="perfect", fixed_charge=FIXED_CHARGE_PERFECT),
    }
    evaluate_registry.PRESS_REGISTRY.update(presses)
    return presses


def settings() -> dict:
    """Recorded in every cell's cost.json."""
    return {"eps0": EPS0, "delta": DELTA, "shortlist_head": SHORTLIST_HEAD,
            "fixed_charge_perfect": FIXED_CHARGE_PERFECT, "vkv_vec": os.environ.get("VKV_VEC", "1"),
            "lambda_block": os.environ.get("VKV_LAMBDA_BLOCK", "8")}
