"""The arms of the quality experiments, registered in the press registry of the evaluation harness.

    vkv_final_kvzip       VerifiedKV(C=KVzip, A=2-bit sketch)
    vkv_final_fastkvzip   VerifiedKV(C=Fast KVzip, A=2-bit sketch)
    vkv_final_perfect     VerifiedKV(C=KVzip, A=perfect sketch), with 2% of the cache charged for the sketch

The harness sets compression_ratio. The sample keeps 1 - compression_ratio - charge of all KV entries, including 64
sink and 64 local tokens. All settings are in verifiedkv/presets.py.
"""

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

import evaluate_registry  # noqa: E402  (evaluation/ is on sys.path, set by run_grid.py)
from kvpress.presses.cached_kvzip_press import CachedKVzipPress  # noqa: E402
from kvpress.presses.fastkvzip_press import FastKVzipScorer  # noqa: E402
from verifiedkv.presets import DELTA, EPS0, FIXED_CHARGE_PERFECT, SHORTLIST_HEAD, VkvFinalGridPress, build  # noqa: E402

ARMS = ("vkv_final_kvzip", "vkv_final_fastkvzip", "vkv_final_perfect")


def _make(scorer, **kw) -> VkvFinalGridPress:
    return build(scorer, **kw)


def register() -> dict:
    presses = {
        "vkv_final_kvzip": _make(CachedKVzipPress()),
        "vkv_final_fastkvzip": _make(FastKVzipScorer()),
        "vkv_final_perfect": _make(CachedKVzipPress(), plan_weights="perfect", fixed_charge=FIXED_CHARGE_PERFECT),
    }
    evaluate_registry.PRESS_REGISTRY.update(presses)
    return presses


def settings() -> dict:
    """Recorded in the cost.json of every cell."""
    return {"eps0": EPS0, "delta": DELTA, "shortlist_head": SHORTLIST_HEAD,
            "fixed_charge_perfect": FIXED_CHARGE_PERFECT, "vkv_vec": os.environ.get("VKV_VEC", "1"),
            "lambda_block": os.environ.get("VKV_LAMBDA_BLOCK", "8")}
