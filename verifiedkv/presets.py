"""The settings used in the paper, and make_press, the entry point for using VerifiedKV."""

from dataclasses import dataclass

from kvpress.presses.bernoulli_press import BernoulliPress
from kvpress.presses.cached_kvzip_press import CachedKVzipPress
from kvpress.presses.fastkvzip_press import FastKVzipScorer

from verifiedkv import press as press_grid

EPS0 = 0.05  # relative error target of the certificate
DELTA = 0.1  # allowed failure probability per decision
SHORTLIST_HEAD = 128  # tail tokens per KV head re-scored exactly
FIXED_CHARGE_PERFECT = 0.02  # memory charged for the perfect sketch


@dataclass
class VkvFinalGridPress(press_grid.VkvGridPress):
    """VkvGridPress whose budget follows compression_ratio. ``fixed_charge`` replaces the size of the sketch as the
    memory paid out of the budget (None uses the real 2-bit sketch)."""

    fixed_charge: float | None = None

    @property
    def draw_frac(self) -> float:
        if self.fixed_charge is not None:
            paid = self.fixed_charge
        else:
            paid = self.sketch_frac if (self.plan_weights == "sketch" and self.charge_sketch) else 0.0
        return (1.0 - self._total) - paid


def build(scorer, eps0=EPS0, delta=DELTA, shortlist_head=SHORTLIST_HEAD, **kw) -> VkvFinalGridPress:
    p = VkvFinalGridPress(bern=BernoulliPress(press=scorer), eps0=eps0, delta=delta, shortlist_head=shortlist_head,
                          **kw)
    p.kept = None  # unused: the budget comes from compression_ratio (draw_frac above)
    return p


def make_press(compression_ratio: float = 0.9375, scorer: str = "kvzip", sketch: str = "2-bit", eps0: float = EPS0,
               delta: float = DELTA, shortlist_head: int = SHORTLIST_HEAD) -> VkvFinalGridPress:
    """VerifiedKV(C=scorer, A=sketch). compression_ratio 0.9375 keeps 1/16 of the cache on the GPU.

    scorer: "kvzip" or "fastkvzip". sketch: "2-bit" or "perfect" (plans on the exact attention weights)."""
    scorers = {"kvzip": CachedKVzipPress, "fastkvzip": FastKVzipScorer}
    if scorer not in scorers:
        raise ValueError(f"scorer must be one of {sorted(scorers)}, got {scorer!r}")
    if sketch not in ("2-bit", "perfect"):
        raise ValueError(f"sketch must be '2-bit' or 'perfect', got {sketch!r}")
    kw = {"plan_weights": "perfect", "fixed_charge": FIXED_CHARGE_PERFECT} if sketch == "perfect" else {}
    p = build(scorers[scorer](), eps0=eps0, delta=delta, shortlist_head=shortlist_head, **kw)
    p.compression_ratio = compression_ratio
    return p
