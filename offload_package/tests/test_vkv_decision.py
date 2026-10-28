"""vkv.decision (capped top-M) == vkv.reference (Algorithm 2, full argsort; the harness's
kstar_fast for the Chernoff check)."""
import pytest
import torch

from offload_package.vkv import reference as ref
from offload_package.vkv import reference_chernoff as chern
from offload_package.vkv.decision import CHERN_MARGIN, attention_weights, decide, reference_rows

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _inputs(R, n, seed, *, p_zero=0.05, p_one=0.1, p_promoted=0.02, spread=4.0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    s = torch.randn(R, n, generator=g, dtype=torch.float64) * spread
    wh = (s - s.amax(-1, keepdim=True)).exp()
    u = torch.rand(R, n, generator=g, dtype=torch.float64)
    r = torch.rand(R, n, generator=g, dtype=torch.float64) ** 3        # many small r
    r = torch.where(u < p_zero, torch.zeros_like(r), r)
    r = torch.where((u >= p_zero) & (u < p_zero + p_one), torch.ones_like(r), r)
    promoted = torch.rand(R, n, generator=g) < p_promoted
    tail = (r < 1) & ~promoted
    sure = torch.rand(R, generator=g, dtype=torch.float64) * 5 + wh.sum(-1) * 0.05
    return wh.to(DEV), r.to(DEV), tail.to(DEV), sure.to(DEV)


def _reference(wh, r, tail, sure, eps0, delta, check, order, trunc, k_max=1024):
    plan, k, _ = reference_rows(wh, r, tail, sure, eps0=eps0, delta=delta, check=check,
                                order=order, trunc=trunc, k_max=k_max)
    F = ref.fetch_mask(plan, k) & tail
    return plan, k, F


def _fmask(dec, n):
    m = torch.zeros(dec.F.shape[0], n, dtype=torch.bool, device=dec.F.device)
    rows, cols = dec.fetch_valid().nonzero(as_tuple=True)
    m[rows, dec.F[rows, cols]] = True
    return m


@pytest.mark.parametrize("check", ["chern", "simple", "tight"])
@pytest.mark.parametrize("order", ["needle", "grad"])
@pytest.mark.parametrize("trunc", [True, False])
@pytest.mark.parametrize("caps", [(4096, 1024), (8, 4), (1, 1), (4096, 16384)])
def test_decide_matches_reference(check, order, trunc, caps):
    t_max, k_max = caps
    for seed, (R, n, eps0) in enumerate([(16, 3000, 0.3), (8, 700, 0.05), (12, 2000, 0.6)]):
        wh, r, tail, sure = _inputs(R, n, seed)
        plan, k, F = _reference(wh, r, tail, sure, eps0, 0.05, check, order, trunc, k_max)
        dec = decide(wh, r, tail, sure, eps0=eps0, delta=0.05, check=check, order=order,
                     trunc=trunc, t_max=t_max, k_max=k_max)
        assert torch.equal(dec.T, plan.T), (seed, (dec.T != plan.T).sum())
        assert torch.equal(dec.k, k), (seed, dec.k, k)
        assert torch.equal(_fmask(dec, n) & tail, F), seed
        assert torch.allclose(dec.Bt, plan.Bt)
        assert torch.allclose(dec.beta_k, ref.curve_at(plan.beta, k), rtol=1e-9, atol=1e-12)


def test_caps_force_fallback_and_stay_exact():
    wh, r, tail, sure = _inputs(16, 4000, 7, spread=0.5)    # flat weights: long T*, large k*
    plan, k, F = _reference(wh, r, tail, sure, 0.02, 0.05, "simple", "needle", True)
    dec = decide(wh, r, tail, sure, eps0=0.02, delta=0.05, check="simple", order="needle",
                 t_max=16, k_max=16)
    assert bool(dec.fallback.any())
    assert torch.equal(dec.T, plan.T) and torch.equal(dec.k, k)
    assert torch.equal(_fmask(dec, 4000) & tail, F)


@pytest.mark.parametrize("k_max", [16, 256, 16384])
def test_chern_k_is_certified_by_the_exact_curve(k_max):
    """Every k* of an unscreened row passes the exact float64 Chernoff curve (head + Bennett and
    the float32 terms only ever make the bound larger, the margin covers the rounding)."""
    for seed, (R, n, eps0, spread) in enumerate([(24, 3000, 0.3, 4.0), (16, 1200, 0.1, 1.0),
                                                   (16, 2500, 0.5, 8.0)]):
        wh, r, tail, sure = _inputs(R, n, 50 + seed, spread=spread)
        dec = decide(wh, r, tail, sure, eps0=eps0, delta=0.05, k_max=k_max)
        plan = ref.prepare(wh, r, tail, sure, 0.05, trunc=True, order="grad")
        P = chern.chernoff_curves(plan, wh, r, tail, [eps0])[0]          # exact, float64
        live = ~dec.screened
        assert bool(live.any())
        assert bool((P.gather(-1, dec.k[:, None])[:, 0][live] <= 0.05).all()), seed
        # the screen is a valid (Bernstein) bound at k = 0
        assert bool((plan.p_fail_tight(eps0)[:, 0][dec.screened] <= 0.05).all())


def test_chern_bennett_head_is_exercised_and_exact():
    """A head shorter than the tail (Bennett on the rest) equals the harness's kstar_fast with the
    same head. Rows that pass within the head pay Bennett's slack (k >= the exact-curve k); rows
    that pass nowhere in it take the reference's exact full curve (``fallback``)."""
    wh, r, tail, sure = _inputs(32, 4000, 21, spread=3.0)
    plan = ref.prepare(wh, r, tail, sure, 0.05, trunc=True, order="grad")
    k_h = chern.kstar_fast(plan, wh, r, tail, 0.3, 0.05, head=128)[0]
    k_full = chern.kstar_fast(plan, wh, r, tail, 0.3, 0.05, head=0)[0]
    dec = decide(wh, r, tail, sure, eps0=0.3, delta=0.05, k_max=128)
    assert torch.equal(dec.k, k_h)
    inh = ~dec.fallback
    assert bool(inh.any()) and bool(dec.fallback.any())
    assert bool((k_h[inh] >= k_full[inh]).all()) and bool((k_h[inh] > k_full[inh]).any())
    assert torch.equal(k_h[dec.fallback], k_full[dec.fallback])


def test_empty_tail_and_all_r1():
    R, n = 4, 300
    wh = torch.rand(R, n, dtype=torch.float64, device=DEV)
    r = torch.ones(R, n, dtype=torch.float64, device=DEV)
    tail = torch.zeros(R, n, dtype=torch.bool, device=DEV)
    dec = decide(wh, r, tail, wh.sum(-1), eps0=0.3, delta=0.05)
    assert int(dec.k.sum()) == 0 and dec.WF == 0 and not bool(dec.T.any())


def test_attention_weights_match_reference_bias():
    wh, r, tail, sure = _inputs(8, 1500, 3)
    g = torch.Generator(device="cpu").manual_seed(11)
    Z = (torch.rand(r.shape, generator=g, dtype=torch.float64).to(DEV) < r)
    plan, k, F = _reference(wh, r, tail, sure, 0.3, 0.05, "simple", "needle", True)
    bias = ref.attention_bias(r, tail, Z, plan.T, F)
    w = attention_weights(r, tail, tail & Z, plan.T, F)
    keep = torch.isfinite(bias)
    assert torch.equal(keep, w > 0)
    assert torch.allclose(w[keep].double().log(), bias[keep], rtol=1e-6, atol=1e-6)
