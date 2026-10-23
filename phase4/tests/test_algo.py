"""Tests for vkv1.algo against brute-force recomputation and Monte Carlo (CPU, float64, synthetic heads)."""

import itertools
import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vkv1 import algo  # noqa: E402

DT = torch.float64


def synth(rows=6, n=400, seed=0, n_sink=4, zero_r=0):
    """Heavy-tailed logits, keep probabilities loosely tied to the weights (some heavy entries get tiny r)."""
    g = torch.Generator().manual_seed(seed)
    s = torch.randn(rows, n, generator=g, dtype=DT) * 1.5
    spikes = torch.rand(rows, n, generator=g, dtype=DT) < 0.01
    s = s + spikes * (4 + 3 * torch.rand(rows, n, generator=g, dtype=DT))
    tau = torch.exp(0.5 * s[0] + torch.randn(n, generator=g, dtype=DT))  # one score per entry (shared by rows)
    tau = tau / tau.sum()
    r = (tau * n * 0.08).clamp(max=1.0)
    r[:n_sink] = 1.0
    if zero_r:
        r[n_sink:n_sink + zero_r] = 0.0
    w = torch.exp(s - s.amax(-1, keepdim=True))
    tail = r < 1
    wx = torch.rand(rows, generator=g, dtype=DT) * w.sum(-1) * 0.05  # question / generated keys
    sure = torch.where(tail, torch.zeros((), dtype=DT), w).sum(-1) + wx
    return w, r, tail, sure


def mse_of(w, r, tail, T):
    pos = tail & (r > 0)
    b = (w * (1 - r))[T & tail].sum()
    v = (w**2 * r * (1 - r))[T & pos].sum() + (w**2 * (1 - r) / r.clamp_min(1e-300))[pos & ~T].sum()
    return b**2 + v


@pytest.mark.parametrize("zero_r", [0, 5])
def test_truncation_is_argmin_of_mse_along_jump_order(zero_r):
    w, r, tail, _ = synth(zero_r=zero_r)
    T = algo.truncation(w, r, tail)
    for row in range(w.shape[0]):
        wr = w[row]
        pos = tail & (r > 0)
        jump = torch.where(pos, wr * (1 - r) / r.clamp_min(1e-300), torch.full_like(wr, -1.0))
        order = jump.argsort(descending=True)
        r0 = tail & (r == 0)
        mses = []
        for i in range(int(pos.sum()) + 1):
            Ti = torch.zeros_like(tail)
            Ti[order[:i]] = True
            mses.append(float(mse_of(wr, r, tail, (Ti & pos) | r0)))
        best = min(range(len(mses)), key=lambda i: (mses[i], i))
        assert int((T[row] & pos).sum()) == best, (row, int((T[row] & pos).sum()), best)
        assert bool((T[row] & r0 == r0).all())


def test_lemma1_truncation_caps_the_jumps():
    w, r, tail, sure = synth(rows=20, seed=3)
    plan = algo.prepare(w, r, tail, sure, delta=0.05)
    pos = tail & (r > 0)
    ht = pos & ~plan.T
    jump = torch.where(ht, w * (1 - r) / r.clamp_min(1e-300), torch.zeros_like(w)).amax(-1)
    assert bool((jump <= 2 * plan.beta_T * (1 + 1e-12)).all())
    assert bool((plan.up[..., 0] <= torch.maximum(2 * plan.beta_T, (w * (1 - r)).amax(-1))).all())
    assert bool((plan.b_plus_before >= plan.up[..., 0]).all())


@pytest.mark.parametrize("order,trunc", list(itertools.product(algo.ORDERS, [True, False])))
def test_sweep_matches_direct_recomputation(order, trunc):
    w, r, tail, sure = synth(rows=3, seed=1, zero_r=3)
    plan = algo.prepare(w, r, tail, sure, delta=0.1, trunc=trunc, order=order)
    pos = tail & (r > 0)
    for row in range(w.shape[0]):
        for k in (0, 1, 7, 50, int(plan.n_tail[row])):
            F = torch.zeros_like(tail)
            F[plan.order[row, :k]] = True
            U = tail & ~F
            T = plan.T[row]
            wr = w[row]
            beta = (wr * (1 - r))[T & U].sum()
            V = (wr**2 * r * (1 - r))[T & U & pos].sum() + (wr**2 * (1 - r) / r.clamp_min(1e-300))[U & pos & ~T].sum()
            up = torch.cat([(wr * (1 - r))[T & U & pos], (wr * (1 - r) / r.clamp_min(1e-300))[U & pos & ~T],
                            torch.zeros(1, dtype=DT)]).max()
            dn = torch.cat([(wr * r)[T & U & pos], wr[U & pos & ~T], torch.zeros(1, dtype=DT)]).max()
            for name, val in (("beta", beta), ("V", V), ("up", up), ("dn", dn)):
                got = getattr(plan, name)[row, k]
                assert torch.isclose(got, val, rtol=1e-9, atol=1e-300), (name, row, k, float(got), float(val))
            assert bool(F[~tail].sum() == 0)


def test_simple_curve_monotone_and_kstar_minimal():
    w, r, tail, sure = synth(rows=12, seed=2)
    for order in algo.ORDERS:
        plan = algo.prepare(w, r, tail, sure, delta=0.05, order=order)
        e = plan.eps_simple()
        assert bool((e[..., 1:] <= e[..., :-1]).all())
        for eps0 in (0.05, 0.1, 0.3, 1.0):
            k, nonmono = algo.kstar(plan, eps0, "simple")
            assert not bool(nonmono.any())
            assert bool((algo.curve_at(e, k) <= eps0).all())
            km1 = (k - 1).clamp_min(0)
            assert bool(((k == 0) | (algo.curve_at(e, km1) > eps0)).all())


def test_tight_never_fetches_more_than_simple():
    w, r, tail, sure = synth(rows=30, seed=4)
    for order, trunc in itertools.product(algo.ORDERS, [True, False]):
        plan = algo.prepare(w, r, tail, sure, delta=0.05, order=order, trunc=trunc)
        for eps0 in (0.05, 0.1, 0.3):
            ks, _ = algo.kstar(plan, eps0, "simple")
            kt, _ = algo.kstar(plan, eps0, "tight")
            assert bool((kt <= ks).all())


def test_attention_bias_implements_the_estimator():
    w, r, tail, sure = synth(rows=4, seed=5, zero_r=2)
    g = torch.Generator().manual_seed(9)
    Z = torch.rand(r.shape, generator=g, dtype=DT) < r
    plan = algo.prepare(w, r, tail, sure, delta=0.05)
    k, _ = algo.kstar(plan, 0.1)
    F = algo.fetch_mask(plan, k)
    bh = algo.estimate(w, r, tail, sure, Z, plan.T, F)
    bias = algo.attention_bias(r, tail, Z, plan.T, F)
    wx = sure - torch.where(tail, torch.zeros((), dtype=DT), w).sum(-1)
    via_bias = (torch.log(w) + bias).logsumexp(-1).exp() + wx
    assert torch.allclose(via_bias, bh, rtol=1e-10)


def test_fetching_everything_is_exact_and_error_curve_matches_estimate():
    w, r, tail, sure = synth(rows=4, seed=6, zero_r=2)
    Z = torch.rand(r.shape, generator=torch.Generator().manual_seed(1), dtype=DT) < r
    plan = algo.prepare(w, r, tail, sure, delta=0.05, order="grad")
    B = sure + torch.where(tail, w, torch.zeros((), dtype=DT)).sum(-1)
    e0 = algo.estimate(w, r, tail, sure, Z, plan.T, torch.zeros_like(w, dtype=torch.bool)) - B
    curve = algo.error_curve(plan, w, r, tail, Z, e0)
    for k in (0, 3, 40):
        kk = torch.full(B.shape, k)
        got = algo.estimate(w, r, tail, sure, Z, plan.T, algo.fetch_mask(plan, kk)) - B
        assert torch.allclose(algo.curve_at(curve, kk), got, rtol=1e-9, atol=1e-12)
    full = algo.estimate(w, r, tail, sure, Z, plan.T, tail.expand_as(w))
    assert torch.allclose(full, B, rtol=1e-12)


@pytest.mark.parametrize("check,order,trunc", list(itertools.product(algo.CHECKS, algo.ORDERS, [True, False])))
def test_theorem4_per_decision_failure_at_most_delta(check, order, trunc):
    """Monte Carlo over 4000 draws per row, exact weights: P(|B hat - B| > eps0 B) <= delta on every row."""
    delta, eps0, N = 0.1, 0.1, 4000
    w, r, tail, sure = synth(rows=10, n=300, seed=7)
    B = sure + torch.where(tail, w, torch.zeros((), dtype=DT)).sum(-1)
    plan = algo.prepare(w, r, tail, sure, delta=delta, order=order, trunc=trunc)
    k, _ = algo.kstar(plan, eps0, check)
    F = algo.fetch_mask(plan, k)
    Z = torch.rand(N, *r.shape, generator=torch.Generator().manual_seed(11), dtype=DT) < r
    bh = algo.estimate(w[None], r, tail, sure[None], Z[:, None, :], plan.T[None], F[None])  # [N, rows]
    fail = ((bh - B).abs() > eps0 * B).double().mean(0)
    tol = 4 * math.sqrt(delta * (1 - delta) / N)
    assert bool((fail <= delta + tol).all()), fail.tolist()
    assert bool((k > 0).any()), "fixture too easy: nothing fetched"


def test_design_rule_sampled_set_leak_breaks_the_guarantee():
    """The draft's red flag: choosing T from the draw (sampled entries at weight 1) biases the estimate down."""
    w, r, tail, sure = synth(rows=6, n=300, seed=8)
    B = sure + torch.where(tail, w, torch.zeros((), dtype=DT)).sum(-1)
    Z = torch.rand(2000, *r.shape, generator=torch.Generator().manual_seed(2), dtype=DT) < r
    leaked = algo.estimate(w[None], r, tail, sure[None], Z[:, None, :], (tail & Z)[:, None, :],
                           torch.zeros_like(w, dtype=torch.bool)[None])
    assert float(((leaked - B) / B).mean()) < -0.2


def test_batched_eps0_equals_scalar_calls():
    w, r, tail, sure = synth(rows=5, seed=12)
    eps = torch.tensor([0.05, 0.1, 0.3], dtype=DT)
    plan = algo.prepare(w[None].expand(3, -1, -1), r, tail, sure[None].expand(3, -1), delta=0.05)
    single = algo.prepare(w, r, tail, sure, delta=0.05)
    for check in algo.CHECKS:
        kb, _ = algo.kstar(plan, eps[:, None], check)
        for i, e in enumerate(eps.tolist()):
            ks, _ = algo.kstar(single, e, check)
            assert torch.equal(kb[i], ks)


@pytest.mark.parametrize("n_sink,window,kept", [(64, 64, 0.0625), (64, 64, 1 / 32), (4, 0, 0.0625)])
def test_residency_forces_sinks_and_window_within_budget(n_sink, window, kept):
    g = torch.Generator().manual_seed(13)
    tau = torch.rand(6, 4, 8000, generator=g, dtype=DT) ** 6  # [layers, KV heads, n], KVzip-like masses
    r = algo.residency(tau, kept, n_sink, window)
    assert bool((r[..., :n_sink] == 1).all())
    if window:
        assert bool((r[..., -window:] == 1).all())
    assert abs(float(r.mean()) - kept) < 1e-6 * kept + 1e-9
    assert bool(((r >= 0) & (r <= 1)).all())


def test_residency_when_forced_entries_exceed_the_budget_keeps_only_them():
    tau = torch.rand(2, 2, 1000, generator=torch.Generator().manual_seed(14), dtype=DT) ** 6
    r = algo.residency(tau, 0.05, 64, 64)  # 128 forced of 1000 = 12.8% > 5%
    assert bool((r[..., :64] == 1).all()) and bool((r[..., -64:] == 1).all())
    assert float(r[..., 64:-64].max()) < 1e-6
