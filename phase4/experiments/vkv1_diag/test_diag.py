"""Tests for diag_lib (CPU): python -m pytest -q test_diag.py"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.expanduser("~"), "phase4", "tests"))
import diag_lib as dl  # noqa: E402
from diag_lib import algo  # noqa: E402
from test_algo import synth  # noqa: E402  (the same synthetic heads as the algo tests)

DT = torch.float64


def test_prepare_order_reproduces_algo_prepare():
    w, r, tail, sure = synth(rows=4, seed=1, zero_r=2)
    ref = algo.prepare(w, r, tail, sure, delta=0.05, order="needle")
    got = dl.prepare_order(w, r, tail, sure, 0.05, ref.order)
    for name in ("beta", "V", "up", "dn", "Bt", "beta_T"):
        assert torch.allclose(getattr(got, name), getattr(ref, name)), name
    assert torch.equal(got.T, ref.T)


def test_chernoff_bounds_the_true_failure_rate():
    delta, N = 0.1, 6000
    w, r, tail, sure = synth(rows=6, n=300, seed=7)
    B = sure + torch.where(tail, w, torch.zeros((), dtype=DT)).sum(-1)
    plan = algo.prepare(w, r, tail, sure, delta=delta, order="grad")
    eps = [0.05, 0.1]
    ch = dl.chernoff_curves(plan, w, r, tail, eps)
    mc = dl.mc_curves(plan, w, r, tail, sure, B, eps, N, torch.Generator().manual_seed(3))
    tight = [plan.p_fail_tight(e) for e in eps]
    for c, m, t in zip(ch, mc, tight):
        assert bool((c + 4 * (m * (1 - m) / N).sqrt() + 1e-3 >= m).all()), "Chernoff below the true failure rate"
        # first passing k: Chernoff fetches no more than the tight Bernstein check (up to the lambda grid)
        kc, kt = dl.first_pass(c, delta, plan.n_tail), dl.first_pass(t, delta, plan.n_tail)
        assert bool((kc <= kt + 2).all()), (kc, kt)


def test_allocation_meets_mean_budget_and_is_cheaper_than_per_row():
    w, r, tail, sure = synth(rows=40, n=400, seed=9)
    plan = algo.prepare(w, r, tail, sure, delta=0.05, order="grad")
    lad = dl.ladder(w.shape[-1], "cpu")
    P = plan.p_fail_tight(0.1)
    cost = dl.expected_cost(plan, r, tail)
    k_row = dl.first_pass(P, 0.05, plan.n_tail)
    Pl, cl, kl = dl.with_row_k(P, cost, lad, k_row)
    idx = dl.allocate(Pl, cl, 0.05)
    assert float(Pl.clamp(max=1).gather(-1, idx[:, None]).mean()) <= 0.05 + 1e-12
    # the per-decision solution is one feasible choice, so the averaged allocation never costs more
    assert float(cl.gather(-1, idx[:, None]).sum()) <= float(cost.gather(-1, k_row[:, None]).sum()) + 1e-9
