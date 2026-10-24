"""Tests for hx_lib (CPU): python -m pytest -q test_hx.py"""

import itertools
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.expanduser("~"), "phase4", "tests"))
import hx_lib as hx  # noqa: E402
from hx_lib import algo, dl  # noqa: E402
from test_algo import synth  # noqa: E402

DT = torch.float64


def all_k(plan):
    n = plan.T.shape[-1]
    return torch.arange(n + 1).expand(*plan.n_tail.shape, -1).contiguous()


def test_m0_is_the_existing_chernoff_bound():
    w, r, tail, sure = synth(rows=5, n=300, seed=4)
    plan = algo.prepare(w, r, tail, sure, delta=0.05, order="grad")
    eps = [0.1, 0.3]
    ch = dl.chernoff_curves(plan, w, r, tail, eps)
    got = hx.hx_at(plan, w, r, tail, eps, all_k(plan), m=0)
    for c, g in zip(ch, got):
        assert torch.allclose(g.clamp(max=1), c.clamp(max=1), rtol=1e-7, atol=1e-10), (g - c).abs().max()


def small_case(seed):
    g = torch.Generator().manual_seed(seed)
    rows, n = 3, 14
    w = torch.exp(torch.randn(rows, n, generator=g, dtype=DT) * 1.5)
    r = 0.05 + 0.6 * torch.rand(n, generator=g, dtype=DT)
    r[:2] = 1.0
    r[2] = 0.0  # an r = 0 tail entry (bias only)
    tail = r < 1
    sure = torch.where(tail, torch.zeros((), dtype=DT), w).sum(-1)
    return w, r, tail, sure


def exact_fail(plan, w, r, tail, sure, eps, k):
    """Brute force over every draw of the random tail entries: P(|B hat - B| > eps B) with F = first k."""
    B = sure + torch.where(tail, w, torch.zeros((), dtype=DT)).sum(-1)
    F = algo.fetch_mask(plan, k)
    rnd = [i for i in range(w.shape[-1]) if 0 < float(r[i]) < 1]
    P = torch.zeros(w.shape[0], dtype=DT)
    for bits in itertools.product([False, True], repeat=len(rnd)):
        Z = r >= 1  # r = 1 always kept; r = 0 never
        pz = 1.0
        for i, b in zip(rnd, bits):
            Z = Z.clone()
            Z[i] = b
            pz *= float(r[i]) if b else 1 - float(r[i])
        err = (algo.estimate(w, r, tail, sure, Z.expand_as(w), plan.T, F) - B).abs()
        P += pz * (err > eps * B).double()
    return P


def test_exact_when_every_term_is_heavy():
    for seed in (0, 1):
        w, r, tail, sure = small_case(seed)
        plan = algo.prepare(w, r, tail, sure, delta=0.05, order="grad")
        for eps in (0.1, 0.25):
            for k in (0, 2, 5):
                kk = torch.full((w.shape[0], 1), k)
                got = hx.hx_at(plan, w, r, tail, [eps], kk, m=32, nb=16384)[0][:, 0]
                ex = exact_fail(plan, w, r, tail, sure, eps, kk[:, 0])
                assert bool((got >= ex - 1e-9).all()), (seed, eps, k, got, ex)  # a bound
                assert bool((got <= ex + 0.02).all()), (seed, eps, k, got, ex)  # and (nearly) exact


def test_bounds_monte_carlo_and_beats_chernoff():
    delta, N = 0.1, 6000
    w, r, tail, sure = synth(rows=6, n=300, seed=7)
    B = sure + torch.where(tail, w, torch.zeros((), dtype=DT)).sum(-1)
    plan = algo.prepare(w, r, tail, sure, delta=delta, order="grad")
    eps = [0.05, 0.1]
    ks = torch.minimum(dl.ladder(300, "cpu").expand(6, -1), plan.n_tail[:, None])
    got = hx.hx_at(plan, w, r, tail, eps, ks, m=32)
    mc = dl.mc_curves(plan, w, r, tail, sure, B, eps, N, torch.Generator().manual_seed(3))
    ch = dl.chernoff_curves(plan, w, r, tail, eps)
    for g, m, c in zip(got, mc, ch):
        m, c = m.gather(-1, ks), c.gather(-1, ks).clamp(max=1)
        assert bool((g + 4 * (m * (1 - m) / N).sqrt() + 1e-3 >= m).all()), "hx below the true failure rate"
        assert float(g.clamp(max=1).sum()) <= float(c.sum()) + 1e-9, "hx looser than Chernoff on average"


def test_kstar_passes_and_never_exceeds_chernoff_much():
    delta = 0.05
    w, r, tail, sure = synth(rows=8, n=400, seed=11)
    plan = algo.prepare(w, r, tail, sure, delta=delta, order="grad")
    eps = [0.1, 0.3]
    lad = dl.ladder(400, "cpu")
    ks, ps = hx.hx_kstar(plan, w, r, tail, eps, delta, lad, m=32)
    ch = dl.chernoff_curves(plan, w, r, tail, eps)
    for e, k, p, c in zip(eps, ks, ps, ch):
        again = hx.hx_at(plan, w, r, tail, [e], k[:, None], m=32)[0][:, 0]
        assert torch.allclose(again, p)
        assert bool(((p <= delta) | (k == plan.n_tail)).all())
        kc = dl.first_pass(c, delta, plan.n_tail)
        assert bool((k <= kc + 2).all()), (k, kc)


def test_store_is_capped_draw_free_and_keeps_highest_priority():
    H, R, n, cap = 2, 3, 50, 5
    g = torch.Generator().manual_seed(0)
    w = torch.rand(H, R, n, generator=g, dtype=DT)
    B = w.sum(-1)
    tail0 = torch.ones(H, n, dtype=torch.bool)
    tail0[:, :3] = False
    F = torch.rand(H, R, n, generator=g) < 0.3
    S = torch.zeros(H, n, dtype=torch.bool)
    prio = torch.zeros(H, n, dtype=DT)
    S, prio = hx.store_update(S, prio, F, w, B, tail0, cap)
    assert bool((S.sum(-1) <= cap).all()) and not bool((S & ~tail0).any())
    assert not bool((S & ~F.any(1)).any())  # only fetched entries enter
    p = torch.where(F, w / B[..., None], torch.zeros((), dtype=DT)).amax(1)
    for h in range(H):  # the kept ones are the top-cap by priority among the candidates
        cand = F.any(1)[h] & tail0[h]
        thr = p[h][cand].sort(descending=True).values[cap - 1]
        assert bool((p[h][S[h]] >= thr).all())
    F2 = torch.rand(H, R, n, generator=g) < 0.3  # a second forward: union of old store and new fetches, still capped
    S2, prio2 = hx.store_update(S, prio, F2, w, B, tail0, cap)
    assert bool((S2.sum(-1) <= cap).all()) and bool((prio2 >= prio).all())
    assert not bool((S2 & ~(S | F2.any(1))).any())
