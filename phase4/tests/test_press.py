"""Tests for vkv1.press.vkv1_bias on synthetic tensors (no model). Needs kvpress on the path:
PYTHONPATH=~/src/kvpress-e5 python -m pytest tests/test_press.py"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
pytest.importorskip("kvpress")
from vkv1 import algo  # noqa: E402
from vkv1.press import new_stats, vkv1_bias  # noqa: E402

DT = torch.float64


def make(n_kv=2, G=2, ctx=300, extra=6, d=16, seed=0, eps0=0.2, delta=0.1, check="simple", order="needle",
         trunc=True):
    g = torch.Generator().manual_seed(seed)
    key = torch.randn(1, n_kv, ctx + extra, d, generator=g) * 1.2
    key[0, :, 40] *= 4  # a heavy entry
    tau = torch.rand(n_kv, ctx, generator=g, dtype=DT) ** 3
    r = (tau / tau.mean() * 0.08).clamp(max=1.0)
    r[:, :4] = 1.0
    keep = torch.rand(n_kv, ctx, generator=g, dtype=DT) < r
    bias = torch.where(keep, -torch.log(r), torch.tensor(-float("inf"), dtype=DT)).float()
    state = {"vkv1": {"eps0": eps0, "delta": delta, "check": check, "order": order, "trunc": trunc,
                      "sketch": "perfect", "persist": True},
             "bias": bias[None], "r": r[None], "keep": keep[None], "stats": new_stats(),
             "promoted": torch.zeros(1, n_kv, ctx, dtype=torch.bool), "layer_idx": 0, "first_layer": 0}
    return key, state, G, d


def reference(query, key, state, scaling, promoted):
    """Row-by-row Algorithm 2 with algo.py directly: bias [H, q_len, ctx] and the union of F per KV head."""
    cfg, r, keep = state["vkv1"], state["r"][0], state["keep"][0]
    n_kv, ctx = r.shape
    H, q_len = query.shape[1], query.shape[2]
    G = H // n_kv
    k_len = key.shape[2]
    out = torch.empty(H, q_len, ctx, dtype=DT)
    union = torch.zeros(n_kv, ctx, dtype=torch.bool)
    for hq in range(H):
        h = hq // G
        tail = (r[h] < 1) & ~promoted[h]
        for t in range(q_len):
            q = query[0, hq, t].double() * scaling
            s = key[0, h, :ctx].double() @ q
            se = key[0, h, ctx: k_len - q_len + t + 1].double() @ q
            m = torch.maximum(s.max(), se.max())
            w, wx = torch.exp(s - m), torch.exp(se - m).sum()
            sure = w[~tail].sum() + wx
            plan = algo.prepare(w, r[h], tail, sure, cfg["delta"], trunc=cfg["trunc"], order=cfg["order"])
            k, _ = algo.kstar(plan, cfg["eps0"], cfg["check"])
            F = algo.fetch_mask(plan, k) & tail
            union[h] |= F
            out[hq, t] = algo.attention_bias(r[h], tail, keep[h], plan.T, F)
    return out, union


@pytest.mark.parametrize("check,order,trunc", [("simple", "needle", True), ("tight", "grad", True),
                                               ("simple", "needle", False)])
def test_question_then_decode_matches_reference_and_persists(check, order, trunc):
    key, state, G, d = make(check=check, order=order, trunc=trunc)
    n_kv, ctx = state["r"].shape[1:]
    scaling = d ** -0.5
    g = torch.Generator().manual_seed(3)
    # question pass: 4 positions, the last 4 extra keys are the question tokens (2 earlier extra keys exist)
    q_len = 4
    query = torch.randn(1, n_kv * G, q_len, d, generator=g)
    kq = key[:, :, : ctx + 2 + q_len]
    ref, union = reference(query, kq, state, scaling, torch.zeros(n_kv, ctx, dtype=torch.bool))
    got = vkv1_bias(query, kq, state, scaling)[0]
    assert torch.allclose(got.double(), ref.float().double(), equal_nan=True)
    assert torch.equal(state["promoted"][0], union)
    assert union.any(), "fixture too easy: the question pass fetched nothing"
    # decode step: persisted entries are exact (bias 0) and leave the tail
    qd = torch.randn(1, n_kv * G, 1, d, generator=g)
    kd = key
    ref_d, union_d = reference(qd, kd, state, scaling, state["promoted"][0].clone())
    prev = state["promoted"][0].clone()
    got_d = vkv1_bias(qd, kd, state, scaling)[0]
    assert torch.allclose(got_d.double(), ref_d.float().double(), equal_nan=True)
    assert bool((got_d[:, 0][prev.repeat_interleave(G, 0)] == 0).all())  # [H, ctx] mask per query head
    assert torch.equal(state["promoted"][0], prev | union_d)
    st = state["stats"]
    assert st["question_decisions"] == n_kv * G * q_len and st["decode_decisions"] == n_kv * G
    # a new question resets persistence
    vkv1_bias(query, kq, state, scaling)
    assert torch.equal(state["promoted"][0], union)


def test_limits_fetch_all_and_no_fetch():
    key, state, G, d = make(eps0=0.0)
    n_kv, ctx = state["r"].shape[1:]
    query = torch.randn(1, n_kv * G, 1, d, generator=torch.Generator().manual_seed(5))
    got = vkv1_bias(query, key, state, d ** -0.5)
    assert bool((got == 0).all()), "eps0 = 0 must fetch every tail entry (bias 0 everywhere)"
    key, state, G, d = make(eps0=1e9, trunc=False)
    got = vkv1_bias(query, key, state, d ** -0.5)[0]
    bern = state["bias"][0].repeat_interleave(G, 0)[:, None]
    assert torch.equal(got, bern.expand_as(got)), "no fetch, no truncation = the Bernoulli bias"


def test_residency_equals_bernoullipress_at_4_sinks_no_window():
    """algo.residency(n_sink=4, window=0) reproduces BernoulliPress.compress_post's keep probabilities exactly."""
    from kvpress.presses import bernoulli_press as bp

    g = torch.Generator().manual_seed(21)
    scores = torch.rand(5, 1, 4, 2000, generator=g, dtype=DT) ** 5
    scores[..., :4] = 1.0  # KVzip marks its 4 sinks with score 1
    kept = 0.0625
    flat = scores.reshape(-1)
    budget = kept * flat.numel()
    c = bp._solve_c(flat, budget)
    ref = bp._keep_prob(flat, c).view(scores.shape)
    ref[..., :4] = 1.0
    if ref.sum().item() > budget:
        lo, hi = 0.0, c
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            rm = bp._keep_prob(flat, mid).view(scores.shape)
            rm[..., :4] = 1.0
            lo, hi = (lo, mid) if rm.sum().item() > budget else (mid, hi)
        ref = bp._keep_prob(flat, hi).view(scores.shape)
        ref[..., :4] = 1.0
    assert torch.equal(algo.residency(scores, kept, 4, 0), ref)
