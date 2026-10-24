"""The final Vkv1 configuration (2026-10-24): check "chern", order "grad", NO persistence, 64 sinks + 64-token window.
PYTHONPATH=~/src/kvpress-e5 python -m pytest -q test_final_config.py (CPU)"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.expanduser("~"), "phase4", "tests"))
import press_chern as pc  # noqa: E402
from press_chern import algo, dl  # noqa: E402
from test_press import make  # noqa: E402


def reference_chern(query, key, state, scaling):
    """Row-by-row chern + grad with an empty resident extra set (no persistence): bias [H, q_len, ctx], fetch union."""
    cfg, r, keep = state["vkv1"], state["r"][0], state["keep"][0]
    n_kv, ctx = r.shape
    H, q_len = query.shape[1], query.shape[2]
    G = H // n_kv
    k_len = key.shape[2]
    out = torch.empty(H, q_len, ctx, dtype=torch.float64)
    union = torch.zeros(n_kv, ctx, dtype=torch.bool)
    for hq in range(H):
        h = hq // G
        tail = r[h] < 1
        for t in range(q_len):
            q = query[0, hq, t].double() * scaling
            s = key[0, h, :ctx].double() @ q
            se = key[0, h, ctx: k_len - q_len + t + 1].double() @ q
            m = torch.maximum(s.max(), se.max())
            w, wx = torch.exp(s - m), torch.exp(se - m).sum()
            plan = algo.prepare(w, r[h], tail, w[~tail].sum() + wx, cfg["delta"], order="grad")
            k = dl.first_pass(dl.chernoff_curves(plan, w, r[h], tail, [cfg["eps0"]])[0], cfg["delta"], plan.n_tail)
            F = algo.fetch_mask(plan, k) & tail
            union[h] |= F
            out[hq, t] = algo.attention_bias(r[h], tail, keep[h], plan.T, F)
    return out, union


def test_chern_grad_nopersist_question_then_decode():
    key, state, G, d = make(eps0=0.2, delta=0.1, order="grad")
    state["vkv1"].update(check="chern", persist=False)
    n_kv, ctx = state["r"].shape[1:]
    scaling = d ** -0.5
    g = torch.Generator().manual_seed(8)
    q_len = 4
    query = torch.randn(1, n_kv * G, q_len, d, generator=g)
    kq = key[:, :, : ctx + 2 + q_len]
    ref, union = reference_chern(query, kq, state, scaling)
    got = pc.vkv1_bias(query, kq, state, scaling)[0]
    assert torch.allclose(got.double(), ref.float().double(), equal_nan=True)
    assert union.any(), "fixture too easy: the question pass fetched nothing"
    assert not state["promoted"].any(), "no persistence: nothing may stay resident after the question pass"
    # decode: starts from the draw alone, so it equals the reference with an empty resident set
    qd = torch.randn(1, n_kv * G, 1, d, generator=g)
    ref_d, _ = reference_chern(qd, key, state, scaling)
    got_d = pc.vkv1_bias(qd, key, state, scaling)[0]
    assert torch.allclose(got_d.double(), ref_d.float().double(), equal_nan=True)
    assert not state["promoted"].any()
    assert state["stats"]["persisted_extra"] == 0

