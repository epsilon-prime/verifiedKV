"""CPU tests for the reference press:
- the certificate code gives k* consistent with the exact curve;
- the press computes Algorithm 2 row by row on the 2-bit sketch (reference built directly from verifiedkv/algo.py);
- the sketch, residency and budget behave as documented."""

import math
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from verifiedkv import chernoff  # noqa: E402
from verifiedkv import press_reference as pf  # noqa: E402
from verifiedkv.press_reference import algo, nfp  # noqa: E402

DT = torch.float64


def make(n_kv=2, G=2, ctx=300, extra=6, d=16, seed=0, eps0=0.3, delta=0.05, shortlist=0, plan_weights="sketch",
         fast=False):
    """A small cache: random keys with one heavy entry, keep probabilities, a draw, the 2-bit sketch, the state."""
    g = torch.Generator().manual_seed(seed)
    key = torch.randn(1, n_kv, ctx + extra, d, generator=g) * 1.2
    key[0, :, 40] *= 4  # a heavy entry
    tau = torch.rand(n_kv, ctx, generator=g, dtype=DT) ** 3
    r = (tau / tau.mean() * 0.08).clamp(max=1.0)
    r[:, :4] = 1.0
    keep = torch.rand(n_kv, ctx, generator=g, dtype=DT) < r
    ch = nfp.top_energy_pairs(key[:, :, :ctx], 2)  # [1, n_kv, 4]: 2 RoPE pairs of 8 (the 16-of-64 ratio)
    k_ctx = key[:, :, :ctx].gather(-1, ch[:, :, None, :].expand(-1, -1, ctx, -1))
    k_hat = nfp.quant_sketch(k_ctx, 2, 128)
    cfg = {"eps0": eps0, "delta": delta, "shortlist_head": shortlist, "plan_weights": plan_weights, "fast": fast,
           "eval_eps": 0.5}
    state = {"vkv_final": cfg, "r": r[None], "keep": keep[None], "k_hat": k_hat, "channels": ch,
             "stats": pf.new_stats(), "layer_idx": 0, "first_layer": 0}
    q = torch.randn(1, n_kv * G, 3, d, generator=torch.Generator().manual_seed(seed + 4))
    return key[:, :, : ctx + 3], state, q, d ** -0.5


def reference(query, key, state, scaling, C_head=None):
    """Algorithm 2 row by row with verifiedkv/algo.py and the exact Chernoff curve: bias [H, q_len, ctx]."""
    cfg, r, keep = state["vkv_final"], state["r"][0], state["keep"][0]
    n_kv, ctx = r.shape
    H, q_len = query.shape[1], query.shape[2]
    G = H // n_kv
    k_len = key.shape[2]
    ch, k_hat = state["channels"][0], state["k_hat"][0]
    out = torch.empty(H, q_len, ctx, dtype=DT)
    for hq in range(H):
        h = hq // G
        tail = r[h] < 1
        for t in range(q_len):
            qv = query[0, hq, t].float() * scaling
            s = key[0, h, :ctx].float() @ qv
            se = key[0, h, ctx: k_len - q_len + t + 1].float() @ qv
            if cfg["plan_weights"] == "perfect":
                s_hat = s
            else:
                s_hat = k_hat[h] @ qv[ch[h]]
                if C_head is not None:
                    s_hat = torch.where(C_head[h], s, s_hat)
            m = torch.maximum(torch.where(tail, s_hat, s).max(), se.max())
            w, wh = torch.exp((s - m).double()), torch.exp((s_hat - m).double())
            sure = torch.where(tail, torch.zeros((), dtype=DT), w).sum() + torch.exp((se - m).double()).sum()
            plan = algo.prepare(wh, r[h], tail, sure, cfg["delta"], trunc=True, order="grad")
            k = chernoff.first_pass(chernoff.chernoff_curves(plan, wh, r[h], tail, [cfg["eps0"]])[0], cfg["delta"],
                                    plan.n_tail)
            F = algo.fetch_mask(plan, k) & tail
            out[hq, t] = algo.attention_bias(r[h], tail, keep[h], plan.T, F)
    return out.float()


def test_fast_kstar_is_always_certified():
    for seed in range(6):
        g = torch.Generator().manual_seed(seed)
        n = 20000 if seed % 2 else 600  # the long rows exercise the head + Bennett path (head 16384)
        wh = torch.rand(4, n, generator=g, dtype=DT) ** 8
        wh[:, 7] *= 50  # heavy entries
        r = (torch.rand(4, n, generator=g, dtype=DT) * 0.15).clamp_min(1e-3)
        tail = r < 1
        plan = algo.prepare(wh, r, tail, wh.sum(-1) * 0.2, 0.05, order="grad")
        for eps0 in (0.3, 0.5):
            k, _, _ = chernoff.kstar_fast(plan, wh, r, tail, eps0, 0.05)
            P = chernoff.chernoff_curves(plan, wh, r, tail, [eps0])[0]
            screened = plan.p_fail_tight(eps0)[..., 0] <= 0.05
            ok = screened | (P.gather(-1, k[..., None])[..., 0] <= 0.05) | (k >= plan.n_tail)
            assert bool(ok.all()), (seed, eps0)


def test_sketch_plan_matches_row_by_row_reference():
    key, st, q, sc = make()
    got = pf.vkv_final_bias(q, key, st, sc)[0]
    assert torch.equal(got, reference(q, key, st, sc))


def test_perfect_plan_matches_row_by_row_reference():
    key, st, q, sc = make(plan_weights="perfect")
    assert torch.equal(pf.vkv_final_bias(q, key, st, sc)[0], reference(q, key, st, sc))


def test_shortlist_is_per_head_exact_and_capped():
    key, st, q, sc = make(shortlist=7)
    got = pf.vkv_final_bias(q, key, st, sc)[0]
    # the expected list: per KV head, the 7 tail tokens with the largest sketch score over all the head's rows
    r, ch, k_hat = st["r"][0], st["channels"][0], st["k_hat"][0]
    n_kv, ctx = r.shape
    G = q.shape[1] // n_kv
    C = torch.zeros(n_kv, ctx, dtype=torch.bool)
    for h in range(n_kv):
        qs = q[0, h * G:(h + 1) * G].reshape(-1, q.shape[-1]).float() * sc
        best = (qs[:, ch[h]] @ k_hat[h].T).amax(0)
        best = torch.where(r[h] < 1, best, torch.full_like(best, -math.inf))
        C[h, best.topk(7).indices] = True
    assert torch.equal(got, reference(q, key, st, sc, C_head=C & (r < 1)))
    assert st["stats"]["question_fwd_keys"] <= 7 * n_kv


def test_full_shortlist_is_the_perfect_plan():
    key, st, q, sc = make(shortlist=10 ** 6)
    key2, st2, q2, _ = make(plan_weights="perfect")
    assert torch.equal(pf.vkv_final_bias(q, key, st, sc), pf.vkv_final_bias(q2, key2, st2, sc))


def test_fast_and_exact_certificates_agree_on_the_fixture():
    key, st, q, sc = make(fast=True, shortlist=7)
    key2, st2, q2, _ = make(fast=False, shortlist=7)
    assert torch.equal(pf.vkv_final_bias(q, key, st, sc), pf.vkv_final_bias(q2, key2, st2, sc))


def test_defaults_and_budget():
    p = pf.vkv_final_algo()
    got = (p.kept, p.eps0, p.delta, p.shortlist_head, p.plan_weights, p.charge_sketch, p.fast, p.n_sink, p.window,
           p.bits, p.n_pairs, p.pairs_per_128_dims, p.group)
    assert got == (0.0625, 0.3, 0.05, 128, "sketch", True, True, 64, 64, 2, 16, True, 128)
    p.head_dim = 128  # set from the model in post_init_from_model
    assert abs(p.sketch_frac - 9 / 512) < 1e-12  # 32 channels x 2 bits + fp16 zero / scale per 128-token group
    assert abs(p.draw_frac - (0.0625 - 9 / 512)) < 1e-12
    assert pf.vkv_final_algo(plan_weights="perfect").draw_frac == 0.0625  # nothing to pay for


def test_other_states_go_to_the_previous_hook():
    seen = []
    prev = pf._prev_hook
    pf._prev_hook = lambda *a: seen.append(a) or "prev"
    try:
        assert pf._route(None, None, {"other": 1}, 1.0) == "prev" and len(seen) == 1
    finally:
        pf._prev_hook = prev
