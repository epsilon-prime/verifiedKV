"""Vkv on the per-head token engine == the research harness (vkv_final) step by step.

A small engine + staging store runs the Vkv round (Algorithm 1: residency with the sketch paid
from the budget, the draw, the sketch) and decode steps through ``VkvDecoder`` (shortlist,
Algorithm 2 with the Chernoff certificate, the sparse kernel). Each step is compared with
``harness_layer``: ``press_final.vkv_final_bias``'s math over the FULL original KV (the harness
never moves anything), followed by a float64 softmax with the additive bias. Nothing persists:
every step starts again from the draw.
"""
from __future__ import annotations

import math
import os

import pytest
import torch

from offload_package.integration.token_mode_heads import HeadTokenOffloadEngine
from offload_package.offload.manager import KVOffloadManager
from offload_package.scoring.paged_eviction import PagedEvictionScorer
from offload_package.staging.pool import KVStagingPool
from offload_package.staging.residency import ResidencyTable
from offload_package.staging.token_store import TokenStagingStore
from offload_package.vkv import reference as ref
from offload_package.vkv import reference_chernoff as chern
from offload_package.vkv.state import VkvConfig, VkvState, prompt_crc

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
P, STAGE, PAGES, L, H, G, D = 8, 28, 52, 2, 2, 2, 64
CTX, GEN = 150, 5                       # context (prompt) and tokens written before the round
Hq = H * G
SM = 1.0 / math.sqrt(D)


def _ops():
    try:
        from sparse_optimized.cuda_ops import get_sparse_decode_ops
        return get_sparse_decode_ops()
    except Exception as e:  # pragma: no cover
        pytest.skip(f"sparse decode kernel unavailable: {e}")


def _world(cfg: VkvConfig, seed=0, kernels=True, stage=STAGE):
    torch.manual_seed(seed)
    n = L * PAGES * H * P * 2 * D
    buf = (torch.randn(n, device="cuda") * 0.6).to(torch.float16)
    caches = list(buf.view(L, PAGES, H, P, 2 * D).unbind(0))
    cpu_cache = torch.zeros((L, 4, 2, P, H, D), dtype=torch.float16, pin_memory=True)
    res = ResidencyTable()
    mgr = KVOffloadManager(scorer=PagedEvictionScorer(), residency=res, gpu_k_caches=caches,
                           gpu_v_caches=caches, cpu_cache=cpu_cache, gpu_budget_blocks=0,
                           kv_layout="HND")
    pool = KVStagingPool(num_slots=stage, staging_base_page=PAGES - stage, gpu_k_caches=caches,
                         gpu_v_caches=caches, cpu_cache=cpu_cache, residency=res,
                         offload_manager=mgr)
    eng = HeadTokenOffloadEngine(kv_caches=caches, page_size=P, max_len=256, num_req_slots=2,
                                 num_cpu_tokens=512, scope="global")
    eng.vkv = VkvState(eng, cfg)
    store = TokenStagingStore.for_token_engine(pool, eng)
    if not kernels:
        store._kern = False
    eng.add_request("r")
    eng.set_pages("r", list(range(PAGES - stage)))
    eng.on_step("r", 0, CTX + GEN)
    eng.on_step("r", CTX + GEN, 0)
    s = eng.slot_of("r")
    g = torch.Generator(device="cuda").manual_seed(seed + 1)
    eng.score[:, s, :CTX + GEN] = torch.rand(L, CTX + GEN, H, device="cuda", generator=g) ** 2
    # Needles: a few strong keys per (layer, head) along a direction the queries share.
    dirs = torch.randn(L, H, D, device="cuda")
    for l in range(L):
        for h in range(H):
            pos = torch.randperm(CTX, device="cuda", generator=g)[:6]   # distinct: one write per cell
            c = eng.cell[l, s, pos, h].long()
            eng.kv_rows[c, :D] = (dirs[l, h] * 1.5 + 0.3 * torch.randn(6, D, device="cuda")).half()
    full = {"K": [], "V": []}
    for l in range(L):
        c = eng.cell[l, s, :CTX + GEN].long()                             # [n, H]
        rows = eng.kv_rows[c]                                             # [n, H, 2D]
        full["K"].append(rows[..., :D].permute(1, 0, 2).float().clone())  # [H, n, D]
        full["V"].append(rows[..., D:].permute(1, 0, 2).float().clone())
    prompt = list(range(1000, 1000 + CTX))
    eng.vkv.set_prompt("r", prompt)
    return eng, store, full, dirs, prompt


def _draw_frac(cfg):
    """kvpress: kept - sketch_bytes_per_entry(D, group, bits, pairs) / (D * KV_BYTES_PER_CHANNEL)."""
    ch = 2 * cfg.pairs_for(D)
    return cfg.kappa - (cfg.bits * ch / 8 + 2 * ch * 2 / cfg.group) / (D * 4)


def _draw(r, prompt, seed):
    """VkvFinalPress._compress_post's draw: BernoulliPress._generator masks the seed to 31 bits."""
    gen = torch.Generator(device="cuda").manual_seed(
        (prompt_crc(prompt) ^ (seed * 7919)) & 0x7FFFFFFF)
    return (torch.rand(r.reshape(-1).shape, generator=gen, dtype=torch.float64,
                       device="cuda") < r.reshape(-1)).view_as(r)


def _harness_residency(eng, cfg, prompt, rid="r", ctx=CTX):
    """VkvFinalPress._compress_post: r and the draw, [L, H, ctx]."""
    s = eng.slot_of(rid)
    tau = eng.score[:, s, :ctx].permute(0, 2, 1).double()
    tau = ref.transform_scores_for_sampling(tau.reshape(-1)).view_as(tau)
    r = ref.residency(tau, _draw_frac(cfg), cfg.n_sink, cfg.window)
    return r, _draw(r, prompt, cfg.seed)


def harness_layer(qs, K, V, r, Z, kh, ch, cfg, ctx=CTX):
    """vkv_final_bias for one layer + the biased softmax. qs [Hq, D] (scaled); K, V [H, n, D] over
    context + generated; r, Z [H, ctx]; kh [H, ctx, nch]; ch [H, nch]. The certificate is
    kstar_fast with head = cfg.k_max (the harness's 16384 whenever k_max >= ctx)."""
    q = qs.view(H, G, D)
    s_all = q @ K.transpose(-1, -2)                                       # [H, G, n]
    s, se = s_all[..., :ctx], s_all[..., ctx:]
    tail = r < 1
    q_s = q.gather(-1, ch[:, None, :].expand(H, G, ch.shape[-1]))
    s_hat = q_s @ kh.transpose(-1, -2)
    if cfg.shortlist:                                                     # (a) shortlist
        best = torch.where(tail, s_hat.amax(1), torch.full_like(s_hat[:, 0], -float("inf")))
        idx = best.topk(min(cfg.shortlist, ctx), -1).indices
        C_head = torch.zeros(H, ctx, dtype=torch.bool, device="cuda").scatter_(-1, idx, True) & tail
        s_hat = torch.where(C_head[:, None], s, s_hat)
    m = torch.maximum(torch.where(tail[:, None], s_hat, s).amax(-1), se.amax(-1))[..., None]
    w = torch.exp((s - m).double())
    wh = torch.exp((s_hat - m).double())
    wx = torch.exp((se - m).double()).sum(-1)
    sure = torch.where(tail[:, None], torch.zeros((), dtype=torch.float64, device="cuda"),
                       w).sum(-1) + wx
    plan = ref.prepare(wh, r[:, None], tail[:, None], sure, cfg.delta, trunc=cfg.trunc,
                       order=cfg.order)
    if cfg.check == "chern":
        k = chern.kstar_fast(plan, wh, r[:, None], tail[:, None], cfg.eps0, cfg.delta,
                             head=cfg.k_max)[0]
    else:
        k, _ = ref.kstar(plan, cfg.eps0, cfg.check)
    F = ref.fetch_mask(plan, k) & tail[:, None]
    bias = ref.attention_bias(r[:, None], tail[:, None], Z[:, None], plan.T, F)
    logits = torch.cat([s.double() + bias, se.double()], -1)
    p = torch.softmax(logits, -1)
    out = torch.einsum("hgn,hnd->hgd", p, V.double())
    return out.reshape(Hq, D), k.reshape(-1), F


def _sketch_ref(Kctx, cfg, ctx=CTX):
    """_attach_sketch: channels + dequantised sketch of the context keys, per layer [H, ...]."""
    k4 = Kctx.unsqueeze(0)
    ch = ref.top_energy_pairs(k4.half(), cfg.pairs_for(D))
    kc = k4.half().gather(-1, ch[:, :, None, :].expand(-1, -1, ctx, -1))
    return ref.quant_sketch(kc, cfg.bits, cfg.group)[0], ch[0]


_C = dict(n_sink=4, window=4, group=32)
CFGS = [
    VkvConfig(kappa=0.3, eps0=0.3, shortlist=16, **_C),                  # vkv_final defaults
    VkvConfig(kappa=0.2, eps0=0.1, shortlist=16, k_max=8, t_max=4, **_C),   # Bennett head, caps
    VkvConfig(kappa=0.25, eps0=0.05, shortlist=0, check="tight", order="needle", t_max=4,
              k_max=3, **_C),
    VkvConfig(kappa=0.3, eps0=0.08, shortlist=32, trunc=False, k_max=2, **_C),
    VkvConfig(kappa=0.2, eps0=0.2, shortlist=16, check="simple", order="needle", **_C),
    VkvConfig(kappa=0.15, eps0=0.05, shortlist=8, charge_sketch=False, seed=3, **_C),
]


@pytest.mark.parametrize("ci", range(len(CFGS)))
def test_round_matches_harness(ci):
    cfg = CFGS[ci]
    eng, store, full, _, prompt = _world(cfg)
    if cfg.charge_sketch:
        assert abs(cfg.draw_frac(D) - _draw_frac(cfg)) < 1e-15
    else:
        assert cfg.draw_frac(D) == cfg.kappa
    r_ref, keep_ref = _harness_residency(eng, cfg, prompt)
    if not cfg.charge_sketch:
        tau = eng.score[:, eng.slot_of("r"), :CTX].permute(0, 2, 1).double()
        r_ref = ref.residency(ref.transform_scores_for_sampling(tau.reshape(-1)).view_as(tau),
                              cfg.kappa, cfg.n_sink, cfg.window)
        keep_ref = _draw(r_ref, prompt, cfg.seed)
    s = eng.slot_of("r")
    freed = eng.run_round("r")
    st = eng.vkv
    # r: stored float32, tail-ness preserved
    r32 = st.r[:, s, :CTX].permute(0, 2, 1)
    assert torch.equal((r32 < 1), (r_ref < 1))
    assert torch.allclose(r32.double(), r_ref, rtol=1e-6, atol=1e-7)
    assert bool((st.r[:, s, CTX:] == 1).all())
    # cells: kept = draw, generated always resident; evicted rows hold the original KV
    cell = eng.cell[:, s, :CTX + GEN]
    assert torch.equal((cell[:, :CTX] >= 0).permute(0, 2, 1), keep_ref)
    assert bool((cell[:, CTX:] >= 0).all())
    for l in range(L):
        c = eng.cell[l, s, :CTX + GEN]
        got = torch.zeros(H, CTX + GEN, 2 * D, dtype=torch.float16, device="cuda")
        res = (c >= 0).T
        got[res] = eng.kv_rows[c.T[res].long()]
        cpu = eng.cpu_row[l, s, :CTX + GEN].T
        off = cpu >= 0
        assert torch.equal(off, ~res)
        got[off] = eng.cpu_pool[cpu[off].long().cpu()].cuda()
        assert torch.equal(got[..., :D].float(), full["K"][l])
        assert torch.equal(got[..., D:].float(), full["V"][l])
        # sketch == harness k_hat (bit-identical), channels identical
        kh_ref, ch_ref = _sketch_ref(full["K"][l][:, :CTX], cfg)
        assert torch.equal(st.sketch.channels[l, s], ch_ref)
        kh = st.sketch.k_hat(l, torch.tensor([s], device="cuda"), CTX)[0]
        assert torch.equal(kh, kh_ref)
    assert eng.run_round("r") == []                                      # once per request
    assert isinstance(freed, list)


def test_draw_seed_is_masked_to_31_bits():
    """A seed whose crc ^ seed * 7919 has bit 31 set: the draw is the masked generator's (the
    harness's), not the unmasked one's."""
    from offload_package.vkv.state import draw_seed
    prompt = list(range(1000, 1000 + CTX))
    crc = prompt_crc(prompt)
    seed = next(x for x in range(1, 10_000) if (crc ^ (x * 7919)) >= 2 ** 31)
    assert draw_seed(crc, seed) == (crc ^ (seed * 7919)) & 0x7FFFFFFF
    cfg = VkvConfig(kappa=0.3, seed=seed, **_C)
    eng, store, full, _, prompt = _world(cfg)
    r_ref, keep_ref = _harness_residency(eng, cfg, prompt)
    eng.run_round("r")
    got = (eng.cell[:, eng.slot_of("r"), :CTX] >= 0).permute(0, 2, 1)
    assert torch.equal(got, keep_ref)
    gen = torch.Generator(device="cuda").manual_seed(crc ^ (seed * 7919))     # unmasked
    other = (torch.rand(r_ref.reshape(-1).shape, generator=gen, dtype=torch.float64,
                        device="cuda") < r_ref.reshape(-1)).view_as(r_ref)
    assert not torch.equal(got, other)


@pytest.mark.parametrize("ci", range(len(CFGS)))
def test_chern_split_is_bit_identical(ci, monkeypatch):
    """The certificate split over entry-block and lambda-part programs gives exactly the
    single-program kernel's k*, statistics and output."""
    import offload_package.vkv.decode as dmod
    cfg = CFGS[ci]
    outs, reps = [], []
    for split, eb in ((1, 512), (2, 512), (32, 64)):        # eb 64: several entry blocks
        monkeypatch.setattr(dmod, "CHERN_SPLIT", split)
        monkeypatch.setattr(dmod, "_CHERN_EB", eb)
        eng, store, full, dirs, prompt = _world(cfg, seed=ci)
        eng.run_round("r")
        dec = _decoder(eng, store)
        eng.set_decode_rows(["r"])
        g = torch.Generator(device="cuda").manual_seed(7)
        o = []
        for step in range(3):
            dec.begin_step(["r"])
            q = (torch.randn(1, Hq, D, device="cuda", generator=g) * 0.5 +
                 2.0 * dirs[0].repeat_interleave(G, 0)).half()
            for l in range(L):
                o.append(dec.layer(l, q, SM)[0].float())
        outs.append(torch.stack(o))
        reps.append(eng.vkv.stats.clone())
    for o, rp in zip(outs[1:], reps[1:]):
        assert torch.equal(o, outs[0]) and torch.equal(rp, reps[0])


def _decoder(eng, store, path="fused"):
    from offload_package.vkv.decode import VkvDecoder
    dec = VkvDecoder(eng, store, _ops())
    dec.torch_path = path == "torch"
    assert dec._pool_mapped          # the fused shortlist reads cold keys from the pinned pool
    return dec


@pytest.mark.parametrize("ci", range(len(CFGS)))
@pytest.mark.parametrize("kernels", [True, False])
@pytest.mark.parametrize("path", ["fused", "torch", "cuda"])
@pytest.mark.parametrize("compact", [True, False])
def test_decode_matches_harness(ci, kernels, path, compact):
    """Every decode path's output = the harness's softmax with Algorithm 2 (e)'s bias
    (``reference.attention_bias``: 0 on exact / F / S cap T*, log 1/r on the other sampled
    entries, -inf on the dropped ones), with and without compaction."""
    from dataclasses import replace
    cfg = replace(CFGS[ci], compact=compact)
    if path == "cuda" and not (cfg.order == "grad" and cfg.check == "chern" and cfg.trunc):
        pytest.skip("the CUDA decision implements the vkv_final defaults only")
    if path != "fused" and not kernels:
        pytest.skip("the staging kernels switch only matters to the fused path's fetch")
    eng, store, full, dirs, prompt = _world(cfg, seed=ci, kernels=kernels)
    s = eng.slot_of("r")
    dec = _decoder(eng, store, "torch" if path == "torch" else "fused")
    dec.cuda = path == "cuda"
    calls = []
    if dec.cuda:
        orig = dec._layer_cuda
        dec._layer_cuda = lambda *a, **k: calls.append(1) or orig(*a, **k)
    eng.set_decode_rows(["r"])
    g = torch.Generator(device="cuda").manual_seed(100 + ci)

    def queries():
        q = torch.randn(1, Hq, D, device="cuda", generator=g) * 0.5
        q += 2.0 * dirs[:, :, None].expand(L, H, G, D).reshape(L, Hq, D)[0]
        return q.half()

    # Before the round: dense, exact.
    dec.begin_step(["r"])
    q = queries()
    out, _ = dec.layer(0, q, SM)
    qd = q[0].float().view(H, G, D)
    p = torch.softmax((qd @ full["K"][0].transpose(-1, -2)) * SM, -1)
    dense = torch.einsum("hgn,hnd->hgd", p, full["V"][0]).reshape(Hq, D)
    assert torch.allclose(out[0].float(), dense, atol=2e-3, rtol=1e-2)

    r_ref, keep_ref = _harness_residency(eng, cfg, prompt)
    eng.run_round("r")
    if not cfg.charge_sketch:
        keep_ref = (eng.cell[:, s, :CTX] >= 0).permute(0, 2, 1)   # (checked by the round test)
    st = eng.vkv
    r_used = st.r[:, s, :CTX].permute(0, 2, 1).double()                # the stored (fp32) r
    r_after = st.r[:, s].clone()
    sk = [_sketch_ref(full["K"][l][:, :CTX], cfg) for l in range(L)]
    total_k = 0
    for step in range(3):
        dec.begin_step(["r"])
        for l in range(L):
            q = queries() if l == 0 else q
            out, lse = dec.layer(l, q, SM)
            o_ref, k_ref, F = harness_layer(q[0].float() * SM, full["K"][l], full["V"][l],
                                            r_used[l], keep_ref[l], sk[l][0], sk[l][1], cfg)
            total_k += int(k_ref.sum())
            assert torch.allclose(out[0].float(), o_ref.float(), atol=3e-3, rtol=2e-2), \
                (step, l, (out[0].float() - o_ref.float()).abs().max())
    assert total_k > 0, "the test must exercise fetches"
    if dec.cuda:
        assert len(calls) == 3 * L                           # every layer took the CUDA path
    assert torch.equal(st.r[:, s], r_after)                  # nothing persists
    rep = st.report()
    if cfg.shortlist:
        assert rep["keys_sum"] > 0                            # cold shortlist keys were read
    if cfg.compact:
        # store invariants (compaction stages through the ring): owner <-> tok_slot bijection
        flat = store.tok_slot.view(-1)
        occ = (store.owner >= 0).nonzero().flatten()
        assert torch.equal(flat[store.owner[occ]].long(), occ)


def test_release_resets():
    cfg = CFGS[0]
    eng, store, full, dirs, prompt = _world(cfg)
    s = eng.slot_of("r")
    eng.run_round("r")
    dec = _decoder(eng, store)
    eng.set_decode_rows(["r"])
    dec.begin_step(["r"])
    q = (2.0 * dirs[0].repeat_interleave(G, 0)).half().unsqueeze(0)
    for l in range(L):
        dec.layer(l, q, SM)
    eng.release("r")
    assert bool((eng.vkv.r[:, s] == 1).all()) and eng.vkv.ctx[s] == 0
    assert int(eng.vkv.acnt[:, s].sum()) == 0
    assert "r" not in eng.vkv._prompts


def test_multipass_when_ring_is_small():
    """More cold fetches than the ring holds: F is split into ring-sized passes whose partial
    softmaxes are LSE-merged -- still the harness's output."""
    cfg = VkvConfig(kappa=0.15, eps0=0.01, shortlist=16, **_C)
    eng, store, full, dirs, prompt = _world(cfg, stage=2)
    _, keep_ref = _harness_residency(eng, cfg, prompt)
    s = eng.slot_of("r")
    eng.run_round("r")
    r_used = eng.vkv.r[:, s, :CTX].permute(0, 2, 1).double()
    dec = _decoder(eng, store)
    eng.set_decode_rows(["r"])
    dec.begin_step(["r"])
    q = (torch.randn(1, Hq, D, device="cuda") * 0.3).half()
    for l in range(L):
        out, _ = dec.layer(l, q, SM)
        kh, ch = _sketch_ref(full["K"][l][:, :CTX], cfg)
        o_ref, k_ref, _ = harness_layer(q[0].float() * SM, full["K"][l], full["V"][l],
                                        r_used[l], keep_ref[l], kh, ch, cfg)
        assert torch.allclose(out[0].float(), o_ref.float(), atol=3e-3, rtol=2e-2)
    assert eng.vkv.report()["chunked"] > 0


@pytest.mark.parametrize("ci", [0, 1, 3, 4])
def test_fused_equals_torch_path(ci):
    """Same world, same queries: the fused and the torch decode agree on every decision and
    statistic and the output."""
    cfg = CFGS[ci]
    outs, reps = [], []
    for path in ("fused", "torch"):
        eng, store, full, dirs, prompt = _world(cfg, seed=ci)
        eng.run_round("r")
        dec = _decoder(eng, store, path)
        eng.set_decode_rows(["r"])
        g = torch.Generator(device="cuda").manual_seed(7)
        o = []
        for step in range(3):
            dec.begin_step(["r"])
            q = (torch.randn(1, Hq, D, device="cuda", generator=g) * 0.5 +
                 2.0 * dirs[0].repeat_interleave(G, 0)).half()
            for l in range(L):
                o.append(dec.layer(l, q, SM)[0].float())
        outs.append(torch.stack(o))
        reps.append(eng.vkv.report())
    assert torch.allclose(outs[0], outs[1], atol=2e-3, rtol=1e-2)
    for key in ("decisions", "fetch_any", "k_sum", "xfer_sum", "T_size_sum", "fallback",
                "screened", "keys_sum", "nonmono"):
        assert reps[0][key] == reps[1][key], (key, reps[0][key], reps[1][key])
    assert abs(reps[0]["beta_rel_sum"] - reps[1]["beta_rel_sum"]) <= 1e-6 * reps[1]["beta_rel_sum"]
    assert reps[0]["k_sum"] > 0


def test_two_request_batch():
    """Two requests decoded together: different contexts, one still before its round (dense),
    then both sampled. Each row equals its own harness step."""
    cfg = VkvConfig(kappa=0.25, eps0=0.1, shortlist=16, **_C)
    torch.manual_seed(5)
    stage, pages = 28, 60
    n = L * pages * H * P * 2 * D
    buf = (torch.randn(n, device="cuda") * 0.6).to(torch.float16)
    caches = list(buf.view(L, pages, H, P, 2 * D).unbind(0))
    cpu_cache = torch.zeros((L, 4, 2, P, H, D), dtype=torch.float16, pin_memory=True)
    res = ResidencyTable()
    mgr = KVOffloadManager(scorer=PagedEvictionScorer(), residency=res, gpu_k_caches=caches,
                           gpu_v_caches=caches, cpu_cache=cpu_cache, gpu_budget_blocks=0,
                           kv_layout="HND")
    pool = KVStagingPool(num_slots=stage, staging_base_page=pages - stage, gpu_k_caches=caches,
                         gpu_v_caches=caches, cpu_cache=cpu_cache, residency=res,
                         offload_manager=mgr)
    eng = HeadTokenOffloadEngine(kv_caches=caches, page_size=P, max_len=256, num_req_slots=2,
                                 num_cpu_tokens=1024, scope="global")
    eng.vkv = VkvState(eng, cfg)
    store = TokenStagingStore.for_token_engine(pool, eng)
    reqs = {"a": (150, 3, list(range(0, 20))), "b": (90, 2, list(range(20, 32)))}
    full, prompts = {}, {}
    g = torch.Generator(device="cuda").manual_seed(9)
    dirs = torch.randn(L, H, D, device="cuda")
    for rid, (ctx, gen, pg) in reqs.items():
        eng.add_request(rid)
        eng.set_pages(rid, pg)
        eng.on_step(rid, 0, ctx + gen)
        eng.on_step(rid, ctx + gen, 0)
        sl = eng.slot_of(rid)
        eng.score[:, sl, :ctx + gen] = torch.rand(L, ctx + gen, H, device="cuda", generator=g) ** 2
        for l in range(L):
            for h in range(H):
                pos = torch.randint(0, ctx, (5,), device="cuda", generator=g)
                c = eng.cell[l, sl, pos, h].long()
                eng.kv_rows[c, :D] = (dirs[l, h] * 1.5).half()
        full[rid] = []
        for l in range(L):
            rows = eng.kv_rows[eng.cell[l, sl, :ctx + gen].long()]
            full[rid].append((rows[..., :D].permute(1, 0, 2).float().clone(),
                              rows[..., D:].permute(1, 0, 2).float().clone()))
        prompts[rid] = list(range(500 + len(rid), 500 + len(rid) + ctx))
        eng.vkv.set_prompt(rid, prompts[rid])

    sk = {rid: [_sketch_ref(full[rid][l][0][:, :reqs[rid][0]], cfg, reqs[rid][0])
                for l in range(L)] for rid in reqs}
    keep, r0 = {}, {}

    def do_round(rid):
        _, keep[rid] = _harness_residency(eng, cfg, prompts[rid], rid, reqs[rid][0])
        eng.run_round(rid)
        r0[rid] = eng.vkv.r[:, eng.slot_of(rid), :reqs[rid][0]].clone()   # stored r

    do_round("a")
    dec = _decoder(eng, store, os.environ.get("VKV_TEST_PATH", "fused"))
    ids = ["a", "b"]
    eng.set_decode_rows(ids)
    for step in range(2):
        if step == 1:
            do_round("b")
        dec.begin_step(ids)
        q = (torch.randn(2, Hq, D, device="cuda", generator=g) * 0.5
             + 2.0 * dirs[0].repeat_interleave(G, 0)).half()
        for l in range(L):
            out, _ = dec.layer(l, q, SM)
            for b, rid in enumerate(ids):
                K_, V_ = full[rid][l]
                ctx = reqs[rid][0]
                if rid not in keep:          # before its round: exact dense attention
                    p = torch.softmax((q[b].float().view(H, G, D) @ K_.transpose(-1, -2)) * SM, -1)
                    o_ref = torch.einsum("hgn,hnd->hgd", p, V_).reshape(Hq, D)
                else:
                    o_ref, _, F = harness_layer(q[b].float() * SM, K_, V_, r0[rid][l].T.double(),
                                                keep[rid][l], sk[rid][l][0], sk[rid][l][1],
                                                cfg, ctx=ctx)
                assert torch.allclose(out[b].float(), o_ref.float(), atol=3e-3, rtol=2e-2), \
                    (step, l, rid)
    assert eng.vkv.report()["k_sum"] > 0


@pytest.mark.parametrize("path", ["fused", "torch"])
def test_fallback_can_shrink_the_fetch_width(path, monkeypatch):
    """Rows that fall back are recomputed by the reference; when the widest row is one of them
    and its reference k* is smaller, the fetch width shrinks (regression: the fused path wrote a
    narrower reference order into the wider buffer). Forced here: t_max = 1 makes most rows fall
    back, and the reference is made to fetch nothing."""
    import offload_package.vkv.decision as DN
    import offload_package.vkv.decode as DC
    cfg = VkvConfig(kappa=0.2, eps0=0.05, shortlist=16, t_max=1, **_C)
    eng, store, full, dirs, prompt = _world(cfg, seed=2)
    eng.run_round("r")
    real = DN.reference_rows

    def zero_k(*a, **kw):
        plan, kf, nm = real(*a, **kw)
        return plan, torch.zeros_like(kf), nm

    monkeypatch.setattr(DC, "reference_rows", zero_k)
    monkeypatch.setattr(DN, "reference_rows", zero_k)
    dec = _decoder(eng, store, path)
    eng.set_decode_rows(["r"])
    dec.begin_step(["r"])
    q = (2.0 * dirs[0].repeat_interleave(G, 0)).half().unsqueeze(0)
    for l in range(L):
        out, _ = dec.layer(l, q, SM)
        assert bool(torch.isfinite(out.float()).all())
    rep = eng.vkv.report()
    assert rep["fallback"] > 0


def test_first_round_runs_before_the_first_token():
    """TokenModeScheduler runs vkv's first round in the step that ends the prompt (wrapped
    ``update_from_output``: after the forward, before the first token is returned). The round's
    page list reaches the runner at the next step, so it must include that step's allocation."""
    from types import SimpleNamespace as N

    from offload_package.integration.token_mode import TokenModeScheduler

    class Blk:
        def __init__(self, i):
            self.block_id = i

    eng, store, full, dirs, prompt = _world(CFGS[0])
    eng.release("r")
    freed = []
    stm = N(req_to_blocks={"r": [Blk(i) for i in range(-(-CTX // P))]})
    kv = N(coordinator=N(single_type_managers=[stm]),
           block_pool=N(free_blocks=lambda bs: freed.extend(b.block_id for b in bs)))
    req = N(num_tokens=CTX, num_computed_tokens=0, num_offloaded_tokens=0,
            prompt_token_ids=prompt, is_finished=lambda: False)
    sched = N(requests={"r": req}, cache_config=N(enable_prefix_caching=False),
              update_from_output=lambda so, out: "outputs")
    ts = TokenModeScheduler(interval_tokens=64, engine_getter=lambda: eng)
    ts.bind(scheduler=sched, kv_cache_manager=kv)

    def so(n_sched, new=None, cached=None):
        return N(preempted_req_ids=set(), num_scheduled_tokens={"r": n_sched},
                 scheduled_new_reqs=[new] if new else [],
                 scheduled_cached_reqs=N(req_ids=[cached[0]] if cached else [],
                                         num_computed_tokens=[cached[1]] if cached else []))

    # Prefill (one chunk), then its forward: the prompt's KV is written.
    s1 = so(CTX, new=N(req_id="r", prompt_len=CTX, num_computed_tokens=0))
    ts.build_connector_meta(s1)
    s = eng.slot_of("r")
    eng.score[:, s, :CTX] = torch.rand(L, CTX, H, device="cuda") ** 2
    req.num_computed_tokens, req.num_tokens = CTX, CTX + 1
    assert not eng.remapped("r")
    assert sched.update_from_output(s1, None) == "outputs"
    assert eng.remapped("r") and freed, "first round did not run with the prefill step"
    off = eng.offset("r")
    assert req.num_offloaded_tokens == off > 0
    assert [b.block_id for b in stm.req_to_blocks["r"]] == eng.pages("r")
    # Next step: vLLM allocates the decode token's block; the runner's page list has it.
    stm.req_to_blocks["r"].append(Blk(PAGES - STAGE - 1))
    ts.build_connector_meta(so(1, cached=("r", CTX)))
    assert eng.offset("r") == off, "a second round ran at the first decode step"
    assert eng.take_row_update("r") == [b.block_id for b in stm.req_to_blocks["r"]]
    assert eng.vkv.ctx[s] == CTX


def test_evict_is_kvzip_eviction():
    """evict=True: kvpress KVzipPress.compress_post at compression 1 - kappa (global bottom-k
    for the per-layer counts, then each layer's bottom-k), and decode attends to exactly the
    kept cells (plus the generated tokens) with weight 1."""
    cfg = VkvConfig(kappa=0.2, evict=True, **_C)
    eng, store, full, dirs, prompt = _world(cfg, seed=4)
    s = eng.slot_of("r")
    score = eng.score[:, s, :CTX].permute(0, 2, 1).double().contiguous()  # [L, H, ctx]
    eng.run_round("r")
    n_pr = int(score.numel() * (1 - cfg.kappa))
    per = torch.bincount(torch.topk(-score.reshape(-1), n_pr).indices // (H * CTX), minlength=L)
    keep = torch.ones_like(score, dtype=torch.bool)
    for l in range(L):
        keep[l].view(-1)[torch.topk(-score[l].reshape(-1), int(per[l])).indices] = False
    got = (eng.cell[:, s, :CTX] >= 0).permute(0, 2, 1)
    assert torch.equal(got, keep)
    dec = _decoder(eng, store)
    eng.set_decode_rows(["r"])
    dec.begin_step(["r"])
    q = (torch.randn(1, Hq, D, device="cuda") * 0.5).half()
    for l in range(L):
        out, _ = dec.layer(l, q, SM)
        att = torch.cat([keep[l], torch.ones(H, GEN, dtype=torch.bool, device="cuda")], -1)
        qd = q[0].float().view(H, G, D)
        lg = (qd @ full["K"][l].transpose(-1, -2)) * SM
        p = torch.softmax(lg.masked_fill(~att[:, None, :], float("-inf")), -1)
        ref_o = torch.einsum("hgn,hnd->hgd", p, full["V"][l]).reshape(Hq, D)
        assert torch.allclose(out[0].float(), ref_o, atol=2e-3, rtol=1e-2)


@pytest.mark.parametrize("ci", list(range(len(CFGS))) + ["evict"])
def test_graph_replay_equals_eager(ci):
    """VKV_GRAPHS: the per-layer graphs (eager first call, capture, replays) give exactly the
    eager decode's outputs, LSEs and statistics at the same padded widths."""
    cfg = VkvConfig(kappa=0.2, evict=True, **_C) if ci == "evict" else CFGS[ci]
    outs, reps, caps = [], [], []
    for graphed in (False, True):
        eng, store, full, dirs, prompt = _world(cfg, seed=3)
        eng.run_round("r")
        dec = _decoder(eng, store)
        dec.graphs = True                   # padded widths in both runs
        eng.set_decode_rows(["r"])
        g = torch.Generator(device="cuda").manual_seed(7)
        o = []
        for step in range(4):
            dec.begin_step(["r"])
            q = (torch.randn(1, Hq, D, device="cuda", generator=g) * 0.5 +
                 2.0 * dirs[0].repeat_interleave(G, 0)).half()
            for l in range(L):
                run = dec.layer_graphed if graphed else dec.layer
                out, lse = run(l, q, SM)
                o.append(torch.cat([out.float().flatten(), lse.float().flatten()]))
        outs.append(torch.stack(o))
        reps.append(eng.vkv.stats.clone())
        caps.append(dec.captures)
    assert caps == [0, L]                   # one capture per layer, then replays
    assert torch.equal(outs[0], outs[1])
    assert torch.equal(reps[0], reps[1])


@pytest.mark.parametrize("case", ["evict", "evict_nocompact", "before_round"])
def test_flash_attention_matches_sparse_decode(case):
    """Attend-only rows (KVzip eviction, with or without compaction; a request before its
    round) through FlashAttention give the sparse decode kernel's output and LSE."""
    from offload_package.vkv.decode import _fa_varlen
    if _fa_varlen is None:
        pytest.skip("vLLM FA3 unavailable")
    if case == "before_round":
        cfg = CFGS[0]
    else:
        cfg = VkvConfig(kappa=0.2, evict=True, compact=case == "evict", **_C)
    eng, store, full, dirs, prompt = _world(cfg, seed=5)
    if case != "before_round":
        eng.run_round("r")
    dec = _decoder(eng, store)
    eng.set_decode_rows(["r"])
    g = torch.Generator(device="cuda").manual_seed(9)
    for step in range(2):
        dec.begin_step(["r"])
        assert dec.rows["n"] == 0
        q = (torch.randn(1, Hq, D, device="cuda", generator=g) * 0.5 +
             2.0 * dirs[0].repeat_interleave(G, 0)).half()
        for l in range(L):
            dec.fa = True
            o_fa, l_fa = dec.layer(l, q, SM)
            dec.fa = False
            o_sp, l_sp = dec.layer(l, q, SM)
            assert torch.allclose(o_fa.float(), o_sp.float(), atol=2e-3, rtol=1e-2)
            assert torch.allclose(l_fa, l_sp, atol=1e-3, rtol=1e-4)
            dec.fa = True
            o_nl, l_nl = dec.layer(l, q, SM, need_lse=False)
            assert l_nl is None and torch.equal(o_nl, o_fa)


def test_layer_full_matches_layer_and_replays():
    """FULL decode graphs (KVzip eviction): ``layer_full`` (rows from the engine's map, a
    max_len-wide page table) gives ``layer``'s output, and a captured call replayed after the
    next token is appended attends to it as an eager call does."""
    from offload_package.vkv.decode import _fa_varlen
    if _fa_varlen is None:
        pytest.skip("vLLM FA3 unavailable")
    cfg = VkvConfig(kappa=0.2, evict=True, compact=False, **_C)
    eng, store, full, dirs, prompt = _world(cfg, seed=6)
    eng.run_round("r")
    dec = _decoder(eng, store)
    dec.full = True
    eng.set_decode_rows(["r"])
    dec.begin_step(["r"])
    q = (torch.randn(1, Hq, D, device="cuda") * 0.5).half()
    for l in range(L):
        a, _ = dec.layer(l, q, SM)
        b, _ = dec.layer_full(l, q, SM)
        assert torch.allclose(a.float(), b.float(), atol=2e-3, rtol=1e-2)
    q_in = q.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        o_g, _ = dec.layer_full(0, q_in, SM, need_lse=False)
    # Next decode step: one more token written, its position appended (eagerly).
    n0 = int(eng._reqs["r"].mapped)
    eng.on_step("r", n0, 1)
    eng.on_step("r", n0 + 1, 0)
    dec.begin_step(["r"])
    graph.replay()
    ref, _ = dec.layer_full(0, q, SM, need_lse=False)
    assert torch.equal(o_g, ref)
    assert int(eng.vkv.acnt[0, eng.slot_of("r")].max()) > 0


@pytest.mark.parametrize("ci", range(len(CFGS)))
def test_restore_in_place_equals_staged_fetch(ci):
    """compact=False: fetched entries restored into their own cells (once) give the staged
    fetch's decode -- same decisions, statistics and output -- and later steps reuse them."""
    from dataclasses import replace
    outs, reps = [], []
    for compact in (True, False):
        cfg = replace(CFGS[ci], compact=compact)
        eng, store, full, dirs, prompt = _world(cfg, seed=ci)
        eng.run_round("r")
        dec = _decoder(eng, store)
        eng.set_decode_rows(["r"])
        g = torch.Generator(device="cuda").manual_seed(7)
        o, nres = [], []
        for step in range(3):
            dec.begin_step(["r"])
            q = (torch.randn(1, Hq, D, device="cuda", generator=g) * 0.5 +
                 2.0 * dirs[0].repeat_interleave(G, 0)).half()
            for l in range(L):
                o.append(dec.layer(l, q, SM)[0].float())
            if not compact:
                nres.append(int(eng.vkv.restored.sum()))
        outs.append(torch.stack(o))
        reps.append(eng.vkv.report())
        if not compact:
            assert eng.vkv.home is not None and nres == sorted(nres)
            s = eng.slot_of("r")
            res = eng.vkv.restored[:, s].bool()
            # a restored cell holds exactly its entry's CPU copy
            l, p, h = res.nonzero(as_tuple=True)
            if l.numel():
                hc = eng.vkv.home[:, s][l, p, h].long()
                cr = eng.cpu_row[:, s][l, p, h].long()
                assert torch.equal(eng.kv_rows[hc].cpu(), eng.cpu_pool[cr.cpu()])
    assert torch.allclose(outs[0], outs[1], atol=1e-3, rtol=1e-3)
    for key in ("decisions", "fetch_any", "k_sum", "xfer_sum", "T_size_sum", "fallback",
                "screened", "nonmono"):
        assert reps[0][key] == reps[1][key], (key, reps[0][key], reps[1][key])


_CUDA_CFGS = [i for i, c in enumerate(CFGS)
              if c.order == "grad" and c.check == "chern" and c.trunc]


@pytest.mark.parametrize("ci", _CUDA_CFGS)
@pytest.mark.parametrize("compact", [False, True])
def test_cuda_decision_equals_triton(ci, compact):
    """VKV_CUDA: the CUDA decide / emit kernels make the Triton path's decisions (k*, T*, the
    fetch order's use, fallback, screening: every statistic) and its output, step after step
    (restored entries reused, or with compaction staged), capped rows included."""
    from dataclasses import replace
    outs, reps = [], []
    for cuda in (False, True):
        cfg = replace(CFGS[ci], compact=compact)
        eng, store, full, dirs, prompt = _world(cfg, seed=ci)
        eng.run_round("r")
        dec = _decoder(eng, store)
        dec.cuda = cuda
        calls = []
        if cuda:
            orig = dec._layer_cuda
            dec._layer_cuda = lambda *a, **k: calls.append(1) or orig(*a, **k)
        eng.set_decode_rows(["r"])
        g = torch.Generator(device="cuda").manual_seed(7)
        o = []
        for step in range(3):
            dec.begin_step(["r"])
            q = (torch.randn(1, Hq, D, device="cuda", generator=g) * 0.5 +
                 2.0 * dirs[0].repeat_interleave(G, 0)).half()
            for l in range(L):
                o.append(dec.layer(l, q, SM)[0].float())
        outs.append(torch.stack(o))
        reps.append(eng.vkv.report())
        if cuda:
            assert len(calls) == 3 * L                  # every layer call took the CUDA path
    for key in ("decisions", "fetch_any", "k_sum", "xfer_sum", "T_size_sum", "fallback",
                "screened", "keys_sum", "nonmono"):
        assert reps[0][key] == reps[1][key], (key, reps[0][key], reps[1][key])
    assert abs(reps[0]["beta_rel_sum"] - reps[1]["beta_rel_sum"]) <= 1e-6 * reps[1]["beta_rel_sum"]
    assert torch.allclose(outs[0], outs[1], atol=1e-3, rtol=1e-3)
    assert reps[0]["k_sum"] > 0
