"""Token offload must not change attention: PQ decode on an offloaded KV
cache is BITWISE equal to PQ decode on the full cache.

Two identical multi-layer KV buffers. A is decoded by the PQ wrapper as-is.
B is offloaded by HeadTokenOffloadEngine (real rounds: offloaded cells copied
to the CPU pool, survivors packed, pages freed) and decoded by the same
wrapper with the offload orchestrator attached (stage -> fetch -> cell decode).
Both use the slot-code path the vLLM backend drives (codes built once from
the uncompacted keys and shared), GQA sharing and SKYLIGHT_PQ_DETERMINISTIC,
so the selection and the gather order are identical and every output and
LSE must match exactly, across decode steps with appends and a second round.
"""
from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

L, H, GROUP, D, P = 3, 2, 2, 64, 16
HQ = H * GROUP
PAGES, SLOTS = 120, 8                    # last SLOTS pages of B are staging
PROMPT, STEPS, ROUND1, ROUND2 = 600, 40, 5, 20
MAXLEN = 1024
TOPK, SINK, LOCAL = 0.2, 16, 32


def _caches(buf):
    return list(buf.view(L, PAGES, H, P, 2 * D).unbind(0))


def _wrapper():
    from pq_optimized import BatchDecodeWithPagedKVCacheWrapper as W
    ws = torch.zeros(256 << 20, dtype=torch.uint8, device="cuda")
    return W(ws, "HND", topk=TOPK, sink_size=SINK, local_size=LOCAL, max_seq_len=MAXLEN,
             num_subspaces=8, codebook_size=64, n_slots=2), ws


def _maint():
    from pq_optimized.slot_codes import SlotCodeMaintainer
    m = SlotCodeMaintainer(h_kv=H, m=8, page=P, n_slots=2, l_max=MAXLEN, max_batch=2)
    m._codebook_size = 64
    m._ch = D
    return m


def _plan(w, pages, phys_len):
    n = len(pages)
    dev = "cuda"
    w.plan(torch.tensor([0, n], dtype=torch.int32, device=dev),
           torch.tensor(pages, dtype=torch.int32, device=dev),
           torch.tensor([phys_len - (n - 1) * P], dtype=torch.int32, device=dev),
           HQ, H, D, P, q_data_type=torch.float16, kv_data_type=torch.float16)


def _run_layer(w, maint, cache, logical_len):
    w._slot_buf = maint.get_buf(cache, MAXLEN)
    w._slot_slots = torch.zeros(1, dtype=torch.int32, device="cuda")
    w._slot_seqlens = torch.tensor([logical_len], dtype=torch.int32, device="cuda")
    w._slot_ptotal = MAXLEN
    w._slot_maint = maint
    w._slot_summary = True
    w._slot_external = True
    w.set_n_keys(logical_len)
    return w


def _orchestrator(caches_b, eng):
    from offload_package.offload.manager import KVOffloadManager
    from offload_package.scoring.paged_eviction import PagedEvictionScorer
    from offload_package.staging.pool import KVStagingPool
    from offload_package.staging.residency import ResidencyTable
    from offload_package.staging.token_store import TokenStagingStore
    cpu_cache = torch.zeros((L, 4, 2, P, H, D), dtype=torch.float16, pin_memory=True)
    res = ResidencyTable()
    mgr = KVOffloadManager(scorer=PagedEvictionScorer(), residency=res, gpu_k_caches=caches_b,
                           gpu_v_caches=caches_b, cpu_cache=cpu_cache, gpu_budget_blocks=0,
                           kv_layout="HND")
    pool = KVStagingPool(num_slots=SLOTS, staging_base_page=PAGES - SLOTS,
                         gpu_k_caches=caches_b, gpu_v_caches=caches_b, cpu_cache=cpu_cache,
                         residency=res, offload_manager=mgr)
    store = TokenStagingStore.for_token_engine(pool, eng)
    return SimpleNamespace(token_offload=eng, token_store=store, staging_pool=pool,
                           residency=res, true_freeing=False), store


def _decode_both(scope, frac, corrupt_cpu=False):
    """Decode A (full cache) and B (offloaded) side by side; return the
    number of (step, layer) pairs whose output or LSE differ, and the store."""
    from offload_package.integration.token_mode_heads import HeadTokenOffloadEngine

    torch.manual_seed(0)
    buf_a = torch.randn(L * PAGES * H * P * 2 * D, dtype=torch.float16, device="cuda")
    buf_b = buf_a.clone()
    ca, cb = _caches(buf_a), _caches(buf_b)
    n0 = -(-PROMPT // P)
    pages_a = list(range(1, 1 + n0))
    pages_b = list(pages_a)
    free_a = list(range(1 + n0, PAGES - SLOTS))
    free_b = list(free_a)

    # PQ codes and codebooks: fitted once on A's (uncompacted) prompt keys and
    # copied to B -- what the backend's slot path holds for both.
    ma, mb = _maint(), _maint()
    for l in range(L):
        ma.init_layer(ma.get_buf(ca[l], MAXLEN), ca[l],
                      {0: torch.tensor(pages_a, device="cuda")}, [(0, 0, PROMPT)], torch.float16)
        mb.get_buf(cb[l], MAXLEN).copy_(ma.get_buf(ca[l], MAXLEN))
        mb._cb[cb[l].data_ptr()].copy_(ma._cb[ca[l].data_ptr()])
        mb._cb_fitted[cb[l].data_ptr()] = True

    eng = HeadTokenOffloadEngine(kv_caches=cb, page_size=P, max_len=MAXLEN, num_req_slots=2,
                                 num_cpu_tokens=MAXLEN, offload_fraction=frac,
                                 sink_tokens=SINK, window_tokens=LOCAL, scope=scope)
    eng.add_request("r")
    eng.set_pages("r", pages_b)
    eng.on_step("r", 0, PROMPT)
    eng.on_step("r", PROMPT, 0)
    s = eng.slot_of("r")
    eng.score[:, s, :PROMPT] = torch.rand((L, PROMPT, H), device="cuda")
    orch, store = _orchestrator(cb, eng)
    mismatches = 0

    wa, _wsa = _wrapper()
    wb, _wsb = _wrapper()
    for t in range(STEPS):
        if t == ROUND1:          # steps before it decode unremapped (identity cells)
            free_b += eng.run_round("r")
            pages_b = eng.pages("r")
            assert eng.remapped("r")
            if corrupt_cpu:                             # negative control
                eng.cpu_pool.add_(1.0)
        p = PROMPT + t                                  # logical position of the new token
        if p // P >= len(pages_a):
            pages_a.append(free_a.pop(0))
        phys = p - eng.offset("r")
        if phys // P >= len(pages_b):
            pages_b.append(free_b.pop(0))
        eng.set_pages("r", pages_b)
        eng.on_step("r", p, 1)
        new_kv = torch.randn((L, H, 2 * D), dtype=torch.float16, device="cuda")
        for l in range(L):
            ca[l][pages_a[p // P], :, p % P] = new_kv[l]
            cb[l][pages_b[phys // P], :, phys % P] = new_kv[l]
        _plan(wa, pages_a, p + 1)
        _plan(wb, pages_b, phys + 1)
        wb.set_offload_orchestrator(orch)
        wb.set_offload_request_ids(["r"])
        for l in range(L):
            q = torch.randn((1, HQ, D), dtype=torch.float16, device="cuda")
            out_a, lse_a = _run_layer(wa, ma, ca[l], p + 1).run(
                q, (ca[l][..., :D], ca[l][..., D:]), return_lse=True)
            out_b, lse_b = _run_layer(wb, mb, cb[l], p + 1).run(
                q, (cb[l][..., :D], cb[l][..., D:]), return_lse=True)
            mismatches += not (torch.equal(out_a, out_b) and torch.equal(lse_a, lse_b))
        eng.on_step("r", p + 1, 0)
        if t == ROUND2:                                   # a second round mid-decode
            free_b += eng.run_round("r")
            pages_b = eng.pages("r")
    return mismatches, store


@pytest.fixture
def _pq_env(monkeypatch):
    pytest.importorskip("pq_optimized")
    monkeypatch.setenv("SKYLIGHT_PQ_GQA_SHARE", "1")
    monkeypatch.setenv("SKYLIGHT_PQ_DETERMINISTIC", "1")


@pytest.mark.parametrize("scope,frac", [("global", 0.7), ("layer", 0.5), ("uniform", 0.9),
                                        ("shared", 0.7)])
def test_offloaded_pq_decode_is_bitwise_equal(scope, frac, _pq_env):
    mismatches, store = _decode_both(scope, frac)
    assert mismatches == 0, f"{mismatches} (step, layer) outputs differ from pq"
    # Nothing went over capacity, and staging actually served offloaded picks.
    assert int(store._ctr[2]) == 0
    assert int((store.tok_slot >= 0).sum()) > 0
    if store._kernels() is not None:      # the sync-free path ran and fetched
        assert int(store._ctr_acc[0]) > 0
        store.check_errors()


def test_harness_detects_a_wrong_fetch(_pq_env):
    """Negative control: corrupt every CPU copy -> the comparison must fail."""
    mismatches, _ = _decode_both("global", 0.7, corrupt_cpu=True)
    assert mismatches > 0


# ---- FULL CUDA graphs: a replayed offloaded decode == eager full-cache PQ ----

def _graph_wrapper():
    from pq_optimized import BatchDecodeWithPagedKVCacheWrapper as W
    ws = torch.zeros(256 << 20, dtype=torch.uint8, device="cuda")
    bufs = dict(paged_kv_indptr_buffer=torch.zeros(2, dtype=torch.int32, device="cuda"),
                paged_kv_indices_buffer=torch.zeros(PAGES, dtype=torch.int32, device="cuda"),
                paged_kv_last_page_len_buffer=torch.zeros(1, dtype=torch.int32, device="cuda"))
    return W(ws, "HND", use_cuda_graph=True, **bufs, topk=TOPK, sink_size=SINK,
             local_size=LOCAL, max_seq_len=MAXLEN, num_subspaces=8, codebook_size=64,
             n_slots=2), (ws, bufs)


def _decode_graph_vs_eager(scope, frac, corrupt_cpu=False):
    """A: eager PQ on the full cache. C: the offloaded cache decoded by one
    CUDA graph per layer, captured after step 0 and only REPLAYED afterwards
    -- across two offload rounds (maps, staging and page lists change under
    the graph). Every replayed output / LSE must equal A's bitwise."""
    from offload_package.integration.token_mode_heads import HeadTokenOffloadEngine

    torch.manual_seed(0)
    buf_a = torch.randn(L * PAGES * H * P * 2 * D, dtype=torch.float16, device="cuda")
    buf_c = buf_a.clone()
    ca, cc = _caches(buf_a), _caches(buf_c)
    n0 = -(-PROMPT // P)
    pages_a, free_a = list(range(1, 1 + n0)), list(range(1 + n0, PAGES - SLOTS))
    pages_c, free_c = list(pages_a), list(free_a)
    ma, mc = _maint(), _maint()
    for l in range(L):
        ma.init_layer(ma.get_buf(ca[l], MAXLEN), ca[l],
                      {0: torch.tensor(pages_a, device="cuda")}, [(0, 0, PROMPT)], torch.float16)
        mc.get_buf(cc[l], MAXLEN).copy_(ma.get_buf(ca[l], MAXLEN))
        mc._cb[cc[l].data_ptr()].copy_(ma._cb[ca[l].data_ptr()])
        mc._cb_fitted[cc[l].data_ptr()] = True

    eng = HeadTokenOffloadEngine(kv_caches=cc, page_size=P, max_len=MAXLEN, num_req_slots=2,
                                 num_cpu_tokens=MAXLEN, offload_fraction=frac,
                                 sink_tokens=SINK, window_tokens=LOCAL, scope=scope)
    eng.add_request("r")
    eng.set_pages("r", pages_c)
    eng.on_step("r", 0, PROMPT)
    eng.on_step("r", PROMPT, 0)
    eng.score[:, eng.slot_of("r"), :PROMPT] = torch.rand((L, PROMPT, H), device="cuda")
    orch, store = _orchestrator(cc, eng)
    eng.alloc_decode_rows(1)
    store.prepare_graphs(1 * HQ * MAXLEN)

    wa, _wsa = _wrapper()
    wc, _keep = _graph_wrapper()
    # Persistent graph inputs / per-row state (the backend's slot buffers).
    q_s = torch.zeros((1, HQ, D), dtype=torch.float16, device="cuda")
    slots_s = torch.zeros(1, dtype=torch.int32, device="cuda")
    seqlen_s = torch.zeros(1, dtype=torch.int32, device="cuda")
    graphs, outs = [], []
    mismatches = 0
    for t in range(STEPS):
        if t in (ROUND1, ROUND2):
            free_c += eng.run_round("r")
            pages_c = eng.pages("r")
            assert eng.remapped("r")
            if corrupt_cpu:                             # negative control
                eng.cpu_pool.add_(1.0)
        p = PROMPT + t
        if p // P >= len(pages_a):
            pages_a.append(free_a.pop(0))
        phys = p - eng.offset("r")
        if phys // P >= len(pages_c):
            pages_c.append(free_c.pop(0))
        eng.set_pages("r", pages_c)
        eng.on_step("r", p, 1)
        new_kv = torch.randn((L, H, 2 * D), dtype=torch.float16, device="cuda")
        for l in range(L):
            ca[l][pages_a[p // P], :, p % P] = new_kv[l]
            cc[l][pages_c[phys // P], :, phys % P] = new_kv[l]
        _plan(wa, pages_a, p + 1)
        _plan(wc, pages_c, phys + 1)
        wc.set_offload_orchestrator(orch)
        wc.set_offload_request_ids(["r"])
        seqlen_s.fill_(p + 1)
        qs = [torch.randn((1, HQ, D), dtype=torch.float16, device="cuda") for _ in range(L)]

        def setup(l):
            wc._slot_buf = mc.get_buf(cc[l], MAXLEN)
            wc._slot_slots, wc._slot_seqlens = slots_s, seqlen_s
            wc._slot_ptotal, wc._slot_maint = MAXLEN, mc
            wc._slot_summary = wc._slot_external = True

        for l in range(L):
            out_a, lse_a = _run_layer(wa, ma, ca[l], p + 1).run(
                qs[l], (ca[l][..., :D], ca[l][..., D:]), return_lse=True)
            q_s.copy_(qs[l])
            if t == 0:                    # eager warm-up step (also the real step 0)
                setup(l)
                out_c, lse_c = wc.run(q_s, (cc[l][..., :D], cc[l][..., D:]), return_lse=True)
            else:
                graphs[l].replay()
                out_c, lse_c = outs[l]
            mismatches += not (torch.equal(out_a, out_c) and torch.equal(lse_a, lse_c))
        if t == 0:                        # capture: records, does not execute
            torch.cuda.synchronize()
            for l in range(L):
                setup(l)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    o = wc.run(q_s, (cc[l][..., :D], cc[l][..., D:]), return_lse=True)
                graphs.append(g)
                outs.append(o)
            fetched0 = int(store._ctr_acc[0])
        eng.on_step("r", p + 1, 0)
        store.check_errors()
    return mismatches, int(store._ctr_acc[0]) - fetched0


@pytest.mark.parametrize("scope,frac", [("global", 0.7), ("uniform", 0.9)])
def test_graph_replayed_offload_decode_is_bitwise_equal(scope, frac, _pq_env, monkeypatch):
    monkeypatch.setenv("SKYLIGHT_PQ_DYN_CG", "1")          # capture-safe live budget
    mismatches, fetched_on_replay = _decode_graph_vs_eager(scope, frac)
    assert mismatches == 0, f"{mismatches} (step, layer) replayed outputs differ from pq"
    assert fetched_on_replay > 0                           # replays fetched rows


def test_graph_harness_detects_a_wrong_fetch(_pq_env, monkeypatch):
    monkeypatch.setenv("SKYLIGHT_PQ_DYN_CG", "1")
    mismatches, _ = _decode_graph_vs_eager("global", 0.7, corrupt_cpu=True)
    assert mismatches > 0


def test_captured_score_write_matches_eager():
    """The fixed-shape masked score write and the hashed random gate under
    a CUDA graph == eager (padded row -1 leaves every score untouched)."""
    from offload_package.integration.token_mode import _write_scores_masked
    from offload_package.scoring.fastkvzip import BatchedRandomGates

    class E:
        score = torch.rand((L, 2, 64, H), device="cuda")
    gate = BatchedRandomGates(seed=3, nhead=H)
    hidden = torch.zeros((L, 2, 8), device="cuda")
    rows = torch.tensor([1, -1], device="cuda")
    pos = torch.tensor([40, 7], device="cuda")
    ref = E.score.clone()
    want = gate(hidden, positions=pos, slots=rows)
    ref[:, 1, 40] = want[:, 0]
    s0 = E.score.clone()
    _write_scores_masked(E, rows, pos, gate(hidden, positions=pos, slots=rows))
    assert torch.equal(E.score, ref)
    E.score.copy_(s0)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        _write_scores_masked(E, rows, pos, gate(hidden, positions=pos, slots=rows))
    assert torch.equal(E.score, s0)                       # capture does not execute
    g.replay()
    assert torch.equal(E.score, ref)
    # Stateless: same ids -> same scores, other positions -> other scores.
    assert torch.equal(gate(hidden, positions=pos, slots=rows), want)
    assert not torch.equal(gate(hidden, positions=pos + 1, slots=rows), want)


def test_dyn_cg_decode_matches_static_budget(_pq_env, monkeypatch):
    """The capture-safe live budget (SKYLIGHT_PQ_DYN_CG: fixed-width gather,
    kernel reads the live prefix) decodes like the eager budget sized by
    set_n_keys: same top-k set, same canonical (deterministic) order. Not
    bitwise: the kernel's split-K depends on the gather width (live vs max),
    so the reduction order differs in the last bits. Both
    graph tests above run DYN_CG on both sides, so this anchors them to the
    static path -- a deterministic sort that pulled the dead tail into the
    live prefix once broke DYN_CG for every config."""
    torch.manual_seed(0)
    buf = torch.randn(L * PAGES * H * P * 2 * D, dtype=torch.float16, device="cuda")
    caches = _caches(buf)
    pages = list(range(1, 1 + -(-PROMPT // P)))
    m = _maint()
    for l in range(L):
        m.init_layer(m.get_buf(caches[l], MAXLEN), caches[l],
                     {0: torch.tensor(pages, device="cuda")}, [(0, 0, PROMPT)], torch.float16)
    w_static, _ws1 = _wrapper()
    monkeypatch.setenv("SKYLIGHT_PQ_DYN_CG", "1")
    w_dyn, _ws2 = _wrapper()
    assert w_dyn._dyn_cg and not w_static._dyn_cg
    _plan(w_static, pages, PROMPT)
    _plan(w_dyn, pages, PROMPT)
    for l in range(L):
        q = torch.randn((1, HQ, D), dtype=torch.float16, device="cuda")
        kv = (caches[l][..., :D], caches[l][..., D:])
        a = _run_layer(w_static, m, caches[l], PROMPT).run(q, kv, return_lse=True)
        # (the append folds the "newest" token, idempotent for the same tail)
        b = _run_layer(w_dyn, m, caches[l], PROMPT).run(q, kv, return_lse=True)
        torch.testing.assert_close(b[0], a[0], atol=2e-3, rtol=0, msg=f"layer {l} out")
        torch.testing.assert_close(b[1], a[1], atol=1e-4, rtol=0, msg=f"layer {l} lse")
