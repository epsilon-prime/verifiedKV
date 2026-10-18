"""Per-(layer, head) cell mode: token-staging kernels == the torch path.

Like test_token_kernels.py, but with HeadTokenOffloadEngine: keys carry the KV
head, the map is ``cell [S, max_len, H]`` and staged rows are single 2C cells
of the flat ``kv_rows``. Two identical worlds run the same selections, one
through the kernels and one through torch; outputs, maps and every staged
byte must match, and each staged cell must hold its CPU-pool row.
"""
from __future__ import annotations

import pytest
import torch

from offload_package.integration.token_mode_heads import HeadTokenOffloadEngine
from offload_package.offload.manager import KVOffloadManager
from offload_package.scoring.paged_eviction import PagedEvictionScorer
from offload_package.staging import token_kernels
from offload_package.staging.pool import KVStagingPool
from offload_package.staging.residency import ResidencyTable
from offload_package.staging.token_store import TokenStagingStore

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
P, SLOTS, PAGES, LAYERS, PROMPT, H, C, GROUP = 8, 2, 20, 2, 100, 2, 32, 2
KSEL = 16


def _pool(layout):
    torch.manual_seed(0)
    n = LAYERS * PAGES * H * P * 2 * C
    buf = torch.randn(n, dtype=torch.float16, device="cuda")
    if layout == "layer_outer":
        caches = list(buf.view(LAYERS, PAGES, H, P, 2 * C).unbind(0))
    elif layout == "token_major":
        caches = list(buf.view(LAYERS, PAGES, P, H, 2 * C).permute(0, 1, 3, 2, 4).unbind(0))
    else:
        caches = list(buf.view(PAGES, LAYERS, H, P, 2 * C).unbind(1))
    cpu_cache = torch.zeros((LAYERS, 4, 2, P, H, C), dtype=torch.float16, pin_memory=True)
    res = ResidencyTable()
    mgr = KVOffloadManager(scorer=PagedEvictionScorer(), residency=res, gpu_k_caches=caches,
                           gpu_v_caches=caches, cpu_cache=cpu_cache, gpu_budget_blocks=0,
                           kv_layout="HND")
    return KVStagingPool(num_slots=SLOTS, staging_base_page=PAGES - SLOTS, gpu_k_caches=caches,
                         gpu_v_caches=caches, cpu_cache=cpu_cache, residency=res,
                         offload_manager=mgr)


def _world(kernels: bool, layout="layer_outer", scope="global"):
    pool = _pool(layout)
    eng = HeadTokenOffloadEngine(kv_caches=pool.gpu_k_caches, page_size=P, max_len=128,
                                 num_req_slots=2, num_cpu_tokens=256, offload_fraction=0.6,
                                 sink_tokens=2, window_tokens=3, scope=scope)
    eng.add_request("r")
    eng.set_pages("r", list(range(13)))                 # 104 token slots
    eng.on_step("r", 0, PROMPT)
    eng.on_step("r", PROMPT, 0)
    s = eng.slot_of("r")
    g = torch.Generator(device="cuda").manual_seed(3)
    eng.score[:, s, :PROMPT] = torch.rand(LAYERS, PROMPT, H, device="cuda", generator=g)
    eng.run_round("r")
    store = TokenStagingStore.for_token_engine(pool, eng)
    if not kernels:
        store._kern = False
    return pool, eng, store, torch.tensor([[s]], device="cuda")


def _kern_or_skip():
    if token_kernels.get() is None:
        pytest.skip(f"token kernels unavailable: {token_kernels.why_unavailable()}")


def _sel(g, n_heads_q=H * GROUP):
    """[1, Hq * KSEL] selection; q heads of a group share (GQA share)."""
    per_kv = [torch.randint(0, PROMPT, (KSEL,), device="cuda", generator=g) for _ in range(H)]
    return torch.cat([per_kv[q // GROUP] for q in range(n_heads_q)]).view(1, -1)


def _staging_rows(store):
    """Every cell of the shared staging pool (all layers' staging pages)."""
    return store.res.engine.kv_rows[store._stage_cell(torch.arange(store.C, device="cuda"))]


def _per_head_sel(eng, layer, n_per_head):
    """[1, Hq * n] selecting n cold positions of each KV head (GQA share)."""
    s = eng.slot_of("r")
    per_kv = [(eng.cell[layer, s, :PROMPT, h] < 0).nonzero().flatten()[:n_per_head]
              for h in range(H)]
    assert all(x.numel() == n_per_head for x in per_kv)
    return torch.cat([per_kv[q // GROUP] for q in range(H * GROUP)]).view(1, -1)


def _check_store(st, layer=None, need_staged=()):
    """Staging-map invariants (the kernel path allocates rows in atomic order,
    so its maps are not byte-equal to the torch path's -- they must be
    CONSISTENT): owner <-> tok_slot is a bijection over occupied rows, and
    every key in `need_staged` (this layer's selected cold keys) is staged."""
    flat = st.tok_slot.view(-1)
    occ = (st.owner >= 0).nonzero().flatten()
    assert torch.equal(flat[st.owner[occ]].long(), occ), "tok_slot[owner[v]] != v"
    staged = (flat >= 0).nonzero().flatten()
    assert torch.equal(st.owner[flat[staged].long()], staged), "owner[tok_slot[k]] != k"
    if layer is not None and len(need_staged):
        off = layer * st._S * st.kps
        keys = torch.as_tensor(list(need_staged), device="cuda") + off
        assert bool((flat[keys] >= 0).all()), "a selected cold key is not staged"


def _compare(a, b, layer, idx):
    (pa, ea, sa, ra), (pb, eb, sb, rb) = a, b
    q = (idx.shape[1] // (H * GROUP), GROUP)
    out_a = sa.stage_tokens(layer, idx, ra, qheads=q)
    out_b = sb.stage_tokens(layer, idx, rb, qheads=q)
    # Kernel side: n_cold of this layer in the device counters (no readback).
    assert int(sa._ctr[3]) == int(out_b[1].sum())
    for x, y in zip(out_a[:3], out_b[:3]):
        assert torch.equal(x, y)
    ma, mb = out_a[3], out_b[3]
    assert (ma is None) == (mb is None)
    if ma is not None:                                   # missing set, any order
        assert torch.equal(ma.sort().values.cpu(), mb.sort().values.cpu())
    da = sa.phys_tokens(layer, *out_a[:3], None, None)
    torch.cuda.synchronize()
    cold_keys = set(out_a[0][out_a[1]].tolist())
    for st in (sa, sb):
        _check_store(st, layer, cold_keys if ma is None else ())
    # Every address reads the right token: resident cells and staged ones.
    s = ea.slot_of("r")
    kvh = TokenStagingStore._kv_heads(idx.shape[1], q, idx.device)
    for j in range(idx.shape[1]):
        pos, h = int(idx[0, j]), int(kvh[0, j])
        c = int(da[0, j])
        if int(ea.cell[layer, s, pos, h]) >= 0:
            assert c == int(ea.cell[layer, s, pos, h])
        elif ma is None or not bool((ma.cpu() == out_a[0][0, j].cpu()).any()):
            r = int(ea.cpu_row[layer, s, pos, h])
            assert torch.equal(ea.kv_rows[c].cpu(), ea.cpu_pool[r]), (layer, pos, h)
    return out_a, out_b


@pytest.mark.parametrize("layout", ["layer_outer", "block_outer", "token_major"])
def test_cell_kernels_match_torch_path(layout):
    _kern_or_skip()
    a, b = _world(True, layout), _world(False, layout)
    assert a[2]._kernels() is not None and b[2]._kernels() is None
    g = torch.Generator(device="cuda").manual_seed(1)
    for _ in range(10):
        for layer in range(LAYERS):
            _compare(a, b, layer, _sel(g))


def test_shared_pool_lets_one_layer_exceed_its_share():
    """Staging is one pool for all layers: layer 0 may stage more cells than
    an equal per-layer split (SLOTS * H * P = 32) would give it."""
    _kern_or_skip()
    a, b = _world(True), _world(False)
    assert a[2].C == SLOTS * LAYERS * H * P                 # 64 cells, all layers
    idx = _per_head_sel(a[1], 0, 20)                        # 40 cold (pos, head) keys
    out_a, out_b = _compare(a, b, 0, idx)
    assert out_a[3] is None and int((a[2].tok_slot[0] >= 0).sum()) == 40
    # Layer 1 then evicts some of layer 0's cells; maps stay consistent.
    _compare(a, b, 1, _per_head_sel(a[1], 1, 20))
    _compare(a, b, 0, idx)


def test_cell_over_capacity():
    """Kernel path (no sync, no chunking): going over capacity stages what
    fits, keeps the maps consistent and raises at the per-step check. The
    torch path still returns the missing set and chunk-fetches."""
    _kern_or_skip()
    a, b = _world(True), _world(False)
    G = b[2].C
    idx = _per_head_sel(a[1], 0, 40)                        # 80 keys > 64 cells
    q = (40, GROUP)
    out_a = a[2].stage_tokens(0, idx, a[3], qheads=q)
    assert out_a[3] is None
    torch.cuda.synchronize()
    _check_store(a[2])
    assert int((a[2].tok_slot[0] >= 0).sum()) == G           # every row used once
    with pytest.raises(RuntimeError, match="no free staging row"):
        a[2].check_errors()
    a[2].check_errors()                                      # reported once
    mb = b[2].stage_tokens(0, idx, b[3], qheads=q)[3]
    assert mb is not None and mb.numel() == 80
    b[2].tick += 1
    assert b[2].fetch(0, mb[:G].to("cuda"), mb[:G].cpu())
    torch.cuda.synchronize()
    _check_store(b[2], 0, set(mb[:G].tolist()))


def test_cell_capacity_check():
    pool, eng, store, rows = _world(True)
    # C = 64 cells, H = 2: <= 32 keys per head (15 + 16 + 1 mid key fits).
    store.check_capacity(topk=0.0, sink=15, local=16, max_len=128, max_seqs=1,
                         gqa_share=True, group=GROUP)
    with pytest.raises(ValueError, match="staging pages"):
        store.check_capacity(topk=0.0, sink=15, local=16, max_len=128, max_seqs=1,
                             gqa_share=False, group=GROUP)


def test_cell_release_and_missing_cpu_copy():
    _kern_or_skip()
    pool, eng, store, rows = _world(True)
    s = eng.slot_of("r")
    cold = (eng.cell[1, s, :PROMPT, 1] < 0).nonzero().flatten()
    idx = cold[:KSEL].repeat(H * GROUP).view(1, -1)
    store.stage_tokens(1, idx, rows, qheads=(KSEL, GROUP))
    assert int((store.tok_slot[1] >= 0).sum()) > 0
    store.release_slot(s)
    assert int((store.tok_slot >= 0).sum()) == 0 and int((store.owner >= 0).sum()) == 0
    eng.cpu_row[1, s, cold[KSEL], 0] = -1
    bad = torch.full((1, H * GROUP * KSEL), int(cold[KSEL]), device="cuda")
    store.stage_tokens(1, bad, rows, qheads=(KSEL, GROUP))
    with pytest.raises(RuntimeError, match="no CPU copy"):          # per-step check
        store.check_errors()
    good = torch.full_like(bad, int(cold[KSEL + 1]))
    eng.cpu_row[1, s, cold[KSEL], 0] = 0                            # restore validity
    store.stage_tokens(1, good, rows, qheads=(KSEL, GROUP))
    store.check_errors()                                            # cleared once reported
