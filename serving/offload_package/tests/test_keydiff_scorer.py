"""token_scorer='keydiff': KeyDiff (arXiv 2504.15364, Eq. 8) at the round.

Per (layer, KV head) the score of a key is -cos(k, anchor), anchor = mean of
the L2-normalized keys; the round keeps the most distinctive keys. Checked
against an independent computation from the KV cache, and the round's
choice against the scores (uniform scope: per (layer, head), every evicted
cell scores <= every kept unprotected cell).
"""
from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

L, H, P, C2, PAGES, PROMPT = 2, 2, 8, 16, 64, 300


def test_keydiff_scores_and_round():
    from offload_package.integration.token_mode_heads import HeadTokenOffloadEngine
    torch.manual_seed(0)
    buf = torch.randn((L, PAGES, H, P, C2), dtype=torch.float16, device="cuda")
    caches = list(buf.unbind(0))
    eng = HeadTokenOffloadEngine(kv_caches=caches, page_size=P, max_len=PAGES * P,
                                 num_req_slots=2, num_cpu_tokens=PAGES * P,
                                 offload_fraction=0.7, sink_tokens=4, window_tokens=8,
                                 scope="uniform")
    eng.keydiff = True
    eng.add_request("r")
    pages = list(range(-(-PROMPT // P)))
    eng.set_pages("r", pages)
    eng.on_step("r", 0, PROMPT)
    eng.on_step("r", PROMPT, 0)
    s = eng.slot_of("r")
    # Reference, straight from the cache (head-major [pages, H, P, 2C] layout).
    pos = torch.arange(PROMPT, device="cuda")
    pg = torch.tensor(pages, device="cuda")[pos // P]
    want = torch.empty((L, PROMPT, H), device="cuda")
    for l in range(L):
        for h in range(H):
            k = caches[l][pg, h, pos % P, : C2 // 2].float()
            kn = torch.nn.functional.normalize(k, dim=-1)
            anchor = torch.nn.functional.normalize(kn.mean(0), dim=-1)
            want[l, :, h] = -(kn @ anchor)
    cells0 = eng.cell[:, s, :PROMPT].clone()
    eng.run_round("r")
    got = eng.score[:, s, :PROMPT]
    torch.testing.assert_close(got, want, atol=1e-5, rtol=0)
    # The round kept the most distinctive (highest-score) unprotected keys.
    evicted = (cells0 >= 0) & (eng.cell[:, s, :PROMPT] < 0)
    prot = (pos < 4) | (pos >= PROMPT - 8)
    assert int(evicted.sum()) > 0 and not bool(evicted[:, prot].any())
    for l in range(L):
        for h in range(H):
            ev = evicted[l, :, h]
            kept = ~ev & ~prot
            assert float(want[l, ev, h].max()) <= float(want[l, kept, h].min())
