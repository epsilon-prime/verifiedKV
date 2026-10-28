"""``topk.row_topk`` == a stable descending sort's head (positions, values, ranks), ties included."""
from __future__ import annotations

import pytest
import torch

from offload_package.vkv.topk import row_topk

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _ref(x, K):
    xd = x.double()
    v, i = torch.sort(torch.where(xd == 0, 0.0, xd), dim=-1, descending=True, stable=True)
    return v[:, :K], i[:, :K]


@pytest.mark.parametrize("R,n,K,dtype", [
    (32, 32768, 4096, torch.float64), (8, 32768, 128, torch.float32), (32, 3000, 2048, torch.float64),
    (5, 5000, 3000, torch.float64), (300, 700, 64, torch.float64), (4, 40, 40, torch.float64),
    (3, 100, 1, torch.float32), (2, 17, 5, torch.float32)])
def test_matches_stable_sort(R, n, K, dtype):
    g = torch.Generator(device="cuda").manual_seed(n + K)
    x = torch.randn(R, n, device="cuda", dtype=dtype, generator=g).exp()
    x[:, ::7] = -1.0                                    # exact ties (masked entries)
    x[0, : min(50, n)] = x[0, n // 2]                   # ties at the top
    x[:, 3 % n] = -0.0
    x[:, 5 % n] = 0.0
    x[-1, :] = float("-inf")                            # an all-tied row
    if dtype == torch.float64 and R > 2:
        x[1] = x[1].float().double()                    # ties below float32 resolution
    rank = torch.zeros((R, n), dtype=torch.int32, device="cuda")
    v, i = row_topk(x, K, values=True, rank=rank)
    rv, ri = _ref(x, K)
    assert torch.equal(i, ri) and torch.equal(v, rv)
    want = torch.zeros_like(rank).scatter_(
        -1, ri, torch.arange(1, K + 1, dtype=torch.int32, device="cuda").expand(R, K).contiguous())
    assert torch.equal(rank, want)
    vc, ic = row_topk(x.cpu(), K, values=True)          # the CPU path: the same order
    assert torch.equal(ic, ri.cpu()) and torch.equal(vc, rv.cpu())
