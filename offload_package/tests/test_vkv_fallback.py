"""``fallback.run`` (the capped rows' decision on the GPU) == ``decision.reference_rows``."""
from __future__ import annotations

import pytest
import torch

from offload_package.vkv import fallback as FB
from offload_package.vkv import reference as ref
from offload_package.vkv import reference_chernoff as chern
from offload_package.vkv.decision import CHERN_MARGIN, reference_rows

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _world(n, B, H, G, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    R = B * H * G
    max_len = n + 37
    slots = torch.tensor([3 * b + 1 for b in range(B)], device="cuda")
    r = torch.rand((3 * B + 2, max_len, H), device="cuda", generator=g) * 0.2
    r[r < 0.03] = 0.0                                               # r = 0 tail entries
    r[:, :8] = 1.0                                                  # sinks
    keep = torch.rand(r.shape, device="cuda", generator=g) < 0.05
    r[keep] = 1.0
    s_hat = torch.randn((R, n), device="cuda", generator=g) * 2.0
    s_hat[:, ::97] += 6.0                                           # a few strong entries
    m = s_hat.amax(-1).contiguous()
    sure = torch.rand((R,), dtype=torch.float64, device="cuda", generator=g) * 50 + 1
    return slots, r.float().contiguous(), s_hat.contiguous(), m, sure, max_len


@pytest.mark.parametrize("check,order,trunc,k_max,eps0", [
    ("chern", "grad", True, 16, 0.3), ("chern", "grad", True, 4096, 0.05),
    ("chern", "needle", False, 64, 0.1), ("chern", "grad", True, 1, 0.02),
    ("tight", "needle", True, 8, 0.1), ("simple", "grad", False, 8, 0.2),
    ("simple", "grad", True, 8, 0.05)])
@pytest.mark.parametrize("n", [700, 5000])
def test_matches_reference_rows(check, order, trunc, k_max, eps0, n):
    B, H, G = 2, 2, 3
    R = B * H * G
    delta = 0.05
    slots, r, s_hat, m, sure, max_len = _world(n, B, H, G, seed=n + k_max)
    rows = torch.tensor([0, 3, 4, 7, 11, 9, 5], device="cuda")    # > groups: some loop
    fbl = torch.zeros(R + 1, dtype=torch.int32, device="cuda")
    fbl[:rows.numel()] = rows.int()
    fbl[R] = rows.numel()
    Tm = torch.zeros((R, n), dtype=torch.int8, device="cuda")
    rank = torch.zeros((R, n), dtype=torch.int32, device="cuda")
    kst = torch.full((R,), -7, dtype=torch.int64, device="cuda")
    beta_k, Bt, tsize = (torch.full((R,), -7.0, dtype=torch.float64, device="cuda")
                         for _ in range(3))
    nonmono, scr = (torch.full((R,), 9, dtype=torch.int8, device="cuda") for _ in range(2))
    L = float(torch.log(torch.tensor(2.0 / delta, dtype=torch.float64)))
    consts = torch.tensor([eps0, delta, delta * (1 - CHERN_MARGIN), L, 1.0 / 3.0],
                          dtype=torch.float64, device="cuda")
    grid = chern.GRID.to("cuda")
    FB.run(fbl, s_hat, r, slots, m, sure, grid, consts, Tm, rank, kst, beta_k, Bt, nonmono,
           tsize, scr, n=n, Kc=min(k_max, n), max_len=max_len, H=H, G=G, trunc=trunc,
           order=order, check=check)
    b = rows // (H * G)
    h = (rows // G) % H
    rr = r[slots[b].view(-1, 1), torch.arange(n, device="cuda").view(1, -1), h.view(-1, 1)].double()
    tail = rr < 1
    wh = torch.exp((s_hat[rows] - m[rows].view(-1, 1)).double())
    plan, kf, nm = reference_rows(wh, rr, tail, sure[rows], eps0=eps0, delta=delta, check=check,
                                  order=order, trunc=trunc, k_max=k_max)
    assert torch.equal(kst[rows], kf)
    assert torch.equal((Tm[rows] != 0) | (tail & (rr <= 0)), plan.T)
    assert torch.equal(nonmono[rows].bool(), nm)
    assert torch.allclose(beta_k[rows], ref.curve_at(plan.beta, kf), rtol=1e-9, atol=1e-12)
    assert torch.equal(tsize[rows], (plan.T & tail).sum(-1).double())
    # the fetch order, where the reference's order is unambiguous (tail keys are distinct)
    nt = tail.sum(-1)
    for i in range(rows.numel()):
        o = plan.order[i, :int(nt[i])]
        assert torch.equal(rank[rows[i]][o].long(), torch.arange(1, o.numel() + 1, device="cuda"))
    others = torch.ones(R, dtype=torch.bool, device="cuda")
    others[rows] = False
    assert bool((kst[others] == -7).all()) and bool((rank[others] == 0).all())
