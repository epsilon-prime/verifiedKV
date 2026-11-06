"""plan_round (counts only) == what HeadTokenOffloadEngine.run_round does.

The multiprocess design lets the scheduler free blocks from plan_round while
the worker runs the real round; they must agree on every round: applied or
skipped, cells evicted, survivors, blocks kept, offset and freed pages.
Random scores, all per-head scopes, several fractions / budgets / windows,
rounds interleaved with decode appends, and prompts small enough to hit the
skip branches (nothing to evict, round frees no page).
"""
from __future__ import annotations

import random

import pytest
import torch

from offload_package.integration.token_plan import plan_round

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

L, H, P, C2 = 2, 2, 8, 16
PAGES = 160


def _engine(scope, fraction, budget, sink, window):
    from offload_package.integration.token_mode_heads import HeadTokenOffloadEngine
    buf = torch.zeros((L, PAGES, H, P, C2), dtype=torch.float16, device="cuda")
    caches = list(buf.unbind(0))                 # views of one backing buffer
    return HeadTokenOffloadEngine(kv_caches=caches, page_size=P, max_len=PAGES * P,
                                  num_req_slots=2, num_cpu_tokens=PAGES * P,
                                  offload_fraction=fraction, token_budget=budget,
                                  sink_tokens=sink, window_tokens=window, scope=scope)


def _run(scope, fraction, budget, sink, window, prompt, steps, W, seed):
    rng = random.Random(seed)
    torch.manual_seed(seed)
    eng = _engine(scope, fraction, budget, sink, window)
    eng.add_request("r")
    free = list(range(PAGES))
    pages = [free.pop(0) for _ in range(-(-prompt // P))]
    eng.set_pages("r", pages)
    eng.on_step("r", 0, prompt)
    s = eng.slot_of("r")
    resident = L * H * prompt                    # the scheduler-side mirror
    n_checked = n_applied = 0
    for t in range(steps):
        Lg = prompt + t
        eng.on_step("r", Lg, 0)
        if t % W == 0:
            eng.score[:, s, :Lg] = torch.rand((L, Lg, H), device="cuda")
            plan = plan_round(req_id="r", scope=scope, L=L, H=H, P=P, Lg=Lg,
                              resident=resident, n_pages=len(eng.pages("r")),
                              extra_tokens=1, fraction=fraction, budget=budget,
                              sink=sink, window=window)
            rounds0, off0 = eng.stats["rounds"], eng.stats["offloaded"]
            n_pages0 = len(eng.pages("r"))
            freed = eng.run_round("r", extra_tokens=1)
            applied = eng.stats["rounds"] > rounds0
            assert applied == (plan is not None), (scope, Lg, resident, plan)
            n_checked += 1
            if plan is not None:
                n_applied += 1
                assert eng.stats["offloaded"] - off0 == plan.E
                assert int((eng.cell[:, s, :Lg] >= 0).sum()) == plan.T
                assert eng.offset("r") == plan.offset
                assert len(eng.pages("r")) == plan.keep_pages
                assert len(freed) == n_pages0 - plan.keep_pages
                resident = plan.T
                free = freed + free
            else:
                assert freed == []
        # decode one token: the scheduler appends a page when the physical
        # tail needs one (vLLM's allocation on the physical length)
        phys = Lg - eng.offset("r")
        if phys // P >= len(eng.pages("r")):
            eng.set_pages("r", eng.pages("r") + [free.pop(0)])
        eng.on_step("r", Lg, 1)
        resident += L * H
        assert int((eng.cell[:, s, :Lg + 1] >= 0).sum()) == resident
    return n_checked, n_applied


@pytest.mark.parametrize("scope", ["global", "layer", "uniform"])
@pytest.mark.parametrize("fraction,budget", [(0.7, None), (0.5, None), (0.95, None),
                                             (0.0, None), (0.0, 40)])
@pytest.mark.parametrize("sink,window", [(4, 8), (0, 0), (16, 64)])
def test_plan_round_matches_run_round(scope, fraction, budget, sink, window):
    checked = applied = 0
    for seed, (prompt, steps, W) in enumerate([(300, 120, 16), (97, 60, 7), (12, 40, 5)]):
        c, a = _run(scope, fraction, budget, sink, window, prompt, steps, W, seed)
        checked += c
        applied += a
    assert checked > 0
    if fraction >= 0.5 and window < 64:
        assert applied > 0                 # the applied branch is exercised


def test_plan_round_skips():
    kw = dict(req_id="r", scope="global", L=L, H=H, P=P, extra_tokens=1,
              fraction=0.7, budget=None, sink=4, window=8)
    assert plan_round(Lg=0, resident=0, n_pages=0, **kw) is None
    # protected (sink + window) >= everything: nothing to evict
    assert plan_round(Lg=10, resident=L * H * 10, n_pages=2, **kw) is None
    # evicts, but the survivors still fill every page: no page to free
    # (fraction 0.05: K = 23 of 24, 92 survivors need ceil(92 / 32) = 3 pages)
    assert plan_round(Lg=24, resident=L * H * 24, n_pages=3,
                      **{**kw, "fraction": 0.05}) is None
    p = plan_round(Lg=400, resident=L * H * 400, n_pages=50, **kw)
    assert p is not None and p.T == L * H * p.K and p.offset == 400 - p.nb * P
    with pytest.raises(ValueError):
        plan_round(Lg=400, resident=1, n_pages=50, **{**kw, "scope": "shared"})
