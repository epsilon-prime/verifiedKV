"""PYTHONPATH=~/src/kvpress-e5 python -m pytest -q test_press_chern.py (CPU)"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.expanduser("~"), "phase4", "tests"))
import press_chern as pc  # noqa: E402
from press_chern import algo, dl  # noqa: E402
from test_press import make  # noqa: E402
from vkv1 import press as orig  # noqa: E402

DT = torch.float64


def test_tight_copy_equals_original_press():
    for check in ("simple", "tight"):
        key, s1, G, d = make(check=check, order="grad")
        _, s2, _, _ = make(check=check, order="grad")
        q = torch.randn(1, key.shape[1] * G, 3, d, generator=torch.Generator().manual_seed(4))
        kq = key[:, :, : s1["r"].shape[-1] + 5]
        assert torch.equal(pc.vkv1_bias(q, kq, s1, d ** -0.5), orig.vkv1_bias(q, kq, s2, d ** -0.5))
        assert torch.equal(s1["promoted"], s2["promoted"])


def test_chern_matches_row_by_row_reference():
    key, st, G, d = make(eps0=0.2, delta=0.1, order="grad")
    st["vkv1"]["check"] = "chern"
    r, keep = st["r"][0], st["keep"][0]
    n_kv, ctx = r.shape
    q_len, scaling = 3, d ** -0.5
    query = torch.randn(1, n_kv * G, q_len, d, generator=torch.Generator().manual_seed(6))
    kq = key[:, :, : ctx + 2 + q_len]
    got = pc.vkv1_bias(query, kq, st, scaling)[0]
    fetched = 0
    for hq in range(n_kv * G):
        h = hq // G
        tail = r[h] < 1
        for t in range(q_len):
            qv = query[0, hq, t].double() * scaling
            s = kq[0, h, :ctx].double() @ qv
            se = kq[0, h, ctx: kq.shape[2] - q_len + t + 1].double() @ qv
            m = torch.maximum(s.max(), se.max())
            w, wx = torch.exp(s - m), torch.exp(se - m).sum()
            plan = algo.prepare(w, r[h], tail, w[~tail].sum() + wx, 0.1, order="grad")
            k = dl.first_pass(dl.chernoff_curves(plan, w, r[h], tail, [0.2])[0], 0.1, plan.n_tail)
            F = algo.fetch_mask(plan, k) & tail
            fetched += int(F.sum())
            ref = algo.attention_bias(r[h], tail, keep[h], plan.T, F).float()
            assert torch.equal(got[hq, t], ref), (hq, t)
    assert fetched > 0, "fixture too easy"
