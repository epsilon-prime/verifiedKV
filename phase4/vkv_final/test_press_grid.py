"""press_grid (statistics stripped) gives bit-identical attention biases to press_final (branch vkv_final_algo).
    PYTHONPATH=../../kvpress python -m pytest -q test_press_grid.py"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import press_grid as pg  # noqa: E402
import press_final as pf  # noqa: E402
from test_press_final import make  # noqa: E402  (the branch's fixture)


@pytest.mark.parametrize("plan_weights,shortlist,fast,q_len", [
    ("sketch", 0, True, 3), ("sketch", 7, True, 3), ("sketch", 7, True, 1), ("sketch", 7, False, 3),
    ("sketch", 10 ** 6, True, 3), ("perfect", 0, True, 3), ("perfect", 0, False, 1)])
def test_bias_identical_to_press_final(plan_weights, shortlist, fast, q_len):
    for seed in range(3):
        key, st, q, sc = make(seed=seed, shortlist=shortlist, plan_weights=plan_weights, fast=fast)
        q = q[:, :, :q_len]
        key = key[:, :, : st["r"].shape[-1] + q_len]
        ref = pf.vkv_final_bias(q, key, st, sc)
        st_g = {k: v for k, v in st.items() if k not in ("vkv_final", "stats")}
        st_g["vkv_grid"] = {k: st["vkv_final"][k] for k in ("eps0", "delta", "shortlist_head", "plan_weights", "fast")}
        assert torch.equal(pg.vkv_grid_bias(q, key, st_g, sc), ref), (seed, plan_weights, shortlist, fast, q_len)


def test_defaults_match_press_final():
    a, b = pg.vkv_grid(), pf.vkv_final_algo()
    for f in ("kept", "eps0", "delta", "shortlist_head", "plan_weights", "charge_sketch", "fast", "n_sink", "window",
              "bits", "n_pairs", "pairs_per_128_dims", "group"):
        assert getattr(a, f) == getattr(b, f), f
