"""Opt-in dumps for replaying vkv against the research harness (``tools/vkv_replay.py``).

VKV_DUMP=<dir> turns it on (eager decode only: run with VKV_GRAPHS=0, VKV_CUDA=1). Written:
  round_<req>.pt        context ids, KVzip scores tau [L, H, ctx] (as the residency read them),
                        our r [L, H, ctx], our draw keep [L, H, ctx], and the full context K / V
                        [H, ctx, D] of the VKV_DUMP_LAYERS (before anything is offloaded)
  step<t>_layer<l>.pt   decode step t (from 0, counted by begin_step) of layer l: the query
                        q [Hq, D] and sm_scale, the K / V of the positions after the context
                        [H, extra, D], and our decision and output per row: plan logits s_hat,
                        m, sure, B~, k*, ranks (F = 1 <= rank <= k*), T*, the attention output.
VKV_DUMP_LAYERS (default "0,9,18,27,35") and VKV_DUMP_STEPS (default "0,1,10,30,60") pick them.
"""
from __future__ import annotations

import os

import torch

DIR = os.environ.get("VKV_DUMP")
ENABLED = bool(DIR)
LAYERS = {int(x) for x in os.environ.get("VKV_DUMP_LAYERS", "0,9,18,27,35").split(",")}
STEPS = {int(x) for x in os.environ.get("VKV_DUMP_STEPS", "0,1,10,30,60").split(",")}


def _save(obj, name):
    os.makedirs(DIR, exist_ok=True)
    torch.save(obj, os.path.join(DIR, name))


def round_(st, req_id, slot, ctx, tau, r, keep, prompt_ids):
    eng = st.eng
    C = st.D
    kv = {}
    for l in sorted(LAYERS):
        if l < st.L:
            cells = eng.cell[l, slot, :ctx].long()                         # [ctx, H]
            rows = eng.kv_rows[cells.reshape(-1)].view(ctx, st.H, 2 * C)
            kv[l] = (rows[..., :C].permute(1, 0, 2).cpu(), rows[..., C:].permute(1, 0, 2).cpu())
    _save({"req": str(req_id), "ctx": ctx, "ids": torch.tensor(list(prompt_ids)),
           "tau": tau.float().cpu(), "r": r.double().cpu(), "keep": keep.cpu(), "kv": kv},
          f"round_{req_id}.pt")


def layer(dec, layer_idx, q, sm_scale, **ours):
    if layer_idx not in LAYERS or dec._step not in STEPS:
        return
    eng, st, rows = dec.eng, dec.st, dec.rows
    if len(rows["slots"]) != 1:
        return                                                    # single-request replays only
    slot, ctx, L = rows["slots"][0], rows["ctx"][0], rows["lens"][0]
    C = st.D
    cells = eng.cell[layer_idx, slot, ctx:L].long()                       # [extra, H]
    ex = eng.kv_rows[cells.reshape(-1)].view(L - ctx, st.H, 2 * C)
    _save({"step": dec._step, "layer": layer_idx, "q": q[0].cpu(), "sm_scale": float(sm_scale),
           "k_extra": ex[..., :C].permute(1, 0, 2).cpu(), "v_extra": ex[..., C:].permute(1, 0, 2).cpu(),
           **{k: (v.cpu() if isinstance(v, torch.Tensor) else v) for k, v in ours.items()}},
          f"step{dec._step}_layer{layer_idx}.pt")
