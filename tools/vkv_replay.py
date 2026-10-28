"""Replay dumped vkv decode steps (``offload_package/vkv/dump.py``) through the research harness's
own code and compare, step by step and layer by layer.

The harness code is used as is, from its checkout (branch vkv_final_algo):
  phase4/vkv_final/press_final.py   vkv_final_bias (shortlist, plan weights, certificate, bias)
  phase4/vkv1/algo.py, chernoff.py  residency, T*, fetch order, k*
  kvpress/presses/needle_fetch_press.py, bernoulli_press.py   sketch, score transform, generator
Only kvpress's package plumbing is stubbed (its adapters do not import with this transformers).

Inputs are our real tensors: the KVzip scores the round read, the context K / V before
offloading, each dumped step's query and the K / V written after the context. Compared:
  round   r (harness residency on our scores) and the draw (harness generator, prompt crc);
  sketch  harness top_energy_pairs + quant_sketch against our plan logits;
  step    per row: k*, F (the fetched set), T*, and the attention output against the harness's
          biased softmax (and both against exact full attention).

    python tools/vkv_replay.py <dump dir> [--harness /home/gaurav_kumar/vkv]
"""
import argparse
import glob
import importlib.util
import os
import re
import sys
import types
import zlib

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("dump")
ap.add_argument("--harness", default="/home/gaurav_kumar/vkv")
ap.add_argument("--kappa", type=float, default=0.0625)
ap.add_argument("--eps0", type=float, default=0.3)
ap.add_argument("--delta", type=float, default=0.05)
ap.add_argument("--k-max", type=int, default=4096, help="our certificate's exact head (VkvConfig.k_max)")
args = ap.parse_args()
H_ROOT = args.harness
dev = "cuda"


# ---- the harness modules, with kvpress's package plumbing stubbed --------------------------------------
def _stub(name, **attrs):
    m = types.ModuleType(name)
    m.__path__ = []
    m.__dict__.update(attrs)
    sys.modules[name] = m
    return m


class _Base:
    def __post_init__(self):
        pass


_stub("kvpress")
_stub("kvpress.presses")
_stub("kvpress.adapters", get_adapter=lambda *a, **k: None)
_stub("kvpress.utils", get_cache_metadata=lambda *a, **k: {}, compute_n_kept=lambda *a, **k: 0)
_stub("kvpress.presses.base_press", BasePress=_Base)
for mod, cls in (("compactor_press", "CompactorPress"), ("fastkvzip_press", "FastKVzipScorer"),
                 ("kvzip_press", "KVzipPress"), ("cached_kvzip_press", "CachedKVzipPress")):
    _stub(f"kvpress.presses.{mod}", **{cls: type(cls, (_Base,), {})})


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


P = f"{H_ROOT}/kvpress/kvpress/presses"
bp = _load("kvpress.presses.bernoulli_press", f"{P}/bernoulli_press.py")
nfp = _load("kvpress.presses.needle_fetch_press", f"{P}/needle_fetch_press.py")
sys.path[:0] = [f"{H_ROOT}/phase4", f"{H_ROOT}/phase4/vkv_final"]
pf = _load("press_final", f"{H_ROOT}/phase4/vkv_final/press_final.py")
algo, chernoff = pf.algo, pf.chernoff

# capture the harness's plan and k* inside vkv_final_bias (one block per layer at these sizes)
CAP = {}
_prep, _kfast = algo.prepare, chernoff.kstar_fast


def _prepare(wh, r, tail, sure, *a, **k):
    plan = _prep(wh, r, tail, sure, *a, **k)
    CAP.update(plan=plan, wh=wh, sure=sure)
    return plan


def _kstar_fast(*a, **k):
    res = _kfast(*a, **k)
    CAP["k"] = res[0]
    CAP["kargs"] = (a, k)
    return res


algo.prepare, chernoff.kstar_fast = _prepare, _kstar_fast


# ---- round: residency and draw ---------------------------------------------------------------------------
rf = glob.glob(f"{args.dump}/round_*.pt")
assert len(rf) == 1, f"one round dump expected, found {rf}"
R0 = torch.load(rf[0])
tau, r_ours, keep_ours = R0["tau"].to(dev), R0["r"].to(dev), R0["keep"].to(dev)
L, Hkv, ctx = tau.shape
D = next(iter(R0["kv"].values()))[0].shape[-1]
pairs = 16 * D // 128
sketch_frac = nfp.sketch_bytes_per_entry(D, 128, 2, pairs) / (D * nfp.KV_BYTES_PER_CHANNEL)
draw = args.kappa - sketch_frac
flat = bp.transform_scores_for_sampling(tau.reshape(-1).double())
r_h = algo.residency(flat.view(L, 1, Hkv, ctx), draw, 64, 64)            # harness [L, 1, H, ctx]
ids = R0["ids"].to(torch.int64).numpy()
crc = zlib.crc32(np.ascontiguousarray(ids).tobytes())
gen = bp.BernoulliPress._generator(r_h.device, crc ^ (0 * 7919))
keep_h = (torch.rand(r_h.reshape(-1).shape, generator=gen, dtype=torch.float64, device=dev)
          < r_h.reshape(-1)).view_as(r_h)
print(f"context {ctx} tokens, {L} layers x {Hkv} KV heads, head_dim {D}; draw fraction {draw:.6f}")
dr = (r_h[:, 0] - r_ours).abs()
print(f"[round] r: max |harness - ours| = {float(dr.max()):.3g} (rel {float((dr / r_h[:, 0].clamp_min(1e-300)).max()):.3g}); "
      f"draw: {int((keep_h[:, 0] != keep_ours).sum())} of {keep_ours.numel()} entries differ; "
      f"kept {float(keep_h.double().mean()):.5f} (ours {float(keep_ours.double().mean()):.5f})")


# ---- steps -----------------------------------------------------------------------------------------------
def kv_for(layer):
    K, V = R0["kv"][layer]
    return K.to(dev), V.to(dev)


sk_cache = {}
files = sorted(glob.glob(f"{args.dump}/step*_layer*.pt"),
               key=lambda f: tuple(int(x) for x in re.findall(r"\d+", os.path.basename(f))))
hdr = (f"{'step':>4} {'layer':>5} | {'k* eq':>6} {'@head':>6} {'F eq':>6} {'T* eq':>6} | {'k* ours/H':>11} "
       f"{'F Jacc':>6} | {'|o - oH|':>9} {'rel':>8} | {'ours vs exact':>13} {'H vs exact':>13} | sketch")
print("\n" + hdr + "\n" + "-" * len(hdr))
tot = {"rows": 0, "k": 0, "F": 0, "T": 0}
for f in files:
    d = torch.load(f)
    l, t = d["layer"], d["step"]
    K, V = kv_for(l)
    Ke, Ve = d["k_extra"].to(dev), d["v_extra"].to(dev)
    q = d["q"].to(dev)                                              # [Hq, D]
    Hq = q.shape[0]
    G = Hq // Hkv
    scale = d["sm_scale"]
    if l not in sk_cache:
        k4 = K.unsqueeze(0)
        ch = nfp.top_energy_pairs(k4, pairs)                        # [1, H, nch]
        kc = k4.gather(-1, ch[:, :, None, :].expand(-1, -1, ctx, -1))
        sk_cache[l] = (nfp.quant_sketch(kc, 2, 128), ch)
    k_hat, ch = sk_cache[l]
    state = {"vkv_final": {"eps0": args.eps0, "delta": args.delta, "shortlist_head": 128,
                           "plan_weights": "sketch", "fast": True, "eval_eps": 0.5},
             "r": r_h[l], "keep": keep_h[l], "k_hat": k_hat, "channels": ch,
             "layer_idx": l, "first_layer": 0, "stats": pf.new_stats()}
    key = torch.cat([K, Ke], 1).unsqueeze(0)                        # [1, H, ctx + extra, D]
    CAP.clear()
    bias = pf.vkv_final_bias(q.view(1, Hq, 1, D), key, state, scale)[0, :, 0]   # [Hq, ctx]
    plan, k_h = CAP["plan"], CAP["k"].reshape(Hq)                   # rows: head h * G + g
    # the harness's certificate with our exact head (kstar_fast's head = 16384 in the press)
    ka, kk = CAP["kargs"]
    k_h4 = _kfast(*ka, **{**kk, "head": args.k_max})[0].reshape(Hq)
    tail = r_h[l, 0] < 1                                            # [H, ctx]
    F_h = (algo.fetch_mask(plan, CAP["k"]) & tail[:, None]).reshape(Hq, ctx)
    T_h = (plan.T & tail[:, None]).reshape(Hq, ctx)
    # harness attention output: softmax over [context + bias | later keys], float64
    qd = q.double().view(Hkv, G, D) * scale
    s = qd @ K.double().transpose(-1, -2)
    se = qd @ Ke.double().transpose(-1, -2)
    lg = torch.cat([s + bias.view(Hkv, G, ctx).double(), se], -1)
    vv = torch.cat([V, Ve], 1).double()
    o_h = torch.einsum("hgn,hnd->hgd", torch.softmax(lg, -1), vv).reshape(Hq, D)
    o_x = torch.einsum("hgn,hnd->hgd", torch.softmax(torch.cat([s, se], -1), -1), vv).reshape(Hq, D)
    # ours
    n = d["n"]
    kst, rank, Tm = d["kst"].to(dev), d["rank"].to(dev)[:, :ctx], d["Tm"].to(dev)[:, :ctx]
    F_o = (rank >= 1) & (rank <= kst[:, None])
    r0 = (r_h[l, 0] <= 0).repeat_interleave(G, 0)                 # r = 0 entries: in T* always
    T_o = ((Tm != 0) | r0) & tail.repeat_interleave(G, 0)
    o = d["out"].to(dev)[0].double()
    k_eq = (kst == k_h).float().mean()
    F_eq = (F_o == F_h).all(-1).float().mean()
    T_eq = (T_o == T_h).all(-1).float().mean()
    inter = (F_o & F_h).sum(-1).double()
    uni = (F_o | F_h).sum(-1).double()
    jac = float(torch.where(uni > 0, inter / uni.clamp_min(1), torch.ones_like(uni)).mean())
    do = (o - o_h).abs().max()
    rel = do / o_h.abs().max()
    eo = (o - o_x).norm(dim=-1) / o_x.norm(dim=-1)
    eh = (o_h - o_x).norm(dim=-1) / o_x.norm(dim=-1)
    ex_o, ex_h = f"{float(eo.median()):.3f}/{float(eo.max()):.2f}", f"{float(eh.median()):.3f}/{float(eh.max()):.2f}"
    k4_eq = (kst == k_h4).float().mean()
    # the harness at our head: its F and its biased-softmax output
    F_h4 = (algo.fetch_mask(plan, k_h4.view_as(CAP["k"])) & tail[:, None]).reshape(Hq, ctx)
    b4 = algo.attention_bias(r_h[l, 0].double()[:, None], tail[:, None], keep_h[l, 0][:, None], plan.T,
                             F_h4.view(Hkv, G, ctx)).reshape(Hq, ctx)
    lg4 = torch.cat([s + b4.view(Hkv, G, ctx).double(), se], -1)
    o_h4 = torch.einsum("hgn,hnd->hgd", torch.softmax(lg4, -1), vv).reshape(Hq, D)
    tot["F4"] = tot.get("F4", 0) + int((F_o == F_h4).all(-1).sum())
    e4 = ((o - o_h4).norm(dim=-1) / o_h4.norm(dim=-1)).max()
    tot["e4"] = max(tot.get("e4", 0.0), float(e4))
    # sketch: our plan logits against the harness's, on the tail (both relative to their own m)
    wh_h = CAP["wh"].reshape(Hq, ctx)
    m_o = d["m"].to(dev).double()
    sh_o = d["s_hat"].to(dev)[:, :ctx].double() - m_o[:, None]
    tl_ = tail.repeat_interleave(G, 0)
    sh_h = torch.log(wh_h.clamp_min(1e-300))
    ds = float(((sh_o - sh_h).abs() * tl_).amax()) if n else 0.0
    print(f"{t:>4} {l:>5} | {float(k_eq):6.3f} {float(k4_eq):6.3f} {float(F_eq):6.3f} {float(T_eq):6.3f} | "
          f"{int(kst.sum()):>5}/{int(k_h.sum()):<5} {jac:6.3f} | {float(do):9.2e} {float(rel):8.2e} | "
          f"{ex_o:>13} {ex_h:>13} | {ds:.2e}")
    tot["rows"] += Hq
    tot["k"] += int((kst == k_h).sum())
    tot["k4"] = tot.get("k4", 0) + int((kst == k_h4).sum())
    bad = (kst != k_h4).nonzero().flatten().tolist()
    if bad:
        print(f"        rows with k* != harness@head: " + ", ".join(
            f"{i}: ours {int(kst[i])} H@head {int(k_h4[i])} H {int(k_h[i])} fb {int(d['fb'][i])}" for i in bad[:6]))
    tot["F"] += int((F_o == F_h).all(-1).sum())
    tot["T"] += int((T_o == T_h).all(-1).sum())
print(f"\nrows {tot['rows']}: k* identical {tot['k']} (to the harness's own certificate at head "
      f"{args.k_max}: {tot['k4']}), F identical {tot['F']} (at head: {tot['F4']}), T* identical {tot['T']}")
print(f"attention output vs the harness at our head: max relative L2 error per row {tot['e4']:.2e}")
print("columns: k*/F/T* eq = fraction of rows identical; F Jacc = mean Jaccard of the fetched sets; "
      "|o - oH| = max abs diff of the attention outputs; 'vs exact' = median / max relative L2 error per row "
      "against full attention; sketch = max |plan logit - m| difference on the tail")
