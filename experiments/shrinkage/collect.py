"""Truncation ablation, panels (a)-(c) (--plan sketch: panel (c), the certificate fetch count k* per truncation): the exact MSE of the attention-denominator estimator as the truncated set T
grows from nothing (unbiased Horvitz-Thompson) to the whole tail (plain eviction), on real attention rows.

For one row (query head x position) with exact tail weights w_i, keep probabilities r_i and truncated set T:
    B_hat = sure + sum_{i in S cap T} w_i + sum_{i in S \\ T} w_i / r_i          (S = the Bernoulli draw)
    MSE(T) = (b0 + sum_T w(1-r))^2 + sum_T w^2 r(1-r) + sum_{tail \\ T} w^2 (1-r)/r
(b0 = mass of r = 0 entries, always lost). T_m = the first m entries by decreasing jump J = w(1-r)/r, the order the
MSE rule (verifiedkv/algo.truncation) walks; T* = that rule's stopping point. Exact: no sampling. --mc N also checks the
closed form against N Monte Carlo draws on a few rows.

The press is vkv_final with plan_weights "perfect" (its plan weights are then the exact weights) and the same draw
fraction as the deployed 2-bit arm (sketch share charged), so r is exactly the final grid's. Nothing in the repo is
edited: algo.prepare and vkv_grid_bias are wrapped at runtime (as vkv_final/profile_grid.py does).

    python -u collect.py --cases cwe:152 qa_1:152 niah_multikey_3:152 niah_multivalue:152 --ratios 0.875 0.9375 0.96875
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "evaluation"))
sys.path.insert(0, str(REPO / "experiments" / "quality"))
import evaluate  # noqa: E402
import arms  # noqa: E402  (VkvFinalGridPress; imports verifiedkv.press, which routes the attention hook)
from verifiedkv import press as pg  # noqa: E402
from verifiedkv import chernoff  # noqa: E402  (kstar_fast is the vectorised one)
from kvpress.presses.cached_kvzip_press import CachedKVzipPress  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
ap.add_argument("--dataset", default="ruler32k")
ap.add_argument("--cases", nargs="+", required=True, help="task:row (row of the seed-42 permutation; dev rows >= 150)")
ap.add_argument("--ratios", nargs="+", type=float, default=[0.875, 0.9375, 0.96875])
ap.add_argument("--decode-steps", type=int, default=32, help="decode steps recorded (and generated)")
ap.add_argument("--rows-per-call", type=int, default=8, help="rows kept per planner call (random, seeded)")
ap.add_argument("--mc", type=int, default=0, help="Monte Carlo draws for the closed-form check (0 = off)")
ap.add_argument("--plan", choices=["perfect", "sketch", "sketch_ab", "sketch_diag"], default="perfect",
                help="perfect: exact plan weights, panels (a)/(b); sketch: the deployed 2-bit sketch + 128/KV-head shortlist, panel (c); "
                     "sketch_ab: deployed plan, truncation order and T* from the sketch weights, MSE from exact weights")
ap.add_argument("--eps0", type=float, default=0.05)
ap.add_argument("--delta", type=float, default=0.1)
ap.add_argument("--out", required=True)
args = ap.parse_args()

GRID = np.concatenate([[0.0], np.logspace(-6, 0, 61)])  # truncated fraction m / n_tail
F64 = torch.float64
SKETCH_FRAC = 0.017578125  # the 2-bit sketch's share for every model of the grid: the deployed arm's draw
gen = torch.Generator(device="cuda").manual_seed(0)
CTX = {"call": 0, "n_layers": 1, "meta": {}}
REC = []
MC = []


def curves(wh, r, tail, w_order=None):
    """wh, r, tail [R, n] (float64 / bool). Returns a dict of per-row arrays (MSE normalised by B_tail^2 later).
    w_order: weights that decide the truncation order and T* (default wh); the MSE always uses wh."""
    zero = torch.zeros((), dtype=F64, device=wh.device)
    w = torch.where(tail, wh, zero)
    wo = w if w_order is None else torch.where(tail, w_order, zero)
    pos = tail & (r > 0)
    rr = r.clamp_min(1e-300)
    b0 = torch.where(tail & (r <= 0), w, zero).sum(-1)
    jump = torch.where(pos, wo * (1 - r) / rr, torch.full_like(w, -1.0))
    order = jump.argsort(-1, descending=True)
    a = torch.where(pos, w * (1 - r), zero).gather(-1, order)            # bias added when truncated
    vT = torch.where(pos, w * w * r * (1 - r), zero).gather(-1, order)   # variance if truncated
    vH = torch.where(pos, w * w * (1 - r) / rr, zero).gather(-1, order)  # variance if HT
    z = torch.zeros_like(a[:, :1])
    A = torch.cat([z, a.cumsum(-1)], -1)
    VT = torch.cat([z, vT.cumsum(-1)], -1)
    VH = torch.cat([z, vH.cumsum(-1)], -1)
    mse = (b0[:, None] + A) ** 2 + VT + (VH[:, -1:] - VH)                # [R, n + 1], m = 0..n (n incl. non-pos)
    n_pos = pos.sum(-1)
    m_star = algo_T(wo, r, tail).sum(-1) - (tail & (r <= 0)).sum(-1)     # T* size among r > 0 entries
    idx = torch.arange(mse.shape[-1], device=w.device)[None]
    mse = torch.where(idx <= n_pos[:, None], mse, torch.full_like(mse, float("inf")))  # past n_pos: undefined
    m_min = mse.argmin(-1)
    g = torch.from_numpy(GRID).to(w.device)
    m_grid = torch.round(g[None] * n_pos[:, None].double()).long()
    return {"mse_grid": mse.gather(-1, m_grid), "mse_ht": mse[:, 0], "mse_evict": mse.gather(-1, n_pos[:, None])[:, 0],
            "mse_star": mse.gather(-1, m_star[:, None])[:, 0], "mse_min": mse.gather(-1, m_min[:, None])[:, 0],
            "m_star": m_star, "m_min": m_min, "n_pos": n_pos, "bias_star": (b0 + A.gather(-1, m_star[:, None])[:, 0]),
            "b_tail": w.sum(-1), "order": order}


def algo_T(w, r, tail):
    return pg.algo.truncation(w, r, tail, True)


def monte_carlo(wh, r, tail, sure, order, m_star, n_pos, n_draws):
    """Empirical MSE of B_hat for T = {}, T*, the whole tail, over n_draws fresh Bernoulli draws (rows of wh)."""
    out = []
    for i in range(wh.shape[0]):
        w, ri, ti = wh[i], r[i], tail[i]
        sel = ti.nonzero().squeeze(-1)
        w, ri = w[sel], ri[sel]
        B = float(sure[i] + w.sum())
        rank = torch.empty_like(order[i]); rank[order[i]] = torch.arange(order.shape[-1], device=w.device)
        rk = rank[sel]
        res = {}
        for name, m in (("ht", 0), ("star", int(m_star[i])), ("evict", int(n_pos[i]))):
            inT = rk < m
            S = torch.rand((n_draws, w.numel()), generator=gen, dtype=F64, device=w.device) < ri
            contrib = torch.where(inT, w, w / ri.clamp_min(1e-300))
            Bh = float(sure[i]) + (S * contrib).sum(-1)
            res[name] = float(((Bh - B) ** 2).mean())
        out.append(res)
    return out


_prepare = pg.algo.prepare


def kstars(W, R, Tl, S0, delta):
    """Certificate fetch count k* (eps0, delta) for T = {} (no truncation), T* (the MSE rule) and T = whole tail,
    on the plan weights the press uses; pure computation on copies, the press's own plan is untouched."""
    out = {}
    orig = pg.algo.truncation
    for name in ("none", "star", "all"):
        try:
            if name == "all":
                pg.algo.truncation = lambda wh, r, tail, trunc=True: tail.clone()
            plan = _prepare(W, R, Tl, S0, delta, trunc=(name != "none"), order="grad")
        finally:
            pg.algo.truncation = orig
        k, _, _ = chernoff.kstar_fast(plan, W, R, Tl, args.eps0, delta)
        out[name] = k
    return out


def exact_block(shape):
    """Exact plan weights and exact mass outside the tail for the current planner block, recomputed from the layer's
    query / key exactly as verifiedkv.press.vkv_grid_bias does for plan_weights "perfect" (same blocking, same masks)."""
    query, key, state, scaling = CTX["qk"]
    r = state["r"]
    _, n_kv, ctx = r.shape
    num_heads, q_len, d = query.shape[1], query.shape[2], query.shape[3]
    rows = (num_heads // n_kv) * q_len
    k_len = key.shape[2]
    dev = query.device
    q_all = query.float().reshape(n_kv, rows, d) * scaling
    t_pos = torch.arange(rows, device=dev) % q_len
    allowed = torch.arange(k_len - ctx, device=dev)[None, :] <= (k_len - q_len - ctx + t_pos)[:, None]
    blk = max(1, pg.CHUNK // (n_kv * ctx))
    a = CTX["blk_i"] * blk
    sl = slice(a, a + blk)
    q = q_all[:, sl]
    assert q.shape[1] == shape[1], (q.shape, shape)
    s = torch.matmul(q, key[0, :, :ctx].float().transpose(-1, -2))
    se = torch.matmul(q, key[0, :, ctx:].float().transpose(-1, -2)).masked_fill(~allowed[sl], -float("inf"))
    m = torch.maximum(s.amax(-1), se.amax(-1))[..., None]
    w = torch.exp((s - m).double())
    tail3 = (r[0].double() < 1)[:, None]
    sure = torch.where(tail3, torch.zeros((), dtype=F64, device=dev), w).sum(-1) + torch.exp((se - m).double()).sum(-1)
    return w, sure



def diagnose(Wsk, We, r, tail):
    """Why the sketch's T* differs from the exact one. Wsk: the planner's weights (sketch off the shortlist), We: exact
    weights, both [R, n]; Wsk is rescaled to We's scale using the r = 1 entries (sinks / window are exact in both)."""
    zero = torch.zeros((), dtype=F64, device=We.device)
    exact_part = ~tail
    c = (torch.where(exact_part, We, zero).sum(-1) / torch.where(exact_part, Wsk, zero).sum(-1))[:, None]
    Wh = Wsk * c
    pos = tail & (r > 0)
    rr = r.clamp_min(1e-300)
    in_sl = pos & torch.isclose(Wh, We, rtol=1e-5, atol=0)  # shortlist entries carry exact scores
    Tsk = pg.algo.truncation(Wh, r, tail, True) & pos
    Tex = pg.algo.truncation(We, r, tail, True) & pos
    a_true = torch.where(pos, We * (1 - r), zero)
    a_hat = torch.where(pos, Wh * (1 - r), zero)
    J_true = torch.where(pos, We * (1 - r) / rr, zero)
    J_hat = torch.where(pos, Wh * (1 - r) / rr, torch.full_like(Wh, -1.0))
    order = J_hat.argsort(-1, descending=True)                  # the sketch's walk
    a_t, J_t = a_true.gather(-1, order), J_true.gather(-1, order)
    beta_before = a_t.cumsum(-1) - a_t                          # true bias collected before each step of the walk
    in_T = Tsk.gather(-1, order)
    bad = in_T & (J_t <= 2 * beta_before)                       # truncations that raised the true MSE
    n_sk = Tsk.sum(-1)
    stop = order.gather(-1, n_sk[:, None].clamp(max=order.shape[-1] - 1))[:, 0]  # first entry the walk did not take
    ar = torch.arange(We.shape[0], device=We.device)
    beta_true_T = a_true.where(Tsk, zero).sum(-1)
    beta_hat_T = a_hat.where(Tsk, zero).sum(-1)
    has_stop = n_sk < pos.sum(-1)
    return {
        "n_sk": n_sk, "n_ex": Tex.sum(-1), "overlap": (Tsk & Tex).sum(-1),
        "sk_in_shortlist": (Tsk & in_sl).sum(-1), "ex_in_shortlist": (Tex & in_sl).sum(-1),
        "beta_hat_over_true": beta_hat_T / beta_true_T.clamp_min(1e-300),
        "n_bad": bad.sum(-1),
        "stop_in_shortlist": in_sl[ar, stop] & has_stop,
        "stop_Jhat_over_J": J_hat[ar, stop] / J_true[ar, stop].clamp_min(1e-300),
        "stop_premature": has_stop & (J_true[ar, stop] > 2 * beta_true_T),   # exact rule would have taken it
        "stop_beta_inflated": beta_hat_T > beta_true_T,
    }


def prepare_wrapped(wh, r, tail, sure, delta, **kw):
    meta = CTX["meta"]
    call = CTX["call"] - 1
    layer, step = call % CTX["n_layers"], call // CTX["n_layers"]  # step 0 = the question forward
    if step <= args.decode_steps:
        n_kv, R, n = wh.shape
        W = wh.reshape(-1, n)
        Rr = r.expand(n_kv, R, n).reshape(-1, n)
        Tl = tail.expand(n_kv, R, n).reshape(-1, n)
        S0 = sure.reshape(-1)
        k = min(args.rows_per_call, W.shape[0])
        pick = torch.randperm(W.shape[0], generator=gen, device=W.device)[:k]
        if args.plan == "sketch_diag":
            Wex, Sex = exact_block(wh.shape)
            CTX["blk_i"] += 1
            d = diagnose(W[pick], Wex.reshape(-1, n)[pick], Rr[pick], Tl[pick])
            for j in range(k):
                REC.append({**meta, "layer": layer, "step": step, "phase": "question" if step == 0 else "decode",
                            **{kk: (float(v[j]) if v.dtype.is_floating_point else int(v[j])) for kk, v in d.items()}})
            return _prepare(wh, r, tail, sure, delta, **kw)
        if args.plan == "sketch_ab":
            Wex, Sex = exact_block(wh.shape)
            CTX["blk_i"] += 1
            We, Se = Wex.reshape(-1, n)[pick], Sex.reshape(-1)[pick]
            ce = curves(We, Rr[pick], Tl[pick])                       # exact order, exact T*
            cs = curves(We, Rr[pick], Tl[pick], w_order=W[pick])      # sketch order, sketch T*, exact MSE
            B = Se + ce["b_tail"]
            for j in range(k):
                REC.append({**meta, "layer": layer, "step": step, "phase": "question" if step == 0 else "decode",
                            "B": float(B[j]), "n_pos": int(ce["n_pos"][j]), "mse_ht": float(ce["mse_ht"][j]),
                            "mse_evict": float(ce["mse_evict"][j]), "m_star": int(ce["m_star"][j]),
                            "mse_star": float(ce["mse_star"][j]), "mse_min": float(ce["mse_min"][j]),
                            "m_star_sk": int(cs["m_star"][j]), "mse_star_sk": float(cs["mse_star"][j]),
                            "mse_min_sk": float(cs["mse_min"][j]),
                            "mse_grid": ce["mse_grid"][j].tolist(), "mse_grid_sk": cs["mse_grid"][j].tolist()})
            return _prepare(wh, r, tail, sure, delta, **kw)
        if args.plan == "sketch":
            ks = kstars(W[pick], Rr[pick], Tl[pick], S0[pick], delta)
            nt = Tl[pick].sum(-1)
            for j in range(k):
                REC.append({**meta, "layer": layer, "step": step, "phase": "question" if step == 0 else "decode",
                            "n_tail": int(nt[j]), **{f"k_{n}": int(v[j]) for n, v in ks.items()}})
            return _prepare(wh, r, tail, sure, delta, **kw)
        c = curves(W[pick], Rr[pick], Tl[pick])
        B = (S0[pick] + c["b_tail"])
        for j in range(k):
            REC.append({**meta, "layer": layer, "step": step, "phase": "question" if step == 0 else "decode",
                        "B": float(B[j]), "n_pos": int(c["n_pos"][j]), "m_star": int(c["m_star"][j]),
                        "m_min": int(c["m_min"][j]), "mse_ht": float(c["mse_ht"][j]),
                        "mse_star": float(c["mse_star"][j]), "mse_min": float(c["mse_min"][j]),
                        "mse_evict": float(c["mse_evict"][j]), "bias_star": float(c["bias_star"][j]),
                        "mse_grid": c["mse_grid"][j].tolist()})
        if args.mc and len(MC) < 24 and step in (0, 1) and layer % 6 == 0:
            for res, j in zip(monte_carlo(W[pick][:2], Rr[pick][:2], Tl[pick][:2], S0[pick][:2], c["order"][:2],
                                          c["m_star"][:2], c["n_pos"][:2], args.mc), range(2)):
                MC.append({"closed": {"ht": float(c["mse_ht"][j]), "star": float(c["mse_star"][j]),
                                      "evict": float(c["mse_evict"][j])}, "mc": res, "draws": args.mc})
    return _prepare(wh, r, tail, sure, delta, **kw)


_bias = pg.vkv_grid_bias


def bias_wrapped(query, key, state, scaling):
    CTX["call"] += 1
    CTX["qk"], CTX["blk_i"] = (query, key, state, scaling), 0
    return _bias(query, key, state, scaling)


pg.algo.prepare = prepare_wrapped
pg.vkv_grid_bias = bias_wrapped  # the route looks the name up at call time

cfg = evaluate.EvaluationConfig(dataset=args.dataset, data_dir="qa_1", model=args.model, press_name="no_press",
                                compression_ratio=0.0, model_kwargs={"attn_implementation": "sdpa"})
runner = evaluate.EvaluationRunner(cfg)
runner.press = None
runner._setup_model_pipeline()
pipe = runner.pipeline
mc = getattr(pipe.model.config, "text_config", pipe.model.config)
CTX["n_layers"] = int(mc.num_hidden_layers)
gpu = torch.cuda.get_device_name(0)
T0 = time.time()
for case in args.cases:
    task, ridx = case.split(":")
    row = evaluate.load_ruler_length_dataset(args.dataset, task).sample(frac=1, random_state=42).iloc[int(ridx)]
    for ratio in args.ratios:
        press = (arms._make(CachedKVzipPress(), plan_weights="perfect", fixed_charge=SKETCH_FRAC) if args.plan == "perfect"
                 else arms._make(CachedKVzipPress()))  # sketch: exactly the deployed vkv_final_kvzip arm
        press.compression_ratio = ratio
        CTX.update(call=0, meta={"task": task, "row": int(ridx), "ratio": ratio})
        n0 = len(REC)
        with torch.inference_mode():
            pipe(row["context"], question=row["question"], answer_prefix=row["answer_prefix"], press=press,
                 max_new_tokens=min(int(row["max_new_tokens"]), args.decode_steps), max_context_length=200000)
        print(f"[{time.time() - T0:7.1f}s] {task}:{ridx} ratio {ratio}: {len(REC) - n0} rows, draw "
              f"{press.kept_log[-1]:.4f}, {gpu}", flush=True)

out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(out, grid=GRID, mse_grid=np.array([r.pop("mse_grid") for r in REC] if args.plan in ("perfect", "sketch_ab") else []),
                    mse_grid_sk=np.array([r.pop("mse_grid_sk") for r in REC] if args.plan == "sketch_ab" else []),
                    meta=np.array(json.dumps(REC)), mc=np.array(json.dumps(MC)), gpu=np.array(gpu))
print(f"wrote {out}: {len(REC)} rows, {len(MC)} Monte Carlo checks", flush=True)
for m in MC:
    print({k: round(m['mc'][k] / m['closed'][k], 3) if m['closed'][k] > 0 else None for k in m['mc']}, flush=True)
