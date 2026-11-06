"""VerifiedKV press with extra statistics for the error adherence experiment.

The decision code is vkv_final_bias from verifiedkv/press_reference.py, copied unchanged. Only statistics are added
(marked "# adh"). Importing this module routes the attention hook for "vkv_final" states here.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from verifiedkv import press_reference as pf  # noqa: E402  (installs its own hook first; ours replaces it below)
from verifiedkv.press_reference import CHUNK, VkvFinalPress, algo, chernoff, nfp, vkv_final_algo  # noqa: E402,F401

EPS_GRID = (0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5)
LOG_EDGES = torch.arange(-6.0, 2.0001, 0.05, dtype=torch.float64)  # 161 edges -> 162 bins (under/overflow at ends)
K_EDGES = torch.tensor([0.5] + [2.0 ** j - 0.5 for j in range(1, 21)], dtype=torch.float64)  # 21 edges -> 22 bins
LAYER_KEYS = ("decisions", "fetch_any", "k_sum", "xfer_sum", "fail", "fail_T0", "fail_HT0", "fwd", "fwd_any",
              "fwd_xfer", "fwd_keys", "fwd_ctx", "n_tail", "n_tail_drawn")
STEP_KEYS = ("decisions", "fetch_any", "k_sum", "xfer_sum", "fail", "fail_T0", "fail_HT0", "fwd_xfer", "fwd_keys")
HISTS = ("hist_post", "hist_T0", "hist_HT", "hist_sketchB")


def _new_x(dev):
    x = {}
    for kind in pf.KINDS:
        x[kind] = {
            **{t: torch.zeros(len(EPS_GRID), dtype=torch.float64, device=dev) for t in ("thr_post", "thr_T0", "thr_HT")},
            **{h: torch.zeros(len(LOG_EDGES) + 1, dtype=torch.float64, device=dev) for h in HISTS},
            "hist_k": torch.zeros(len(K_EDGES) + 1, dtype=torch.float64, device=dev),
            "nofetch_n": 0, "nofetch_fail": 0, "fetch_fail": 0,
            "layer": {},
        }
    x["decode"]["step"] = []
    return x


def _hist(h, v, edges):
    h += torch.bincount(torch.bucketize(v.reshape(-1), edges), minlength=h.numel()).to(h.dtype)


def _log_err(e):
    return torch.log10(e.clamp_min(1e-300))


def adh_bias(query: torch.Tensor, key: torch.Tensor, state: dict, scaling: float) -> torch.Tensor:
    cfg, r, keep = state["vkv_final"], state["r"], state["keep"]
    bsz, n_kv, ctx = r.shape
    assert bsz == 1, "batch size 1 only"
    num_heads, q_len, d = query.shape[1], query.shape[2], query.shape[3]
    G = num_heads // n_kv
    rows = G * q_len
    k_len = key.shape[2]
    dev = query.device
    q_all = query.float().reshape(n_kv, rows, d) * scaling  # row = g * q_len + t
    t_pos = torch.arange(rows, device=dev) % q_len
    kind = "decode" if q_len == 1 else "question"
    st = state["stats"]
    if "_x" not in st:  # adh
        st["_x"] = _new_x(dev)
    X = st["_x"][kind]  # adh
    li = int(state["layer_idx"])  # adh
    L = X["layer"].setdefault(li, dict.fromkeys(LAYER_KEYS, 0))  # adh
    if q_len == 1 and state["layer_idx"] == state["first_layer"]:
        st["decode_steps"] += 1
        st["_step_any"] = False
        X["step"].append(dict.fromkeys(STEP_KEYS, 0))  # adh
    S = X["step"][-1] if kind == "decode" else None  # adh
    edges, kedges, grid = LOG_EDGES.to(dev), K_EDGES.to(dev), torch.tensor(EPS_GRID, dtype=torch.float64, device=dev)  # adh
    r_h = r[0].double()  # [n_kv, ctx], float64 (a float32 r could round 1 - 1e-9 up to 1)
    Z = keep[0]  # the draw (bool) [n_kv, ctx]; r = 1 entries are always kept
    tail = r_h < 1  # the random part
    tail3, r3, Z3 = tail[:, None], r_h[:, None], Z[:, None]
    allowed = torch.arange(k_len - ctx, device=dev)[None, :] <= (k_len - q_len - ctx + t_pos)[:, None]  # causal
    out = torch.empty(n_kv, rows, ctx, dtype=torch.float32, device=dev)
    xfer_union = torch.zeros(n_kv, ctx, dtype=torch.bool, device=dev)
    keys_union = torch.zeros_like(xfer_union)
    blk = max(1, CHUNK // (n_kv * ctx))
    sketched = cfg["plan_weights"] == "sketch"
    k_hat, ch = (state["k_hat"][0], state["channels"]) if sketched else (None, None)

    def sketch_scores(q):  # q [n_kv, R, d] -> s_hat [n_kv, R, ctx] from the sketched channels only
        q_s = q if ch is None else q.gather(-1, ch[0][:, None, :].expand(-1, q.shape[1], -1))
        return torch.matmul(q_s, k_hat.transpose(-1, -2))

    # (a) shortlist: one list of M tail tokens per KV head, by the largest sketch score over the head's rows
    C_head = None
    if sketched and cfg["shortlist_head"]:
        best = torch.full((n_kv, ctx), -float("inf"), device=dev)
        for a in range(0, rows, blk):
            best = torch.maximum(best, sketch_scores(q_all[:, a: a + blk]).amax(1))
        best = torch.where(tail, best, torch.full_like(best, -float("inf")))
        idx = best.topk(min(int(cfg["shortlist_head"]), ctx), -1).indices
        C_head = torch.zeros(n_kv, ctx, dtype=torch.bool, device=dev).scatter_(-1, idx, True) & tail
        keys_union |= C_head & ~Z  # keys read from the host (the sampled ones are already resident)

    zero = torch.zeros((), dtype=torch.float64, device=dev)
    for a in range(0, rows, blk):
        sl = slice(a, a + blk)
        q = q_all[:, sl]
        s = torch.matmul(q, key[0, :, :ctx].float().transpose(-1, -2))  # [n_kv, R, ctx], exact (for the bias / stats)
        se = torch.matmul(q, key[0, :, ctx:].float().transpose(-1, -2)).masked_fill(~allowed[sl], -float("inf"))
        # (b) plan weights
        if not sketched:  # reference setting: plan on the exact weights ("perfect sketch")
            m = torch.maximum(s.amax(-1), se.amax(-1))[..., None]
            w = torch.exp((s - m).double())
            wh = w
        else:
            s_hat = sketch_scores(q)
            if C_head is not None:
                s_hat = torch.where(C_head[:, None], s, s_hat)  # shortlisted tokens: exact scores
            m = torch.maximum(torch.where(tail3, s_hat, s).amax(-1), se.amax(-1))[..., None]
            w = torch.exp((s - m).double())
            wh = torch.exp((s_hat - m).double())
        wx = torch.exp((se - m).double()).sum(-1)  # question / generated tokens: always exact
        sure = torch.where(tail3, zero, w).sum(-1) + wx  # exact mass outside the random part
        # (c) certificate: T*, grad order, smallest k with P(|B_hat - B| > eps0 B) <= delta
        plan = algo.prepare(wh, r3, tail3, sure, cfg["delta"], trunc=True, order="grad")
        if cfg["fast"]:
            k, _, _ = chernoff.kstar_fast(plan, wh, r3, tail3, cfg["eps0"], cfg["delta"])
        else:
            P = chernoff.chernoff_curves(plan, wh, r3, tail3, [cfg["eps0"]])[0]
            k = chernoff.first_pass(P, cfg["delta"], plan.n_tail)
        F = algo.fetch_mask(plan, k) & tail3
        # (d) attention bias for this block of rows
        out[:, sl] = algo.attention_bias(r3, tail3, Z3, plan.T, F).float()
        xf = F & ~Z3
        xfer_union |= xf.any(1)
        # statistics: the realised error against the exact weights (the full cache is on the GPU)
        B = w.sum(-1) + wx
        no = torch.zeros_like(F)
        e = (algo.estimate(w, r3, tail3, sure, Z3, plan.T, F) - B).abs() / B
        e0 = (algo.estimate(w, r3, tail3, sure, Z3, plan.T, no) - B).abs() / B
        eh = (algo.estimate(w, r3, tail3, sure, Z3, tail3 & (r3 <= 0), no) - B).abs() / B
        n_dec, n_fa, n_k, n_xf = k.numel(), int((k > 0).sum()), int(k.sum()), int(xf.sum())  # adh
        n_f, n_f0, n_fh = int((e > cfg["eps0"]).sum()), int((e0 > cfg["eps0"]).sum()), int((eh > cfg["eps0"]).sum())  # adh
        st[f"{kind}_decisions"] += n_dec
        st[f"{kind}_fetch_any"] += n_fa
        st[f"{kind}_k_sum"] += n_k
        st[f"{kind}_xfer_sum"] += n_xf
        st[f"{kind}_fail"] += n_f
        st[f"{kind}_fail_eval"] += int((e > cfg["eval_eps"]).sum())
        st[f"{kind}_fail_T0"] += n_f0
        st[f"{kind}_fail_HT0"] += n_fh
        st[f"{kind}_beta_rel_sum"] += float((algo.curve_at(plan.beta, k) / plan.Bt).sum())
        st[f"{kind}_T_size_sum"] += int((plan.T & tail3).sum())
        # adh: thresholds, histograms, fetch / no-fetch split, per layer, per decode step
        X["thr_post"] += (e[..., None] > grid).sum((0, 1)).double()
        X["thr_T0"] += (e0[..., None] > grid).sum((0, 1)).double()
        X["thr_HT"] += (eh[..., None] > grid).sum((0, 1)).double()
        _hist(X["hist_post"], _log_err(e), edges)
        _hist(X["hist_T0"], _log_err(e0), edges)
        _hist(X["hist_HT"], _log_err(eh), edges)
        _hist(X["hist_sketchB"], _log_err((plan.Bt - B).abs() / B), edges)
        _hist(X["hist_k"], k.double(), kedges)
        nf = k == 0
        X["nofetch_n"] += int(nf.sum())
        X["nofetch_fail"] += int((nf & (e > cfg["eps0"])).sum())
        X["fetch_fail"] += int((~nf & (e > cfg["eps0"])).sum())
        for tgt in (L, S) if S is not None else (L,):
            tgt["decisions"] += n_dec
            tgt["fetch_any"] += n_fa
            tgt["k_sum"] += n_k
            tgt["xfer_sum"] += n_xf
            tgt["fail"] += n_f
            tgt["fail_T0"] += n_f0
            tgt["fail_HT0"] += n_fh
    u = xfer_union.sum(-1)
    n_fwd_xfer, n_fwd_keys = int(u.sum()), int((keys_union & ~xfer_union).sum())  # adh
    st[f"{kind}_fwd"] += n_kv
    st[f"{kind}_fwd_any"] += int((u > 0).sum())
    st[f"{kind}_fwd_xfer"] += n_fwd_xfer
    st[f"{kind}_fwd_keys"] += n_fwd_keys
    st[f"{kind}_fwd_ctx"] += n_kv * ctx
    L["fwd"] += n_kv  # adh
    L["fwd_any"] += int((u > 0).sum())
    L["fwd_xfer"] += n_fwd_xfer
    L["fwd_keys"] += n_fwd_keys
    L["fwd_ctx"] += n_kv * ctx
    L["n_tail"] += int(tail.sum())
    L["n_tail_drawn"] += int((tail & Z).sum())
    if S is not None:
        S["fwd_xfer"] += n_fwd_xfer
        S["fwd_keys"] += n_fwd_keys
    if q_len == 1 and not st["_step_any"] and bool((u > 0).any()):
        st["decode_steps_any"] += 1
        st["_step_any"] = True
    return out.view(n_kv, G, q_len, ctx).reshape(1, num_heads, q_len, ctx)


def export(stats: dict) -> dict:
    """press.stats["_x"] as JSON-ready lists (empty dict if the press never ran)."""
    x = stats.get("_x")
    if x is None:
        return {}
    outd = {"eps_grid": list(EPS_GRID), "log_edges": [round(v, 4) for v in LOG_EDGES.tolist()],
            "k_edges": K_EDGES.tolist()}
    for kind in pf.KINDS:
        X = x[kind]
        o = {kk: (v.tolist() if torch.is_tensor(v) else v) for kk, v in X.items() if kk not in ("layer", "step")}
        o["layer"] = {str(li): v for li, v in sorted(X["layer"].items())}
        if kind == "decode":
            o["step"] = X["step"]
        outd[kind] = o
    return outd


def _route(query, key, state, scaling):
    if "vkv_final" in state:
        return adh_bias(query, key, state, scaling)
    return pf._prev_hook(query, key, state, scaling)


nfp.needle_decode_bias = _route
