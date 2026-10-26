"""vkv_final_algo: the final Vkv press (2026-10-27). Read ../VKV_FINAL_ALGO.md first; this docstring is the short
version.

What happens to one context (the code path is in brackets):

1. Prefill and scoring [kvpress BernoulliPress around CachedKVzipPress]: the full KV cache is built and KVzip scores
   every (layer, KV head, token).
2. Residency and draw [VkvFinalPress._compress_post -> vkv1/algo.residency]: keep probabilities r = min(1, c * score)
   with ONE global c, chosen so that the expected kept fraction equals the budget left after paying for the sketch:
   (1 - compression_ratio) - sketch_frac. The first 64 (sinks) and last 64 (window) context tokens get r = 1.
   Each token is then kept with probability r (one Bernoulli draw, seeded by the context tokens).
3. Sketch [kvpress NeedleFetchPress._attach_sketch -> top_energy_pairs, quant_sketch]: per layer and KV head, the 16
   RoPE pairs (32 of 128 channels) with the most key energy, stored at 2 bits per channel in groups of 128 tokens
   (fp16 zero and scale per group). About 1.76% of the full cache, paid from the budget in step 2.
4. Every forward over the cache (the question pass, then each decode step), every layer [vkv_final_bias, routed in
   through kvpress's attention hook needle_decode_bias]:
   a. Shortlist: per KV head, the 128 tail tokens with the largest sketch score over all of that head's query rows.
      Their keys are read from the host and scored exactly (fixed buffer: 128 keys per KV head per forward).
   b. Plan weights per row: exact weights for resident-for-sure tokens and the shortlist, sketch weights for the rest
      of the tail (tail = tokens with r < 1).
   c. Certificate [vkv1/algo.prepare + chernoff.kstar_fast]: truncation set T* (MSE rule), fetch order by the
      epsilon-gradient ("grad"), and the smallest k such that the Chernoff bound gives
      P(|B_hat - B| > eps0 * B) <= delta. The first k tail tokens of the order are fetched.
   d. Attention bias [vkv1/algo.attention_bias]: 0 on resident-for-sure, fetched and sampled-truncated tokens,
      log(1/r) on the other sampled tail tokens (Horvitz-Thompson reweighting), -inf on everything else.
   Nothing persists: every forward starts again from the draw.

Defaults: eps0 = 0.3, delta = 0.05, shortlist 128, 16x (compression_ratio 0.9375).

This is a research harness on kvpress: the full cache stays on the GPU, "dropping" is masking and "fetching" is
unmasking, and the transfers are counted. Because the full cache is present, every decision also records its realised
error |B_hat - B| / B (statistics in ``press.stats``).

Import this module before running a model with the press (it routes kvpress's needle hook for states carrying
"vkv_final"; other states go to the previous hook unchanged).
"""

import os
import sys
import zlib
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch

import kvpress.presses.needle_fetch_press as nfp
from kvpress.presses.bernoulli_press import BernoulliPress, transform_scores_for_sampling
from kvpress.presses.cached_kvzip_press import CachedKVzipPress
from kvpress.utils import get_cache_metadata

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # phase4/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # phase4/vkv_final/
from vkv1 import algo  # noqa: E402  (estimator, truncation, fetch order, attention bias)
import chernoff  # noqa: E402  (the certificate: exact and fast Chernoff k*)

CHUNK = 6_000_000  # KV heads x rows x ctx elements per block of the float64 plan
KINDS = ("question", "decode")
STATS = ("decisions", "fetch_any", "k_sum", "xfer_sum", "fail", "fail_eval", "fail_T0", "fail_HT0", "beta_rel_sum",
         "T_size_sum", "fwd", "fwd_any", "fwd_xfer", "fwd_keys", "fwd_ctx")


def new_stats() -> dict:
    """Counters per phase ("question" = the multi-token question forward, "decode" = all decode steps):
    decisions   rows planned (query heads x positions x layers)
    fetch_any   rows with k > 0
    k_sum       sum of k
    xfer_sum    sum over rows of fetched entries that were not in the draw
    fail        rows with realised |B_hat - B| > eps0 B (the promise: <= delta)
    fail_eval   the same at eval_eps (default 0.5), to compare runs with different eps0
    fail_T0     the same with nothing fetched (T* kept): what the certificate had to fix
    fail_HT0    the same with nothing fetched and no truncation (plain Horvitz-Thompson)
    beta_rel_sum, T_size_sum   known bias left at k / B~, and |T*|, summed over rows
    fwd, fwd_any, fwd_ctx      (forward, layer, KV head) cells, those that fetched, and their context sizes
    fwd_xfer    entries read from the host per cell (union over the cell's rows): the K+V traffic
    fwd_keys    shortlist entries whose KEY was read and that were not fetched anyway (keys only: half an entry)
    decode_steps, decode_steps_any   decode steps, and those in which anything was fetched"""
    st = {f"{k}_{m}": 0.0 for k in KINDS for m in STATS}
    st.update(decode_steps=0, decode_steps_any=0, _step_any=False)
    return st


def vkv_final_bias(query: torch.Tensor, key: torch.Tensor, state: dict, scaling: float) -> torch.Tensor:
    """One forward of one layer: the attention bias [1, num_heads, q_len, ctx] for every row, and the statistics.

    query [1, num_heads, q_len, d]; key [1, n_kv, ctx + extra, d] (context keys, then question / generated keys).
    state: per-layer metadata set by VkvFinalPress (r, keep, k_hat, channels, cfg under "vkv_final", stats)."""
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
    if q_len == 1 and state["layer_idx"] == state["first_layer"]:
        st["decode_steps"] += 1
        st["_step_any"] = False
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
        st[f"{kind}_decisions"] += k.numel()
        st[f"{kind}_fetch_any"] += int((k > 0).sum())
        st[f"{kind}_k_sum"] += int(k.sum())
        st[f"{kind}_xfer_sum"] += int(xf.sum())
        st[f"{kind}_fail"] += int((e > cfg["eps0"]).sum())
        st[f"{kind}_fail_eval"] += int((e > cfg["eval_eps"]).sum())
        st[f"{kind}_fail_T0"] += int((e0 > cfg["eps0"]).sum())
        st[f"{kind}_fail_HT0"] += int((eh > cfg["eps0"]).sum())
        st[f"{kind}_beta_rel_sum"] += float((algo.curve_at(plan.beta, k) / plan.Bt).sum())
        st[f"{kind}_T_size_sum"] += int((plan.T & tail3).sum())
    u = xfer_union.sum(-1)
    st[f"{kind}_fwd"] += n_kv
    st[f"{kind}_fwd_any"] += int((u > 0).sum())
    st[f"{kind}_fwd_xfer"] += int(u.sum())
    st[f"{kind}_fwd_keys"] += int((keys_union & ~xfer_union).sum())
    st[f"{kind}_fwd_ctx"] += n_kv * ctx
    if q_len == 1 and not st["_step_any"] and bool((u > 0).any()):
        st["decode_steps_any"] += 1
        st["_step_any"] = True
    return out.view(n_kv, G, q_len, ctx).reshape(1, num_heads, q_len, ctx)


_prev_hook = nfp.needle_decode_bias


def _route(query, key, state, scaling):
    if "vkv_final" in state:
        return vkv_final_bias(query, key, state, scaling)
    return _prev_hook(query, key, state, scaling)


nfp.needle_decode_bias = _route  # kvpress's attention patch looks the hook up in the module at call time


@dataclass
class VkvFinalPress(nfp.NeedleFetchPress):
    """See the module docstring. Built through ``vkv_final_algo``; the fields are the knobs.

    kept            total memory budget as a fraction of the full KV cache (1 - compression_ratio); the draw keeps
                    kept - sketch_frac when charge_sketch
    eps0, delta     the promise per decision: P(|B_hat - B| > eps0 B) <= delta
    shortlist_head  keys read and scored exactly per KV head and forward (0 = off)
    plan_weights    "sketch" (the algorithm) or "perfect" (plan on exact weights: the reference setting)
    fast            chernoff.kstar_fast (True) or the exact float64 curve (False); same certificate
    eval_eps        the realised failure is also counted at this eps (stats only)
    bits / n_pairs / pairs_per_128_dims / group   the 2-bit sketch: 16 RoPE pairs per 128 dims, 2 bits, groups of 128
    """

    bern: BernoulliPress = field(default_factory=lambda: BernoulliPress(press=CachedKVzipPress()))
    kept: float = 0.0625
    eps0: float = 0.3
    delta: float = 0.05
    shortlist_head: int = 128
    plan_weights: str = "sketch"
    charge_sketch: bool = True
    fast: bool = True
    eval_eps: float = 0.5
    n_sink: int = 64
    window: int = 64
    bits: int = 2
    n_pairs: int | None = 16
    pairs_per_128_dims: bool = True

    def __post_init__(self):
        super().__post_init__()
        assert self.plan_weights in ("sketch", "perfect"), self.plan_weights
        # KVzipPress only scores when its ratio is > 0; NeedleFetchPress derives that ratio from _total. The value is
        # otherwise unused: ``kept`` sets the residency.
        self._total = self._total or 0.9375
        self.stats = new_stats()
        self.kept_log: list[float] = []
        self.kept_draw = None

    def reset(self):
        self.stats.clear()
        self.stats.update(new_stats())
        self.kept_log.clear()

    @property
    def draw_frac(self) -> float:
        """Fraction of the full cache the random draw keeps (the budget minus the sketch, when charged)."""
        paid = self.sketch_frac if (self.plan_weights == "sketch" and self.charge_sketch) else 0.0
        return self.kept - paid

    @contextmanager
    def __call__(self, model):
        self.post_init_from_model(model)
        self.bern.compress_post = self._compress_post  # BernoulliPress hands it to the scorer after prefill
        try:
            with self.bern(model):
                yield
        finally:
            del self.bern.compress_post

    def _compress_post(self, model):
        sc = self.bern.press
        scores = sc.score_val  # [n_rows, 1, n_kv, ctx] KVzip scores
        n_rows, bsz, n_kv, ctx = scores.shape
        assert bsz == 1
        kept = self.draw_frac
        assert kept > 0, f"the sketch ({self.sketch_frac:.4f}) leaves nothing of the budget {self.kept}"
        self.kept_draw = kept
        flat = transform_scores_for_sampling(scores.reshape(-1).double())
        r = algo.residency(flat.view(n_rows, 1, n_kv, ctx), kept, self.n_sink, self.window)
        crc = zlib.crc32(sc._context_ids.detach().cpu().numpy().tobytes())
        gen = self.bern._generator(r.device, crc ^ (self.bern.seed * 7919))
        keep = (torch.rand(r.reshape(-1).shape, generator=gen, dtype=torch.float64, device=r.device)
                < r.reshape(-1)).view_as(r)
        bias = torch.where(keep, -torch.log(r.clamp_min(1e-300)), torch.tensor(-float("inf"), device=r.device))
        self.kept_log.append(float(keep.double().mean()))
        metadata = get_cache_metadata(sc._cache)
        ab = metadata.setdefault("attention_bias", {})
        rows = sc._score_row_by_layer
        for module in self.bern._attention_modules(model):
            ab[int(module.layer_idx)] = bias[rows[int(module.layer_idx)]].float()
            module.masked_key_indices = None
        self._attach_sketch(model)  # NeedleFetchPress: moves the bias into metadata["needle"] with k_hat, channels
        cfg = {"eps0": self.eps0, "delta": self.delta, "shortlist_head": self.shortlist_head,
               "plan_weights": self.plan_weights, "fast": self.fast, "eval_eps": self.eval_eps}
        for li, stt in metadata["needle"].items():
            row = rows[int(li)]
            stt.update({"vkv_final": cfg, "r": r[row], "keep": keep[row], "stats": self.stats})


def vkv_final_algo(compression_ratio: float = 0.9375, eps0: float = 0.3, delta: float = 0.05,
                   shortlist_head: int = 128, **kw) -> VkvFinalPress:
    """The final Vkv configuration: chern-grad certificate, no persistence, 64 sinks + 64-token window, planned on the
    2-bit sketch (16 RoPE pairs, paid from the budget) with a per-KV-head shortlist of 128 exactly scored keys.
    Total memory = 1 - compression_ratio of the full cache. ``kw`` overrides any VkvFinalPress field."""
    return VkvFinalPress(kept=1 - compression_ratio, eps0=eps0, delta=delta, shortlist_head=shortlist_head, **kw)
