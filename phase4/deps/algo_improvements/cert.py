"""Shared pieces for the eps-certificate tests: residency, the needle-fetch sketch, rigorous logit bounds.

The sketch is NeedleFetchPress's own (top_energy_pairs + quant_sketch, imported read-only), so every bound
here is about the sketch the final runs carry. Upper bounds on a logit s_i = <q, k_i> (q scaled) split it
into the sketched channels S and the rest R:

    <q_S, k_S> <= <q_S, k_hat_S> + sum_{j in S} |q_j| e_{g,j}     e = per-group max |k - k_hat| (stored)
    <q_R, k_R> <= one of
        pn : sum_p ||q_p|| M_{g,p}        M = per-group max RoPE pair norm over the unsketched pairs
        rn : ||q_R|| ||k_{R,i}||          per-token residual norm
        qm : sum_j max(q_j lo_{g,j}, q_j hi_{g,j})   per-group per-channel min / max (Quest)
        best : min of the three
All are sure bounds (no assumption). Bytes per entry are against a 16-bit K+V entry of 4 d bytes.
"""

import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.expanduser("~"), "src", "kvpress-e5"))
from kvpress.presses.needle_fetch_press import quant_sketch, top_energy_pairs  # noqa: E402

GROUP, BITS, N_PAIRS = 128, 2, 16


def solve_c(tau: torch.Tensor, kept: float, iters: int = 100) -> float:
    """Global c with sum min(1, c tau) = kept * numel (sinks are already tau = 1)."""
    flat = tau.reshape(-1).double()
    pos = flat[flat > 0]
    lo, hi = 1e-12 / pos.max().item(), 1.0 / pos.min().item()
    budget = kept * flat.numel()
    for _ in range(iters):
        mid = math.sqrt(lo * hi)
        lo, hi = (lo, mid) if (mid * flat).clamp(max=1).sum().item() >= budget else (mid, hi)
    return hi


def keep_prob(tau: torch.Tensor, c) -> torch.Tensor:
    r = (c * tau.double()).clamp(max=1.0)
    r[..., :4] = 1.0
    return r


def _groups(x: torch.Tensor, group: int):
    """[n, C] -> ([G, group, C] padded by repeating the last row, n)."""
    n = x.shape[0]
    g = -(-n // group)
    pad = g * group - n
    if pad:
        x = torch.cat([x, x[-1:].expand(pad, -1)])
    return x.view(g, group, -1), n


class HeadSketch:
    """Sketch and bound metadata for one (layer, KV head). keys: [n, d] post-RoPE."""

    def __init__(self, keys: torch.Tensor, n_pairs: int = N_PAIRS, group: int = GROUP, bits: int = BITS):
        k = keys.float()
        n, d = k.shape
        half = d // 2
        self.n, self.d, self.group = n, d, group
        self.S = top_energy_pairs(k[None, None], n_pairs)[0, 0]  # [2 n_pairs] channel ids
        mask = torch.ones(d, dtype=torch.bool)
        mask[self.S] = False
        self.R = mask.nonzero().flatten()
        kS = k[:, self.S]
        self.k_hat = quant_sketch(kS[None, None], bits, group)[0, 0]  # [n, |S|]
        err, _ = _groups((kS - self.k_hat).abs(), group)
        self.e = err.amax(1).half().float() * (1 + 2**-10)  # [G, |S|], rounded up so it stays a sure bound
        kR = k[:, self.R]
        gR, _ = _groups(kR, group)
        self.r_lo = gR.amin(1).half().float() - 1e-3 * gR.abs().amax(1)  # [G, |R|], widened past fp16 rounding
        self.r_hi = gR.amax(1).half().float() + 1e-3 * gR.abs().amax(1)
        pairs_R = torch.tensor([j for j in range(half) if j in set(self.R.tolist()) and j + half in set(self.R.tolist())])
        self.pairs_R = pairs_R
        pn = (k[:, pairs_R] ** 2 + k[:, pairs_R + half] ** 2).sqrt()  # [n, P]
        gpn, _ = _groups(pn, group)
        self.M = gpn.amax(1).half().float() * (1 + 2**-10)  # [G, P]
        self.res_norm = kR.norm(dim=-1).half().float() * (1 + 2**-10)  # [n]
        self.gid = torch.arange(n) // group

    @staticmethod
    def bytes_per_entry(d: int, n_sketch_ch: int, group: int = GROUP) -> dict:
        n_r = d - n_sketch_ch
        return {"sketch": BITS * n_sketch_ch / 8 + 2 * n_sketch_ch * 2 / group,
                "err": n_sketch_ch * 2 / group, "pn": (n_r // 2) * 2 / group, "rn": 2.0,
                "qm": n_r * 2 * 2 / group}

    def logits(self, q: torch.Tensor, keys: torch.Tensor):
        """q [Q, d] scaled. Returns exact s [Q, n], sketch estimate s_hat, and upper bounds per variant."""
        s = q @ keys.float().T
        qS, qR = q[:, self.S], q[:, self.R]
        s_hat = qS @ self.k_hat.T
        slackS = (qS.abs()[:, None, :] * self.e[None]).sum(-1)[:, self.gid]  # [Q, n]
        half = self.d // 2
        qpn = (q[:, self.pairs_R] ** 2 + q[:, self.pairs_R + half] ** 2).sqrt()  # [Q, P]
        pn = (qpn @ self.M.T)[:, self.gid]
        rn = qR.norm(dim=-1, keepdim=True) * self.res_norm[None]
        qm = torch.maximum(qR[:, None, :] * self.r_lo[None], qR[:, None, :] * self.r_hi[None]).sum(-1)[:, self.gid]
        hi_S = s_hat + slackS
        ub = {"pn": hi_S + pn, "rn": hi_S + rn, "qm": hi_S + qm}
        ub["best"] = torch.minimum(torch.minimum(ub["pn"], ub["rn"]), ub["qm"])
        return s, s_hat, ub
