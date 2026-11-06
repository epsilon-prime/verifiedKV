# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Per-head upstream error statistics for KV cache compression.

For one query row this measures, per ATTENTION (query) head, the three pieces of the attention
computation separately:

    denominator  B     = sum_j w_j                      w_j = exp(<q, k_j> * scaling)
    numerator    A     = sum_j w_j v_j                  (a vector)
    attention    Attn  = A / B                          (what the model consumes)

against the same quantities computed with the compressed cache's weights. A correct B does not
imply a correct A/B, so all three are recorded rather than inferred from one another.

Two conventions, both deliberate:

* The context region [0, ctx_len) is the only compressible part. The suffix [ctx_len, k_len) is the
  question plus the generated tokens and is identical in both estimates, so B and A are reported
  split into a context part and a suffix part. The ERRORS are computed on the full key set, because
  that is what the softmax actually normalises over.
* Everything is computed from the REAL keys, before any press mutates them, and never from a
  second forward pass. That keeps the numbers independent of kvpress's fake-key trick (which
  overwrites the cache in place) and gives per-head values that a Frobenius norm over a whole
  layer cannot.

Numerics: B_sparse and B_dense are derived from ONE logits tensor, so the bf16 -> fp32 error is
shared and cancels in the difference. Logits and exp are fp32; every reduction over the ~10^5 key
axis is fp64, because naive fp32 accumulation drifts ~sqrt(N)*eps ~ 2e-5 relative, the same order
as the effect being measured. Results are stored in log space, so the max-shift cancels exactly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch

logger = logging.getLogger(__name__)

NEG = -float("inf")


@dataclass
class HeadStats:
    """Per-(batch, query head) statistics. Every tensor is [bsz, n_q_heads], float64 on CPU."""

    log_b_sparse: torch.Tensor  # log sum of the compressed weights over the CONTEXT region
    log_b_dense: torch.Tensor  # log sum of the true weights over the CONTEXT region
    log_b_suffix: torch.Tensor  # log sum over the uncompressed suffix, identical in both
    n_kept_ctx: torch.Tensor  # distinct context keys with non-zero weight under the estimate
    shift: torch.Tensor  # the max-shift used, for diagnostics only
    num_err: torch.Tensor  # ||A_hat - A|| / ||A||, full key set
    num_err_abar: torch.Tensor  # ||A_hat - A|| / sum_j w_j ||v_j||, full key set
    attn_err: torch.Tensor  # ||Attn_hat - Attn|| / ||Attn||, full key set
    norm_a_dense: torch.Tensor  # ||A||
    abar_dense: torch.Tensor  # sum_j w_j ||v_j||, the quantity bounds are usually stated against


def resolve_scaling(module, query: torch.Tensor, kwargs: Optional[dict] = None) -> float:
    """
    The attention scale, taken from the kwargs or the module, never from the config.

    For Qwen3-4B-Instruct-2507 num_attention_heads * head_dim = 4096 != hidden_size = 2560, so
    hidden_size // num_heads gives 80 instead of 128 and would silently rescale every logit.
    """
    if kwargs is not None and kwargs.get("scaling") is not None:
        return float(kwargs["scaling"])
    scaling = getattr(module, "scaling", None)
    if scaling is not None:
        return float(scaling)
    return float(query.shape[-1] ** -0.5)


def valid_key_mask(
    attention_mask: Optional[torch.Tensor],
    *,
    q_index: int,
    q_len: int,
    k_len: int,
    bsz: int,
    device: torch.device,
) -> torch.Tensor:
    """
    [bsz, 1, k_len] bool, True where this query row may attend. NEVER mutates its input.

    transformers hands SDPA a None mask when q_len == 1 or k_len == q_len, and a BOOLEAN mask
    otherwise, which is an .expand() view with stride 0 on the batch and query axes. Writing to it
    in place would corrupt many rows at once, and adding a float to it would turn True/False into
    1.0/0.0 and silently destroy the causal structure. So: branch on dtype BEFORE any arithmetic,
    and stay out of place throughout.
    """
    if attention_mask is None:
        # Bottom-right alignment: query row i sees keys [0, k_len - q_len + i].
        pos = torch.arange(k_len, device=device)
        return (pos <= (k_len - q_len + q_index)).view(1, 1, k_len).expand(bsz, 1, k_len)

    m = attention_mask
    if m.dim() == 4:
        m = m[:, :, q_index : q_index + 1, :]  # a slice is a view; no copy, no write
    if m.shape[-1] < k_len:
        raise ValueError(f"attention_mask covers {m.shape[-1]} keys, need {k_len}")
    m = m[..., :k_len]

    if m.dtype == torch.bool:
        return m.reshape(m.shape[0], 1, k_len)
    finfo = torch.finfo(m.dtype)
    return (m > finfo.min / 2).reshape(m.shape[0], 1, k_len)


def keep_mask_from_indices(
    masked_key_indices, *, bsz: int, n_kv_heads: int, ctx_len: int, device: torch.device
) -> torch.Tensor:
    """
    [bsz, n_kv_heads, ctx_len] bool, True where the key is kept.

    Built from the index set rather than from the fake keys: search_hyperplane only makes
    exp(<q,k>) approximately zero, and only for the queries it was solved against, so reading the
    sparse denominator off the fake keys would measure kvpress's numerical slop instead of the
    estimator.
    """
    keep = torch.ones((bsz, n_kv_heads, ctx_len), dtype=torch.bool, device=device)
    b_idx, h_idx, s_idx = masked_key_indices
    b_idx, h_idx, s_idx = b_idx.to(device), h_idx.to(device), s_idx.to(device)
    inside = s_idx < ctx_len
    keep[b_idx[inside], h_idx[inside], s_idx[inside]] = False
    return keep


@torch.no_grad()
def compute_head_stats(
    *,
    query: torch.Tensor,  # [bsz, n_q_heads, q_len, head_dim]
    key: torch.Tensor,  # [bsz, n_kv_heads, k_len, head_dim]
    value: torch.Tensor,  # [bsz, n_kv_heads, k_len, head_dim]
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    ctx_len: int,
    bias: Optional[torch.Tensor] = None,  # [bsz, n_kv_heads, ctx_len] additive, -inf == dropped
    keep_mask: Optional[torch.Tensor] = None,  # [bsz, n_kv_heads, ctx_len] bool, True == kept
    q_index: int = -1,
    compute_dtype: torch.dtype = torch.float32,
) -> HeadStats:
    """
    Exactly one of `bias` (the estimator's log-weights, already carrying its reweighting) or
    `keep_mask` (plain top-k, no reweighting) must be given.

    `compute_dtype` is the precision of the logits and the weights; reductions are always float64.
    The default float32 leaves a ~2e-8 relative error on each log denominator, four orders of
    magnitude below the effects this measures, and float64 is ~30x slower for no useful gain. The
    tests use float64 to check the algebra exactly, separately from precision.
    """
    if (bias is None) == (keep_mask is None):
        raise ValueError("pass exactly one of bias= or keep_mask=")

    bsz, n_q_heads, q_len, head_dim = query.shape
    n_kv_heads, k_len = key.shape[1], key.shape[2]
    groups = n_q_heads // n_kv_heads
    device = query.device
    if q_index < 0:
        q_index += q_len
    assert 0 <= q_index < q_len, f"q_index {q_index} out of range for q_len {q_len}"
    assert 0 < ctx_len <= k_len, f"ctx_len {ctx_len} outside (0, {k_len}]"

    # Grouped einsum, no key expansion. The (n_kv_heads, groups) -> n_q_heads reshape is exactly
    # the inverse of repeat_interleave(groups, dim=1), i.e. the layout add_attention_bias assumes.
    q = query[:, :, q_index, :].to(compute_dtype).view(bsz, n_kv_heads, groups, head_dim)
    k32 = key.to(compute_dtype)
    logits = torch.einsum("bhgd,bhnd->bhgn", q, k32) * scaling
    logits = logits.reshape(bsz, n_q_heads, k_len)

    valid = valid_key_mask(
        attention_mask, q_index=q_index, q_len=q_len, k_len=k_len, bsz=bsz, device=device
    )
    logits = logits.masked_fill(~valid, NEG)

    shift = logits.amax(dim=-1, keepdim=True)
    shift = torch.where(torch.isfinite(shift), shift, torch.zeros_like(shift))
    w = torch.exp(logits - shift)  # in (0, 1]; exp(-inf) == 0 exactly

    w_ctx, w_suf = w[..., :ctx_len], w[..., ctx_len:]

    # The compressed context weights
    if bias is not None:
        bias_q = bias.to(compute_dtype).repeat_interleave(groups, dim=1)
        w_ctx_sparse = torch.exp(logits[..., :ctx_len] - shift + bias_q)
        w_ctx_sparse = torch.where(torch.isfinite(w_ctx_sparse), w_ctx_sparse, torch.zeros_like(w_ctx_sparse))
    else:
        keep_q = keep_mask.repeat_interleave(groups, dim=1)
        w_ctx_sparse = torch.where(keep_q, w_ctx, torch.zeros_like(w_ctx))

    f64 = torch.float64
    b_ctx_dense = w_ctx.sum(-1, dtype=f64)
    b_ctx_sparse = w_ctx_sparse.sum(-1, dtype=f64)
    b_suffix = w_suf.sum(-1, dtype=f64) if k_len > ctx_len else torch.zeros_like(b_ctx_dense)
    n_kept = (w_ctx_sparse > 0).sum(-1)

    # Numerators. value is per KV head; the grouped einsum expands it implicitly.
    v32 = value.to(compute_dtype)
    v_ctx, v_suf = v32[:, :, :ctx_len], v32[:, :, ctx_len:]

    def _num(weights: torch.Tensor, vals: torch.Tensor, n: int) -> torch.Tensor:
        if n == 0:
            return torch.zeros((bsz, n_q_heads, head_dim), dtype=f64, device=device)
        wg = weights.view(bsz, n_kv_heads, groups, n)
        return torch.einsum("bhgn,bhnd->bhgd", wg.to(f64), vals.to(f64)).reshape(bsz, n_q_heads, head_dim)

    n_suf = k_len - ctx_len
    a_dense = _num(w_ctx, v_ctx, ctx_len) + _num(w_suf, v_suf, n_suf)
    a_sparse = _num(w_ctx_sparse, v_ctx, ctx_len) + _num(w_suf, v_suf, n_suf)

    # Abar = sum_j w_j ||v_j||, the quantity the bounds are usually stated against. E6 found
    # ||A|| ~ 0.4 Abar because the value vectors cancel, so both denominators are reported.
    vnorm = v32.norm(dim=-1).to(f64)  # [bsz, n_kv_heads, k_len]

    def _abar(weights: torch.Tensor, vn: torch.Tensor, n: int) -> torch.Tensor:
        if n == 0:
            return torch.zeros((bsz, n_q_heads), dtype=f64, device=device)
        wg = weights.view(bsz, n_kv_heads, groups, n)
        return torch.einsum("bhgn,bhn->bhg", wg.to(f64), vn).reshape(bsz, n_q_heads)

    abar_dense = _abar(w_ctx, vnorm[:, :, :ctx_len], ctx_len) + _abar(w_suf, vnorm[:, :, ctx_len:], n_suf)

    b_full_dense = b_ctx_dense + b_suffix
    b_full_sparse = b_ctx_sparse + b_suffix
    attn_dense = a_dense / b_full_dense.clamp_min(1e-300).unsqueeze(-1)
    attn_sparse = a_sparse / b_full_sparse.clamp_min(1e-300).unsqueeze(-1)

    d_num = (a_sparse - a_dense).norm(dim=-1)
    norm_a = a_dense.norm(dim=-1)
    d_attn = (attn_sparse - attn_dense).norm(dim=-1)
    norm_attn = attn_dense.norm(dim=-1)

    def _log(x: torch.Tensor) -> torch.Tensor:
        return torch.log(x) + shift.squeeze(-1).to(f64)

    cpu = dict(device="cpu")
    return HeadStats(
        log_b_sparse=_log(b_ctx_sparse).to(**cpu),
        log_b_dense=_log(b_ctx_dense).to(**cpu),
        log_b_suffix=_log(b_suffix).to(**cpu),
        n_kept_ctx=n_kept.to(**cpu),
        shift=shift.squeeze(-1).to(f64).to(**cpu),
        num_err=(d_num / norm_a.clamp_min(1e-300)).to(**cpu),
        num_err_abar=(d_num / abar_dense.clamp_min(1e-300)).to(**cpu),
        attn_err=(d_attn / norm_attn.clamp_min(1e-300)).to(**cpu),
        norm_a_dense=norm_a.to(**cpu),
        abar_dense=abar_dense.to(**cpu),
    )
