# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Press-agnostic micro-metrics for attention / KV sparsity and output error.

Mirrors sparse_attention_hub's density / output-error pair:
- ``attention_sparsity`` is the dropped fraction (``1 - density``)
- ``attention_output_error`` is ``||sparse - dense||_F / ||dense||_F`` on attention outputs
- ``denominator_error`` is ``|D̂ - D| / D`` for VKvWR's IS residual-denominator estimate

Sparsity is logged from attention masks/biases, head-wise masked keys, or cache
eviction (see metadata ``source``). Relative output error needs a dense reference:
bias / masked-key paths keep the full KV in the live cache; eviction presses store
a prefill copy under ``metadata["full_kv"]`` for the attention patch to use.
Denominator error is logged on the VKvWR decode path.
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from kvpress.metric_logging.logger import MicroMetricLogger

ATTENTION_SPARSITY = "attention_sparsity"
ATTENTION_OUTPUT_ERROR = "attention_output_error"
DENOMINATOR_ERROR = "denominator_error"
FULL_KV_METADATA_KEY = "full_kv"

DEFAULT_MICRO_METRICS = (ATTENTION_SPARSITY, ATTENTION_OUTPUT_ERROR, DENOMINATOR_ERROR)

MicroMetricLogger.register_metric(ATTENTION_SPARSITY, float)
MicroMetricLogger.register_metric(ATTENTION_OUTPUT_ERROR, float)
MicroMetricLogger.register_metric(DENOMINATOR_ERROR, float)


def bias_sparsity(bias: torch.Tensor) -> float:
    """Fraction of context KV positions dropped (non-finite / ``-inf`` bias)."""
    return float((~torch.isfinite(bias)).float().mean().item())


def masked_key_sparsity(
    masked_key_indices: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    batch_size: int,
    num_kv_heads: int,
    seq_len: int,
) -> float:
    """Fraction of KV positions nullified via ``masked_key_indices``."""
    total = batch_size * num_kv_heads * seq_len
    if total == 0:
        return 0.0
    return float(masked_key_indices[0].numel()) / float(total)


def cache_sparsity(seq_len_before: int, seq_len_after: int) -> float:
    """Fraction of tokens removed from the KV cache."""
    if seq_len_before <= 0:
        return 0.0
    return float(max(0, seq_len_before - seq_len_after)) / float(seq_len_before)


def relative_attention_output_error(sparse_output: torch.Tensor, dense_output: torch.Tensor) -> float:
    """Relative Frobenius error of sparse vs dense attention outputs."""
    dense = dense_output.float()
    sparse = sparse_output.float()
    denom = torch.norm(dense)
    if denom.item() == 0.0:
        return 0.0
    return float((torch.norm(sparse - dense) / denom).item())


def _attn_output(result: Any) -> torch.Tensor:
    return result[0] if isinstance(result, tuple) else result


def maybe_log_sparsity(
    value: float,
    *,
    layer_idx: int,
    q_len: Optional[int] = None,
    k_len: Optional[int] = None,
    source: str,
) -> None:
    logger = MicroMetricLogger()
    if not logger.is_metric_enabled(ATTENTION_SPARSITY):
        return
    metadata: dict[str, Any] = {"layer_idx": layer_idx, "source": source}
    if q_len is not None:
        metadata["q_len"] = q_len
    if k_len is not None:
        metadata["k_len"] = k_len
    logger.log(ATTENTION_SPARSITY, float(value), metadata=metadata)


def maybe_log_attention_output_error(
    sparse_result: Any,
    dense_result: Any,
    *,
    layer_idx: int,
    q_len: int,
    k_len: int,
    source: str,
) -> None:
    logger = MicroMetricLogger()
    if not logger.is_metric_enabled(ATTENTION_OUTPUT_ERROR):
        return
    logger.log(
        ATTENTION_OUTPUT_ERROR,
        relative_attention_output_error(_attn_output(sparse_result), _attn_output(dense_result)),
        metadata={"layer_idx": layer_idx, "q_len": q_len, "k_len": k_len, "source": source},
    )


def want_attention_output_error() -> bool:
    return MicroMetricLogger().is_metric_enabled(ATTENTION_OUTPUT_ERROR)


def want_store_full_kv() -> bool:
    """Whether eviction presses should snapshot the uncompressed prefill KV into metadata."""
    return want_attention_output_error()


def store_full_kv(
    metadata: dict[str, Any],
    *,
    layer_idx: int,
    keys: torch.Tensor,
    values: torch.Tensor,
    compressed_ctx_len: int,
) -> None:
    """Persist a detached copy of the pre-compression KV for dense micro-metric forwards."""
    metadata.setdefault(FULL_KV_METADATA_KEY, {})[int(layer_idx)] = {
        "keys": keys.detach(),
        "values": values.detach(),
        "ctx_len": int(keys.shape[2]),
        "compressed_ctx_len": int(compressed_ctx_len),
    }


def get_full_kv(metadata: Optional[dict[str, Any]], layer_idx: int) -> Optional[dict[str, Any]]:
    if metadata is None:
        return None
    return metadata.get(FULL_KV_METADATA_KEY, {}).get(int(layer_idx))


def reconstruct_dense_kv(
    full_kv: dict[str, Any],
    key: torch.Tensor,
    value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build dense (key, value) as full prefill context plus post-compression suffix from ``key``/``value``."""
    compressed_ctx = int(full_kv["compressed_ctx_len"])
    if key.shape[2] < compressed_ctx:
        raise ValueError(
            f"live cache length {key.shape[2]} is shorter than compressed_ctx_len {compressed_ctx} "
            "(stale full_kv metadata?)"
        )
    if key.shape[2] == compressed_ctx:
        return full_kv["keys"], full_kv["values"]
    return (
        torch.cat([full_kv["keys"], key[:, :, compressed_ctx:]], dim=2),
        torch.cat([full_kv["values"], value[:, :, compressed_ctx:]], dim=2),
    )


def relative_denominator_error(d_hat: float, d_true: float) -> float:
    """Relative error ``|D̂ - D| / D`` of the IS residual-denominator estimate."""
    d_true = float(d_true)
    d_hat = float(d_hat)
    if d_true <= 0.0:
        return 0.0 if d_hat == 0.0 else float("inf")
    return abs(d_hat - d_true) / d_true


def maybe_log_denominator_error(
    d_hat: float,
    d_true: float,
    *,
    layer_idx: int,
    q_len: Optional[int] = None,
    k_len: Optional[int] = None,
    source: str = "vkvwr_bias",
    error: Optional[float] = None,
    n_heads: Optional[int] = None,
) -> None:
    """Log ``denominator_error`` with ``d_hat`` / ``d_true`` in metadata.

    If ``error`` is provided it is logged as-is (e.g. mean per-head relative error);
    otherwise ``|D̂ - D| / D`` is computed from ``d_hat`` and ``d_true``.
    """
    logger = MicroMetricLogger()
    if not logger.is_metric_enabled(DENOMINATOR_ERROR):
        return
    value = float(error) if error is not None else relative_denominator_error(d_hat, d_true)
    metadata: dict[str, Any] = {
        "layer_idx": layer_idx,
        "source": source,
        "d_hat": float(d_hat),
        "d_true": float(d_true),
    }
    if q_len is not None:
        metadata["q_len"] = q_len
    if k_len is not None:
        metadata["k_len"] = k_len
    if n_heads is not None:
        metadata["n_heads"] = int(n_heads)
    logger.log(DENOMINATOR_ERROR, value, metadata=metadata)


def want_denominator_error() -> bool:
    return MicroMetricLogger().is_metric_enabled(DENOMINATOR_ERROR)


# ---------------------------------------------------------------------------------------------------------------------
# Per-head upstream error statistics (denominator, numerator and the attention ratio).
#
# Separate from the JSON Lines metrics above: these are per (layer, query head) rather than per
# layer, and they go to a parquet sidecar because there are ~200k rows per run. The JSONL output
# of the three metrics above is unchanged whether or not this is enabled.
# ---------------------------------------------------------------------------------------------------------------------


def head_stats_enabled() -> bool:
    """Whether per-head statistics are being collected at all. Cheap, and does not advance the
    collector's step bookkeeping, unlike want_head_stats."""
    from kvpress.metric_logging.head_stats_sink import HeadStatsCollector

    return HeadStatsCollector().enabled


def want_head_stats(layer_idx: int) -> bool:
    """True when per-head statistics should be computed for this layer, before doing the work."""
    from kvpress.metric_logging.head_stats_sink import HeadStatsCollector

    return HeadStatsCollector().should_record(layer_idx)


def record_head_stats(
    *,
    module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    kwargs: Optional[dict] = None,
    layer_idx: int,
    source: str,
    ctx_len: int,
    bias: Optional[torch.Tensor] = None,
    keep_mask: Optional[torch.Tensor] = None,
) -> None:
    """
    Compute and record the per-head statistics. Must be called with the REAL keys, i.e. before any
    press overwrites them with fake keys.
    """
    from kvpress.metric_logging.head_stats import compute_head_stats, resolve_scaling
    from kvpress.metric_logging.head_stats_sink import HeadStatsCollector

    collector = HeadStatsCollector()
    if not collector.enabled:
        return
    q_len, k_len = query.shape[2], key.shape[2]
    try:
        stats = compute_head_stats(
            query=query,
            key=key,
            value=value,
            attention_mask=attention_mask,
            scaling=resolve_scaling(module, query, kwargs),
            ctx_len=ctx_len,
            bias=bias,
            keep_mask=keep_mask,
            q_index=-1,
        )
    except Exception as exc:  # a metric must never take the run down
        logger.warning(f"head stats skipped at layer {layer_idx}: {exc}")
        return
    collector.record(
        stats,
        layer_idx=layer_idx,
        source=source,
        ctx_len=ctx_len,
        k_len=k_len,
        q_len=q_len,
        q_pos=k_len - 1,
        groups=query.shape[1] // key.shape[1],
    )
