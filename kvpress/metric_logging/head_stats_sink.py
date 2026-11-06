# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Collector and on-disk sink for per-head upstream error statistics.

One row per (context, question, decode step, layer, query head). At 32 layers x 32 heads x ~4
decode steps x ~50 questions that is ~200k rows per run, which is why this writes parquet beside
``micro_metrics.jsonl`` rather than going through the JSON Lines logger: ~7 MB raw and ~1.5 MB
compressed per run, loadable in under a second, against ~60 MB of JSON.

The run configuration is written into the parquet schema metadata, so a file is interpretable
without reference to the directory it came from.

Striding is evaluated BEFORE the statistics are computed, so raising a stride saves the einsum and
not merely the write. MicroMetricLogger.sampling_factor is deliberately not reused for this: it
samples after the compute and independently per record, which would break up the per-head tuples
and make a bias/variance decomposition impossible.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1"

METRIC_DEFINITION = (
    "Per query head, for one query row: log_b_* = log sum_j exp(<q,k_j>*scaling) over the stated "
    "key subset (natural log); ctx = [0,ctx_len) is compressible, suffix = [ctx_len,k_len) is not. "
    "num_err/attn_err are relative errors of the numerator sum_j w_j v_j and of the ratio A/B, both "
    "over the FULL key set. num_err_abar divides by sum_j w_j ||v_j|| instead of ||A||."
)

_COLUMNS = {
    "context_idx": np.int32,
    "context_crc": np.uint32,
    "question_idx": np.int16,
    "decode_step": np.int16,
    "layer": np.int16,
    "query_head": np.int16,
    "kv_head": np.int16,
    "log_b_sparse": np.float32,
    "log_b_dense": np.float32,
    "log_b_suffix": np.float32,
    "n_kept_ctx": np.int32,
    "ctx_len": np.int32,
    "k_len": np.int32,
    "q_len": np.int16,
    "q_pos": np.int32,
    "shift": np.float32,
    "num_err": np.float32,
    "num_err_abar": np.float32,
    "attn_err": np.float32,
    "norm_a_dense": np.float32,
    "abar_dense": np.float32,
    "source_id": np.int8,
}

SOURCES = ("attention_bias", "vkvwr_bias", "masked_keys", "full_kv")


class HeadStatsCollector:
    """Singleton. Disabled by default, so importing it changes nothing."""

    _instance: Optional["HeadStatsCollector"] = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._init()
        return cls._instance

    def _init(self):
        self.enabled = False
        self.path: Optional[Path] = None
        self.layer_stride = 1
        self.step_stride = 1
        self.question_stride = 1
        self.max_rows: Optional[int] = None
        self.flush_every = 20000
        self.run_config: dict = {}
        self._buf: list[dict] = []
        self._rows = 0
        self._truncated = False
        self._writer = None
        self._context_idx = -1
        self._context_crc = 0
        self._question_idx = -1
        self._decode_step = -1
        self._last_layer = None

    # ---------------------------------------------------------------- configuration

    def configure(self, *, log_path, run_config=None, layer_stride=1, step_stride=1,
                  question_stride=1, max_rows=None, flush_every=20000):
        self.close()
        self._init()
        self.enabled = True
        self.path = Path(log_path) / "head_stats.parquet"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.layer_stride = max(1, int(layer_stride))
        self.step_stride = max(1, int(step_stride))
        self.question_stride = max(1, int(question_stride))
        self.max_rows = max_rows
        self.flush_every = max(1, int(flush_every))
        self.run_config = dict(run_config or {})
        logger.info(f"Head stats -> {self.path} (strides L{self.layer_stride}/S{self.step_stride}/Q{self.question_stride})")

    @classmethod
    def reset_for_testing(cls):
        with cls._lock:
            if cls._instance is not None:
                cls._instance.close()
            cls._instance = None

    # ---------------------------------------------------------------- run structure

    def begin_context(self, context_ids=None):
        import zlib

        self._context_idx += 1
        self._question_idx = -1
        self._decode_step = 0
        self._last_layer = None
        if context_ids is not None:
            arr = context_ids.detach().cpu().numpy() if torch.is_tensor(context_ids) else np.asarray(context_ids)
            self._context_crc = zlib.crc32(arr.tobytes()) & 0xFFFFFFFF

    def begin_question(self, question_idx=None):
        self._question_idx = self._question_idx + 1 if question_idx is None else int(question_idx)
        self._decode_step = 0
        self._last_layer = None

    def begin_step(self):
        """Optional: steps are otherwise inferred from the layer index cycling back round."""
        self._decode_step += 1
        self._last_layer = None

    # ---------------------------------------------------------------- gating

    def should_record(self, layer_idx: int) -> bool:
        if not self.enabled:
            return False
        if self.max_rows is not None and self._rows >= self.max_rows:
            if not self._truncated:
                logger.warning(f"head stats truncated at max_rows={self.max_rows}")
                self._truncated = True
            return False
        # A forward pass walks the layers in increasing order, so the index going back down is a
        # new step. This needs no instrumentation in the generation loop and works for callers
        # that drive model(...) directly, which the tests do.
        if self._last_layer is not None and layer_idx <= self._last_layer:
            self._decode_step += 1
        self._last_layer = layer_idx
        return (
            layer_idx % self.layer_stride == 0
            and max(self._decode_step, 0) % self.step_stride == 0
            and max(self._question_idx, 0) % self.question_stride == 0
        )

    # ---------------------------------------------------------------- recording

    def record(self, stats, *, layer_idx: int, source: str, ctx_len: int, k_len: int,
               q_len: int, q_pos: int, groups: int):
        if not self.enabled:
            return
        n_heads = stats.log_b_dense.shape[1]
        src = SOURCES.index(source) if source in SOURCES else -1
        row = {
            "context_idx": self._context_idx,
            "context_crc": self._context_crc,
            "question_idx": max(self._question_idx, 0),
            "decode_step": max(self._decode_step, 0),
            "layer": layer_idx,
            "query_head": np.arange(n_heads),
            "kv_head": np.arange(n_heads) // groups,
            "log_b_sparse": stats.log_b_sparse[0].numpy(),
            "log_b_dense": stats.log_b_dense[0].numpy(),
            "log_b_suffix": stats.log_b_suffix[0].numpy(),
            "n_kept_ctx": stats.n_kept_ctx[0].numpy(),
            "ctx_len": ctx_len,
            "k_len": k_len,
            "q_len": q_len,
            "q_pos": q_pos,
            "shift": stats.shift[0].numpy(),
            "num_err": stats.num_err[0].numpy(),
            "num_err_abar": stats.num_err_abar[0].numpy(),
            "attn_err": stats.attn_err[0].numpy(),
            "norm_a_dense": stats.norm_a_dense[0].numpy(),
            "abar_dense": stats.abar_dense[0].numpy(),
            "source_id": src,
        }
        self._buf.append(row)
        self._rows += n_heads
        if self._rows % self.flush_every < n_heads:
            self.flush()

    # ---------------------------------------------------------------- output

    def _columns(self):
        out = {}
        for name, dtype in _COLUMNS.items():
            parts = []
            for row in self._buf:
                val = row[name]
                n = len(row["query_head"])
                parts.append(np.asarray(val) if np.ndim(val) else np.full(n, val))
            out[name] = np.concatenate(parts).astype(dtype, copy=False)
        return out

    def flush(self):
        if not self._buf:
            return
        cols = self._columns()
        self._buf.clear()
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq

            table = pa.table({k: pa.array(v) for k, v in cols.items()})
            if self._writer is None:
                meta = {
                    b"kvpress_head_stats_schema_version": SCHEMA_VERSION.encode(),
                    b"kvpress_metric_definition": METRIC_DEFINITION.encode(),
                    b"kvpress_run_config": json.dumps(self.run_config, default=str).encode(),
                    b"kvpress_sources": json.dumps(list(SOURCES)).encode(),
                }
                self._writer = pq.ParquetWriter(
                    self.path, table.schema.with_metadata(meta), compression="zstd"
                )
            self._writer.write_table(table)  # one row group per flush: a killed run stays readable
        except ImportError:
            self._npz_append(cols)

    def _npz_append(self, cols):
        path = self.path.with_suffix(".npz")
        if path.exists():
            old = dict(np.load(path, allow_pickle=False))
            cols = {k: np.concatenate([old[k], v]) for k, v in cols.items()}
        np.savez_compressed(path, config=np.array(json.dumps(self.run_config, default=str)), **cols)

    def close(self):
        if not getattr(self, "enabled", False):
            return
        self.flush()
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        logger.info(f"Head stats: wrote {self._rows} rows to {self.path}")
        self.enabled = False
