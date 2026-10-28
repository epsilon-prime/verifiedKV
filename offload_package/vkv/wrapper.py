"""Skylight decode wrapper for Vkv (``SKYLIGHT_SPARSE_METHOD=vkv``).

The Skylight backend (``skylight.backend``) builds one decode wrapper per batch size and hands it
the offload orchestrator and this step's decode request ids every build; ``run`` is called per
layer. All the work is ``VkvDecoder``: this class only adapts the FlashInfer-style wrapper
interface (plan / run / sparsity stats). Requires per-head token offloading with
``token_policy='vkv'``. vLLM must not capture it (eager or piecewise); ``VKV_GRAPHS=1`` replays
each layer's attention from the decoder's own CUDA graphs.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, replace
from typing import Any, Optional

import torch
from original_optimized.batch_decode_with_paged_kv_cache_wrapper import (
    BatchDecodeWithPagedKVCacheWrapper as _DenseOptimizedWrapper,
    _unpack_paged_kv_cache,
)
from sparse_optimized.cuda_ops import get_sparse_decode_ops

from .decode import VkvDecoder


@dataclass
class SparsityStats:
    """Same fields as the other Skylight wrappers' (the backend logs them)."""

    last_selected: int = 0
    last_total: int = 0
    last_fraction: float = 0.0
    cumulative_selected: int = 0
    cumulative_total: int = 0
    call_count: int = 0
    selected_is_upper_bound: bool = True


class BatchDecodeWithPagedKVCacheWrapper(_DenseOptimizedWrapper):
    def __init__(self, float_workspace_buffer: torch.Tensor, kv_layout: str = "NHD",
                 use_cuda_graph: bool = False, use_tensor_cores: bool = False,
                 paged_kv_indptr_buffer: Optional[torch.Tensor] = None,
                 paged_kv_indices_buffer: Optional[torch.Tensor] = None,
                 paged_kv_last_page_len_buffer: Optional[torch.Tensor] = None,
                 backend: str = "auto", jit_args: Optional[list[Any]] = None,
                 **_sparse_cfg: Any) -> None:
        # FULL decode graphs: only the attend-only rows of KVzip eviction (VKV_FULL=1, checked
        # by the worker connector); every other vkv decode runs outside vLLM's graphs.
        self._full = bool(use_cuda_graph)
        if use_cuda_graph and os.environ.get("VKV_FULL") != "1":
            raise RuntimeError("vkv decode in FULL CUDA graphs needs VKV_FULL=1 (evict only)")
        super().__init__(
            float_workspace_buffer=float_workspace_buffer, kv_layout=kv_layout,
            use_cuda_graph=use_cuda_graph, use_tensor_cores=use_tensor_cores,
            paged_kv_indptr_buffer=paged_kv_indptr_buffer,
            paged_kv_indices_buffer=paged_kv_indices_buffer,
            paged_kv_last_page_len_buffer=paged_kv_last_page_len_buffer,
            backend=backend, jit_args=jit_args)
        self._sparse_ops = get_sparse_decode_ops()
        self._orch = None
        self._req_ids: list[str] | None = None
        self._decoder_obj: VkvDecoder | None = None
        self._step_ready = False
        self._sparsity_stats = SparsityStats()

    # ---- what the Skylight backend calls ----------------------------------------------------------

    def get_sparsity_stats(self) -> SparsityStats:
        return replace(self._sparsity_stats)

    def set_n_keys(self, n_keys) -> None:
        pass

    def set_offload_orchestrator(self, orchestrator) -> None:
        self._orch = orchestrator

    def set_offload_request_ids(self, request_ids) -> None:
        self._req_ids = list(request_ids) if request_ids else None
        eng = getattr(self._orch, "token_offload", None)
        if eng is not None and getattr(eng, "headwise", False):
            eng.set_decode_rows(self._req_ids)   # the staging kernels read the rows map
        self._step_ready = False
        if self._full and self._req_ids and not torch.cuda.is_current_stream_capturing():
            # The step's host work, eagerly, before vLLM replays the captured decode.
            self._decoder_for_step()

    def plan(self, *args: Any, num_qo_heads: int = 0, num_kv_heads: int = 0, head_dim: int = 0,
             logits_soft_cap: Optional[float] = None, sm_scale: Optional[float] = None,
             **kwargs: Any) -> None:
        """Vkv decodes from the engine's cells, never through the paged table, so the base
        class's split-KV plan (host work that grows with the context) is skipped: only what
        ``run`` reads is recorded."""
        self._num_qo_heads, self._num_kv_heads, self._head_dim = (
            int(num_qo_heads), int(num_kv_heads), int(head_dim))
        self._logits_soft_cap = 0.0 if logits_soft_cap is None else float(logits_soft_cap)
        self._sm_scale = sm_scale
        self._planned = True
        self._step_ready = False

    def _decoder(self) -> VkvDecoder:
        orch = self._orch
        eng = getattr(orch, "token_offload", None)
        store = getattr(orch, "token_store", None)
        if eng is None or store is None or getattr(eng, "vkv", None) is None:
            raise RuntimeError("SKYLIGHT_SPARSE_METHOD=vkv needs OffloadPackageConnector with "
                               "offload_granularity='token' and token_policy='vkv'")
        if self._decoder_obj is None or self._decoder_obj.eng is not eng:
            self._decoder_obj = VkvDecoder(eng, store, self._sparse_ops)
            self._decoder_obj.full = self._full
        return self._decoder_obj

    def _decoder_for_step(self) -> VkvDecoder:
        dec = self._decoder()
        if not self._step_ready:
            if not self._req_ids:
                raise RuntimeError("vkv decode: no request ids for this step")
            dec.begin_step(self._req_ids, getattr(self, "_num_qo_heads", 0) or None)
            self._step_ready = True
        return dec

    # ---- run -----------------------------------------------------------------------------------------

    def run(self, q: torch.Tensor, paged_kv_cache, *args: Any, out: Optional[torch.Tensor] = None,
            return_lse: bool = False, **kwargs: Any):
        del args, kwargs
        if not self._planned:
            raise RuntimeError("plan() must be called before run().")
        capturing = torch.cuda.is_current_stream_capturing()
        if capturing and not self._full:
            raise RuntimeError("vkv decode cannot be captured in a CUDA graph (use enforce_eager)")
        if self._full and (capturing or self._req_ids):
            return self._run_full(q, paged_kv_cache, out, return_lse)
        if not self._req_ids:
            # A dummy batch (vLLM's warm-up / piecewise-graph capture runs): no request.
            o = torch.zeros_like(q) if out is None else out.zero_()
            lse = torch.full(q.shape[:2], float("-inf"), dtype=torch.float32, device=q.device)
            return (o, lse) if return_lse else o
        k_cache = _unpack_paged_kv_cache(paged_kv_cache, self._kv_layout)[0]
        dec = self._decoder_for_step()
        if k_cache.dtype != dec.eng.kv_rows.dtype:
            raise RuntimeError("vkv decode: KV dtype differs from the engine's cells (fp8 KV "
                               "is not supported)")
        layer = self._orch.staging_pool.layer_of(k_cache.data_ptr())
        sm_scale = self._sm_scale if self._sm_scale is not None else 1.0 / math.sqrt(q.shape[-1])
        run = dec.layer_graphed if dec.graphs else dec.layer
        o, lse = run(layer, q, float(sm_scale), float(self._logits_soft_cap or 0.0),
                     need_lse=return_lse)
        st = self._sparsity_stats
        st.call_count += 1
        if out is not None:
            out.copy_(o.to(out.dtype))
            o = out
        else:
            o = o.to(q.dtype)
            if dec.graphs:                  # graph outputs are shared by all layers
                o = o.clone()
        if return_lse and dec.graphs:
            lse = lse.clone()
        return (o, lse) if return_lse else o

    def _run_full(self, q, paged_kv_cache, out, return_lse):
        """FULL-graph decode (captured once, replayed every step): fixed shapes, rows from the
        engine's persistent map, no Python state read at replay."""
        if getattr(getattr(self._orch, "token_offload", None), "vkv", None) is None:
            # vLLM's graph-memory profiling runs before the KV cache (and the token engine)
            # exists: a stand-in; the real capture comes after initialisation.
            o = torch.zeros_like(q) if out is None else out.zero_()
            lse = torch.full(q.shape[:2], float("-inf"), dtype=torch.float32, device=q.device)
            return (o, lse) if return_lse else o
        dec = self._decoder()
        dec.full = True
        k_cache = _unpack_paged_kv_cache(paged_kv_cache, self._kv_layout)[0]
        layer = self._orch.staging_pool.layer_of(k_cache.data_ptr())
        sm_scale = self._sm_scale if self._sm_scale is not None else 1.0 / math.sqrt(q.shape[-1])
        o, lse = dec.layer_full(layer, q, float(sm_scale), float(self._logits_soft_cap or 0.0),
                                need_lse=return_lse)
        if out is not None:
            out.copy_(o.to(out.dtype))
            o = out
        else:
            o = o.to(q.dtype)
        return (o, lse) if return_lse else o

    run_return_lse = lambda self, *a, **kw: self.run(*a, **kw, return_lse=True)
