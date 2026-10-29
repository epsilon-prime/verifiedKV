"""Prefill-only FastKVzip scoring under torch.compile (vLLM piecewise CUDA graphs).

The score hooks of ``token_mode.install_score_hooks`` are Python forward pre-hooks, which torch.compile
cannot trace. Here every ``Qwen3Attention.forward`` instead starts with the custom op
``vllm::offload_fkvz_score(hidden, positions, tok)``, listed in ``splitting_ops`` (``splitting_ops()``):
piecewise compilation splits the graph at it and runs it eagerly between the compiled pieces, as
it runs attention. The op scores the layer's attention input with ``token_mode.layer_score_fn``
(``prefill_only``: a decode-only forward returns at once), for consumers that read the scores once,
at the end of prefill (vkv's round).

``tok`` is a one-element buffer of each attention module, declared mutated so the op is kept in
the graph (it is never written); it also names the op's layer (``bind`` maps its address).

    fkvz_op.install()            # before the engine builds the model (bench_1req_off99 does it)
    LLM(..., compilation_config={..., "splitting_ops": fkvz_op.splitting_ops()})
    fkvz_op.bind(runner, engine, gates, batched)    # worker_connector, once the engine exists
"""
from __future__ import annotations

import torch

OP = "vllm::offload_fkvz_score"
_STATE = {"installed": False, "score": None, "layer": {}}


def _op(hidden: torch.Tensor, positions: torch.Tensor, tok: torch.Tensor) -> None:
    score = _STATE["score"]
    if score is None:
        return                                   # profiling / before the engine is bound
    li = _STATE["layer"].get(tok.data_ptr())
    if li is not None:
        score(li, hidden, positions)


def _op_fake(hidden: torch.Tensor, positions: torch.Tensor, tok: torch.Tensor) -> None:
    return None


def splitting_ops() -> list[str]:
    """vLLM's default piecewise split points (attention, the KV cache update) plus the op."""
    from vllm.config.compilation import CompilationConfig
    return list(CompilationConfig._attention_ops) + [
        "vllm::unified_kv_cache_update", "vllm::unified_mla_kv_cache_update", OP]


def install() -> None:
    """Register the op and make ``Qwen3Attention`` call it (idempotent). Must run before the
    model is built and compiled."""
    if _STATE["installed"]:
        return
    from vllm.model_executor.models.qwen3 import Qwen3Attention
    from vllm.utils.torch_utils import direct_register_custom_op

    direct_register_custom_op(op_name="offload_fkvz_score", op_func=_op,
                              mutates_args=["tok"], fake_impl=_op_fake)
    init, fwd = Qwen3Attention.__init__, Qwen3Attention.forward

    def __init__(self, *a, **k):
        init(self, *a, **k)
        self.register_buffer("_fkvz_tok", torch.zeros(1), persistent=False)

    def forward(self, positions, hidden_states):
        torch.ops.vllm.offload_fkvz_score(hidden_states, positions, self._fkvz_tok)
        return fwd(self, positions, hidden_states)

    Qwen3Attention.__init__, Qwen3Attention.forward = __init__, forward
    _STATE["installed"] = True


def installed() -> bool:
    return _STATE["installed"]


def bind(runner, engine, gates, batched=None) -> None:
    """Score through ``engine`` from now on (prefill forwards only)."""
    from offload_package.integration.token_mode import _decoder_layers, layer_score_fn
    layers = _decoder_layers(runner.model)
    _STATE["layer"] = {l.self_attn._fkvz_tok.data_ptr(): i for i, l in enumerate(layers)}
    _STATE["score"] = layer_score_fn(runner, engine, gates, batched, prefill_only=True)
