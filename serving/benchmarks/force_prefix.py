"""Per-request forced prefix of the output: a request whose ``SamplingParams.extra_args`` holds
``{"force_ids": [...]}`` samples exactly those tokens first, one decode step each, then decodes
freely. ``ruler_vllm.py`` uses it to run the question as decode steps on top of the compressed
context (the harness's question pass: every question position gets its own decision, and the
question keys are generated tokens, so exact and resident).

Add it with ``LLM(..., logits_processors=[ForcePrefix])`` (in-process engine) and call
``install_stop_guard()`` once: the forced tokens may contain stop tokens (the chat suffix's
``<|im_end|>``), and vLLM's ``check_stop`` tests EOS / stop ids before anything else, so the guard
skips it until the forced prefix is out (the length cap still applies). No host sync: masking is
two device writes per forced row.
"""
import torch

from vllm.v1.sample.logits_processor.interface import (
    BatchUpdate, LogitsProcessor, MoveDirectionality)


def _forced_len(request) -> int:
    sp = request.sampling_params
    ids = (sp.extra_args or {}).get("force_ids") if sp is not None else None
    return len(ids) if ids else 0


def install_stop_guard() -> None:
    import vllm.v1.core.sched.scheduler as sched_mod
    orig = sched_mod.check_stop
    if getattr(orig, "_force_guard", False):
        return

    def check_stop(request, max_model_len):
        Q = _forced_len(request)
        if (Q and request.num_output_tokens <= Q and request.num_tokens < max_model_len
                and request.num_output_tokens < request.max_tokens):
            return False
        return orig(request, max_model_len)

    check_stop._force_guard = True
    sched_mod.check_stop = check_stop


class ForcePrefix(LogitsProcessor):
    def __init__(self, vllm_config, device, is_pin_memory):
        self.rows: dict[int, tuple[list[int], list[int]]] = {}   # batch index -> (forced, out)

    def is_argmax_invariant(self) -> bool:
        return False

    def update_state(self, batch_update: BatchUpdate | None) -> None:
        if batch_update is None:
            return
        for i in batch_update.removed:
            self.rows.pop(i, None)
        for i, params, _prompt, out in batch_update.added:
            ids = (params.extra_args or {}).get("force_ids") if params is not None else None
            if ids:
                self.rows[i] = (list(ids), out)
            else:
                self.rows.pop(i, None)
        for a, b, d in batch_update.moved:
            ra, rb = self.rows.pop(a, None), self.rows.pop(b, None)
            if ra is not None:
                self.rows[b] = ra
            if d == MoveDirectionality.SWAP and rb is not None:
                self.rows[a] = rb

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        for i, (ids, out) in self.rows.items():
            t = len(out)                      # output tokens so far = the step's index
            if t < len(ids):
                logits[i].fill_(float("-inf"))
                logits[i, ids[t]] = 0.0
        return logits
