# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Model-loading helpers for checkpoints the stock ``AutoModelForCausalLM`` route does not handle."""

from __future__ import annotations

import logging
from typing import Any, Optional

from transformers import AutoConfig, PreTrainedModel, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

# Ministral 3 ships as Mistral3ForConditionalGeneration (a VLM) with the text model stored under
# ``language_model.``; Mistral3Config has no AutoModelForCausalLM mapping.
_MISTRAL3_KEY_MAPPING = {r"^language_model\.model\.": "model.", r"^language_model\.lm_head\.": "lm_head."}


def needs_text_only_load(model_name: str, trust_remote_code: bool = False) -> bool:
    """True for checkpoints whose causal LM must be loaded from inside a multimodal wrapper."""
    return AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code).model_type == "mistral3"


def load_text_only_causal_lm(
    model_name: str, model_kwargs: Optional[dict[str, Any]] = None, device_map: Any = "auto"
) -> PreTrainedModel:
    """Load the text decoder of a multimodal checkpoint as a causal LM (the vision tower is not loaded).

    Raises if any language-model weight is missing, so a renamed checkpoint cannot silently load with
    randomly initialised layers.
    """
    from transformers import Ministral3ForCausalLM

    config = AutoConfig.from_pretrained(model_name)
    assert config.model_type == "mistral3", f"no text-only loader for model_type {config.model_type!r}"
    kwargs = dict(model_kwargs or {})
    kwargs.setdefault("dtype", "auto")
    model, info = Ministral3ForCausalLM.from_pretrained(
        model_name,
        config=config.text_config,
        device_map=device_map,
        key_mapping=_MISTRAL3_KEY_MAPPING,
        output_loading_info=True,
        **kwargs,
    )
    if info["missing_keys"]:
        missing = sorted(info["missing_keys"])
        raise RuntimeError(f"{model_name}: language-model weights missing after remapping: {missing[:8]}")
    logger.info(f"{model_name}: loaded the text model only ({len(info['unexpected_keys'])} vision weights skipped)")
    return model


def add_chat_eos(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase) -> None:
    """Stop generation at the tokenizer's end-of-turn token as well as the configured ones.

    Qwen3.5's config stops only at ``<|endoftext|>`` while its chat turns end with ``<|im_end|>`` (the
    tokenizer's eos), and it ships no generation config. Models whose generation config already lists
    the tokenizer's eos (Llama 3.1, Qwen3, Mistral) are unchanged.
    """
    eos = tokenizer.eos_token_id
    if eos is None:
        return
    current = model.generation_config.eos_token_id
    current = [] if current is None else (list(current) if isinstance(current, (list, tuple)) else [current])
    if eos not in current:
        model.generation_config.eos_token_id = current + [eos]
        logger.info(f"generation stops at {model.generation_config.eos_token_id} (added the tokenizer's eos {eos})")


def suppress_think(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase) -> None:
    """Never generate ``<think>``: evaluation runs in non-thinking mode (the pipeline's default).

    Qwen3.5's template closes an empty think block when ``enable_thinking=False``, yet on LOFT the model
    still opens a fresh ``<think>`` after the "Final Answer: " prefix on most questions and spends its
    token budget reasoning (exact match 0/10 on dev; 4/10 with the token suppressed, same prompt).
    Models without a ``<think>`` token are unchanged; non-thinking models that have one never emit it.
    """
    think = tokenizer.convert_tokens_to_ids("<think>")
    if think is None or think == tokenizer.unk_token_id:
        return
    current = list(getattr(model.generation_config, "suppress_tokens", None) or [])
    if think not in current:
        model.generation_config.suppress_tokens = current + [think]
        logger.info(f"generation never picks <think> ({think})")
