# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch
from transformers import GenerationConfig

from kvpress.model_loading import add_chat_eos, load_text_only_causal_lm, needs_text_only_load, suppress_think
from tests.adapters.test_ministral3_adapter import tiny_ministral3_config


def _fake(eos_config, eos_tokenizer):
    model = SimpleNamespace(generation_config=GenerationConfig(eos_token_id=eos_config))
    return model, SimpleNamespace(eos_token_id=eos_tokenizer)


def test_add_chat_eos_appends_the_tokenizer_eos():
    """Qwen3.5: config stops at <|endoftext|> (248044) only; chat turns end with <|im_end|> (248046)."""
    model, tok = _fake(248044, 248046)
    add_chat_eos(model, tok)
    assert model.generation_config.eos_token_id == [248044, 248046]


@pytest.mark.parametrize("eos_config", [[128001, 128008, 128009], 128009])
def test_add_chat_eos_leaves_models_that_already_stop(eos_config):
    """Llama 3.1 style: the tokenizer eos is already listed, nothing changes."""
    model, tok = _fake(eos_config, 128009)
    add_chat_eos(model, tok)
    assert model.generation_config.eos_token_id == eos_config


@pytest.fixture(scope="module")
def tiny_mistral3_checkpoint(tmp_path_factory):
    """A randomly initialised Mistral3ForConditionalGeneration (VLM wrapper around a Ministral 3 text model),
    rewritten into the Hub layout of mistralai/Ministral-3-*-Instruct-2512: text weights under
    ``language_model.model.`` and ``language_model.lm_head.`` (transformers 5 itself saves other names),
    with an untied lm_head as in the real 8B/14B checkpoints."""
    from safetensors.torch import load_file, save_file
    from transformers import Mistral3Config, Mistral3ForConditionalGeneration, PixtralVisionConfig

    vision = PixtralVisionConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2, head_dim=16,
        image_size=32, patch_size=16,
    )
    text = tiny_ministral3_config(tie_word_embeddings=False)
    config = Mistral3Config(text_config=text, vision_config=vision, image_token_index=127, tie_word_embeddings=False)
    torch.manual_seed(0)
    vlm = Mistral3ForConditionalGeneration(config).eval()
    path = tmp_path_factory.mktemp("mistral3")
    vlm.save_pretrained(path)

    hub = {}
    for name, tensor in vlm.state_dict().items():  # runtime names -> Hub checkpoint names
        if name.startswith("model.language_model."):
            hub["language_model.model." + name[len("model.language_model."):]] = tensor
        elif name.startswith("lm_head."):
            hub["language_model." + name] = tensor
        elif name.startswith("model."):
            hub[name[len("model."):]] = tensor  # vision_tower.*, multi_modal_projector.*
        else:
            hub[name] = tensor
    for f in path.glob("*.safetensors"):
        f.unlink()
    (path / "model.safetensors.index.json").unlink(missing_ok=True)
    save_file({k: v.contiguous() for k, v in hub.items()}, str(path / "model.safetensors"), metadata={"format": "pt"})
    assert any(k.startswith("language_model.model.layers.") for k in load_file(str(path / "model.safetensors")))
    return str(path), vlm


def test_text_only_load_matches_the_vlm_on_text(tiny_mistral3_checkpoint):
    path, vlm = tiny_mistral3_checkpoint
    assert needs_text_only_load(path)
    model = load_text_only_causal_lm(path, {"dtype": torch.float32}, device_map="cpu")
    assert type(model).__name__ == "Ministral3ForCausalLM"
    ids = torch.randint(0, 120, (1, 24))
    with torch.no_grad():
        expected = vlm(input_ids=ids).logits
        actual = model(input_ids=ids).logits
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def _fake_tok(vocab, unk=0):
    return SimpleNamespace(convert_tokens_to_ids=lambda t: vocab.get(t, unk), unk_token_id=unk)


def test_suppress_think_adds_the_think_token():
    model = SimpleNamespace(generation_config=GenerationConfig())
    suppress_think(model, _fake_tok({"<think>": 248068}))
    assert model.generation_config.suppress_tokens == [248068]
    suppress_think(model, _fake_tok({"<think>": 248068}))  # idempotent
    assert model.generation_config.suppress_tokens == [248068]


def test_suppress_think_leaves_models_without_the_token():
    model = SimpleNamespace(generation_config=GenerationConfig())
    suppress_think(model, _fake_tok({}))
    assert not model.generation_config.suppress_tokens
