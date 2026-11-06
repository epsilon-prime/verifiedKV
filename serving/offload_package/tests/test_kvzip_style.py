"""KVzipScorer's two input conventions. ``style="kvpress"`` must be kvpress's KVzipPress exactly
(the Vkv harness scores with it): its chat affixes, prompt texts, 2048-token chunks, the per-chunk
sink ``min(4, chunk start)`` and its score layout. Bit-for-bit parity of the scores with kvpress
itself was checked on Qwen3-4B (757 / 5917 / 18039-token prompts, every score equal); this test
pins the conventions on a tiny random Qwen3 so it runs without the harness checkout."""
from __future__ import annotations

import pytest
import torch

from offload_package.scoring import kvzip as KZ

NAME = "Qwen/Qwen3-4B-Instruct-2507"
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    transformers = pytest.importorskip("transformers")
    try:
        tok = transformers.AutoTokenizer.from_pretrained(NAME, local_files_only=True)
    except Exception as e:  # pragma: no cover
        pytest.skip(f"{NAME} tokenizer not cached: {e}")
    cfg = transformers.Qwen3Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128,
                                   num_hidden_layers=2, num_attention_heads=4,
                                   num_key_value_heads=2, head_dim=16,
                                   max_position_embeddings=32768)
    torch.manual_seed(0)
    model = transformers.AutoModelForCausalLM.from_config(cfg)
    d = tmp_path_factory.mktemp("tiny_qwen3")
    model.save_pretrained(d)
    tok.save_pretrained(d)
    return str(d), tok


def _prompt(tok, n_words):
    text = " ".join(f"w{i % 97}" for i in range(n_words))
    ids = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                  tokenize=True, enable_thinking=False)
    return ids if isinstance(ids, list) else ids["input_ids"]


def _kvpress_affixes(tok):
    """KVzipPress._ensure_chat_affixes, transcribed."""
    dummy = "dummy context"
    sep = "\n" + "#" * len(dummy)
    t = tok.apply_chat_template([{"role": "user", "content": dummy + sep}],
                                add_generation_prompt=True, tokenize=False, enable_thinking=False)
    context, suffix = t.split(sep)
    prefix = context.split(dummy)[0]
    return (len(tok.encode(prefix, add_special_tokens=False)),
            tok.encode(suffix, add_special_tokens=False))


def _kvpress_pairs(tok, ids, prefix_len, suffix):
    """KVzipPress.prepare (chunk 2048, prev postfix 8), transcribed."""
    ctx = ids[prefix_len:]
    chunks = [ctx[i:i + 2048] for i in range(0, len(ctx), 2048)] if len(ctx) > 2048 else [ctx]
    out = []
    for i, a in enumerate(chunks):
        if i == 0:
            q = tok.encode("\n\nRepeat the previous context exactly.", add_special_tokens=False)
        else:
            q = tok.encode("\n\nRepeat the part of the previous context exactly, starting with",
                           add_special_tokens=False) + chunks[i - 1][-8:]
        out.append((a, q + suffix + a))
    return out


def test_kvpress_inputs(tiny):
    path, tok = tiny
    sc = KZ.KVzipScorer(path, "cuda", dtype=torch.float32, style="kvpress")
    pl, suf = _kvpress_affixes(tok)
    assert (sc.prefix_length, sc.postfix_ids, sc.sink, sc.chunk) == (pl, suf, 4, 2048)
    ids = _prompt(tok, 3000)
    assert len(ids) - pl > 2 * 2048                                  # three chunks
    assert sc._repeat_inputs(ids[pl:]) == _kvpress_pairs(tok, ids, pl, suf)
    # the offload convention keeps its own texts / sink / chunking
    so = KZ.KVzipScorer(path, "cuda", dtype=torch.float32, sink=64, chunk_size=2000)
    assert (so.prefix_length, so.sink, so.chunk) == (0, 64, 2000)
    assert so._repeat_inputs(ids[64:])[1][1][:len(tok.encode(KZ._PROMPTS["offload"][1],
                                                              add_special_tokens=False))] == \
        tok.encode("\n\nRepeat the part of the previous context exactly, starting with ",
                   add_special_tokens=False)


@pytest.mark.parametrize("style", ["kvpress", "offload"])
def test_score_layout_and_chunk_sinks(tiny, monkeypatch, style):
    path, tok = tiny
    sc = KZ.KVzipScorer(path, "cuda", dtype=torch.float32, style=style, sink=64, chunk_size=2000)
    ids = _prompt(tok, 3000)
    seen = []
    orig = KZ._score

    def spy(layer_idx, query, key):
        if layer_idx == 0:
            seen.append((KZ.STATE.sink, KZ.STATE.start, KZ.STATE.end))
        return orig(layer_idx, query, key)

    monkeypatch.setattr(KZ, "_score", spy)
    res = sc.score(list(ids))
    n = len(ids)
    assert tuple(res.shape) == (2, 2, n)
    if style == "kvpress":
        pl = sc.prefix_length
        starts = [pl + 2048 * i for i in range(len(seen))]
        assert [s for _, s, _ in seen] == starts
        assert [k for k, _, _ in seen] == [min(4, s) for s in starts]   # per-chunk sink
        assert seen[-1][2] == n
        assert bool((res[..., :min(4, pl)] == 1.0).all())
        assert bool((res[..., min(4, pl):pl] == 0.0).all())
        sc_ = res[..., pl:]
    else:
        assert all(k == 64 for k, _, _ in seen) and seen[0][1] == 64
        assert bool(torch.isinf(res[..., :64]).all())
        sc_ = res[..., 64:]
    assert bool(torch.isfinite(sc_).all()) and bool((sc_ >= 0).all()) and bool((sc_ <= 1).all())
    assert float(sc_.max()) > 0


def test_vkv_defaults_to_kvpress_scoring():
    """token_policy='vkv' scores with KVzip in the kvpress convention unless told otherwise;
    other policies keep the offload convention."""
    from types import SimpleNamespace

    from offload_package.integration.connector import build_offload_config

    def cfg(**extra):
        kv = SimpleNamespace(kv_connector_extra_config=dict(
            offload_granularity="token", token_budget_scope="global", **extra))
        return build_offload_config(SimpleNamespace(kv_transfer_config=kv), None)

    v = cfg(token_policy="vkv")
    assert (v.token_scorer, v.kvzip_style) == ("kvzip", "kvpress")
    assert cfg(token_policy="vkv", kvzip_style="offload").kvzip_style == "offload"
    lo = cfg(token_scorer="kvzip")
    assert (lo.token_scorer, lo.kvzip_style) == ("kvzip", "offload")
