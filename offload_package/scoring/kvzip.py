"""KVzip: query-agnostic KV importance by context reconstruction.

Port of "KVzip: Query-Agnostic KV Cache Compression with Context
Reconstruction" (Kim et al., arXiv 2505.23416, github.com/snu-mllab/KVzip,
``attention/score.py::KVScore._get_score`` and ``model/wrapper.py``).

The model is asked to repeat its own context. For every context chunk the
input ``"\\n\\nRepeat the previous context exactly." + <chat postfix> + chunk``
(later chunks: ``"... part of the previous context exactly, starting with"``
+ the last 8 tokens of the previous chunk) is run on top of the prefilled
context cache, and in every layer each KV pair is scored by the largest
attention probability any repeat query (over all query heads of its KV group)
gives it. The softmax runs over ``[sink | chunk | repeat tokens]`` only, with
the repeat tokens causal. The result is one score per (layer, KV head, token),
independent of any downstream query.

Two input conventions (``style``):

* ``"offload"`` (default): the first ``sink`` prompt tokens are the attention
  sink and are never scored (+inf), chunks of ``chunk_size``.
* ``"kvpress"``: exactly kvpress's ``KVzipPress`` (what the Vkv harness's
  ``CachedKVzipPress`` scores with): the prompt's chat-template prefix
  (``prefix_length``, from the tokenizer's template as kvpress derives it) is
  not scored, the sink of a chunk starting at ``start`` is the first
  ``min(4, start)`` tokens (the prefix's for the first chunk), chunks of 2048, kvpress's prompt texts and assistant suffix; positions before the
  prefix end score 1.0 for the first 4 and 0.0 after, as kvpress's
  ``score_val``.

Unlike the FastKVzip gate (scores a token from its own hidden state, free at
decode time) this is a separate teacher-forced pass over the whole prompt, so
it runs once per request, right before the request's first offload round, on
a private HF copy of the model (``KVzipScorer.score``). Tokens generated
after that have no reconstruction score and use a neutral per-layer default
(``DefaultScoreGates``).
"""
from __future__ import annotations

import math
import threading

import torch
import torch.nn as nn

_ATTN_NAME = "kvzip_sdpa"
STYLES = ("offload", "kvpress")
# Repeat prompts (first chunk, later chunks); "kvpress" = KVzipPress.prepare verbatim.
_PROMPTS = {
    "offload": ("\n\nRepeat the previous context exactly.",
                "\n\nRepeat the part of the previous context exactly, starting with "),
    "kvpress": ("\n\nRepeat the previous context exactly.",
                "\n\nRepeat the part of the previous context exactly, starting with"),
}
KVPRESS_N_SINK = 4          # KVzipPress.n_sink default
KVPRESS_CHUNK = 2048        # KVzipPress.prepare chunk_size default
_registered = False
_reg_lock = threading.Lock()


class _ScoreState:
    """Shared by the attention function of every layer during one pass."""

    def __init__(self) -> None:
        self.active = False
        self.sink = 0
        self.start = 0
        self.end = 0
        self.q_len = 0
        self.out: list[torch.Tensor] = []     # per layer [1, Hkv, ctx_chunk]


STATE = _ScoreState()


def _register_attention() -> None:
    global _registered
    with _reg_lock:
        if _registered:
            return
        from transformers import AttentionInterface
        from transformers.masking_utils import AttentionMaskInterface, sdpa_mask
        from transformers.integrations.sdpa_attention import sdpa_attention_forward

        def kvzip_attention(module, query, key, value, attention_mask, **kwargs):
            if STATE.active:
                _score(module.layer_idx, query, key)
            return sdpa_attention_forward(module, query, key, value, attention_mask, **kwargs)

        AttentionInterface.register(_ATTN_NAME, kvzip_attention)
        AttentionMaskInterface.register(_ATTN_NAME, sdpa_mask)
        _registered = True


def _score(layer_idx: int, query: torch.Tensor, key: torch.Tensor) -> None:
    """``KVScore._get_score``: query [1, Hq, q, d] (post-RoPE), key
    [1, Hkv, kv, d] = cache (sink + context) + this pass's q repeat tokens."""
    s = STATE
    _, hq, q_len, d = query.shape
    hkv = key.shape[1]
    q = query.view(1, hkv, hq // hkv, q_len, d)
    k = torch.cat([key[:, :, :s.sink], key[:, :, s.start:s.end], key[:, :, -q_len:]], dim=2)
    k = k.unsqueeze(2).transpose(-2, -1)                       # 1, Hkv, 1, d, k
    ctx_len = s.end - s.start
    w = torch.matmul(q, k) / math.sqrt(d)                      # 1, Hkv, G, q, k
    mask = torch.full((q_len, q_len), torch.finfo(w.dtype).min, device=w.device, dtype=w.dtype)
    mask = torch.triu(mask, diagonal=1)
    w[..., -q_len:, -q_len:] += mask
    w = torch.softmax(w, dim=-1)                               # model dtype, as upstream
    w = w[..., s.sink:s.sink + ctx_len]
    s.out[layer_idx] = w.amax(dim=(-3, -2))                    # max over group, q


class KVzipScorer:
    """Holds a private HF copy of the model and scores prompts."""

    def __init__(self, model_name: str, device, dtype=torch.bfloat16, *,
                 sink: int = 64, chunk_size: int = 2000, prev_postfix: int = 8,
                 trust_remote_code: bool = False, style: str = "offload",
                 tokenizer=None) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if style not in STYLES:
            raise ValueError(f"KVzip style must be one of {STYLES}")
        _register_attention()
        self.device = torch.device(device)
        self.style = style
        if style == "kvpress":
            sink, chunk_size = KVPRESS_N_SINK, KVPRESS_CHUNK
        self.sink, self.chunk, self.prev_postfix = int(sink), int(chunk_size), int(prev_postfix)
        self.tok = tokenizer or AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=dtype, attn_implementation=_ATTN_NAME,
            trust_remote_code=trust_remote_code).to(self.device).eval()
        cfg = self.model.config
        self.num_layers = int(cfg.num_hidden_layers)
        self.num_kv_heads = int(cfg.num_key_value_heads)
        if style == "kvpress":
            self.prefix_length, self.postfix_ids = self._kvpress_affixes()
        else:
            self.prefix_length, self.postfix_ids = 0, self._chat_postfix()

    def _enc(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False)

    def _chat_postfix(self) -> list[int]:
        """Tokens a chat template appends after the user turn (assistant
        header), as upstream ``postfix_ids``; empty without a template."""
        if not getattr(self.tok, "chat_template", None):
            return []
        mark = "KVZIPMARK"
        text = self.tok.apply_chat_template(
            [{"role": "user", "content": mark}], tokenize=False, add_generation_prompt=True)
        return self._enc(text.split(mark, 1)[1])

    def _kvpress_affixes(self) -> tuple[int, list[int]]:
        """``KVzipPress._ensure_chat_affixes``: the chat-template prefix length and the
        assistant suffix ids (no template: no prefix, suffix "\\n")."""
        if getattr(self.tok, "chat_template", None) is None:
            prefix_text, suffix_text = "", "\n"
        else:
            dummy_context = "dummy context"
            separator = "\n" + "#" * len(dummy_context)
            temp_context = self.tok.apply_chat_template(
                [{"role": "user", "content": dummy_context + separator}],
                add_generation_prompt=True, tokenize=False, enable_thinking=False)
            context, suffix_text = temp_context.split(separator)
            prefix_text = context.split(dummy_context)[0]
        return len(self._enc(prefix_text)), self._enc(suffix_text)

    def _repeat_inputs(self, ctx: list[int]) -> list[tuple[list[int], list[int]]]:
        chunks = [ctx[i:i + self.chunk] for i in range(0, len(ctx), self.chunk)]
        first, cont = _PROMPTS[self.style]
        out = []
        for i, c in enumerate(chunks):
            if i == 0:
                q = self._enc(first)
            else:
                q = self._enc(cont)
                q = q + chunks[i - 1][-self.prev_postfix:]
            out.append((c, q + self.postfix_ids + c))
        return out

    @torch.inference_mode()
    def score(self, prompt_ids: list[int]) -> torch.Tensor:
        """``[L, Hkv, n_prompt]`` float32 on ``self.device``. "offload": the
        first ``sink`` positions (the sink, never scored) get +inf. "kvpress":
        scoring starts after the chat prefix, each chunk's sink is the first
        ``min(4, chunk start)`` tokens, unscored positions are 1.0 (first 4)
        or 0.0 (KVzipPress.score_val)."""
        from transformers import DynamicCache

        n = len(prompt_ids)
        if self.style == "kvpress":
            start = min(self.prefix_length, n)
            sink = min(self.sink, start)
            res = torch.zeros((self.num_layers, self.num_kv_heads, n), dtype=torch.float32,
                              device=self.device)
            res[..., :self.sink] = 1.0
        else:
            start = sink = min(self.sink, n)
            res = torch.full((self.num_layers, self.num_kv_heads, n), float("inf"),
                             dtype=torch.float32, device=self.device)
        ctx = prompt_ids[start:]
        if not ctx:
            return res
        ids = torch.tensor([prompt_ids], device=self.device)
        cache = DynamicCache(config=self.model.config)
        STATE.active = False
        self.model(input_ids=ids, past_key_values=cache, use_cache=True,
                   logits_to_keep=1)                        # prefill the cache
        s = STATE
        s.sink, s.active = sink, True
        s.start = start
        scores = [None] * self.num_layers
        try:
            for chunk_ids, rep in self._repeat_inputs(ctx):
                if self.style == "kvpress":
                    # KVzipPress.score_kvzip: sink = min(n_sink, start_idx) of the chunk, so
                    # chunks after the first also attend to context tokens below n_sink.
                    s.sink = min(self.sink, s.start)
                s.end = s.start + len(chunk_ids)
                s.q_len = len(rep)
                s.out = [None] * self.num_layers
                self.model(input_ids=torch.tensor([rep], device=self.device),
                           past_key_values=cache, use_cache=True, logits_to_keep=1)
                cache.crop(-len(rep))                       # drop the repeat tokens
                for li in range(self.num_layers):
                    part = s.out[li][0].float()             # Hkv, chunk
                    scores[li] = part if scores[li] is None else torch.cat([scores[li], part], -1)
                s.start = s.end
        finally:
            s.active = False
            s.out = []
        for li in range(self.num_layers):
            res[li, :, start:] = scores[li]
        return res

    def close(self) -> None:
        self.model = None
        torch.cuda.empty_cache()


class DefaultScoreGates(nn.Module):
    """Gate interface for tokens the reconstruction pass never sees (decode
    tokens): a constant per-layer (per-head) score. ``set_default`` refreshes
    it from each scored request's median, so new tokens rank mid-pack rather
    than always first or last to be offloaded."""

    def __init__(self, num_layers: int, nhead: int, per_head: bool, device) -> None:
        super().__init__()
        self.per_head = per_head
        self.register_buffer("value", torch.full((num_layers, nhead), 0.0, device=device))

    def set_default(self, scores: torch.Tensor) -> None:     # scores [L, H, n]
        self.value.copy_(scores.float().median(dim=-1).values)

    @torch.no_grad()
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:  # [L, n, D]
        L, n = hidden.shape[:2]
        v = self.value.unsqueeze(1).expand(L, n, -1)          # L, n, H
        return v if self.per_head else v.amax(-1)


class DefaultScoreGate(nn.Module):
    """Per-layer view of ``DefaultScoreGates`` (large prefill chunks)."""

    def __init__(self, batched: DefaultScoreGates, layer: int) -> None:
        super().__init__()
        self._b, self._li = [batched], layer                  # list: not a submodule

    @torch.no_grad()
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:  # [n, D] -> [H, n]
        return self._b[0].value[self._li].unsqueeze(1).expand(-1, hidden.shape[0])


def write_prompt_scores(engine, slot: int, scores: torch.Tensor) -> None:
    """Install ``scores`` [L, Hkv, n] as request ``slot``'s keep scores."""
    n = scores.shape[-1]
    if getattr(engine, "headwise", False):
        engine.score[:, slot, :n] = scores.permute(0, 2, 1).to(engine.score.device)
    else:
        engine.score[:, slot, :n] = scores.amax(1).to(engine.score.device)
