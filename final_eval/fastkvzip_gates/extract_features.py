"""Training data for Fast KVzip gates: per-layer attention-input hidden states and KVzip scores.

    MODEL=mistralai/Ministral-3-8B-Instruct-2512-BF16 sbatch -p dgx --qos=dgx job_extract.sbatch

Reproduces the data step of the official Fast KVzip code (``prefill/feature.py`` +
``prefill/data/load.py::load_fineweb`` at commit e04afaa) inside kvpress, so it works for every model
kvpress supports, including Ministral 3 (text-only load) and Qwen3.5 (only its full-attention layers
hold a KV cache, so only those get gates):

- contexts: FineWeb-Edu ``sample/10BT/000_00000.parquet``; ``fineweb_10k`` = the first 29 single
  documents with 10k <= token_count < 30k, ``fineweb_10k_cat`` = the first 5 concatenations of
  consecutive such documents until >= 100k tokens ("\\n\\n"-joined), as in ``load_fineweb``;
- prompt: the model's own chat template, exactly as the kvpress pipeline builds it at evaluation time
  (the official code uses hand-written per-family templates and has none for Ministral 3);
- target: kvpress ``KVzipPress`` scores (the same algorithm, ported by the same author), over the
  context tokens after the template prefix (the official ``kv.start_idx:kv.end_idx``);
- input: each compressible attention module's ``hidden_states`` during the prefill forward.

Writes ``data/<model>/<folder>/{hidden,score}_<i>.pt`` in the layout ``vendor/load_data.py`` reads:
hidden ``[n_layer, n_ctx_tokens, hidden]`` (model dtype), score ``[n_layer, n_kv, n_ctx_tokens]`` (fp32).
Resumable: an index whose two files exist is skipped.
"""

import argparse
import glob
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

# this repo's kvpress, ahead of any other installed copy
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "kvpress"))
HF_HOME = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
from transformers import AutoTokenizer, pipeline  # noqa: E402

import kvpress  # noqa: E402,F401  registers the pipeline
from kvpress.adapters import get_adapter  # noqa: E402
from kvpress.model_loading import add_chat_eos, load_text_only_causal_lm, needs_text_only_load  # noqa: E402
from kvpress.presses.kvzip_press import KVzipPress  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
ap.add_argument("--n_single", type=int, default=29)
ap.add_argument("--n_cat", type=int, default=5)
args = ap.parse_args()
T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)


# ---------------------------------------------------------------- contexts, as in load_fineweb
pq = glob.glob(os.path.join(HF_HOME, "hub/datasets--HuggingFaceFW--fineweb-edu/snapshots/*/sample/10BT/000_00000.parquet"))
assert pq, f"FineWeb-Edu sample/10BT/000_00000.parquet not under {HF_HOME} (see README.md, downloads)"
pq = pq[0]
df = pd.read_parquet(pq, columns=["text", "token_count"])
length = df["token_count"].to_numpy()
valid = np.arange(len(length))[(length >= 10000) & (length < 30000)]


def fineweb(cat: bool, n: int) -> list[str]:
    out, text, count = [], "", 0
    if n == 0:
        return out
    for i in valid:
        if cat:
            if count < 100000:
                text += "\n\n" + df["text"].iat[i].strip()
                count += int(df["token_count"].iat[i])
                continue
            out.append(text)
            text, count = "", 0
            # the official loop drops the document that closes a concatenation (its `continue` is skipped
            # once count >= 100k and the sample is not appended); reproduced as is
        else:
            out.append(df["text"].iat[i].strip())
        if len(out) == n:
            break
    return out


sets = [("fineweb_10k", fineweb(False, args.n_single)), ("fineweb_10k_cat", fineweb(True, args.n_cat))]
log(f"contexts: {[(name, len(c)) for name, c in sets]}")

# ---------------------------------------------------------------- model
kw = {"dtype": torch.bfloat16, "attn_implementation": "sdpa"}
if needs_text_only_load(args.model):
    model = load_text_only_causal_lm(args.model, kw, device_map="cuda")
    pipe = pipeline("kv-press-text-generation", model=model, tokenizer=AutoTokenizer.from_pretrained(args.model))
else:
    pipe = pipeline("kv-press-text-generation", model=args.model, device_map="cuda", model_kwargs=kw)
add_chat_eos(pipe.model, pipe.tokenizer)
model, tok = pipe.model, pipe.tokenizer
adapter = get_adapter(model)
modules = adapter.compressible_modules(model)
log(f"{args.model}: {len(modules)} compressible layers")


class ScoreGrab(KVzipPress):
    """KVzipPress that keeps a copy of its scores (the parent clears them when its context exits)."""

    def compress_post(self, model):
        self.grabbed = self.score_val.detach().float().cpu().clone()  # [n_layer, 1, n_kv, ctx]
        self.grabbed_prefix = int(self.prefix_length)
        super().compress_post(model)


hidden = {}


def pre_hook(module, a, kwargs):
    h = kwargs["hidden_states"]
    if module.layer_idx not in hidden and h.shape[1] == ctx_len:  # the prefill call, not a repeat pass
        hidden[module.layer_idx] = h[0].detach().to("cpu", copy=True)
    return None


handles = [m.register_forward_pre_hook(pre_hook, with_kwargs=True) for m in modules]

name = args.model.split("/")[-1].lower()
for folder, contexts in sets:
    path = os.path.join(args.out, name, folder)
    os.makedirs(path, exist_ok=True)
    for i, context in enumerate(contexts):
        f_h, f_s = os.path.join(path, f"hidden_{i}.pt"), os.path.join(path, f"score_{i}.pt")
        if os.path.exists(f_h) and os.path.exists(f_s):
            log(f"{folder}/{i}: done, skipping")
            continue
        inputs = pipe.preprocess(context, [""], answer_prefix="", max_context_length=10**9)
        context_ids = inputs["context_ids"].to(model.device)
        ctx_len = context_ids.shape[1]
        hidden.clear()
        press = ScoreGrab(compression_ratio=0.5)  # > 0 so the scoring passes run; the eviction is discarded
        cache = adapter.make_cache(model)
        with torch.inference_mode(), press(model):
            model.model(input_ids=context_ids, past_key_values=cache)
        start = press.grabbed_prefix
        h = torch.stack([hidden[m.layer_idx] for m in modules], dim=0)[:, start:]  # [n_layer, n, hidden]
        s = press.grabbed[:, 0, :, start:]  # [n_layer, n_kv, n]
        assert h.shape[1] == s.shape[-1], (h.shape, s.shape)
        torch.save(h, f_h)
        torch.save(s, f_s)
        log(f"{folder}/{i}: {ctx_len} tokens (prefix {start}), hidden {tuple(h.shape)}, score {tuple(s.shape)}, "
            f"score mean {s.mean():.4f}, frac>=0.05 {(s >= 0.05).float().mean():.3f}")
        del cache, h, s
        torch.cuda.empty_cache()

for hd in handles:
    hd.remove()
log("extract done")
