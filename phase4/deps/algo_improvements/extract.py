"""Extract what the eps-certificate tests need, on RULER dev rows, with the full cache.

Per (task, dev row): post-RoPE context keys (every --stride-th layer, all KV heads), KVzip and Fast KVzip scores
for those layers, the exact queries attention receives for the question pass and for every step of the
full-cache greedy answer, the keys of the question and generated tokens, the answer and the gold.

Imports kvpress-e5 read-only: the model is loaded exactly as evaluate.py loads it, the prompt is tokenised by
the pipeline's own preprocess, KVzip scores come from CachedKVzipPress (shared cache), Fast KVzip scores from
FastKVzipScorer with its compression step disabled on this instance only.

    python -u extract.py --model Qwen/Qwen3-4B-Instruct-2507 --dataset ruler32k --rows 150:152 \
        --tasks niah_single_3 qa_1 --out out/extracts/qwen3-4b
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

HOME = os.path.expanduser("~")
sys.path.insert(0, os.path.join(HOME, "src", "kvpress-e5", "evaluation"))

import evaluate  # noqa: E402  (also patches attention through kvpress)
from kvpress.adapters import get_adapter  # noqa: E402
from kvpress.presses.cached_kvzip_press import CachedKVzipPress  # noqa: E402
from kvpress.presses.fastkvzip_press import FastKVzipScorer  # noqa: E402
from transformers import DynamicCache  # noqa: E402
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--dataset", default="ruler32k")
ap.add_argument("--tasks", nargs="+", required=True)
ap.add_argument("--rows", default="150:152", help="rows a:b of the seed-42 permutation (dev rows only)")
ap.add_argument("--stride", type=int, default=4, help="keep every stride-th compressible layer")
ap.add_argument("--max-new", type=int, default=64)
ap.add_argument("--max-context-length", type=int, default=200000)
ap.add_argument("--no-fast", action="store_true", help="skip Fast KVzip scores (no gates for this model)")
ap.add_argument("--out", required=True)
args = ap.parse_args()
OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)
T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:8.1f}s] {m}", flush=True)


# ------------------------------------------------------------------ model, exactly as evaluate.py loads it
cfg = evaluate.EvaluationConfig(dataset=args.dataset, data_dir=args.tasks[0], model=args.model,
                                press_name="no_press", compression_ratio=0.0,
                                model_kwargs={"attn_implementation": "sdpa"})
runner = evaluate.EvaluationRunner(cfg)
runner.press = None
runner._setup_model_pipeline()
pipe = runner.pipeline
model, tok = pipe.model, pipe.tokenizer
mcfg = getattr(model.config, "text_config", model.config)
adapter = get_adapter(model)
modules = list(adapter.compressible_modules(model))
LAYERS = [int(m.layer_idx) for m in modules][:: args.stride]
MOD = {int(m.layer_idx): m for m in modules}
ROW = {int(m.layer_idx): i for i, m in enumerate(modules)}
head_dim = int(getattr(mcfg, "head_dim", None) or mcfg.hidden_size // mcfg.num_attention_heads)
rope = getattr(mcfg, "rope_parameters", None) or {}
rotary_dim = int(head_dim * (rope.get("partial_rotary_factor", getattr(mcfg, "partial_rotary_factor", 1.0)) or 1.0))
eos = model.generation_config.eos_token_id
eos = set(eos if isinstance(eos, list) else [eos])
log(f"{args.model}: {len(modules)} KV layers, kept layers {LAYERS}, head_dim {head_dim}, rotary {rotary_dim}, "
    f"GPU {torch.cuda.get_device_name(0)}")

# ------------------------------------------------------------------ query capture at the attention function
CAP = {"on": False, "q": {}, "check": None}
_inner = ALL_ATTENTION_FUNCTIONS["sdpa"]


def capture(module, query, key, value, attention_mask, scaling=None, **kw):
    out = _inner(module, query, key, value, attention_mask, scaling=scaling, **kw)
    li = int(getattr(module, "layer_idx", -1))
    if CAP["on"] and li in LAYERS:
        s = scaling if scaling is not None else query.shape[-1] ** -0.5
        CAP["q"].setdefault(li, []).append((query[0].float() * s).cpu())  # [H, q_len, d]
        if CAP["check"] is None and query.shape[2] == 1:  # first decode step: keep for the reconstruction check
            CAP["check"] = (li, (query[0].float() * s), key[0].float(), value[0].float(), out[0][0].float())
    return out


ALL_ATTENTION_FUNCTIONS["sdpa"] = capture

# ------------------------------------------------------------------ rows
a, b = (int(x) for x in args.rows.split(":"))
assert a >= 150 or args.dataset != "ruler32k", "ruler32k dev rows start at 150"
manifest = OUT / "manifest.jsonl"

for task in args.tasks:
    df = evaluate.load_ruler_length_dataset(args.dataset, task).sample(frac=1, random_state=42).iloc[a:b]
    for ridx, (_, row) in zip(range(a, b), df.iterrows()):
        tag = f"{task}_{ridx}"
        if (OUT / f"{tag}.pt").exists():
            log(f"{tag}: exists, skipping")
            continue
        max_new = min(args.max_new, int(row["max_new_tokens"]))
        enc = pipe.preprocess(row["context"], [row["question"]], row["answer_prefix"], args.max_context_length)
        ctx_ids, q_ids = enc["context_ids"].to(model.device), enc["questions_ids"][0].to(model.device)
        n = ctx_ids.shape[1]

        # KVzip scores: one pipeline call populates the shared cache (or hits it); read the file back
        press = CachedKVzipPress(compression_ratio=0.9375)
        pipe(row["context"], question=row["question"], answer_prefix=row["answer_prefix"], press=press,
             max_new_tokens=1, max_context_length=args.max_context_length)
        press._context_ids = ctx_ids
        kv = torch.load(press._score_cache_path(model), map_location="cpu").float()  # [rows, 1, KV, n]
        assert kv.shape[-1] == n, (kv.shape, n)
        tau_kvzip = torch.stack([kv[ROW[li], 0] for li in LAYERS])

        # prefill with the full cache; Fast KVzip gates score the same prefill (compression step disabled)
        cache = DynamicCache()
        fast = None if args.no_fast else FastKVzipScorer(compression_ratio=0.9375)
        with torch.inference_mode():
            if fast is not None:
                fast.compress_post = lambda m: None  # instance attribute: keep the scores, prune nothing
                with fast(model):
                    model.model(input_ids=ctx_ids, past_key_values=cache)
                fs = torch.stack([s.float() for s in fast.score_val]) if isinstance(fast.score_val, list) \
                    else fast.score_val.float()
                tau_fast = torch.stack([fs[ROW[li], 0] for li in LAYERS]).cpu()
            else:
                model.model(input_ids=ctx_ids, past_key_values=cache)
                tau_fast = None
            keys = torch.stack([adapter.get_keys_values(cache, MOD[li])[0][0, :, :n].to(torch.bfloat16).cpu()
                                for li in LAYERS])  # [L, KV, n, d]

            # question pass (one multi-token forward), then greedy decode, capturing queries
            CAP.update(on=True, q={}, check=None)
            pos = torch.arange(n, n + q_ids.shape[1], device=model.device)[None]
            out = model(input_ids=q_ids, past_key_values=cache, position_ids=pos, logits_to_keep=1)
            nq = q_ids.shape[1]
            gen = [int(out.logits[0, -1].argmax())]
            p = n + nq
            while len(gen) < max_new and gen[-1] not in eos:
                out = model(input_ids=torch.tensor([[gen[-1]]], device=model.device), past_key_values=cache,
                            position_ids=torch.tensor([[p]], device=model.device), logits_to_keep=1)
                gen.append(int(out.logits[0, -1].argmax()))
                p += 1
            CAP["on"] = False
            extra = torch.stack([adapter.get_keys_values(cache, MOD[li])[0][0, :, n:].to(torch.bfloat16).cpu()
                                 for li in LAYERS])  # question + generated keys

        q_question = torch.stack([CAP["q"][li][0] for li in LAYERS])  # [L, H, nq, d]
        q_decode = torch.stack([torch.cat(CAP["q"][li][1:], dim=1) for li in LAYERS]) \
            if len(CAP["q"][LAYERS[0]]) > 1 else None  # [L, H, steps, d]

        # reconstruction check: the captured query reproduces the model's own attention output
        li, q1, k1, v1, o1 = CAP["check"]
        g = q1.shape[0] // k1.shape[0]
        w = torch.softmax(q1 @ k1.repeat_interleave(g, 0).transpose(-1, -2), dim=-1)
        recon = float((w @ v1.repeat_interleave(g, 0)).transpose(0, 1).sub(o1).norm() / o1.norm())

        blob = {"model": args.model, "dataset": args.dataset, "task": task, "row": ridx, "n": n, "nq": nq,
                "layers": LAYERS, "head_dim": head_dim, "rotary_dim": rotary_dim,
                "n_kv": int(mcfg.num_key_value_heads), "n_heads": int(mcfg.num_attention_heads),
                "keys": keys, "keys_extra": extra, "q_question": q_question, "q_decode": q_decode,
                "tau_kvzip": tau_kvzip, "tau_fast": tau_fast,
                "answer": tok.decode(gen, skip_special_tokens=True), "gold": str(row.get("answer")),
                "recon_rel_err": recon, "recon_layer": li}
        torch.save(blob, OUT / f"{tag}.pt")
        with manifest.open("a") as fh:
            fh.write(json.dumps({k: blob[k] for k in ("task", "row", "n", "nq", "answer", "gold",
                                                       "recon_rel_err")}) + "\n")
        log(f"{tag}: n={n}, question {nq} tok, {len(gen)} generated, recon {recon:.1e} (layer {li}), "
            f"answer {blob['answer'][:60]!r} | gold {blob['gold'][:60]}")
        del cache, keys, extra
        torch.cuda.empty_cache()
log("done")
