"""RULER downstream on vLLM, comparable with the Vkv harness's run_downstream.py.

Same contexts (seed-42 permutation of xAlg-AI/att-hub-ruler-32k, rows a:b per task), the same prompt
as kvpress's pipeline (context through the chat template, then question + chat suffix + answer
prefix; enable_thinking=False), greedy decoding with each row's max_new_tokens. Rows are written as
out/<run>/<arm>/rows.jsonl in the harness's format, so its score_runs.py scores them:

    BENCH_MODEL=Qwen/Qwen3-4B-Instruct-2507 BENCH_DTYPE=bfloat16 BENCH_MAX_MODEL_LEN=34000 \\
    BENCH_GPU_UTIL=0.5 python ruler_vllm.py --arm vkv --rows 152:154 --run r32k
    python /home/gaurav_kumar/vkv/phase4/vkv_final/score_runs.py out/r32k

Arms are bench_1req_off99.CONFIGS names (dense, vkv, ...). Resumable.

--question forced (default): the prompt is the context alone (the round compresses it), and
question + chat suffix + answer prefix are forced as the first output tokens, one decode step
each (``force_prefix.ForcePrefix``), as the harness's question pass runs over the compressed
cache; ``pred`` is what follows. TTFT then runs to the first answer token (prefill, round and
the question steps), and TPOT is over the answer's later steps. --question prefill: the whole
prompt is prefilled (the question sees the full cache; the first token comes from prefill).
"""
import argparse
import json
import os
import time
from pathlib import Path

TASKS = ["cwe", "fwe", "niah_multikey_1", "niah_multikey_2", "niah_multikey_3", "niah_multiquery",
         "niah_multivalue", "niah_single_1", "niah_single_2", "niah_single_3", "qa_1", "qa_2", "vt"]

ap = argparse.ArgumentParser()
ap.add_argument("--arm", required=True)
ap.add_argument("--rows", default="152:154")
ap.add_argument("--tasks", nargs="+", default=TASKS)
ap.add_argument("--dataset", default="xAlg-AI/att-hub-ruler-32k")
ap.add_argument("--run", default="ruler32k")
ap.add_argument("--tag", default=None, help="output arm name (default --arm), e.g. vkv-compact")
ap.add_argument("--question", choices=("forced", "prefill"), default="forced")
args = ap.parse_args()

import torch  # noqa: E402
from datasets import load_dataset  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402

import bench_1req_off99 as B  # noqa: E402

out_dir = Path("out") / args.run / (args.tag or args.arm)
out_dir.mkdir(parents=True, exist_ok=True)
rows_f = out_dir / "rows.jsonl"
done = set()
if rows_f.exists():
    done = {(r["task"], r["row"]) for r in map(json.loads, open(rows_f))}

kwargs = B.engine_kwargs(**B.CONFIGS[args.arm])
if args.question == "forced":
    from force_prefix import ForcePrefix, install_stop_guard
    kwargs["logits_processors"] = [ForcePrefix]
    install_stop_guard()
llm = LLM(**kwargs)
tok = llm.get_tokenizer()
# Engine-step stamps (the offline engine returns no request metrics): one (time, scheduled
# tokens) per schedule(); prefill steps schedule >1 token, decode steps 1.
steps = []
_sched = llm.llm_engine.engine_core.engine_core.scheduler
_orig_schedule = _sched.schedule


def _schedule(*a, **k):
    t = time.perf_counter()
    out = _orig_schedule(*a, **k)
    steps.append((t, int(getattr(out, "total_num_scheduled_tokens", 0) or 0)))
    return out


_sched.schedule = _schedule
vkv = None
if B.CONFIGS[args.arm].get("token_policy") == "vkv":
    import skylight.backend as _skb
    _orch = _skb.SkylightSparseMetadataBuilder._offload_orchestrator
    vkv = _orch.token_offload.vkv
    # bytes of one entry (one token's K+V of one KV head of one layer)
    entry_bytes = _orch.token_offload.C2 * _orch.token_offload.kv_caches[0].element_size()
    store = getattr(_orch, "token_store", None)


def copied_now():
    """(released-copy counter, copies still pending release, staging misses): restore-in-place
    copies are counted when the request's slot is released, which happens in a later engine
    step than generate() returns, so a row's copies = its pending restored flags + what the
    stats counter gained since the previous row - the previous row's pending flags."""
    from offload_package.vkv.state import STATS
    rel = float(vkv.stats[STATS.index("copied")])
    pend = float(vkv.restored.sum()) if vkv.restored is not None else 0.0
    miss = float(store._ctr_acc[0]) if getattr(store, "_ctr_acc", None) is not None else 0.0
    return rel, pend, miss


def prompt_ids(context, question, answer_prefix):
    """kvpress KVPressTextGenerationPipeline.preprocess (chat-template branch): (context ids,
    question + chat suffix + answer prefix ids)."""
    sep = "#" * (len(context) + 10)
    t = tok.apply_chat_template([{"role": "user", "content": context + sep}],
                                add_generation_prompt=True, tokenize=False, enable_thinking=False)
    ctx, suffix = t.split(sep)
    return (tok.encode(ctx, add_special_tokens=False),
            tok.encode(question + suffix + answer_prefix, add_special_tokens=False))


a, b = (int(x) for x in args.rows.split(":"))
prev = copied_now() if vkv is not None else None
for task in args.tasks:
    df = load_dataset(args.dataset, task, split=task).to_pandas()
    df = df.sample(frac=1, random_state=42).iloc[a:b]
    for ridx, (_, row) in zip(range(a, b), df.iterrows()):
        if (task, ridx) in done:
            continue
        c_ids, q_ids = prompt_ids(row["context"], row["question"], row["answer_prefix"])
        if args.question == "forced":
            ids, Q = c_ids, len(q_ids)
            sp = SamplingParams(temperature=0.0, max_tokens=Q + int(row["max_new_tokens"]),
                                extra_args={"force_ids": q_ids})
        else:
            ids, Q = c_ids + q_ids, 0
            sp = SamplingParams(temperature=0.0, max_tokens=int(row["max_new_tokens"]))
        before = vkv.stats.clone() if vkv is not None else None
        steps.clear()
        t0 = time.perf_counter()
        out = llm.generate([{"prompt_token_ids": ids}], sp, use_tqdm=False)[0]
        sec = time.perf_counter() - t0
        toks = list(out.outputs[0].token_ids)
        if Q:
            assert toks[:Q] == q_ids, f"{task}_{ridx}: the question was not forced"
            pred = tok.decode(toks[Q:], skip_special_tokens=True)
        else:
            pred = out.outputs[0].text
        n_out = len(toks) - Q                                  # answer tokens
        rec = {"task": task, "row": ridx, "pred": pred, "gold": list(row["answer"]),
               "sec": round(sec, 1), "n_prompt": len(ids), "n_question": Q,
               "question": args.question, "n_out": n_out, "gpu": torch.cuda.get_device_name(0)}
        # Step i lasts from its schedule() to the next one (the last to generate()'s return).
        # Prefill steps schedule > 1 token; decode step j (from 1) produces output token j, so
        # with Q forced tokens decode steps 1..Q end with the first answer token.
        dur = [(b[0] - a_[0], a_[1]) for a_, b in zip(steps, steps[1:] + [(t0 + sec, 0)])]
        pre = [x for x, n in dur if n > 1]
        dec = [x for x, n in dur if n == 1]
        qdec, adec = dec[:Q], dec[Q:]
        rec["prefill_s"] = (steps[0][0] - t0) + sum(pre) if steps else None
        rec["question_s"] = sum(qdec)
        rec["ttft"] = rec["prefill_s"] + rec["question_s"] if steps else None
        rec["tpot"] = sum(adec) / len(adec) if adec else None
        rec["tpot_median"] = sorted(adec)[len(adec) // 2] if adec else None
        rec["question_step_median"] = sorted(qdec)[len(qdec) // 2] if qdec else None
        rec["latency"] = sec
        rec["decode_tok_s"] = len(adec) / sum(adec) if adec else None   # after the first answer token
        rec["tok_s"] = n_out / max(sec, 1e-9)
        if vkv is not None:
            from offload_package.vkv.state import STATS
            d = dict(zip(STATS, (vkv.stats - before).tolist()))
            rec["vkv"] = d
            rec["kept"] = vkv.round_log[-1]["kept"] if vkv.round_log else None
            cur = copied_now()
            n_cp = cur[1] + (cur[0] - prev[0]) - prev[1] + (cur[2] - prev[2])
            prev = cur
            rec["fetch_entries"] = n_cp
            rec["fetch_bytes"] = n_cp * entry_bytes
        with open(rows_f, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"{task}_{ridx} {args.arm}: {rec['sec']}s n={len(ids)}+{Q} pred={rec['pred'][:60]!r} "
              f"gold={rec['gold'][:2]}", flush=True)
B.shutdown_llm(llm)
