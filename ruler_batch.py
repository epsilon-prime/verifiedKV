"""Batched RULER on vLLM at a fixed KV pool: N requests submitted at once, ``max_num_seqs = B``.

Same contexts / prompt / forced question as ruler_vllm.py (--question forced). Every request then
generates exactly ``--new`` answer tokens (EOS ignored), so all arms and batch sizes decode the
same amount; ``pred`` is scored on the answer cut where a normal run would have stopped (the
first stop token, at most the row's max_new_tokens), which greedy decoding leaves unchanged.

    BENCH_GPU_BLOCKS=5850 ... python ruler_batch.py --arm dense --tag dense --batch 4 --run rb32k

Writes out/<run>/<tag>_b<B>/rows.jsonl (harness format, scored by score_runs.py) and
summary.json. Per request (from per-step token timestamps; every request arrives at t0):
  ttft      t0 -> first answer token (queueing, prefill, round and the forced question steps)
  tpot      mean time per answer token after the first
  latency   t0 -> last token
Per run: decode throughput (answer + question tokens of decode-only steps / their time, and
at the highest concurrency reached), e2e throughput (answer tokens / wall), bytes fetched from
the CPU (vkv), peak device memory (cudaMemGetInfo, every step), KV blocks, preemptions.
"""
import argparse
import json
import os
import statistics as S
import time
from pathlib import Path

TASKS = ["cwe", "fwe", "niah_multikey_1", "niah_multikey_2", "niah_multikey_3", "niah_multiquery",
         "niah_multivalue", "niah_single_1", "niah_single_2", "niah_single_3", "qa_1", "qa_2", "vt"]

ap = argparse.ArgumentParser()
ap.add_argument("--arm", required=True)
ap.add_argument("--tag", default=None)
ap.add_argument("--batch", type=int, required=True)
ap.add_argument("--rows", default="152:156")
ap.add_argument("--tasks", nargs="+", default=TASKS)
ap.add_argument("--new", type=int, default=512)
ap.add_argument("--dataset", default="xAlg-AI/att-hub-ruler-32k")
ap.add_argument("--run", default="rb32k")
args = ap.parse_args()

import torch  # noqa: E402
from datasets import load_dataset  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402

import bench_1req_off99 as B  # noqa: E402
from force_prefix import ForcePrefix, install_stop_guard  # noqa: E402

BATCH = args.batch
PROF = os.environ.get("RB_PROF") == "1"
out_dir = Path("out") / args.run / f"{args.tag or args.arm}_b{BATCH}"
out_dir.mkdir(parents=True, exist_ok=True)

kw = B.engine_kwargs(**B.CONFIGS[args.arm])
kw["max_num_seqs"] = BATCH
kw["enable_prefix_caching"] = False
kw["logits_processors"] = [ForcePrefix]
if os.environ.get("BENCH_GPU_BLOCKS"):
    kw["num_gpu_blocks_override"] = int(os.environ["BENCH_GPU_BLOCKS"])
cc = kw.get("compilation_config")
if cc is not None and "cudagraph_capture_sizes" in cc:
    cc["cudagraph_capture_sizes"] = sorted({min(1 << i, BATCH) for i in range(8)} | {BATCH})
install_stop_guard()
llm = LLM(**kw)
tok = llm.get_tokenizer()
core = llm.llm_engine.engine_core.engine_core
sched = core.scheduler
pool = sched.kv_cache_manager.block_pool
devs = sorted({torch.cuda.current_device()} | (
    {torch.device(os.environ["OFFLOAD_PACKAGE_KVZIP_DEVICE"]).index}
    if os.environ.get("OFFLOAD_PACKAGE_KVZIP_DEVICE") else set()))

# ---- per-step recording ---------------------------------------------------------------------
rec = {"on": False, "steps": [], "tok": {}, "mem": {d: 0 for d in devs}, "free": [], "pre": 0}
_schedule, _update, _preempt = sched.schedule, sched.update_from_output, sched._preempt_request


def schedule(*a, **k):
    t = time.perf_counter()
    out = _schedule(*a, **k)
    if rec["on"]:
        rec["cur"] = (t, out)
        rec["free"].append(pool.get_num_free_blocks())
    return out


def update_from_output(scheduler_output, model_runner_output):
    res = _update(scheduler_output, model_runner_output)
    if rec["on"]:
        t1 = time.perf_counter()
        t0, so = rec.pop("cur", (t1, scheduler_output))
        n = dict(so.num_scheduled_tokens)
        rec["steps"].append((t0, t1, sum(n.values()), len(n)))
        if PROF and sum(n.values()) == len(n) == BATCH:
            # RB_PROF=1: torch.profiler over decode steps 5..15 at full batch (kernel table in
            # <out>/prof.txt; run with VKV_GRAPHS=0 to see the vkv kernels one by one)
            rec["full"] = rec.get("full", 0) + 1
            if rec["full"] == 5:
                torch.cuda.synchronize()
                rec["prof"] = torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CUDA])
                rec["prof"].__enter__()
                from offload_package.vkv import cuda_decide as _cd
                if _cd.TPROF is not None:            # VKV_TPROF=1: decide phase counters
                    _cd.TPROF.zero_()
            elif rec["full"] == 15 and "prof" in rec:
                torch.cuda.synchronize()
                p = rec.pop("prof")
                p.__exit__(None, None, None)
                (out_dir / "prof.txt").write_text(p.key_averages().table(
                    sort_by="cuda_time_total", row_limit=40, max_name_column_width=70))
                from offload_package.vkv import cuda_decide as _cd
                if _cd.TPROF is not None:
                    t = _cd.TPROF.tolist()
                    tot = max(sum(t[:len(_cd.TP_PHASES)]), 1)
                    rows_n = max(t[-2], 1)
                    lines = [f"rows {t[-2]}  certificate rows {t[-1]}  cycles/row {tot / rows_n:.0f}"]
                    lines += [f"{nm:24s} {100 * c / tot:6.2f}%  {c / rows_n:10.0f} cyc/row"
                              for nm, c in zip(_cd.TP_PHASES, t)]
                    (out_dir / "tprof.txt").write_text("\n".join(lines) + "\n")
        for rid, ids in zip(model_runner_output.req_ids, model_runner_output.sampled_token_ids):
            if ids is not None and len(ids):
                # the engine's id is the request id + "-<random suffix>"
                rec["tok"].setdefault(rid.rsplit("-", 1)[0], []).extend([t1] * len(ids))
        for d in devs:
            free, total = torch.cuda.mem_get_info(d)
            rec["mem"][d] = max(rec["mem"][d], total - free)
    return res


def preempt(*a, **k):
    rec["pre"] += 1
    return _preempt(*a, **k)


sched.schedule, sched.update_from_output, sched._preempt_request = schedule, update_from_output, preempt

vkv = store = None
entry_bytes = 0
if B.CONFIGS[args.arm].get("token_policy") == "vkv":
    import skylight.backend as _skb
    _orch = _skb.SkylightSparseMetadataBuilder._offload_orchestrator
    vkv = _orch.token_offload.vkv
    entry_bytes = _orch.token_offload.C2 * _orch.token_offload.kv_caches[0].element_size()
    store = getattr(_orch, "token_store", None)


def copied_now():
    """Entries copied CPU -> GPU so far (see ruler_vllm.copied_now)."""
    from offload_package.vkv.state import STATS
    rel = float(vkv.stats[STATS.index("copied")])
    pend = float(vkv.restored.sum()) if vkv.restored is not None else 0.0
    miss = float(store._ctr_acc[0]) if getattr(store, "_ctr_acc", None) is not None else 0.0
    return rel + pend + miss


def prompt_ids(context, question, answer_prefix):
    sep = "#" * (len(context) + 10)
    t = tok.apply_chat_template([{"role": "user", "content": context + sep}],
                                add_generation_prompt=True, tokenize=False, enable_thinking=False)
    ctx, suffix = t.split(sep)
    return (tok.encode(ctx, add_special_tokens=False),
            tok.encode(question + suffix + answer_prefix, add_special_tokens=False))


# Where a normal run stops: the tokenizer's EOS, <|im_end|> and generation_config's eos ids.
from transformers import GenerationConfig  # noqa: E402

gen_eos = GenerationConfig.from_pretrained(B.MODEL).eos_token_id
stop_ids = {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")}
stop_ids |= set(gen_eos if isinstance(gen_eos, (list, tuple)) else [gen_eos] if gen_eos else [])

a, b = (int(x) for x in args.rows.split(":"))
reqs = []
for task in args.tasks:
    df = load_dataset(args.dataset, task, split=task).to_pandas()
    df = df.sample(frac=1, random_state=42).iloc[a:b]
    for ridx, (_, row) in zip(range(a, b), df.iterrows()):
        c_ids, q_ids = prompt_ids(row["context"], row["question"], row["answer_prefix"])
        reqs.append(dict(task=task, row=ridx, ids=c_ids, q=q_ids, gold=list(row["answer"]),
                         mnt=int(row["max_new_tokens"])))

# Warm-up outside the timing: JIT / graph shapes on a short request.
llm.generate([{"prompt_token_ids": reqs[0]["ids"][:2048]}],
             SamplingParams(temperature=0.0, max_tokens=8, ignore_eos=True), use_tqdm=False)
for d in devs:
    torch.cuda.reset_peak_memory_stats(d)
sps = [SamplingParams(temperature=0.0, max_tokens=len(r["q"]) + args.new, ignore_eos=True,
                      extra_args={"force_ids": r["q"]}) for r in reqs]
cp0 = copied_now() if vkv is not None else 0.0
st0 = vkv.stats.clone() if vkv is not None else None
rec["on"] = True
t0 = time.perf_counter()
outs = llm.generate([{"prompt_token_ids": r["ids"]} for r in reqs], sps, use_tqdm=False)
wall = time.perf_counter() - t0
rec["on"] = False
torch.cuda.synchronize()
n_copied = (copied_now() - cp0) if vkv is not None else 0.0

rows = []
for r, o in zip(reqs, outs):
    toks = list(o.outputs[0].token_ids)
    Q = len(r["q"])
    assert toks[:Q] == r["q"], f"{r['task']}_{r['row']}: the question was not forced"
    ans = toks[Q:]
    nat = next((i for i, x in enumerate(ans) if x in stop_ids), len(ans))
    nat = min(nat, r["mnt"])
    ts = rec["tok"][o.request_id]
    assert len(ts) == len(toks), (len(ts), len(toks))
    ttft = ts[Q] - t0
    rows.append({"task": r["task"], "row": r["row"],
                 "pred": tok.decode(ans[:nat], skip_special_tokens=True), "gold": r["gold"],
                 "n_prompt": len(r["ids"]), "n_question": Q, "n_out": len(ans), "n_natural": nat,
                 "ttft": ttft, "tpot": (ts[-1] - ts[Q]) / max(len(ans) - 1, 1),
                 "latency": ts[-1] - t0, "first_token": ts[0] - t0,
                 "sec": round(ts[-1] - t0, 1), "gpu": torch.cuda.get_device_name(0)})
with open(out_dir / "rows.jsonl", "w") as fh:
    for x in rows:
        fh.write(json.dumps(x) + "\n")

steps = rec["steps"]
dec = [(t1 - t0_, n) for t0_, t1, n, m in steps if n == m]           # decode-only steps
conc = max((n for _, n in dec), default=0)
full = [d for d, n in dec if n == conc][2:]
used = [pool.num_gpu_blocks - f for f in rec["free"]]
n_ans = sum(x["n_out"] for x in rows)
summ = {
    "arm": args.tag or args.arm, "batch": BATCH, "n_req": len(rows), "new_tokens": args.new,
    "pool_blocks": pool.num_gpu_blocks, "wall_s": wall,
    "ttft_mean": S.mean(x["ttft"] for x in rows), "ttft_median": S.median(x["ttft"] for x in rows),
    "tpot_mean_ms": 1e3 * S.mean(x["tpot"] for x in rows),
    "tpot_median_ms": 1e3 * S.median(x["tpot"] for x in rows),
    "latency_mean": S.mean(x["latency"] for x in rows),
    "latency_median": S.median(x["latency"] for x in rows),
    "decode_tput": sum(n for _, n in dec) / max(sum(d for d, _ in dec), 1e-9),
    "decode_tput_steady": conc / S.median(full) if full else None, "max_concurrency": conc,
    "mean_decode_concurrency": S.mean(n for _, n in dec) if dec else 0,
    "e2e_tput": n_ans / wall, "e2e_tput_all": sum(len(o.outputs[0].token_ids) for o in outs) / wall,
    "fetch_entries": n_copied, "fetch_bytes": n_copied * entry_bytes,
    "fetch_bytes_per_req": n_copied * entry_bytes / len(rows),
    "peak_dev_mem_gb": {str(d): v / 1e9 for d, v in rec["mem"].items()},
    "peak_torch_alloc_gb": {str(d): torch.cuda.max_memory_allocated(d) / 1e9 for d in devs},
    "kv_blocks_peak": max(used, default=0), "kv_blocks_mean": S.mean(used) if used else 0,
    "preemptions": rec["pre"],
}
if vkv is not None:
    from offload_package.vkv.state import STATS
    summ["vkv"] = dict(zip(STATS, (vkv.stats - st0).tolist()))
json.dump(summ, open(out_dir / "summary.json", "w"), indent=1)
json.dump([(t0_ - t0, t1 - t0, n, m) for t0_, t1, n, m in steps], open(out_dir / "steps.json", "w"))
print(json.dumps({k: v for k, v in summ.items() if k != "vkv"}, indent=1), flush=True)
B.shutdown_llm(llm)
