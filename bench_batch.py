#!/usr/bin/env python3
"""Batched decode: dense vs pq vs pq + token offload with B concurrent requests.

Same engine as bench_1req_off99 (``engine_kwargs`` + ``CONFIGS``: pq_optimized
slot path, fixed top-k budget, FULL decode CUDA graphs with BENCH_CUDAGRAPH=1),
but ``max_num_seqs = B`` and graphs captured for batch sizes 1, 2, 4, .. B.
B requests share the document (BENCH_PROMPT_FILE) behind a unique first line
and sample with their own seed, so texts and selections differ per request.

Reported:
  steady step       median engine step while all B requests decode (one
                    token each): the batched TPOT
  decode tput       B / steady step
  e2e tput          output tokens / wall time of the timed generate()
  TTFT              per request (mean / max): later requests queue behind
                    earlier prefills
  KV blocks         peak / mean in use (+ staging), as in bench_1req_off99

    BENCH_BATCH=8 BENCH_CUDAGRAPH=1 ... python bench_batch.py pq
"""
import os
import statistics as st
import sys
import time

import bench_1req_off99 as B1                      # sets the vLLM / Skylight env

from vllm import LLM, SamplingParams

BATCH = int(os.environ.get("BENCH_BATCH", "4"))
NEW = int(os.environ.get("BENCH_NEW_TOKENS", "1024"))


def main(name: str) -> None:
    print(f"\n{'=' * 70}\nCONFIG: {name}  batch={BATCH}\n{'=' * 70}", flush=True)
    kw = B1.engine_kwargs(**B1.CONFIGS[name])
    kw["max_num_seqs"] = BATCH
    kw["enable_prefix_caching"] = False
    kw["disable_log_stats"] = False
    if B1.CUDAGRAPH:
        sizes = sorted({min(1 << i, BATCH) for i in range(8)} | {BATCH})
        kw["compilation_config"] = {"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY",
                                    "cudagraph_capture_sizes": sizes}
    llm = LLM(**kw)
    sched = llm.llm_engine.engine_core.engine_core.scheduler
    pool = sched.kv_cache_manager.block_pool
    rec = {"t": [], "nt": [], "free": [], "prof": None}
    orig = sched.schedule

    def schedule(*a, **k):
        if rec.get("prof") is not None:
            rec["prof"].step()                # one profiler step per engine step
        rec["t"].append(time.perf_counter())
        out = orig(*a, **k)
        rec["nt"].append(int(getattr(out, "total_num_scheduled_tokens", 0) or 0))
        rec["free"].append(pool.get_num_free_blocks())
        return out

    sched.schedule = schedule
    temp = float(os.environ.get("BENCH_TEMPERATURE", "0.7"))
    sp = [SamplingParams(max_tokens=NEW, ignore_eos=True, temperature=temp,
                         top_p=float(os.environ.get("BENCH_TOP_P", "0.8")),
                         top_k=int(os.environ.get("BENCH_TOP_K", "20")), seed=1000 + i)
          for i in range(BATCH)]
    prompts = [f"Request {i}: read the document and continue it.\n" + B1.PROMPT
               for i in range(BATCH)]
    # Warm-up: codebook fit / JIT on a short request, outside the timing.
    llm.generate([prompts[0]], SamplingParams(max_tokens=8, ignore_eos=True))
    for k in ("t", "nt", "free"):
        rec[k].clear()
    # BENCH_PROFILE_STEPS="start,count": torch-profile engine steps
    # [start, start + count) of the timed run into BENCH_PROFILE_TRACE.
    prof = None
    if os.environ.get("BENCH_PROFILE_STEPS"):
        from torch.profiler import ProfilerActivity, profile, schedule as psched
        ps, pc = (int(x) for x in os.environ["BENCH_PROFILE_STEPS"].split(","))
        trace = os.environ.get("BENCH_PROFILE_TRACE", f"/tmp/trace_batch_{name}.json")
        prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                       schedule=psched(skip_first=ps, wait=0, warmup=2, active=pc, repeat=1),
                       on_trace_ready=lambda p: p.export_chrome_trace(trace))
        prof.__enter__()
        rec["prof"] = prof
    t0 = time.perf_counter()
    outs = llm.generate(prompts, sp, use_tqdm=False)
    wall = time.perf_counter() - t0
    if prof is not None:
        rec["prof"] = None
        prof.__exit__(None, None, None)
        print(f"  profile: steps {ps}+2..{ps + 2 + pc} -> {trace}")

    t, nt = rec["t"], rec["nt"]
    # Decode-only steps schedule one token per running request (prefill chunks
    # are far larger than BATCH). When KV memory caps concurrency below BATCH
    # (e.g. 128k requests), vLLM queues the rest: the steady step is taken at
    # the largest concurrency actually reached.
    dec = [(b - a, n) for a, b, n in zip(t, t[1:], nt) if 0 < n <= BATCH]
    conc = max((n for _, n in dec), default=0)
    full = [d for d, n in dec if n == conc]
    steady = st.median(full[2:]) if len(full) > 4 else float("nan")
    n_out = sum(len(o.outputs[0].token_ids) for o in outs)
    ttft = [o.metrics.first_token_latency for o in outs
            if getattr(o, "metrics", None) and o.metrics.first_token_latency]
    used = [pool.num_gpu_blocks - f for f in rec["free"]]
    print(f"  prompt tokens: {len(outs[0].prompt_token_ids)}  x {BATCH} requests, "
          f"{NEW} new tokens each")
    mean_conc = st.mean(n for _, n in dec) if dec else 0
    print(f"  concurrency: max {conc} requests decoding together "
          f"(mean over decode steps {mean_conc:.1f}; {BATCH} submitted)")
    print(f"  steady batched step: {steady * 1e3:.2f} ms  ({len(full)} steps at {conc} decoding)")
    print(f"  decode throughput: {conc / steady:.1f} tok/s (at {conc} concurrent)")
    print(f"  e2e throughput: {n_out / wall:.1f} tok/s  (wall {wall:.1f} s)")
    if ttft:
        print(f"  TTFT mean {st.mean(ttft):.2f} s  max {max(ttft):.2f} s")
    print(f"  KV blocks in use: peak {max(used)}, mean {int(st.mean(used))} "
          f"(pool {pool.num_gpu_blocks})")
    B1.shutdown_llm(llm)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "pq")
