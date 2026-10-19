# Plan: token-level offload with a multiprocess vLLM engine

Goal: run per-head token offload (`HeadTokenOffloadEngine`, every scope
except the old shared engine) under `VLLM_ENABLE_V1_MULTIPROCESSING=1` —
offline `LLM` and `vllm serve` (so LongBench's `pred.py` works unchanged) —
then async scheduling, then TP > 1. Correctness bar is unchanged: decode
equals pq (bitwise at unit level; forced end to end within the pq-vs-pq
floor), and the new protocol must be **bitwise identical to today's
in-process path** before any process boundary is introduced.

## 1. Why it fails today

The connector is one Python object shared by both roles: the worker builds
`OffloadPackageWorker` first and the scheduler role reuses the same instance
(`connector.py:262-283`). `TokenModeScheduler` (`token_mode.py:283`) runs
inside `schedule()` → `build_connector_meta` and reaches the worker's engine
through `engine_getter=lambda: self._worker_handler._token_engine`
(`connector.py:368-371`). Per step it:

| # | Operation | Data it touches | Lives in |
|---|-----------|-----------------|----------|
| a | `eng.add_request / set_pages / on_step / release` | engine maps | worker |
| b | `eng.prompt_scorer(rid, prompt_ids)` (KVzip, first round) | GPU | worker |
| c | `eng.run_round(rid, extra_tokens)`: scores → evict → D2H → pack survivors | GPU | worker |
| d | trim `req_to_blocks[rid]`, `block_pool.free_blocks(tail)` | block pool | scheduler |
| e | `request.num_offloaded_tokens = eng.offset(rid)` (read by `kv_cache_manager.py:460` to size allocations) | `Request` | scheduler |

With a separate EngineCore process the scheduler-side connector has no
worker handler: `engine_getter()` returns None and `build_connector_meta`
silently returns — no offload at all. `vllm serve` cannot even avoid it:
`AsyncLLM` requires the multiprocess client (`core_client.py:98`).
Everything else is already worker-local and survives a process split:
the decode fetch (PQ wrapper + `TokenStagingStore`), score hooks, the
runner's row update (`gpu_model_runner.py:1468`) and physical seq lens
(`:2138`), the backend's `_cached_request_ids` (runner input batch).

Rows a–c must run on the worker; d–e must run on the scheduler. Today d–e
consume c's *result*. That dependency is the whole problem.

## 2. Key observation: the scheduler can predict a round exactly

`run_round` (`token_mode_heads.py`) decides **which** cells to evict from
GPU scores, but **how many pages survive** depends only on counts:

```
K  = max(keep_count(Lg), n_protected(Lg))            # pure functions of Lg
R  = resident cells before the round
n  = scope-dependent evict count:
       global : max(0, R - L*H*K)
       layer  : sum_l max(0, R_l - H*K)              # R_l = R / L (symmetric)
       uniform: sum_lh max(0, R_lh - K)              # R_lh = R / (L*H)
E  = n           (exact: K >= n_protected, so n <= unprotected resident)
T  = R - E
nb = ceil(T / (L*H*P))
skip if E == 0 or Lg - nb*P <= 0
offset = Lg - nb*P;  keep_pages = min(len(pages), nb + ceil(extra/P));  freed = pages[keep_pages:]
```

Every written token adds exactly one resident cell to every (layer, head),
and every scope's budget is symmetric across layers/heads, so after any
sequence of rounds the per-unit residents stay equal (uniform/layer) or only
their total matters (global). A per-request mirror of `(written, R,
last_round)` on the scheduler is therefore sufficient for all three
per-head scopes. (The shared engine `TokenOffloadEngine` has its own
arithmetic — out of scope here, keep refusing it under multiprocessing.)

## 3. Design: scheduler plans, worker executes

### 3.1 `plan_round` — one pure function used by both sides
* New `offload_package/integration/token_plan.py`:
  `RoundPlan(rid, Lg, K, E, T, nb, offset, keep_pages)` and
  `plan_round(scope, L, H, P, Lg, R, n_pages, extra_tokens, keep_count, n_protected) -> RoundPlan | None`.
* `HeadTokenOffloadEngine.run_round` is refactored to call it for
  `K / nb / skip` and to **assert** its mask agrees (`E == plan.E`,
  `T == plan.T`). Behaviour unchanged; any drift fails immediately.
* Unit test (property-style): random scores, fractions 0.1–0.95, budgets,
  sink/window sizes, all three scopes, prompts that make `E == 0` and
  `Lg - nb*P <= 0` — `plan_round` == what `run_round` does, every time.

### 3.2 Scheduler side (`TokenModeScheduler`, scheduler process)
* No engine access at all. Keeps `_mirror[rid] = (written, R, last_round,
  prompt_len)`, updated from `SchedulerOutput` exactly as `on_step` updates
  `written` today.
* Due-round rule unchanged (first decode step after prefill, then every W,
  only while decoding).
* For a due round: `plan = plan_round(...)`; if not None: trim
  `req_to_blocks`, free the tail blocks (d), set `num_offloaded_tokens =
  plan.offset` (e), update the mirror (`R = T`).
* Emits per-step `TokenStepMeta` inside `OffloadPackageMetadata`:
  * `new`: `(rid, prompt_len, prompt_ids?)` — prompt ids only when the
    scorer needs them (KVzip);
  * `steps`: `(rid, num_computed, n_sched, pages_delta)` — pages as a delta
    (`keep_n`, appended block ids) to avoid sending 8k ids per request per
    step at 128k;
  * `rounds`: `RoundPlan` list (with `extra_tokens`);
  * `released`: finished / preempted rids.
* Staging tail reserve is already config-derived on both sides
  (`scheduler_connector.py:215`, `worker_connector.py:119`): no change.

### 3.3 Worker side (`OffloadPackageWorker`, worker process)
* Must apply the step meta **before** `_update_states` (which calls
  `take_row_update`) and `_prepare_inputs` (which reads `offset`).
  vLLM binds connector metadata only later, at forward time
  (`kv_connector_model_runner_mixin.py:80`, called from
  `gpu_model_runner.py:4552`, after `_update_states` at `:4320`). Use the
  `execute_model` wrapper we already install (`connector.py`,
  `_patched_execute_model`) to call `worker.apply_token_meta(meta)` first.
* `apply_token_meta`: in order — `released` → `release`; `new` →
  `add_request`; `steps` → `set_pages` / `on_step(num_computed, 0)`;
  `rounds` → (`prompt_scorer` on the first) `run_round(rid, plan=plan)`,
  asserting `nb / offset / kept pages` equal the plan; then
  `on_step(num_computed, n_sched)` — the exact call order of today's
  `build_connector_meta`, so the engine state evolves identically.
* GPU ordering: the round's D2H + packing is enqueued on the worker's
  stream before this step's forward. A freed tail page re-allocated to
  another request in the same step is only written by that forward, after
  the packing has read it — same guarantee as today.
* Cross-check channel: the worker reports `(rid, nb, offset)` of each
  executed round in `build_connector_worker_meta`; the scheduler's
  `update_connector_output` compares with its plan and raises on mismatch
  (one step late, a second safety net besides the worker assert).

### 3.4 Plumbing / config
* `OffloadPackageMetadata` gains an optional `token: TokenStepMeta`
  (msgspec/pickle-friendly: plain lists / ints).
* The scheduler-side connector no longer needs `_worker_handler` in token
  mode; keep the in-process shared-object path only for block offload
  (still refused under multiprocessing).
* Skylight backend registration in worker processes: today the benches
  patch vLLM at import time (`register_backend(CUSTOM, ...)` +
  `_get_attn_backend_class` patch). Move it into a vLLM general plugin
  (`vllm.general_plugins` entry point in the offload/skylight package), so
  `vllm serve` workers load it.
* `check_token_mode_env`: allow `VLLM_ENABLE_V1_MULTIPROCESSING` for
  per-head token mode once 3.1–3.3 land; keep refusing spec decode
  (rejected tokens roll `num_computed` back), world_size > 1 (until step 5)
  and async scheduling (until step 4).

## 4. Async scheduling (after multiprocessing works)
Step N+1 is scheduled while step N runs. Plans use `num_computed` that
already counts N's tokens; the worker applies metas in step order, so N's
KV is written before N+1's round packs. Work items: verify the mirror
against vLLM's optimistic `num_computed` (no rollback without spec decode),
`_update_states` deferred corrections, and the decode-row map write
(`set_decode_rows`) racing an in-flight graph replay (double-buffer +
event already used — confirm). Gate: forced exact check with async on.

## 5. Tensor parallel (world_size > 1)
Each rank holds `H/TP` heads; per-rank engines. uniform / layer: per-rank
counts are identical → identical `nb`. global: `nb = ceil(K/P)` regardless
of H, so pages stay consistent, but the ranking becomes per-rank (global
within a rank). Either accept that (document) or all-gather scores before
selection. Staging, CPU pool and kernels are per rank already.

## 6. Steps and gates

| Step | Work | Gate |
|------|------|------|
| 1 | `plan_round` + engine refactor + property test | property test; all existing tests; exact test bitwise |
| 2 | **Split protocol in-process** (`OFFLOAD_PACKAGE_TOKEN_SPLIT=1`): scheduler never touches the engine; metadata applied by the worker before `_update_states` | forced runs **bitwise identical** to today's in-process path (same rounds, same tokens, same NLL vectors) at 8k/32k |
| 3 | Enable `VLLM_ENABLE_V1_MULTIPROCESSING=1` (offline LLM), plugin registration | same forced vectors as step 2 (within pq-vs-pq floor; ideally identical); exact check; TPOT |
| 4 | `vllm serve` + LongBench `pred.py` smoke (5 items) | runs end to end, offload stats non-zero, answers parse |
| 5 | Async scheduling | forced exact check with async on |
| 6 | TP = 2 | exact check vs TP=2 pq |

Step 2 is the important one: it changes the protocol without changing the
process topology, so any difference is a protocol bug, cheap to bisect.

## 7. Risks
* **Mirror drift** (scheduler's counts ≠ engine's): caught by the worker
  assert (same step) and the worker→scheduler report (next step). Main
  sources: preemption/resume, chunked prefill boundaries, requests finished
  mid-step — covered by tests in step 2.
* **Metadata size** at long context with many requests: page deltas, not
  full lists; prompt ids only for KVzip.
* **Worker-side failures** (CPU pool exhausted, staging error flag): today
  they raise inside `schedule()`; with a split they raise in the worker,
  which vLLM surfaces as an engine failure — acceptable, but the scheduler
  can pre-check CPU-pool capacity from counts (E rows per round) to fail
  earlier.
* **Graph mode**: unchanged — rounds and lifecycle stay eager, before the
  replayed `_model_forward`.
