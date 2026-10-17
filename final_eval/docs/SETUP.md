# Final experiments: setup and readiness

Date 2026-10-13. Scope from the user: needle fetch and Bernoulli against kvpress baselines (KVzip or
Fast KVzip, KVzap, others recommended), on RULER 32k / 128k and LOFT 32k / 128k, models Mistral 3,
Qwen3, Qwen3.5, at 8x / 16x / 32x. Nothing below has been run as an experiment; this is the setup
check. Smoke scripts and outputs: `setup/` (`smoke_models.py`, `job_smoke.sbatch`, `out/*.json`).

## 1. What is in place

| Item | State |
|---|---|
| Models (in `~/hf`) | `Qwen/Qwen3.5-9B` (19 GB), `mistralai/Ministral-3-8B-Instruct-2512-BF16` (shards only; its duplicate 17.8 GB `consolidated.safetensors` skipped), `Qwen/Qwen3-8B`; `Qwen/Qwen3-4B-Instruct-2507` already there |
| Baseline weights | KVzap mlp + linear for Qwen3-8B; Fast KVzip gates for qwen3-8b and qwen3-4b-instruct-2507; RestoreKV(+) adapters for Qwen3-8B |
| Data, offline-loadable | LOFT RAG `f20180301/loft-rag-{nq,hotpotqa,musique,qampari,quest}-{32k,128k}`; RULER-32k all 13 tasks (`xAlg-AI/att-hub-ruler-32k`); RULER-128k all 13 (`SaylorTwift/RULER-131072-llama-3.2-tokenizer`) |
| Env | `kvpress-final` = clone of `kvpress-pr` (transformers 5.2.0, torch 2.11 cu128, flash-attn 2.8.3) + `flash-linear-attention` 0.5.2 (Qwen3.5's Gated DeltaNet kernels; without it transformers falls back to a slow torch loop). `kvpress-pr` untouched, so every earlier result stays reproducible |
| Code | `src/kvpress-e5` at tag `fetch_v3-rev6` + `9ca8502`, clean. Already has `KVzapPress`, `FastKVzipPress`, `RestoreKVPress`, a LOFT loader and scorer, and a Qwen3.5 adapter |

## 2. Smoke results (RULER-4k, one niah_single_1 and one qa_1 row per arm, 8x)

| Model | Loads | Arms that run and answer correctly | Fails | Multi-question consistent |
|---|---|---|---|---|
| Qwen3-8B (A100) | yes, stock route | no_press, kvzip, Bernoulli, needle fetch, ExpectedAttention, KVzap, Fast KVzip | RestoreKV (adapter repo not cached, now downloaded); SnapKV miscopies the needle (expected without the question) | yes (3 arms) |
| Qwen3-4B-Instruct-2507 (A100) | yes | no_press, kvzip, Bernoulli, needle fetch, ExpectedAttention, Fast KVzip | KVzap: no published weights | yes |
| Ministral-3-8B (A100) | stock route **hangs** (no `AutoModelForCausalLM` mapping for `mistral3`); text-only `Ministral3ForCausalLM` + key remap loads in 34 s, 0 missing keys (222 unused = vision tower) | kvzip, needle fetch, ExpectedAttention answer correctly | none; no_press and Bernoulli answered verbosely (to the 128-token cap), scored correct (P13) | yes |
| Qwen3.5-9B (A40) | yes, stock route (`Qwen3_5ForCausalLM`, 8 of 32 layers compressible) | no_press, kvzip, Bernoulli | **needle fetch crashes** (P11); **ExpectedAttention crashes** (P12); Fast KVzip: no gates | yes (no_press, Bernoulli) |

## 3. Problems found, and what each needs before a final run

### 3a. Blockers (results would be wrong, not just slow)

| # | Where | Problem | Fix |
|---|---|---|---|
| P1 | Qwen3.5, all KVzip-based arms (kvzip, Bernoulli, needle) | KVzip scores by re-feeding the context in 2048-token "repeat" chunks through the same cache. transformers' Gated DeltaNet ignores the cached recurrent state for any multi-token forward (`initial_state=None`) and then **overwrites** it. So the repeat pass scores with linear layers that have no memory of the context, and generation afterwards starts from the last repeat chunk's state | Patch the DeltaNet forward to continue from the cached conv and recurrent state when `seq_len > 1`; snapshot the recurrent state after prefill and restore it after scoring. Test: multi-token continuation == token-by-token to bf16 tolerance. (Invisible in the 4k smoke: there the last repeat chunk is the end of the context itself, so the wrong state is close to the right one) |
| P2 | Qwen3.5, LOFT | The adapter cannot rewind recurrent state between questions (its own docstring), so on LOFT question k sees questions 1..k-1 and their answers | Same snapshot: restore the post-prefill state before every question |
| P3 | Qwen3.5, needle fetch | `supports_multi_token_continuation = False`, so the question is fed one token at a time. Needle fetch then treats every question token as a decode step (`q_len == 1`): theta 0.1 instead of the batched theta_question 0.003. **A different algorithm runs** | Goes away with P1 (multi-token continuation becomes exact) |
| P4 | Qwen3.5, every arm | Its config's `eos_token_id` is `<|endoftext|>` (248044) but chat turns end with `<|im_end|>` (248046), and the repo has no `generation_config.json`. In the smoke the answers still stopped (10 tokens), so the model emits `<|endoftext|>` after `<|im_end|>` at least on short answers; not guaranteed on long ones | Set `eos_token_id = [248046, 248044]` at load (cheap insurance) |
| P5 | Ministral 3 | Ships as `Mistral3ForConditionalGeneration` (VLM wrapper, model_type `mistral3`), which is not in `AutoModelForCausalLM`'s mapping; the stock pipeline load hung for 27 min | Load `Ministral3ForCausalLM` with `config=text_config` and `key_mapping` (`language_model.model.` -> `model.`, `language_model.lm_head.` -> `lm_head.`); verified, 0 missing keys. `run_check.py` needs this as a load hook. Side effect: `model.config` is the text config, so P7 does not arise for Ministral |
| P6 | Ministral 3, KVzip scores (so kvzip, Bernoulli, needle) | Ministral multiplies queries by a position-dependent factor `1 + 0.1 log(1 + floor(pos / 16384))` (1.22 at 128k). KVzip recomputes queries itself (`get_prerope_query_states` + `apply_rope`) and omits it, so every tau beyond position 16k comes from a too-flat softmax | A `mistral3`/`ministral3` adapter that applies the factor in `apply_rope` (or KVzip reads the scaled query) |
| P7 | needle fetch, cached KVzip on Ministral 3 and Qwen3.5 | `NeedleFetchPress` reads `model.config.head_dim`/`hidden_size`, `CachedKVzipPress` reads `model.config.num_key_value_heads`. On both VLM wrappers these live in `text_config` | Read `getattr(config, "text_config", config)` |
| P8 | needle fetch, Qwen3.5 | Sketch pairs channels as (j, j + d/2). Qwen3.5 rotates only the first 64 of 256 dims, pairs (j, j + 32); the other 192 do not rotate. The pairing is wrong (energy ranking becomes position-dependent) and 16 pairs is 12.5% of channels instead of 25% | Generalise `top_energy_pairs` to partial RoPE; decide the channel budget (section 4) |
| P9 | LOFT | The kvpress loader concatenates the `dev` (10) and `test` (100) splits; `n_samples` would mix them | Filter `split == "test"` for runs; use `dev` for any tuning |
| P10 | needle fetch, LOFT | Audit B2: `promoted` is never reset between questions on one context. LOFT asks 100 questions per context, so it would matter a lot more than on SCBench | Reset per question. The 4k smoke test passed but is too short to exercise it |
| P11 | needle fetch, Qwen3.5 | Crashes: `needle_decode_bias`/`_attach_sketch` read K/V through `extract_keys_and_values` (`cache.layers[i].keys`), which `Qwen3_5DynamicCache` does not have. The press bypasses the adapter | Read K/V through `get_adapter(model).get_keys_values` |
| P12 | ExpectedAttention, Qwen3.5 | Crashes in `_avg_rope`: builds a full-width rotation from a 64-dim (partial) cos/sin | Port to partial RoPE, or drop ExpectedAttention on Qwen3.5 |
| P13 | Ministral 3, no_press | **Not a defect.** The smoke "miss" was a verbose answer that still contains the needle. Measured (`setup/ministral_prompt_check.py`, stock template, RULER-4k x 20): niah_single_1 100, niah_multikey_1 100, qa_1 80, vt 100. Ministral rambles (answers run to the token cap), which costs decode time on long-answer tasks | Keep the official template (it prepends a ~520-token "Le Chat / use tools" system prompt; it is constant, so the KVzip score cache keys are stable) |
| P14 | Qwen3.5, LOFT | Even with `enable_thinking=False` (the template closes an empty think block) Qwen3.5 opens a fresh `<think>` after the "Final Answer: " prefix on most LOFT questions and spends its budget reasoning: no_press exact match 0/10 on nq_32k dev (`setup/qwen35_think_check.py`); suppressing the `<think>` token at decode, same prompt: 4/10 (dropping the prefix: 6/10, but that changes the prompt for one model only) | `model_loading.suppress_think`, wired into `evaluate.py`: `<think>` is never generated, for every model that has the token (Qwen3-4B-2507 never emits it anyway; Ministral has none). Same prompt for all models |
| P15 | Qwen3.5, 128k | Peaks at ~50 GB on a 173k-token row (A100, `setup/out/resmoke_Qwen_Qwen3.5-9B.json`) and **runs out of memory on an A40** (44.4 GB usable). Its small KV cache does not help: prefill activations dominate | Every 128k cell on dgx (A100 80 GB), for all three models; 32k cells fit A40 / L40S |
| P16 | LOFT-32k dev split | **The 32k dev questions are mostly not answerable from the 32k corpus**: a gold answer appears in the context for 0% of quest, 10% of qampari, 80% of hotpotqa / musique and 90% of nq dev questions; the 128k dev questions and **every test question at both lengths: 100%** (checked 2026-10-14). LOFT builds each corpus around its test queries. The models' `[]` ("no document mentions this") on 32k dev was correct behaviour, so rehearsal LOFT-32k dev scores (and parse rates) measure nothing on qampari / quest | Use LOFT-128k dev (or 32k nq) for any sanity check; never tune on LOFT-32k dev. The final grid uses the test split, unaffected |

### 3b. Facts that shape the design

- **LOFT has one context per (task, length)**: 100 test questions (60 for qampari/quest at 32k) share
  one corpus of 133-883 documents. So there is one compression draw per (model, task, length,
  method). Query-level SEs do not cover draw variance for Bernoulli and needle fetch: run **3 seeds**
  for the stochastic arms on LOFT, and report "one context per task" as a limitation.
- **Real token lengths** (the RULER sets were generated with the Llama tokenizer):

  | | Llama-3.1 | Qwen3 | Qwen3.5 | Ministral 3 |
  |---|---|---|---|---|
  | LOFT "32k" | 42-46k | 44-50k | 44-49k | 45-50k |
  | LOFT "128k" | 111-114k | 118-121k | 118-120k | 120-123k |
  | RULER-32k | 30-32k | 32-45k | 34-45k | 32-45k |
  | RULER-128k | 130-131k | 131-171k | 136-173k | 131-174k |

  Qwen3.5, Ministral 3 and Qwen3-4B-2507 have 262k windows: set `max_context_length` to 200k and
  nothing is cut (fixes audit B3). **Qwen3-8B has a 32k native window** (131k with YaRN x4): it needs
  YaRN even for RULER-32k and LOFT-32k, and still truncates RULER-128k at 131k.
- **Qwen3.5 is hybrid**: 8 of 32 layers hold a KV cache (head_dim 256, 4 KV heads); the rest are
  fixed-size DeltaNet states. "16x" means 16x on the full-attention layers' cache; the recurrent
  state (constant, ~1 MB/layer) is reported separately.
- **Budget allocation**: kvpress KVzip and Fast KVzip default to **global** top-k across layers and
  heads, the same family as Bernoulli's global `c`. KVzap in the registry is either threshold-based
  (`kvzap_mlp`, DMSPress: does not hit a fixed ratio) or uniform per head (`kvzap_mlp_head`, the
  allocation Phase 2 E6 showed loses). For a fair fixed-ratio comparison, run KVzap scores through
  the same global top-k.
- **Needle fetch at 32x**: total 3.125%, sketch 1.76%, reserve 0.1-0.4%, so it keeps **1.0-1.3%**
  against Bernoulli's 3.125%. More than half the budget is sketch.

## 4. Recommendations

### 4a. Models

| Family | Recommended | Why | Alternative |
|---|---|---|---|
| Mistral 3 | `Ministral-3-8B-Instruct-2512` (BF16 repo) | Mistral 3 family, same size class as the others, 262k window; Mistral Large 3 (675B) and Mistral Small 3.2 (24B, 128k window, 48 GB weights) do not fit the budget | Ministral-3-14B: about 2x the cost |
| Qwen3 | `Qwen3-4B-Instruct-2507` | native 262k window, so no YaRN and no truncation; continuity with every fetch_v3 number; Fast KVzip gates exist | `Qwen3-8B`: KVzap, Fast KVzip **and** RestoreKV weights all exist, but its native window is 32k: YaRN x4 for every task >32k (static YaRN also changes short-context behaviour) and RULER-128k still truncates (171k tokens > 131k) |
| Qwen3.5 | `Qwen3.5-9B` | closest size; hybrid, so it is the architecture-generality test | Qwen3.5-4B |

### 4b. Baselines (decided by the user, 2026-10-13: **KVzip and Fast KVzip only**)

| Baseline | Status |
|---|---|
| KVzip (`kvzip_c`, global top-k on repeat-context scores) | runs on all four models; on Qwen3.5 only after P1 |
| Fast KVzip (`fastkvzip`, authors' gates, kvpress default: global top-k, sinks + local window kept) | released gates: Qwen3-8B / 14B / 4B-2507 only. **Gates are being trained for Ministral-3-8B and Qwen3.5-9B** in `fastkvzip_gates/` (section 6) |

Not run: KVzap, ExpectedAttention, SnapKV, RestoreKV (cite RestoreKV and KVzap in related work).
Still worth stating in the paper: needle fetch reads the query and keeps a host copy, while both
baselines are query-agnostic and discard what they drop (audit A4).

**Two properties of kvpress's Fast KVzip that matter at our ratios:**
- At contexts >= 32k tokens it always keeps the last `window_size = 4096` tokens. At 128k and 32x the
  whole budget is 4096 tokens (3.125% of 131k), so the window fills it and the gates choose nothing;
  at 16x the window is half the budget. This is how kvpress implements the authors' method; keep it
  (it is their baseline), but say so in the paper.
- The authors' training code adds the per-head bias `b` twice (`optim.py::Weight.forward`); kvpress's
  inference gate adds it once. Our gates are trained with the unmodified trainer, so they carry the same
  quirk as the released ones (assuming those came from this code).

### 4c. Method, before freezing

1. **Fix the decode trigger (audit B1)** and re-measure recall on the new models with dev rows only.
2. **Decide the sketch at 32x**: at 16 pairs it takes over half the budget (kept 1.0-1.3% vs 3.125%).
   Candidates: 8 pairs, or 1 bit, at 32x only. Dev-tune, then pre-register.
3. **Qwen3.5 sketch** (P8): partial-RoPE pairing; and 16 pairs (0.88% of the cache, 12.5% of channels) or 32 (1.76%, 25%).
4. **Reset `promoted` per question** (P10) before any LOFT run.
5. Freeze, tag the code, write `DESIGN.md` (arms, criteria, significance rule: > 1 SE and Holm < 0.05),
   then smoke at full length (one 128k row per model per arm) to check memory and timing.

### 4d. Protocol

- **Dev rows disjoint from test** (audit A1). RULER: tune only on rows outside the test sample
  (`df.sample(n, random_state=42)` takes a prefix of one permutation, so take dev from beyond it).
  LOFT: tune on the `dev` split only.
- **Samples**: RULER 50 per task at both 32k and 128k (128k raised from 20 by the user, 2026-10-14; section 13);
  LOFT all test queries. **3 seeds** (seed 0 + `_s1`, `_s2`) for the stochastic arms on LOFT (one context per
  task), at 16x only (section 12).
- **Metrics, checked against the official code (2026-10-14):**
  - RULER: `string_match_all` for niah / vt / cwe / fwe and `string_match_part` for qa, identical to NVIDIA/RULER
    `scripts/eval/synthetic/constants.py` (kvpress `benchmarks/ruler/calculate_metrics.py`); report the mean over
    the 13 tasks, as RULER does.
  - LOFT: the official primary metric is **subspan_em for all five RAG datasets, qampari and quest included**
    (LOFT README dataset table; `multi_value_rag` uses one-to-one matching via `linear_sum_assignment`, 1 only if
    every gold answer is matched). kvpress's `benchmarks/loft/calculate_metrics.py` has the same SCORING rules, but
    **not** the same answer parser (corrected after the audit, 2026-10-14). LOFT's `utils.extract_prediction` reads
    only the first line containing "[...]" and scores it empty if it does not parse. kvpress also tries later lines
    and falls back to the plain text after "Final Answer". On the 715 rehearsal LOFT cells, 1.6% of answers were
    accepted only by kvpress, raising 8 cells by 10-20 points, all of them compressed arms (KVzip, Bernoulli).
    **All LOFT numbers therefore come from `loft_official/score_loft.py`**, which rescores the saved predictions
    with LOFT's parser (no rerun, no kvpress change). The `subspan_em` in each cell's metrics.json is NOT the
    official score.
    **Decided by the user (2026-10-14): LOFT is reported with the official metric only, subspan_em on all five
    tasks** (qampari / quest included; no coverage column). The rehearsal tables and plots used coverage for
    qampari / quest and are not the reporting format.
  - kvpress leaderboard: RULER **4k** only (`evaluation/leaderboard.sh`), Qwen3-8B and Llama-3.1-8B-Instruct,
    compression 0.25 / 0.50 / 0.75 / 0.875, average string match; submission = PR to the HF space with the
    output dirs (predictions.csv, metrics.json, config.yaml) plus the press merged in kvpress. The final grid is
    NOT leaderboard-compatible (other models, lengths, ratios): a separate RULER-4k run is needed after the paper.
- **Report**: memory including sketch and reserve; fetch events **per generated token** (audit A3);
  question-time transfer per head; recurrent state for Qwen3.5.

### 4e. For the paper (decided by the user, 2026-10-13)

- **Positioning**: a verified Bernoulli eviction algorithm (one global c), with needle fetch as an add-on.
  The guarantee is stated on average across heads, which is what a single global c controls.
- **Systems numbers**: a colleague is building the real runtime (evicted cache, host copy, timed fetches).
- **No dynamic sparse attention baselines** (Quest, vAttention, ...): they keep the full cache and choose
  what to read, so they are not KV cache compression. Related work should still say how needle fetch
  differs from offloading with GPU-resident sketches (ShadowKV, InfiniGen), since reviewers will ask.
- **LOFT seeds**: 2-3 for Bernoulli and needle fetch (one context per task).
- Theorem 1 still describes the retired ladder (audit A2): restate it for static Bernoulli.

## 5. Compute

Measured cost from `fetch_v3/ruler_full_16x`: **25-30 GPU-min per (task, arm) at 128k, 20 rows**
(A100 and L40S alike; long-answer tasks dominate). Arms with the decided baselines: KVzip, Fast KVzip,
Bernoulli, needle fetch at 8/16/32x + no_press = **13 arms** (plus 2 extra seeds of the two stochastic
arms on LOFT).

| Block | GPU-h per model |
|---|---|
| RULER-128k, 13 tasks x 20 | ~75 |
| RULER-32k, 13 tasks x 50 | ~28 |
| LOFT-128k, 5 tasks, all test queries, 3 seeds for Bernoulli and needle fetch | ~20 |
| LOFT-32k | ~8 |
| **Total** | **~130** (3 models ~390) |

GPU types at ~175k tokens: Qwen3-4B (~48 GB) and Ministral-3-8B (~60 GB) need **A100** (dgx: 4 running
jobs per user): ~260 GPU-h / 4 slots, **about 2.7 days of dgx** if the queue is free (today it was not).
Qwen3.5's KV cache is small but its 128k prefill is not: it **also needs A100 at 128k** (P15), so all
128k blocks (~95 GPU-h per model) queue on dgx; the 32k blocks run on A40 / L40S (7 slots). Cheapest cut if needed: RULER-128k
at 32x for Fast KVzip is degenerate (window = whole budget, section 4b), keep it but it is quick to drop.

## 6. Status of the fixes (branch `final-eval-setup` in `src/kvpress-e5`, uncommitted)

| # | Fix | Test |
|---|---|---|
| P1-P3 | Qwen3.5: Gated DeltaNet forward continues exactly from the cached conv and recurrent state on multi-token forwards; adapter `snapshot_state`/`restore_state`; KVzip restores the post-prefill state before every repeat chunk and after scoring; the pipeline restores it after every answer; `supports_multi_token_continuation = True` | `tests/adapters/test_qwen3_5_adapter.py` (continuation == single pass; stock forward fails the same check; KVzip leaves the prefill state; questions do not see each other) |
| P4 | `model_loading.add_chat_eos`: generation also stops at the tokenizer's eos | `tests/test_model_loading.py` |
| P5 | `model_loading.load_text_only_causal_lm` for `mistral3` (raises on any missing weight), wired into `evaluate.py` | `tests/test_model_loading.py` |
| P6 | `Ministral3Adapter.scale_queries` (llama-4 scaling, same inputs as transformers: `cache_position`, `original_max_position_embeddings`); KVzip applies it | `tests/adapters/test_ministral3_adapter.py` |
| P7, P11 | needle fetch and cached KVzip read `text_config`; needle fetch reads K/V through the adapter | `tests/presses/test_needle_fetch_press.py` |
| P8 | `top_energy_pairs` with partial RoPE (rotary pairs atomic, pass-through channels single) | same |
| P9 | `run_check.py`: LOFT keeps one split (`loft_split`, default `test`) | |
| P10 | needle fetch resets `promoted` at every multi-token forward (a new question) | same |
| Fast KVzip | adapter-aware (one gate per KV-holding layer, count asserted); local gates from `$KVPRESS_FASTKVZIP_GATES` before the hub | `test_qwen3_5_adapter.py::test_fastkvzip_*`, `tests/presses/test_fastkvzip_local_gates.py` |
| LOFT seeds | registry arms `bernoulli_kvzip_c_s{1,2}`, `needle_final_c_s{1,2}` | |
| P14 | `model_loading.suppress_think`, wired into `evaluate.py` | `tests/test_model_loading.py` (6 pass) |
| dev rows | `run_check.py`: `ruler_rows: "a:b"` takes RULER rows a..b-1 of the seed-42 permutation (beyond every test prefix); refuses to combine with `n_samples` | readiness smoke |

Full kvpress suite, branch vs base `9ca8502`, same A40 node and env: **0 existing tests change
outcome**; 20 new tests pass (base: 945 pass, 4 fail + 4 error, all offline Hub downloads).

## 7. Fast KVzip gates for Ministral-3-8B and Qwen3.5-9B (`fastkvzip_gates/`)

- Data: the authors' recipe (FineWeb-Edu `sample/10BT/000_00000.parquet`; 29 documents of 10-30k tokens +
  5 concatenations of ~100k), extracted inside kvpress (`extract_features.py`): attention-input hidden
  states and KVzip scores per KV layer, each model's own chat template.
- Training: the authors' `optim.py` unmodified (`vendor/`, commit e04afaa), via `train_gates.py`.
- Validation: train our own gate for **Qwen3-4B-2507** the same way and compare with the released one
  (held-out agreement with KVzip scores, and Fast KVzip accuracy on RULER dev rows).
- Feature sizes: Qwen3.5 ~72 GB (8 layers), Ministral ~300 GB (34 layers): training needs a dgx node
  (2 TB RAM), `--mem=700G`.

## 8. Readiness checks before the grid (all on dev data; nothing here is a result)

Asked for 2026-10-13: "before the grid, more tests, so we know for sure all the models are working".
Until now only Qwen3-4B had passed everything; Ministral and Qwen3.5 had passed the 4k smoke, the LOFT-32k
multi-question check and one 128k row.

| # | Check | What would make it fail | Where | Status |
|---|---|---|---|---|
| R1 | **Rehearsal**: every final arm x ratio through `run_check.py` (the grid's runner) on RULER-32k (13 tasks x 2 dev rows), RULER-128k (13 x 1), LOFT-32k/128k (5 tasks, dev split) | a cell fails; memory accounting off (C2); no_press not a working long-context model on our prompts (C3); answers open `<think>` or run to the cap (C4) | `readiness/` (`make_sweeps.py`, `summarize.py` holds the criteria) | **submitted 2026-10-13**: 128k on dgx (qwen3-4b 350885, ministral 350886) and qwen3.5 (queued on dgx by a watcher once a slot frees; dgx only, user 2026-10-13: no dgx-mpi), 32k on a40 (350887-350889). Ministral / Qwen3.5 `fastkvzip` cells are in separate `*_fastkvzip.yaml` sweeps (same groups), to run once their gates exist |
| R2 | **Questions do not leak at 128k**: LOFT nq_128k (~115k tokens), question 3 alone == after questions 1 and 2, for no_press, KVzip, Bernoulli, needle fetch (+ Fast KVzip) | any mismatch (recurrent state, persisted fetches, cache rewind at real length) | `setup/resmoke.py --sections multi --loft_task nq_128k` | ready |
| R3 | **KVzip scores find the needle** on every model (lift of needle tokens above the global keep threshold) | a model far below the others = scoring bug | `score_sanity/` | **done, not diagnostic**: lift ~1.0-1.5 on Qwen3-4B, ~0.7-1.3 on Ministral, 1.2-1.9 on Qwen3.5 (32k and 128k). KVzip scores every copied token alike, so needles are not singled out even on Qwen3-4B, which still answers. No sign of a Qwen3.5 scoring bug (its query reconstruction also matches transformers line by line); the cause of its failures is left to R7 |
| R4 | **Decode trigger** recall / precision / estimate bias, 3 models x 3 ratios | (diagnostic; sketches stay as they are) | `trigger_check/` | running |
| R5 | **Fast KVzip gates**: retrain Qwen3-4B with our pipeline into `fastkvzip_gates/gates_validation/` (never `gates/`, which would shadow the released gate) and compare with the released gate; then train Ministral and Qwen3.5 | our Qwen3-4B gate clearly worse than the released one (held-out accuracy vs KVzip scores; RULER dev accuracy) | `fastkvzip_gates/` | **submitted 2026-10-13 (user go)**: Qwen3.5 train 350891 (l40, 300 GB: its features are 61 GB); Ministral extract 350883 -> train 350884 (afterok); Qwen3-4B validation extract + train into `gates_validation/` queued by a watcher when a dgx slot frees |
| R6 | Fast KVzip cells of R1 for Ministral and Qwen3.5 | as R1 | rerun the same sweeps (resumable) | after R5 |
| R7 | **Mini rehearsal** (user, 2026-10-13: "a few tasks, fewer samples, to figure out what is happening"): RULER-32k niah_single_1/3, multikey_2, qa_1, vt x 4 dev rows; RULER-128k niah_single_3, qa_1 x 2; LOFT nq/hotpotqa 32k + nq 128k dev; all arms (incl. `needle_uncharged_c`: sketch and reserve not charged, same draw as Bernoulli) x 8/16/32x + no_press | no_press fails (model or prompt problem); every compressed arm fails alike (selection or model problem); cliff between ratios | `readiness/make_mini.py`, groups `mini_<model>` | submitted (jobs 350765-350770) |

Cost of R1: roughly 6-8 GPU-h per model (about 5% of the grid); the 128k sweeps need dgx (P15). Run order:
R3 + R4 (cheap, already running) -> R1 32k on A40 in parallel with R5 -> R1 128k and R2 on dgx -> R6.

Decisions (user, 2026-10-13, later):
- **Fast KVzip window**: the final `fastkvzip` arm is `FastKVzipPress(window_size=None)`, the authors' 2% window
  at every length. The stock press protects 4096 recent tokens from 32k on, which is the whole kept budget at
  32k for 8x (4037 tokens), 16x and 32x and at 128k for 32x; in the smoke Qwen3-4B with the released gate
  then scored 0 on niah_single_3 and LOFT nq at 16x. The stock press stays registered as `fastkvzip_w4096`.
  State the change in the paper.
- **Qwen3.5 sketch: 32 pairs** (`pairs_per_128_dims=True` on every needle arm): 16 pairs per 128 head dims,
  so 128-dim models are unchanged (16) and Qwen3.5's 256-dim heads get 32, the same 1.76% of the cache. With
  16 its decode trigger was blind (recall 0, estimates 150-400x low). The trigger check for Qwen3.5 must be
  rerun with the new sketch.

Note: the KVzip score cache (`~/phase3/cache/kvzip_scores`) is keyed by model and context only, not
by code version. Every Qwen3.5 / Ministral file there was written after the P1/P6 fixes (15:51); if
scoring code changes again, delete that model's files.

### 8a. R4 result: decode trigger (needle_final_c, dev rows, `trigger_check/out/`), 2026-10-13

Medians over the 32k cells (6 tasks x 3 ratios, 8 rows each) plus 128k at 16x (2 tasks x 3 rows).
"Recall" = share of freshly dropped needles (true weight >= 0.1) whose sketch estimate reaches theta.

| Model | recall theta=0.1 (deployed) | recall theta=0.003 | precision theta=0.1 | log(est/true) | needles covered | lost mass Bernoulli -> after fetch |
|---|---|---|---|---|---|---|
| Qwen3-4B-2507 | 0.06-0.30 (0.00-0.05 on vt) | 0.38-0.74 | 0.55-0.78 | -0.3 to -0.7 (vt/fwe-8x: -3) | 0.6-0.9 | 0.6-0.8 -> 0.4-0.6 |
| Ministral-3-8B | 0.00-0.07 | 0.11-0.68 | 0.46-0.87 | -2.5 to -4.6 (12-100x low) | 0.16-0.78 | 0.7-0.9 -> 0.5-0.75 |
| Qwen3.5-9B | **0.00 everywhere** | 0.00-0.20 | (few flags) | **-4.6 to -8.9** | 0.0-0.37 | barely reduced |

Reading: the trigger works on Qwen3-4B (estimates ~1.5x low), is nearly blind on Ministral (as on Llama,
audit B1), and blind on Qwen3.5 (16 pairs = ~12% of a 256-dim head). Needle-fetch dev accuracy follows:
Ministral niah_single_3 100 / 12.5 / 0 at 8/16/32x; Qwen3.5 0 on niah_single_3 and niah_multivalue at
every ratio (whether no_press solves those rows is what the mini rehearsal shows). Sketches unchanged.

### 8b. R3 result so far: KVzip score sanity (`score_sanity/out/score_Qwen_Qwen3.5-9B.json`)

Qwen3.5 lift of needle tokens above the global keep threshold (1 = random): 32k 1.37 / 1.59 / 1.91 at
12.5 / 6.25 / 3.1% kept, 128k 1.15 / 1.31 / 1.61; niah_multikey_3 at or below 1. Query reconstruction
checked line by line against transformers' Qwen3_5Attention (gated q split, q_norm, partial RoPE,
1/sqrt(256)): no bug found. The Qwen3-4B / Ministral reference runs were cancelled (20:4x) before
finishing; rerun with `MODEL=... EXTRA="--tasks128k none" sbatch -p a40 --qos=a40 score_sanity/job_score.sbatch`.

### R8. Is Qwen3.5's failure under compression a bug? (`qwen35_probe/`, 2026-10-13) — **No.**

The mini rehearsal had every compressed arm miss Qwen3.5's niah_single_3 (UUID) dev rows at 8x (all of them
at 128k), while the full cache answered every one. Probe on the same 6 dev rows (32k 150-153, 128k 400-401),
KVzip top-k and Bernoulli at 8x, Qwen3-4B as the reference:

| Check | Qwen3.5-9B | Qwen3-4B |
|---|---|---|
| Gated DeltaNet conv + recurrent state after a compressed run == no_press (bit-identical) | 6/6, both arms | n/a (no linear layers) |
| 1% eviction: answer identical to no_press | 6/6, both arms | 6/6, both arms |
| plain 8x correct | KVzip 2/6, Bernoulli 0/6 | 6/6, 6/6 |
| 8x with the needle tokens forced kept (all else evicted as usual) | **12/12** | 12/12 |
| needle tokens kept (Bernoulli 8x, share of layer x head x token) | 11.5-14.5% | 12.2-14.4% |
| needle tokens kept by >= 1 head of a layer (mean over layers and rows) | 0.42 | 0.65 |

Reading: the machinery is right (prefill untouched, tiny eviction is an identity, a kept needle is read
correctly). KVzip's scores do not favour needle tokens on EITHER model (kept at the 12.5% base rate). The
difference is redundancy: per layer a needle token survives in some head with probability 1 - (1 - 0.125)^h,
0.41 for Qwen3.5's 4 KV heads and 0.66 for Qwen3-4B's 8 (measured 0.42 and 0.65), and Qwen3.5 has 8
attention layers to Qwen3-4B's 36. A UUID needs ~30 tokens copied in order, so Qwen3.5 copies fragments
(`5fa5e0aa-bd24-4530-9b11-...`) or a placeholder (`12345678-1234-...`). It is a property of the hybrid, and
it is where needle fetch has the most to recover (LOFT-128k 8x: Bernoulli 30, needle fetch 70, uncharged 80).

## 9. The final grid (`grid/`, written 2026-10-14, NOT submitted)

`grid/make_grid.py` writes 90 sweeps (2268 cells) for `harness_check/run_check.py`, groups `final_<model>`, results in
`grid/results/`. Per model, length (32k / 128k) and ratio (r8 / r16 / r32):

| Sweep | Arms | Cells |
|---|---|---|
| `_ruler` | no_press (r8 only), kvzip_c, bernoulli_kvzip_c, needle_final_c, needle_uncharged_c | 13 RULER tasks x `n_samples` 50 (32k) / 20 (128k) test rows |
| `_loft` | same | 5 LOFT tasks, every test question (`loft_split: test`) |
| `_fastkvzip`, `_fastkvzip_loft` | fastkvzip (2% window; our gates for Ministral / Qwen3.5) | as above |
| `_seeds_loft` | `{bernoulli_kvzip_c, needle_final_c, needle_uncharged_c}_s{1,2}` | LOFT only (one context per task) |

Test rows are `df.sample(n, random_state=42)`; every readiness / tuning row came from beyond that prefix.

Compute (`grid/estimate.py`, per-row times from the mini rehearsal; LOFT scaled by questions, an upper bound):
**~554 GPU-h**, of which 128k ~316 GPU-h must run on dgx (4 running jobs per user, 5 submitted) and 32k ~238 GPU-h
fits A40 / L40S (3 + 4 slots). With free queues: ~3.5 days for 128k, ~1.5 days for 32k, in parallel. The
LOFT seed sweeps are the largest optional block (~227 GPU-h by this estimate, an overestimate: it scales the
whole dev-context time, KVzip scoring and prefill included, by the 10x question count, while only decoding
grows). Cutting to one extra seed halves it.

### 7b. Gate training: what broke overnight and the fixes (2026-10-14)

- **NaN at step ~1800 (Qwen3.5).** The authors' trainer keeps the gate weights in the features' dtype (bf16) and
  runs SGD at lr 0.2; a prediction that saturates to exactly 0 or 1 in bf16 gives a ~1e12 BCE gradient and an
  inf weight. The data was checked clean (`check_data.py`: 0 non-finite or out-of-range of 988,657 tokens).
  Fix in `train_gates.py`: the same optimiser, loss, schedule and data, gate arithmetic in fp32 (features stay
  bf16 in RAM), saved back in the released gates' dtypes (bf16 projections and bias, fp32 norms and sink keys,
  checked against the released Qwen3-4B file). `--bf16_params` restores the authors' behaviour.
  Qwen3.5 then trained cleanly: held-out agreement 93.1%, loss 1.688.
- **"Gates not released" at load.** kvpress loads gate files with `weights_only=True`, whose allowlist names
  numpy 1's array pickle; our files stored the training curves as numpy 2 arrays. Fix: curves saved as torch
  tensors and weights on CPU (`train_gates.py`); files trained before the fix are rewritten by
  `fix_gate_file.py` (keeps a `.orig`). The Ministral gate (job 350884, started before the fix) is rewritten
  automatically when it finishes, then a `.ok` marker releases its Fast KVzip rehearsal cells.
- Qwen3.5's training moved to L40S (features 61 GB, 300 GB RAM requested); Ministral's (262 GB) needs dgx.
- **Gates trained (2026-10-14, fp32 trainer):** Qwen3.5 held-out agreement 93.0% / rank corr 0.825 (8 layers);
  Ministral 94.2% (34 layers); Qwen3-4B validation 94.6%. **Pipeline validated** on 9,824 held-out Qwen3-4B
  tokens (`eval_gate.py`): ours loss 1.064 / agreement 94.63% / rank corr 0.666, the authors' released gate
  1.069 / 94.49% / 0.605 (ours sees data from its own extraction, so a small edge is expected). Our gates are as
  good as the released ones. The released gate stays in use for Qwen3-4B; `gates_validation/` is only for this
  check (downstream comparison on the mini rows: job 351002, group `gateval_qwen3-4b`).
- **Downstream gate check** (Qwen3-4B, mini rows: RULER-32k 5 tasks x 4 dev rows + LOFT nq/hotpotqa 32k dev,
  Fast KVzip at 8x and 16x): our gate equals the released one in 10 of 14 cells and differs by 1-2 items in the
  rest (mean 67.9 vs 73.6 at 8x, 40.0 vs 43.6 at 16x). Comparable, with a slight edge to the authors' gate:
  the paper should say Ministral / Qwen3.5 gates were trained by us with the authors' recipe and code (fp32
  arithmetic), validated on Qwen3-4B, and may slightly understate Fast KVzip.

### R1 rehearsal, RULER-32k (orientation only: DEV rows, 13 tasks x 2 rows; 2026-10-14 06:45)

| Model | no_press | Bernoulli 8/16/32x | needle fetch 8/16/32x | uncharged 8/16/32x | KVzip 8/16/32x | Fast KVzip 8/16/32x |
|---|---|---|---|---|---|---|
| Qwen3-4B | 89.5 | 89 / 59 / 26 | 88 / 85 / 78 | 90 / 84 / 83 | 87 / 54 / 21 | 78 / 46 / 23 |
| Ministral-3-8B | 87.3 | 82 / 59 / 21 | 78 / 65 / 45 | 82 / 86 / 45 | 66 / 55 / 45 | 79 / 41 / 21 |
| Qwen3.5-9B | 92.3 | 37 / 33 / 31 | 74 / 53 / 61 | 73 / 69 / 56 | 59 / 20 / 19 | 31 / 30 / 26 |

All 32k cells ran (0 failures, memory accounting exact, no `<think>`). Not results: 2 rows per task.

## 10. Submitting the final grid (staged 2026-10-14) — SUPERSEDED by section 14 (32k split, fox, limits)

Nothing below has been run. On the go:

1. **dgx (128k, 24 jobs)**: `cp queue/grid_dgx_pending.txt queue/pending.txt` and start `queue/drain.sh` in the
   background; it submits the next line whenever fewer than 5 of our jobs sit on dgx (normal `dgx` only, never
   `dgx-mpi`). Order: RULER main (Qwen3.5, Qwen3-4B, Ministral x 8/16/32x), LOFT main, Fast KVzip, LOFT seeds.
2. **32k (15 jobs)**: the lines of `queue/grid_32k_commands.txt` directly (A40 main sweeps, L40S Fast KVzip and
   seeds; the per-user caps queue them).
3. Monitor with `readiness/summarize.py`-style checks (C1 failures, C2 memory) on `grid/results/final_*`;
   results only get read once every cell of a block is in.

Estimated ~551 GPU-h (~342 on dgx): about 3.5 days if dgx gives us 4 GPUs, less with fewer LOFT seeds.
Decisions still open for the user: LOFT headline metric (section 4d), number of LOFT seeds (s1 only halves the
largest optional block).

## 11. Readiness verdict (2026-10-14 10:50)

Full rehearsal complete: **3 x 576 = 1,728 cells** (every model x RULER 32k/128k x LOFT 32k/128k x every arm x
8/16/32x, dev rows), **0 failures, memory accounting exact in every cell, no `<think>`**. Fast KVzip runs on all
three models (our gates for Ministral / Qwen3.5). The setup is ready for the final grid.

Realistic grid cost from measured LOFT per-question times (`grid/time_loft.py`, `grid/out/time_*.json`):
**~464 GPU-h** (RULER 152, LOFT main + Fast KVzip 150, LOFT extra seeds 163; Ministral 223, Qwen3.5 147,
Qwen3-4B 95). Compressed answers on LOFT are long (100-170 tokens per question vs 5-50 uncompressed) and masked
decoding is slow (~0.16 s/token at 32k, ~0.4 s/token at 128k on Ministral). Profile (`grid/profile_decode.py`):
with an attention mask transformers' SDPA path repeats K/V to every query head and uses the efficient-attention
kernel; grouped-query attention without the copy would be ~1.5-2x faster (a code change: not made, user to decide).
Savings proposed: LOFT seeds only at 16x (~-110 GPU-h), uncharged needle arm only at 16x (~-45), together ~310 GPU-h.

## 12. Final grid as decided (user, 2026-10-14 ~11:30): options A + B

- **A**: LOFT extra seeds (`*_s1`, `*_s2` of Bernoulli, needle fetch, uncharged) at **16x only**.
- **B**: uncharged needle arm (`needle_uncharged_c`) at **16x only** (main sweeps and seeds).
- Everything else as in section 9: 3 models x RULER 32k (50 rows/task) / 128k (20) x LOFT 32k / 128k (all test
  questions) x 8/16/32x; no_press, KVzip, Fast KVzip (2% window), Bernoulli, needle fetch at every ratio.
- `grid/make_grid.py` -> **78 sweeps, 1,692 cells**. Staged commands regenerated: `queue/grid_dgx_pending.txt`
  (24 dgx jobs, 128k), `queue/grid_32k_commands.txt` (15 jobs, A40 / L40S). Not submitted.
- Cost from measured times (`grid/estimate_measured.py`): **~321 GPU-h** = 205 on dgx (128k, ~2.1 days on 4
  GPUs) + 116 at 32k (under a day on 7 A40 / L40S slots, in parallel). By model: Ministral 149, Qwen3.5 101,
  Qwen3-4B 71.
- LOFT metric decided: official subspan_em on all five tasks (section 4d). Still open: the explicit "submit".

## 13. Changes after section 12 (user, 2026-10-14 afternoon; NOT submitted)

- **New arm, fast needle fetch (`fast_needle_final`)**: `needle_final_c` with Fast KVzip's gate scores in place
  of KVzip's. Budget, sketch (2 bits, 16 pairs per 128 dims), thresholds, reserve and draw seeds are the same,
  so it differs from `needle_final_c` only in the scorer.
  - Code (kvpress branch, uncommitted): `FastKVzipScorer` in `fastkvzip_press.py`. It hands the stacked gate
    scores to `BernoulliPress`, which takes them on the path it uses for KVzip's (one global c, 4 sinks at
    r = 1). `BernoulliPress` only gained the scorer type in its type check and branch. Registry:
    `fast_needle_final` and `_s1` / `_s2`.
  - Tests: `tests/presses/test_fast_needle.py`.
  - **No 2% window** in this arm. Under needle's charged budget at 32x, the kept part is about 1% (3.1% minus
    ~1.8% sketch and ~0.4% reserve), so a 2% window would be twice the budget. The `fastkvzip` baseline keeps
    its window.
  - Grid placement: in the `_fastkvzip` sweeps (the arms that need gates) at 8/16/32x, and LOFT seeds at 16x
    in `_seeds_loft`, like `needle_final_c`.
  - Not yet run on any real model: it needs a dev-row rehearsal before "submit".
- **RULER-128k: 50 test rows per task** (was 20), the same as 32k. The test rows are positions 0-49 of the
  seed-42 permutation and the dev rows start at 400, so they stay disjoint.
- `make_grid.py` -> 78 sweeps, **2,076 runs**. The cost estimate counts the new arm at `needle_final_c`'s
  measured cost (an upper bound: that includes some KVzip scoring).
  - Total **~541 GPU-h** (was 321): fast needle ~92, the 128k rows ~128.
  - 397 h on dgx at 128k is ~4.1 days on 4 GPUs. The 144 h at 32k run in parallel on A40 / L40S.
  - By model: Ministral 246, Qwen3.5 151, Qwen3-4B 144.
- Queue: `grid_dgx_pending.txt` now has 25 jobs. Ministral's 128k gate job was ~52 h, over the 48 h limit, so
  it is split into `-fastkvzip-ruler` (~32 h) and `-fastkvzip-loft` (~20 h). Every other job is <= 32 h
  (largest: the two Qwen 128k gate jobs at ~32 h each).

### 13a. Fast needle fetch: dev-row rehearsal (2026-10-14 evening) — **passes**

- Jobs 351705-351710: the full rehearsal's dev rows, `fast_needle_final` only, 8/16/32x, 3 models.
  - 324 cells, **0 failures** (C1), memory accounting exact (C2).
  - Same peak memory as `needle_final_c`, and about the same time per cell.
- Fetching is almost the same as `needle_final_c`:

  | Model | Tokens that fetch | Question-time heads that fetch |
  |---|---|---|
  | Qwen3-4B | 96.7% (needle 96.0%) | 99.8% |
  | Ministral | 50.9% (50.8%) | 96.1% |
  | Qwen3.5 | 20.1% (20.5%) | 99.7% |

- Scores (orientation only, 1-2 rows per task) are close to `needle_final_c` and well above `fastkvzip` at
  16-32x. Qwen3.5 at 128k / 32x is lower (25 vs 33).
- `summarize.py` now checks memory accounting for any arm with "needle" in its name. `table.py`,
  `report.py` and `plot_rehearsal.py` include the arm (palette slot 6, validated in slot order).

## 14. Final run plan: Prajna + fox (2026-10-15; replaces section 10; waits for "submit")

**Where each job runs** (user's choice: 4 jobs to fox). All 78 sweeps are planned exactly once, checked by
script against `grid/sweeps/`.

| Where | Jobs | GPU-h | Wall time |
|---|---|---|---|
| Prajna dgx (A100 80 GB) | all 128k: 25 jobs, `queue/grid_dgx_pending.txt` | ~397 | ~4.1 days at 4 running (critical path) |
| Prajna A40 | 32k: main r8/r16/r32 of all 3 models, plus Qwen3.5 `fastkvzip` and `seeds` (11 jobs), `queue/grid_a40_pending.txt` | ~97 | ~32 h at 3 running |
| fox GPU 2 (A6000) | Ministral 32k `fastkvzip` + `fastkvzip_loft` (r8/r16/r32) | ~15 | ~15 h |
| fox GPU 3 (A6000) | Ministral 32k r16 `seeds_loft` | ~14 | ~14 h |
| fox GPU 4 (A6000) | Qwen3-4B 32k `fastkvzip` + `fastkvzip_loft`, then r16 `seeds_loft` | ~18 | ~18 h |

**Why this split**
- **No L40S any more.** The section-10 plan put every model's Fast KVzip and seeds jobs on L40S (Ada) and its
  main sweeps on A40, mixing GPU types inside one model's table. Now every 32k cell runs on an A40, or on an
  A6000, which reproduces the A40 answer-for-answer (memory: fox, verified 2026-10-14).
- **Qwen3.5 stays on Prajna.** Its fla Triton kernels (e.g. `l2norm_fwd_kernel`) autotune on a length-dependent
  key, so new test questions retune on whichever machine runs them. Splitting Qwen3.5 across machines could
  therefore change its answers. Qwen3-4B and Ministral do not use these kernels.
- **Fox computes no KVzip scores.** The gate arms do not need them, and the seeds reuse the cached LOFT-32k
  scores (fox's cache holds every short-context entry for our three models). Fox's cells and Prajna's are
  disjoint, so the results merge without collisions.

**Limits checked (2026-10-15)**
- **Prajna** (live `scontrol` / `sacctmgr`):
  - dgx: MaxTime 6 days, QOS 4 running / 5 submitted, up to 4 GPUs per job.
  - a40: MaxTime 4 days, QOS 3 running / 6 submitted, up to 2 GPUs per job.
  - Account: 12 running / 20 submitted.
  - Our jobs ask for 1 GPU, 16 CPUs, 192 GB and `--time=2-00:00:00`. The longest estimates are ~32 h on dgx
    (the two Qwen 128k gate jobs) and ~12 h on A40.
  - `drain.sh` caps match the QOS (dgx 5, a40 6, total 19 < 20). At most 4 + 3 = 7 run at once (< 12).
  - Note: 192 GB per job means at most 2 of our jobs per A40 node (~515 GB). That limits packing only.
- **fox** (`InfolabUserGuide.pdf`, read in full):
  - Rules we follow: no SLURM; `CUDA_VISIBLE_DEVICES` (`run_fox.sh`); tmux; `/opt/anaconda3` with
    `conda create -p` on the NAS (no personal conda); no sudo.
  - `$HOME` holds only symlinks to the NAS (`phase3`, `hf`, `src`, `tmp`, `.triton`); it uses 606 KB, and no
    logs or data are kept there.
  - **"Assume primary group = infolab before writing" to the NAS:** every fox job now starts under
    `sg infolab`.
  - The guide sets no GPU-count or time cap. The jobs use GPUs 2-4 only, and `run_fox.sh` refuses a GPU with
    less than 36 GB free.
  - NAS usage: 58 of 100 GB.

**Fox pre-flight** (dev rows, `fox/smoke_fastneedle/`)
- `fast_needle_final` and `fastkvzip` at 16x on Qwen3-4B and Ministral are compared answer-for-answer with the
  Prajna rehearsal's A40 cells. Ministral's `fastkvzip` comparison cells ran on L40S, so that pair is
  informational only.

**On "submit"**
1. Prajna: `cat queue/grid_dgx_pending.txt queue/grid_a40_pending.txt > queue/pending.txt`, then start
   `queue/drain.sh` in the background.
2. fox: `ssh fox`, check that GPUs 2-4 are free, then run the three lines of `fox/grid_fox_commands.txt`.
3. Monitor:
   - Prajna: `squeue -A 23b1835`, plus C1/C2 checks on `grid/results/final_*`.
   - fox: at most one `ssh fox` check every few hours (it rate-limits ssh).
4. After the fox jobs, run `fox/fetch_results.sh` (dry run), then `fox/fetch_results.sh go`. It copies into
   Prajna's result folders; fox's `failures.json`, if any, is saved as `failures_fox.json`.
5. Every cell's `cost.json` records its GPU name. Fox cells must read "NVIDIA RTX A6000".
6. Score LOFT only with `loft_official/score_loft.py` (section 4d).

**Retired:** `grid/submit_plan.sh` (stale: seeds at every ratio, A40 / L40S alternation; old copy in
`submit_plan.sh.stale`) and `queue/grid_32k_commands.txt` (marked superseded).
- **Pre-flight result (2026-10-15 10:40-10:52, fox GPU 2): passes.** Every cell comparable to a Prajna A40 cell
  is answer-for-answer identical, with the same metrics and the same kept fraction: 9 cells, 30 answers
  (Qwen3-4B `fast_needle_final` and `fastkvzip`, Ministral `fast_needle_final`). Ministral `fastkvzip` against
  its L40S reference: 7 of 14 answers differ, which confirms that L40S results do not mix with A40 / A6000 ones.

**If a job fails, times out or is cancelled** (nothing resubmits it automatically): check `squeue -A 23b1835`
first so the same job never runs twice, then put the job's line back. On Prajna, copy it from
`queue/done.txt` (or the `grid_*_pending.txt` file) into `queue/pending.txt`, and keep `drain.sh` running. On
fox, rerun the job's line from `fox/grid_fox_commands.txt` on the same GPU, or on another free one of 2-4.
`run_check` skips every cell that already has `metrics.json`. A crash loses at most the cell in progress
(<= ~0.8 GPU-h). Do not rely on `failures.json`: there is one per model folder and the next sweep with
failures overwrites it. Completeness is judged by every expected cell having `metrics.json` and `cost.json`.

### 14a. SUBMITTED (user: "submit", 2026-10-15 12:09)
- Prajna:
  - `queue/pending.txt` holds 36 lines (25 dgx, 11 A40), drained by `drain.sh` (pid 1173114, `queue/drain.log`).
  - First jobs 352289-352297. Submitted job ids are recorded in `queue/done.txt`.
- fox: tmux `fox-g2` / `fox-g3` / `fox-g4` started 12:10 under group infolab. Logs:
  `grid/logs/fox-gpu<N>-*.out` on fox.
- 12:35: the 20 not-yet-submitted dgx lines in `queue/pending.txt` were reordered longest-first (same lines;
  the drainer was paused for the edit and restarted as pid 1435403). Simulated 128k finish moves from
  Wed 21 ~06:00 to Tue 20 ~16:00, using SLURM's estimated first starts (Fri 16 04:26 / 06:14 / 10:55 / 21:54).
  Expected: fox done Fri 16 ~06:15, A40 done Fri 16 ~23:00.
- 17:31: the drainer had been killed when the previous session ended (nothing was missed: the dgx 5/5 and A40 6/6
  submission caps were full). Restarted as `flock -n queue/.drain.lock bash drain.sh`. A crontab entry on login2
  (every 10 min, same flock, PATH incl. /opt/slurm/bin) restarts it if it dies again. **Remove it with
  `crontab -r` after the grid.**
- 19:58: found a second, unlocked drainer on login1 (the 12:35 one). The cron and locked drainer live on login2. The
  login1 one was stopped, so one drainer remains. Nothing had been double-submitted. Check from any login node with
  `flock -n queue/.drain.lock true` (exit 1 = a drainer is alive).
- **2026-10-16 00:10: the cost estimate was wrong (my bug).** `grid/estimate_measured.py` counted RULER rows as
  lines in `predictions.csv`, and multi-line answers inflate that count. The grid's measured per-row speeds match
  the rehearsal. Corrected: **~972 GPU-h** (128k on dgx ~730, 32k ~242), not ~541. 8 dgx jobs need more than the
  48 h `--time`: Qwen3-4B r8 / r16 RULER (53 / 57 h), Ministral r8 / r16 / r32 RULER (59 / 65 / 50 h), and the
  Qwen3.5 / Qwen3-4B / Ministral 128k gate-arm RULER jobs (49 / 69 / 85 h). Running job 352292 (Qwen3-4B r8) ends at its limit
  Sat 17 14:40 with about one task left. Fix proposed to the user (not applied).
- 00:07: killed a second, unlocked drainer on login1 (pid 2892348, started 20:01 from this Claude session). The
  locked drainer on login2 remains.
- **2026-10-16 00:28: time limits fixed (user approved targeted limits).** Rule: whole days >= 1.3 x the corrected
  estimate, minimum 2 days. Running jobs come in 5-15% over their estimates, hence 1.3.
  - The 25 lines in `queue/pending.txt` differ from before only in `--time` (checked by diff). dgx lines are
    re-sorted longest-first by the corrected estimate.
  - New limits: Ministral 128k gate-arm RULER 5 days; Qwen3-4B 128k gate-arm, Ministral r8 / r16 RULER and Qwen3-4B r16
    RULER 4 days; Qwen3.5 128k gate-arm, Ministral r32 and Qwen3-4B r32 RULER 3 days. All others stay at 2 days. The
    dgx maximum is 6 days.
  - 352293 (Qwen3-4B r16 RULER, still pending, never ran) was cancelled and requeued as **352868** with 4 days.
  - Follow-up for running 352292 (Qwen3-4B r8 RULER, 2-day limit, ~53-59 h needed): a queue line with
    `--dependency=singleton` and the same job name, 6th in line. SLURM starts it only after 352292 ends, and
    run_check finishes the remaining cells.
  - Verified: all 78 sweeps covered (only the r8 follow-up is intentionally doubled); one flag set per line; qos =
    partition; no duplicate names; 352868 waits on QOSMaxJobsPerUserLimit, not a time limit. Prajna has
    EnforcePartLimits=NO, so after each submission check that the job's Reason is not a limit.
  - `queue/grid_dgx_pending.txt` is marked stale. `grid/estimate_measured.py` still has the line-count bug; a
    corrected copy is in `~/tmp/grid_interim/est_fixed.py` (not applied to the script without the user's OK).
- **2026-10-16 09:49: Ministral 32k main sweeps moved to fox (user OK).** Fox GPU 3 had finished Ministral's LOFT seeds
  cleanly (21.3 h). The `p3grid32-ministral3-8b-r16` and `-r32` lines were removed from `queue/pending.txt` (they had not
  been submitted; only those 2 lines changed, 21 remain).
  - fox `fox-g3b`: r16 RULER + LOFT on GPU 3, started 09:49.
  - fox `fox-g4b`: `fox/after_g4.sh` waits for `fox-g4` (Qwen3-4B) to end, then runs r32 RULER + LOFT on GPU 4. It retries
    every 10 min for up to 6 h if the GPU is busy. Log: `grid/logs/fox-after-g4.out`.
  - Ministral r8 main stays on Prajna A40 (353041). A6000 and A40 give identical KVzip scores and answers (smokes,
    2026-10-14), so Ministral's main table can mix the two.
- 2026-10-16 14:52: the fox-g4b waiter never fired (tmux `-t fox-g4` prefix-matched `fox-g4b`), so fox GPU 4 was idle
  13:01-14:52. Ministral 32k r32 main was then started directly on GPU 4 as tmux `fox-g4m`. All three fox GPUs are busy
  (g2 Ministral gate-arm LOFT, g3b Ministral r16 main, g4m Ministral r32 main). The watcher now also checks fox every 30 min.
- **2026-10-16 23:43: Ministral 128k put on hold (user: "qwen 3 4b and qwen 3.5 is higher priority rn ... only after that
  should any ministral 128k job run").** The drainer was stopped while `queue/pending.txt` was rewritten (backup in
  `~/tmp/grid_interim/pending_before_ministral_hold_*.txt`): the 11 Qwen3-4B / Qwen3.5 lines keep their order, and all 9
  Ministral 128k lines go last, each prefixed `need:<queue>/ministral_go` (drain.sh's existing hold, no code change), so
  none is submitted until that file is created. Ministral r16 RULER (353982, pending) and Ministral Fast KVzip RULER
  (353962, running 1.3 h, 1 cell finished; resume skips it) were cancelled and their lines added to the held set. Drainer
  restarted under flock; it submitted 354042 (Qwen3-4B r8 singleton follow-up) and 354043 (Qwen3.5 128k Fast KVzip).
  **To release Ministral 128k: `touch queue/ministral_go`.** Ministral 32k (A40 353041, fox) is unaffected.
- 2026-10-17 11:47: A40 node cn33 failed (NODE_FAIL) under Qwen3-4B 32k r32 (352979, 14.7 h in, 43 cells finished);
  SLURM requeued it, the relaunch on cn25 failed, and the job was left held (launch_failed_requeued_held). Released with
  `scontrol release 352979`; it restarted on cn26 at 11:55 and resumed at LOFT hotpotqa (finished cells skipped). The
  requeue overwrote the job log; the first run's log is in `~/tmp/grid_interim/p3grid32-qwen3-4b-r32-352979.run1.out`.
- 2026-10-17 ~06:20-08:21: the whole infolab network (fox, bee, dog, ant, cat) became unreachable from Prajna ("No route
  to host"; CSE web still reachable). Fox's Ministral 32k r16/r32 RULER had finished; their LOFT parts were in progress.
  User: wait for fox (no fallback on Prajna for now).
