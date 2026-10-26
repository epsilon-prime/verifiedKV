# Phase 4: Vkv1

> **Final algorithm (2026-10-27): `vkv_final_algo`** - chern-grad planned on the 2-bit key sketch with a per-KV-head
> shortlist of 128 exactly scored keys; default 16x, eps0 0.3, delta 0.05. Code in `vkv_final/`, explanation and
> results in [VKV_FINAL_ALGO.md](VKV_FINAL_ALGO.md). It supersedes `vkv_chern_pq` (the PQ-4 sketch fails: see section 6
> there).

Vkv1 = truncated Horvitz-Thompson estimation of the attention denominator, with a Bernstein certificate and a
certified needle-order fetch. Opened 2026-10-23.

**Final configuration for testing (2026-10-24): `VKV1_CHERN_GRAD.md`.** It uses the Chernoff certificate, grad order and
no persistence. That file has the commands, the results and the open problems.

- `theory/SOURCES/vkv_draft0.{tex,pdf}`: the user's algorithm explainer "Vkv" (23 Oct 2026, 7 pp). The original PDF
  arrived as a chat attachment and was not on disk. This copy is a verbatim LaTeX transcription, compiled with
  `~/tools/tectonic/tectonic -X compile vkv_draft0.tex`. Replace it with the original file if one is copied in.

## Where it comes from (Phase 3, `phase3/experiments/algo_improvements/`)

- T4 (2026-10-20): the MSE truncation rule (draft section 3) as a fetch / no-fetch decision with a Markov bound.
  Report: `phase3/theory/mse_rule/`.
- T5 (2026-10-21): the same truncation set with Hoeffding / Bernstein / Chernoff bounds (`t5_bounds.py`), including
  the one-sided form that the draft's Part III calls the tight certificate.
- New in Vkv1: the fetch sweep (needle order by sketch weight, smallest passing k), the frozen T* after fetching, and
  the per-decision guarantee (Theorem 4).

Data for offline tests: `phase3/experiments/algo_improvements/out/extracts/qwen3-4b_all` (13 RULER-32k tasks x dev
rows 150-151, all 36 layers, exact queries for the question pass and the reference greedy answer).

## Code (2026-10-23)

| path | what |
|---|---|
| `vkv1/algo.py` | the algorithm, pure torch, vectorised over rows: `truncation` (MSE rule), `prepare` (per-entry bias / variance / moves, sweep curves along the fetch order), `kstar` (simple or tight check), `fetch_mask`, `estimate`, `attention_bias`, `error_curve` / `oracle_k` (diagnostics) |
| `vkv1/press.py` | `Vkv1Press`: real generation through kvpress (imported read-only). Residency = BernoulliPress's c and draw (+ optional local window); per-row bias through the needle hook (additive mask, sdpa); fetched entries persist (resident, exact) until the next question; online realised error |
| `tests/` | 25 tests: truncation = brute-force MSE argmin, Lemma 1, sweep = direct recomputation, monotone simple curve and minimal k*, tight <= simple, bias implements the estimator, Theorem 4 by Monte Carlo (all check x order x truncation), the design-rule red flag, press = row-by-row reference incl. persistence and reset |
| `experiments/vkv1_sim/` | upstream: `sim.py` replays Algorithm 2 on the algo_improvements extracts (Qwen3-4B, RULER-32k rows 150-151, all layers, reference greedy answer), configs check {simple, tight} x order {needle, grad} x {trunc, ht}, persistence along the answer; `summarize.py` prints the draft's section 10 checks |
| `experiments/vkv1_down/` | downstream: `run_downstream.py` (arms full / bern / needle_final_c / vkv1:<kept>:<eps0>:<delta>[:tight][:grad][:ht][:sketch][:nopersist][:win=N]), `summarize.py` (scores, online stats, `--same` identity checks) |

Tests: `cd ~/phase4 && CUDA_VISIBLE_DEVICES="" PYTHONPATH=~/src/kvpress-e5 ~/miniforge3/envs/kvpress-final/bin/python -m pytest -q tests`

Fetch order "grad" = decreasing b_i + v_i dt/dV at k = 0 (first-order drop of eps(0) when entry i is fetched). Entries with
r = 0 sit in T* (bias w, no variance). Local window default 0 (BernoulliPress has none; the draft gives it r = 1).

## Smokes (2026-10-23), all passed

- sim `out/smoke1` (cwe_150, layers 0/18/35, 16 decode steps, 1 draw, kept 1/16 and 1/32, all 8 configs, true weights):
  fail 0 everywhere; Lemma 1 0 violations; tight never > simple; simple monotone, tight non-monotone on 1-4% of rows;
  b+ 2.2 -> 0.14 of B tilde; k*/oracle 30-900x. `smoke2` (sketch weights, window 64, 2 contexts) and `smoke3`
  (no persistence) run.
- down `out/smoke1` (niah_single_3 + qa_1 row 152, 64 tokens): vkv1 eps0 = 0 == full (2/2 predictions); vkv1 no fetch +
  no truncation == bern on the same draw (2/2); vkv1 1/16 eps0 .3 recovers the needle bern loses, realised fail 0.
- What the smokes already show: the question pass unions to 40-95% of the context per KV head (each question row has
  its own F) and persistence keeps it resident; decode is cheap afterwards. Without persistence decode fetches on 85% of
  decisions. The draft's Remark 2 / a shared question-pass plan is the open design lever.
- Cost: sim ~40 s per layer for 8 configs x 2 residencies x 16 steps x 1 draw on an A40; a full context (36 layers, all
  steps, 3 residencies, 2 draws) ~5-6 GPU-h, so trim configs before a full run. Downstream ~2x Bernoulli's time per row.

## 64 sinks + 64-token local window (default from 2026-10-23, user)

`algo.residency` (shared by press and sim): first 64 and last 64 context entries at r = 1, paid from the kept budget.
Downstream flags `sink=N` / `win=N` (default 64); `sink=4:win=0` reproduces BernoulliPress. Tests: 30 pass.
2-context run (`vkv1_sim/out/sw64_2ctx`: cwe_150 + niah_single_3_150, all 36 layers, all decode steps, 1 draw;
`vkv1_down/out/sw64_2ctx`: niah_single_3 + qa_1 row 152, full answer length):
- upstream, exact weights: realised failure <= 0.0001 in all 96 rows (delta .05); Lemma 1 0 violations; tight never > simple.
  Question pass: union per KV head 24-83% of the context (best tight-grad-trunc eps .5 at 1/16: 24%); decode per step
  <= 0.09%. Gradient order roughly halves k* vs needle order; truncation cuts question fetch rate (e.g. 0.93 -> 0.69 at
  1/16, eps .3).
- downstream: all vkv1 arms 100 on both contexts (bern 0); eps0 = 0 == full and nofetch+ht+sink=4:win=0 == bern (2/2 each).
- Same 2-context run with needle fetch's 2-bit 16-pair sketch as plan weights (`vkv1_sim/out/sw64_2ctx_sketch`; downstream
  arms `:sketch` in `vkv1_down/out/sw64_2ctx`): the guarantee breaks at small eps0. Question-pass fail up to 10.4%
  (tight-grad-trunc, 1/32, eps .1), 6.0% for the draft's config at 1/16 eps .1; within delta at eps0 >= .3 for
  simple-needle-trunc. The configs with the least slack (tight + grad: k*/oracle 70-120x with true weights, 2-6x with the
  sketch) miss most; plain HT misses least. Traffic only 0-10% lower. Downstream: still 100 on both contexts, fail 0.5-2.9%.

## vkv1_diag (2026-10-24): where does the fetch volume come from? (exact weights)

`experiments/vkv1_diag/` (new; existing code untouched, `vkv1/algo.py` imported read-only): `diag_lib.py` (exact
two-point Chernoff curves, Monte Carlo failure curves, externally ordered plans, Remark-2 allocation),
`diag.py` (methods bern_simple_needle, bern_tight_grad, chern_grad, mc_grad, avg_tight/avg_chern/avg_mc, shared_tight,
needle, oracle; each keeps its own persisted set), `summarize.py`, `test_diag.py` (3 tests). Smoke (cwe_150, layers 9
and 27, 4 steps): question-pass union vs the needle rule on the same decisions, eps0 0.3: draft 195x, tight-grad 103x,
Chernoff 15x, best per-decision rule (MC) 2.3x, Remark-2 with tight curves 1.0x (20% of rows given up), shared set
208x. Full two-context run: `out/two_ctx` (jobs 362474 l40, 362475 a40).

## vkv1_chern (2026-10-24): exact Chernoff certificate downstream

`experiments/vkv1_chern/`: copies of `vkv1/press.py` (`press_chern.py`, adds check "chern" = exact two-point Chernoff
from `vkv1_diag/diag_lib.py`) and of the downstream runner (flag `chern`); originals untouched. Tests
`test_press_chern.py`: copy == original press for simple/tight, chern == row-by-row reference. Motivation (vkv1_diag,
niah_single_3_150, all layers): chern_grad question transfer 9.9x needle at eps0 .3 / 1.9x at .5 (tight-grad 46.5x /
24.8x), 399 / 78 MB per answer (needle 38.6), realised fail 0.1-0.2%. Run `out/two_ctx` (job 362549): full, bern, tight:grad
(must reproduce vkv1_down sw64_2ctx), chern:grad at 1/16 and 1/32, eps0 .3 / .5.
- Result (`vkv1_chern/out/two_ctx`): copy's tight:grad == vkv1_down sw64_2ctx (predictions and stats identical, 2/2).
  chern:grad all 100 on both contexts (bern 0). 1/16: eps0 .3 Q 9.3% of ctx, 449 MB/answer; eps0 .5 Q 1.9%, 91 MB
  (tight:grad .3: 40%, 1899 MB). 1/32: eps0 .3 17.4% / 830 MB, .5 5.8% / 281 MB. Realised fail Q 0.1-0.27%, D 0.01%.
  Cost: ~170 s per row vs 35 s for tight (32-point lambda grid at every layer and step); not optimised.
- Full two-context diagnostic (`vkv1_diag/out/two_ctx`, cwe_150 + niah_single_3_150, all layers and steps, 1/16, 100 MC draws).
  MB per answer (needle rule 42.6) at eps0 .3 / .5: draft 3483 / 2671; tight-grad 2347 / 1378; chern_grad 562 / 128;
  mc_grad (best per-decision draw-free) 123 / 7.9; avg_tight 406 / 28; avg_chern 94 / 1.4; avg_mc 5.3 / 0.1;
  shared_tight 3690 / 2950; oracle 26 / 2.8. Realised fail: certified methods <= 0.3% (avg_chern 1.25%), needle 2.0%
  (eps0 .3, no promise). avg_* give up on ~20% of question rows.

## All 13 tasks, Chernoff + grad (2026-10-24; submitted 02:27, done 07:01)

- Upstream `vkv1_diag/out/all13_chern` (26 contexts, rows 150-151, all layers / steps, 1/16, exact weights): MB/answer
  eps0 .3 / .5: chern_grad 441 / 84, tight-grad 1849 / 1037, needle rule 52.2, oracle 15.5 / 1.6. Q union 8.5% / 1.6%
  of ctx (tight 36 / 20). Realised fail <= 0.15% (Q) / 0.01% (D). Persisted 8.4% / 1.6%.
- Downstream `vkv1_chern/out/all13` (13 tasks x rows 152-155 = 52 contexts): mean full 88.1, bern 57.0, needle_final_c
  85.1, chern .3 86.3, chern .5 86.5. Paired vs full: chern .5 -1.60 +- 0.80, chern .3 -1.73 +- 1.94, needle -2.95 +- 3.10;
  chern vs needle +1.2 / +1.4 (SE 3-3.7, not significant). MB/answer: chern .3 461, .5 91, needle 35.8. Online fail Q
  0.15% / 0.06%, D 0.01%. 240 s/row (needle 19). Caveat: vkv1 plans with EXACT weights (perfect sketch), needle with its
  real 2-bit sketch; memory not matched (vkv1 6.25% kept + 1.8-9% persisted; needle 1/16 total incl. sketch).

## No persistence (2026-10-24, user: persistence made the memory comparison unfair)

`experiments/vkv1_nopersist/`:
- `diag_np.py`: a copy of `vkv1_diag/diag.py`. Every forward starts from the draw; it adds `fwd_peak` (the worst
  single-forward buffer); decode shares one forward across methods.
- Downstream uses the existing `nopersist` flag through `vkv1_chern/run_downstream.py`.

Checks:
- The question pass is bit-identical to the persistent run (A40).
- Online `persist%` is 0.
- The first L40S run differed only in draw-dependent metrics: the CUDA RNG output depends on the GPU model. The run was
  repeated on A40 (`out/two_ctx_a40`).

Results:
- Offline (2 contexts, all layers and steps), chern_grad MB per answer at eps0 0.3 / 0.5: 1702 / 226, against 562 / 128
  with persistence.
  - Decode now re-fetches at every step.
  - Worst buffer: 68% / 39% of the context in the question pass.
  - Other methods: mc_grad 285 / 11, needle 46.5, oracle 47.5 / 3.6. Fail at most 0.22%.
- Downstream (niah_single_3 + qa_1, row 152): all arms 100 except bern 0. MB per answer: chern 1171 / 165, needle about
  36.

## vkv1_hx (2026-10-24): hx bound and capped store (negative)

`experiments/vkv1_hx/`; theory in `theory/hx_store_note.md`.
- **hx**: exact DP on the 64 heaviest live terms (jumps rounded up for the upper tail, down for the lower), plus
  Chernoff on the rest. `test_hx.py`: 5 tests, including m = 0 equal to chernoff_curves and an exact match to brute
  force.
- **store**: draw at 1/16 - rho plus a store of rho n slots, filled draw-free.

Smoke (2 layers, 4 steps), MB per answer at eps0 0.3 / 0.5:
- chern 78.9 / 15.0; hx 76.9 / 13.7.
- With a 1% store: chern 90.3 / 20.0. Chern's question-pass set is about 14% of the context, which a 1% store cannot
  hold.

The full run was cancelled (user). `slack.py` shows why hx cannot help much:
- At chern's k*, the true P(fail) is about 0.5% (MC with 2,000 draws), the normal approximation agrees, hx reports
  2-4.5% and Chernoff about 5%.
- The slack is the moderate-deviation looseness of exponential bounds, not heavy jumps. Berry-Esseen is useless here
  (C rho / sigma^3 is about 0.4).

## Chern-grad downstream, 20 rows per task (2026-10-24/25; `experiments/vkv1_r20/`)

**Setup:** Qwen3-4B-Instruct-2507, RULER-32k, 13 tasks x rows 152-171 (260 contexts), 16x, perfect sketch, A40 only.
Six jobs; `analyze.py` merges them into `out/r20_all`.

**Mean score:**

| arm | score |
|---|---|
| full | 91.46 |
| bern | 64.78 |
| needle_final_c | 87.53 |
| chern-grad eps0 0.3 (no persistence) | 89.99 |
| chern-grad eps0 0.5 (no persistence) | 90.21 |

**Paired differences:**
- Against full: chern .3 -1.47 +- 0.60, chern .5 -1.25 +- 0.67, needle -3.93 +- 1.21, bern -26.7 +- 2.5.
- Chern against needle: +2.46 +- 1.23 (.3) and +2.68 +- 1.27 (.5).
- Chern .3 against .5: -0.22 +- 0.78.

**Loss by task:** most of chern's loss is on cwe (full 79, chern 66.5 / 64, needle 47.5) and qa_1 (75 against 70).

**Online fail:** 0.15% (.3) and 0.07% (.5) in both the question pass and decode.

**Cost per answer:**

| arm | MB per answer | question pass, % of context per KV head | decode, % per step | time |
|---|---|---|---|---|
| chern .3 | 1089 | 8.7 | 0.36 | about 245 s |
| chern .5 | 152 | 1.7 | 0.035 | about 245 s |
| needle | 36 | | | 18 s |

**Caveat:** chern plans with exact weights; needle uses its 2-bit sketch, which is charged to its budget.
