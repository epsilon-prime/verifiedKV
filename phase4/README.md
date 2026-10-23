# Phase 4: Vkv1

Vkv1 = truncated Horvitz-Thompson estimation of the attention denominator, with a Bernstein certificate and a
certified needle-order fetch. Opened 2026-10-23.

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
