# Final evaluation setup: audit (2026-10-14)

Scope: everything `SETUP.md` points to before the final grid is submitted. That covers the uncommitted
`final-eval-setup` branch of `~/src/kvpress-e5` (including the new `FastKVzipScorer` / `fast_needle_final` arm), the
shared runner `harness_check/run_check.py`, `grid/`, `queue/`, `readiness/`, `fastkvzip_gates/`, `trigger_check/`, the
Qwen3.5 probe and the metric code. Everything was read-only: no code, sweep or queue file was changed, and no job was
submitted. The only script written is `audit/loft_parser_check.py`.

## Verdict

The code changes are correct, and the grid matches the decisions in SETUP §12 and §13. Two things should be dealt
with before "submit":

1. **The LOFT scorer is not LOFT's official parser** (A1). This can be fixed after the runs by rescoring the saved
   predictions, but SETUP §4d's claim that it is "line-for-line the same rule" is wrong as written.
2. **The 128k question-leak check (R2) was never run.** The 32k check that was run predates four changes (A2).

Everything else is either verified below or is a caveat for the paper.

## A. Findings that need a decision

### A1. LOFT metric: kvpress's answer parser differs from LOFT's official one

LOFT's official `extract_prediction` (`google-deepmind/loft`, `utils.py`) stops at the **first** line that contains
`[...]`. If that list does not parse, the answer gets no prediction, and there is no other fallback. kvpress's copy
(`evaluation/benchmarks/loft/calculate_metrics.py`) does two extra things:

- after a failed parse, it keeps scanning later lines;
- if no list is found, it falls back to the text after "final answer".

The scoring functions (subspan EM, one-to-one matching for qampari / quest, normalisation) and the per-dataset
choice of evaluator do match LOFT's `rag.py` exactly.

Measured on every rehearsal LOFT cell (`audit/loft_parser_check.py`). The rescoring reproduces all 715 saved
`metrics.json` files exactly before the parser is swapped.

| | |
|---|---|
| answers parsed only by kvpress | 106 of 7,150 (1.5%): 61 from the prefix fallback, 45 from a later-line list |
| cells whose subspan EM changes | 8 of 715 (+10 to +20 points on 10-question cells; mean over all cells +0.14) |
| where | compressed arms on Qwen3.5 and Qwen3-4B (KVzip, Bernoulli); hotpotqa / nq 32k |

The extra credit goes to answers whose format has broken down, which is more common under compression, so the
deviation is arm-dependent, even though it is small.

**Fix options:**
- (a) **Recommended:** rescore the grid's saved `predictions.csv` with the official parser at analysis time. This
  needs no rerun and no change to the code that runs the grid.
- (b) Change kvpress's scorer before the grid (a code change).

Either way, correct SETUP §4d.

### A2. R2 (do questions leak into each other at 128k?) was never run

`setup/out/resmoke_*.json` holds the multi-question check, and it holds the following.

- It covers only **LOFT nq_32k** (10-13, 16:33-17:37).
- It covers only the arms `no_press`, `bernoulli_kvzip_c` and `needle_final_c`. All passed.
- It predates:
  - the `<think>` suppression (P14): Qwen3.5's answers in it still open `<think>`;
  - the 32-pair Qwen3.5 sketch;
  - our Fast KVzip gates;
  - `fast_needle_final`.
- `fastkvzip`, `fast_needle_final` and `needle_uncharged_c` were never checked.

The unit tests do cover the leak mechanisms (recurrent-state restore, `promoted` reset, cache rewind). Every LOFT
cell, however, answers 60-100 questions from one compressed cache.

**Recommended:** before "submit", run `setup/resmoke.py --sections multi --loft_task nq_128k` with every final arm,
once per model on dgx. This is a smoke test of a few minutes per arm.

## B. Documentation that is stale or wrong in SETUP.md

- **§4d:** "kvpress's LOFT metric is line-for-line the same rule" is wrong (A1). "RULER identical to NVIDIA" is
  true for the scoring functions. As far as I recall NVIDIA's `postprocess_pred`, though, RULER *replaces* control
  characters with `\n`, while kvpress *deletes* them, which joins lines. That can only add matches, so it is
  negligible. Say "same scoring functions".
- **§8a:** the Qwen3.5 row is the old 16-pair result. The rerun with 32 pairs (`trigger_check/out/`, 10-13 22:28)
  gives:

  | | recall at decode θ 0.1 | recall at question θ 0.003 | needles covered |
  |---|---|---|---|
  | 16 pairs (old) | median 0.000 | 0.12 | 0.19 |
  | 32 pairs (current) | median 0.013 (max 0.11) | 0.58 | 0.46 |

  So the Qwen3.5 decode trigger is still almost blind; its fetches come from the question pass.
- **§4b, §5, §9, §10:** these describe states that were later superseded: the 4096-token Fast KVzip window, 20
  RULER-128k rows, 90 sweeps / 554 GPU-h, and "LOFT metric still open". They are not marked as superseded. A
  one-line "superseded by §12-13" under each would prevent misreading.
- **§8, R2:** still says "ready" (A2).

## C. Caveats for the paper and analysis (not bugs)

1. **`fast_needle_final` vs `fastkvzip` differ in three things at once:**
   - sampling plus the log(1/r) correction, instead of top-k;
   - no 2% window, instead of 2%;
   - fetching.

   No "Bernoulli on Fast KVzip scores" arm exists (the analogue of `bernoulli_kvzip_c`), so the effect of fetching
   cannot be separated for this scorer. `FastKVzipScorer` already supports such an arm, so it would be one registry
   line plus about one more `fastkvzip`-sized block of compute. That is the user's decision, and it is only needed
   if the paper claims more than "needle fetch also works on Fast KVzip scores".
2. **What the guarantee covers with gate scores.** Horvitz-Thompson unbiasedness holds for any r > 0. The
   (eps, delta) denominator certificate, however, assumes the score envelopes the attention weight (w_i <= tau_i B).
   That assumption is argued for KVzip's scores, not for a trained gate's prediction of them. State the guarantee
   for the KVzip-scored arms only, or justify it separately.
3. **Memory accounting** charges the sketch and the reserve, but not:
   - the per-kept-entry log(1/r) value, about 0.4% of the kept cache at fp16, or 3.1% of it if stored as the 8
     extra key dims of the planned runtime;
   - the transient question-time fetches, already known from the fetch_v3 audit.

   Say so where memory is reported.
4. **GPU types.**
   - At 32k, the main sweeps run on A40, while Fast KVzip, the fast needle arm and the extra LOFT seeds run on
     L40S. At 128k, everything runs on A100.
   - Comparisons that share a draw (Bernoulli vs uncharged, at the same seed) are always in the same job, so they
     are truly paired (same GPU, same draw).
   - Comparisons across GPU types carry small numerical differences. `cost.json` records the GPU of every cell.
   - If fox is used (RTX 6000 Ada / A6000), it is a fourth GPU type *and* a separate copy of the code. Before any
     run there, compare a checksum of fox's `src/kvpress-e5` and `harness_check/` against Prajna's.
5. **Cost estimate.** The LOFT per-question times come from one task (nq dev) at one ratio (16x) with 10 questions.
   At 128k, Bernoulli on Qwen3-4B produced 10 tokens for 10 questions, so the 128k LOFT hours are probably
   underestimated. The largest job is estimated at about 32 h against the 48 h `--time`. The sweeps are resumable,
   so the worst case is a resubmission.
6. **Gate validation.** The "held-out" tokens are every 100th token of the *same* training documents, so the
   check is in-distribution. Downstream, our Qwen3-4B gate was slightly below the released one (67.9 vs 73.6 at
   8x on the mini rows). This is already stated in SETUP §7b, so keep the "may slightly understate Fast KVzip"
   sentence. Both our gates and the released ones carry the authors' double-bias quirk (`optim.py` lines 101 and
   103 add `b` twice; the press adds it once), so the treatment is consistent.

## D. Verified correct

- **Qwen3.5 Gated DeltaNet patch (P1-P3).** Compared line by line with stock transformers 5.2:
  - prefill (states are `None`) and single-token decode run the stock path;
  - a multi-token forward over a filled cache prepends the last kernel-1 cached conv inputs and starts the chunked
    delta rule from the cached recurrent state, which is what a token-by-token feed computes.
  - Snapshot and restore cover KVzip's repeat chunks, the end of scoring, and every answer (P2).
- **Ministral (P5, P6).**
  - The text-only load fails loudly on any missing weight.
  - The llama-4 query factor matches `Ministral3Attention`: same formula, applied after RoPE, `cache_position`
    passed as a keyword by the decoder layer. Real config: beta 0.1, original max 16384.
  - The test compares against the query the attention kernel actually receives.
  - Generation stops at eos 2 via `generation_config.json`.
- **Needle fetch (P7, P8, P10, P11).**
  - It reads K/V through the adapter and `text_config`.
  - Qwen3.5's `partial_rotary_factor` 0.25 is read from `rope_parameters` (rotary_dim 64), and the pairing
    (j, j+32) is correct for its `rotate_half` RoPE.
  - `promoted` is zeroed at a question's forward before that question's own fetches are added.
  - The trigger sees the kernel's (scaled) query.
  - Budget, recomputed: sketch 1.76% on both head sizes; kept fraction at 32x 1.07-1.28% (SETUP: 1.0-1.3%).
- **Fast KVzip and FastKVzipScorer.**
  - One gate per KV-holding layer (count asserted); sinks forced to 1; global bottom-k by layer row.
  - With the window off it protects only the sinks.
  - The scorer hands BernoulliPress stacked scores in [0, 1], used unchanged, with the 4 sinks forced to r = 1.
  - `gates/` holds only Ministral and Qwen3.5, so Qwen3-4B uses the released gate; `gates_validation/` is never
    on the load path.
- **Draws and seeds.** BernoulliPress seeds each draw from the context's CRC and its seed, never from global RNG
  state, so results do not depend on run order. `needle_uncharged_c` has the same r and the same uniforms as
  `bernoulli_kvzip_c` (the same draw); `_s1` / `_s2` change only the draw.
- **Dev and test are disjoint.**
  - Test rows are `df.sample(n, random_state=42)`, which is the first n rows of the `frac=1` permutation; dev rows
    come from positions 150+ (32k) and 400+ (128k). No length filter runs before sampling for RULER or LOFT.
  - LOFT test and dev are separated by `loft_split`. Gold answers are flat string lists, as the metric assumes.
- **Grid and queue.**
  - `make_grid.py` matches §12-13: uncharged arm and extra seeds at 16x only, no_press once per model and length,
    50 RULER rows at both lengths, `fast_needle_final` in the gate sweeps and the seeds.
  - Output folder names are unique per arm and ratio.
  - 128k runs only on dgx (never dgx-mpi). Nothing is submitted: `pending.txt` is empty and `squeue` shows no jobs.
- **Score cache.** Every Ministral and Qwen3.5 KVzip score file is newer than the last scoring-code edit (10-13
  15:51).
- **Rehearsal.** `readiness/summarize.py` rerun today: 684 cells per model, 0 failures, memory accounting ok, no
  `<think>`.
- **Tests.** The 6 new or changed test files: 79 passed.
- **Qwen3.5 "not a bug" reasoning (R8).** 1-(7/8)^4 = 0.414 and 1-(7/8)^8 = 0.656, matching the measured 0.42 and
  0.65.
