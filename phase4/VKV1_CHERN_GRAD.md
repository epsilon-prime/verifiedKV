# Vkv1, final configuration: Chernoff certificate + grad order, no persistence (2026-10-25)

The final version of Vkv1. **Recommended setting: eps0 = 0.5, delta = 0.05, kept 1/16.** Over 260 RULER-32k
contexts it scores the same as eps0 = 0.3 at about 1/7 of the traffic (see "Final results"). The algorithm source is `theory/SOURCES/vkv_draft0.pdf`. Two
changes from the draft were tested: an exact two-point Chernoff certificate replaces Bernstein, and the fetch order is
by eps-gradient instead of by weight. Fetched tokens no longer persist.

## The algorithm

Model used so far: Qwen3-4B-Instruct-2507 (36 layers, 32 query heads, 8 KV heads, GQA groups of 4), RULER-32k.

1. **Residency, once per context.**
   - Keep probabilities r_i = min(1, c tau_i) on KVzip scores tau, with one global c over all layers and KV heads so
     that mean r = kept = 1/16.
   - The first 64 tokens (sinks) and the last 64 context tokens (local window) have r = 1 and are paid from the same
     budget.
   - One Bernoulli draw Z_i ~ Bern(r_i) per (layer, KV head, token), shared by that KV head's 4 query heads.
   - Kept tokens get the attention bias log(1/r); dropped tokens are masked.
2. **Every forward over the cache** (the question pass, then each decode step), for every layer, query head and query
   position (one "decision"):
   - T* = the truncation set from the MSE rule (draft section 3). Truncated tokens count at weight w, not w/r, and
     their bias beta is known.
   - Fetch order: decreasing eps-gradient b_i + v_i dt/dV (`order="grad"`).
   - k* = the smallest k whose exact two-point Chernoff bound on P(|B_hat - B| > eps0 B) is <= delta.
     - The bound is `diag_lib.chernoff_curves`: both tails, each minimised over a 32-point lambda grid.
     - It is used in place of the draft's Bernstein check: same truncation, same sweep.
   - Fetch the first k* tokens of the order (they become exact) and build the attention bias from them.
3. **No persistence.** Fetched tokens are used for that forward only. GPU memory stays at the kept set plus one
   forward's fetch buffer.

**Guarantee.** For each decision, P(|B_hat - B| > eps0 B) <= delta, given exact weights. With the perfect sketch
(below), this holds exactly in the offline replay and in the real question pass. In real decode it is approximate: the
decode query depends on earlier outputs, which used the same draw.

**Not yet realistic:** the plan uses EXACT attention weights ("perfect sketch"; the full cache is on the GPU, and
fetching is unmasking). Needle fetch's 2-bit sketch can be switched in (`sketch`), but with it failures went above
delta (up to 10%). Choosing the sketch is open.

## Code

| piece | file |
|---|---|
| algorithm core (truncation, sweep curves, estimator, bias) | `vkv1/algo.py` |
| exact Chernoff curves | `experiments/vkv1_diag/diag_lib.py` (`chernoff_curves`, `first_pass`) |
| press for real generation (kvpress needle hook, additive mask, sdpa) | `experiments/vkv1_chern/press_chern.py` (copy of `vkv1/press.py` plus `check="chern"`) |
| downstream runner | `experiments/vkv1_chern/run_downstream.py` |
| downstream summary (scores, online stats) | `experiments/vkv1_down/summarize.py` |
| offline (upstream) diagnostic, no persistence | `experiments/vkv1_nopersist/diag_np.py` (+ `summarize.py`) |

The press defaults are still the draft's (`check="simple"`, `order="needle"`, `persist=True`). The final configuration
must therefore be selected explicitly:
- **press:** `Vkv1Press(kept=0.0625, eps0=..., delta=0.05, check="chern", order="grad", persist=False)`
- **runner arm:** `vkv1:0.0625:<eps0>:0.05:chern:grad:nopersist`

Forgetting `nopersist` gives the old persistent variant.

**Paths and environment.**
- On Prajna: conda env `kvpress-final`, `PYTHONPATH=~/src/kvpress-e5` (the repo's `kvpress/`). The code expects to
  live at `~/phase4` (several `sys.path` inserts use `~/phase4`); see `WIP.md`.
- Offline input: attention extracts from `deps/algo_improvements/extract.py` (Qwen3-4B, 13 tasks x rows 150-151, about
  64 GB, not in git).

## How to run

```bash
# tests (CPU)
cd ~/phase4 && CUDA_VISIBLE_DEVICES="" PYTHONPATH=~/src/kvpress-e5 python -m pytest -q tests \
    experiments/vkv1_diag/test_diag.py experiments/vkv1_chern/test_press_chern.py experiments/vkv1_chern/test_final_config.py

# downstream: real generation (rows >= 152; 150-151 are the offline extracts)
cd ~/phase4/experiments/vkv1_nopersist
ARGS="--rows 152:153 --tasks niah_single_3 qa_1 --arms full bern:0.0625 needle_final_c:0.9375 \
      vkv1:0.0625:0.3:0.05:chern:grad:nopersist vkv1:0.0625:0.5:0.05:chern:grad:nopersist --run two_ctx" \
  sbatch -p a40 --qos=a40 job_down.sbatch          # writes out/two_ctx/<arm>/rows.jsonl
python ../vkv1_down/summarize.py out/two_ctx

# upstream: offline replay on the extracts (all layers, all decode steps)
ARGS="<extract dir> --out out/<run> --methods chern_grad needle oracle mc_grad" sbatch -p a40 --qos=a40 job_diag.sbatch
python summarize.py out/<run>
```

**Reproducibility.** The draw comes from PyTorch's CUDA generator. Its output for a fixed seed depends on the GPU
model: A40 and L40S give different draws. Compare draw-dependent numbers (fail, transfers) only across runs on the same
GPU type. Draw-free numbers (k*, the fetch decisions) do not depend on it.

## Final results: 13 tasks x 20 rows (2026-10-25)

**Setup:** Qwen3-4B-Instruct-2507, RULER-32k, 16x (kept 1/16), perfect sketch, rows 152-171 of every task (260
contexts), all on A40. Code is in `experiments/vkv1_r20/`: `job_down.sbatch` plus `analyze.py`, which merges the six
jobs into `out/r20_all` (in git: one `rows.jsonl` per arm, 260 rows each) and prints the tables.

| arm | mean score | paired vs full (+- SE) | MB per answer | sec per answer |
|---|---|---|---|---|
| full | 91.46 | – | – | 10 |
| Bernoulli 1/16 | 64.78 | -26.69 +- 2.54 | 0 | 20 |
| needle_final_c 16x | 87.53 | -3.93 +- 1.21 | 36 | 18 |
| **chern-grad eps0 0.3** | **89.99** | -1.47 +- 0.60 | 1089 | 243 |
| **chern-grad eps0 0.5** | **90.21** | -1.25 +- 0.67 | 152 | 245 |

- **Against needle:** chern-grad is +2.46 +- 1.23 (eps0 0.3) and +2.68 +- 1.27 (0.5), about 2 SE.
- **eps0 0.3 against 0.5:** -0.22 +- 0.78, no difference.
- **Online fail:** 0.15% (eps0 0.3) and 0.07% (0.5) of decisions, against delta = 5%.
- **Per task:** almost all of the loss is on cwe (full 79.0, chern 66.5 / 64.0, needle 47.5) and qa_1 (75 against 70).
  The other 11 tasks match the full cache.
- **Consistency with the Phase 3 final eval** (`vkv1_r20/plot/compare.py`). That eval used test rows 0-49 at 16x; this
  sweep used dev rows. The shared arms agree within 1 SE:
  - full 92.84 there, 91.46 here;
  - Bernoulli 62.01 there, 64.78 here;
  - needle 88.22 there, 87.53 here.

  `vkv1_r20/plot/` has the final-eval figure with the chern-grad points added.
- **Caveats:** chern-grad plans with the exact weights while needle uses its 2-bit sketch; traffic is 4x needle at
  eps0 0.5; about 245 s per answer in this unoptimised simulation.

## Earlier results (perfect sketch, kept 1/16, delta 0.05)

**Offline, no persistence.** 2 contexts (cwe_150, niah_single_3_150), all 36 layers and decode steps, A40
(`vkv1_nopersist/out/two_ctx_a40`).

| method | eps0 | question pass, % of context per KV head | decode, % per step | worst single-forward buffer (Q / D) | fail (Q / D) | MB per answer |
|---|---|---|---|---|---|---|
| chern_grad | 0.3 | 9.48 | 0.39 | 68% / 29% | 0.19% / 0.22% | 1702 |
| chern_grad | 0.5 | 2.14 | 0.035 | 39% / 8.7% | 0.11% / 0.07% | 226 |
| mc_grad (ideal per-decision rule, not a bound) | 0.3 / 0.5 | 2.02 / 0.12 | 0.057 / 0.0014 | | 1.4% / 0.4% (Q) | 285 / 11 |
| needle rule (exact weights) | – | 0.75 | 0.0014 | 5.6% / 0.02% | 2.0% / 6.0% (eps0 0.3) | 46.5 |
| oracle (sees the draw; lower bound) | 0.3 / 0.5 | 0.39 / 0.04 | | | 0 | 47.5 / 3.6 |

**Downstream, no persistence.** niah_single_3 + qa_1, row 152, l40 (`vkv1_nopersist/out/two_ctx`).
- Every arm scores 100 on both contexts except Bernoulli (0): full, needle_final_c, and chern-grad at eps0 0.3 and 0.5.
- MB per answer: chern-grad 1171 (eps0 0.3) and 165 (0.5); needle_final_c about 36.
- Online fail: 0.10-0.20% of decisions.
- About 90 s per answer, against 4-22 s for full or needle_final_c (an unoptimised simulation: the 32-point lambda grid
  runs at every layer and step).

**With persistence (the earlier variant, for reference only).** All 13 tasks x 4 rows (rows 152-155), downstream:
- Mean score: full 88.1, bern 57.0, needle 85.1, chern-grad 86.3 (eps0 0.3) and 86.5 (0.5).
- Paired difference vs full: -1.6 to -1.7 points.
- The no-persistence variant is in "Final results" above.

## Known problems and open questions

1. **Traffic.** At eps0 0.5 chern-grad reads about 4-5x needle per answer; at 0.3 about 30-37x.
   - Even the ideal per-decision rule (mc_grad) reads 285 MB at eps0 0.3, so 0.3 cannot reach needle level with a
     draw-free per-decision certificate under this residency.
2. **Where the slack is.** `experiments/vkv1_hx/slack.py` measured the bound against the true failure probability at
   chern's k*:
   - The true probability is about 0.5%, while Chernoff reports about 5%.
   - A normal approximation matches the truth, but Berry-Esseen is useless (its correction term is about 0.4).
   - An exact-heavy-terms bound (hx) and a 1% budget-charged store were tried. Both gave under 10% gain, or worse in
     the smoke (`theory/hx_store_note.md`, `experiments/vkv1_hx/`).
   - Untried idea: certify k by fresh simulated draws (independent of Z) with a Clopper-Pearson bound and a union over
     the k ladder. This is rigorous and per-decision, and its cost approaches mc_grad's.
3. **Fetch buffer.** Without persistence, the worst single forward of one (layer, KV head) needs a buffer of 39-68% of
   the context during the question pass (needle: 5.6%).
4. **Sketch.** Everything above uses exact weights.
5. **Decode adaptivity.** In real generation the decode guarantee is approximate (see "Guarantee").
