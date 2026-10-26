# vkv_final_algo

The final Vkv configuration: **chern-grad planned on a 2-bit key sketch, with a per-KV-head shortlist of 128 exactly
scored keys**. Default: 16x compression, eps0 = 0.3, delta = 0.05.

Code: `phase4/vkv_final/` (self-contained on this branch; it uses this repository's `kvpress/` and `phase4/vkv1/algo.py`).

---

## 1. What problem it solves

At 16x compression the GPU keeps 1/16 of the KV cache; the full cache lives in host memory. For every attention row
(one query head at one position, in one layer) the algorithm must decide which missing tokens to read back, so that
the softmax denominator B = sum_i exp(q . k_i) is estimated within a factor eps0 with high probability:

    P( |B_hat - B| > eps0 * B ) <= delta        for every decision (row) separately

The estimate B_hat comes from a random sample of the cache (Horvitz-Thompson reweighting) plus whatever is fetched.
To choose what to fetch without reading the keys, the algorithm plans on a small sketch of every key that stays on
the GPU. The plan is only as good as the sketch's ability to see the few heavy tokens (the "needle" an answer depends
on). That is what this configuration gets right.

## 2. The algorithm, step by step

Per context (once, after prefill):

1. **Score** every (layer, KV head, token) with KVzip (`CachedKVzipPress`).
2. **Budget**: total memory = (1 - compression_ratio) of the full cache = 6.25% at 16x. The sketch costs 9 bytes per
   token and KV head out of 512 (1.76%), paid from that budget. The random draw keeps the rest: **4.49%**.
3. **Residency**: keep probability r_i = min(1, c * score_i), one global c over all layers and heads so that the
   expected kept fraction is 4.49%; the first 64 tokens (attention sinks) and the last 64 (local window) get r = 1.
   One Bernoulli draw Z_i ~ Bernoulli(r_i) decides what stays on the GPU. Tokens with r < 1 are the **tail**.
4. **Sketch** (2-bit, per layer and KV head): keep the 16 RoPE channel pairs (32 of 128 channels) with the largest
   mean key energy, and store each at 2 bits (4 uniform levels between the min and max of its group of 128 tokens,
   fp16 zero and scale per group). RoPE rotates within a pair, so a pair's contribution to q . k does not depend on
   position, and the choice needs only the keys. Every token keeps its own rounded values, so an unusual key stays
   unusual in the sketch.

Per forward (the question pass, then each decode step), per layer:

5. **Shortlist** (per KV head): the 128 tail tokens with the largest sketch score over all of the head's query rows.
   Their keys are read from the host and scored exactly. This is a fixed buffer of 128 keys per KV head per forward,
   read unconditionally.
6. **Plan weights** (per row): exact weights for the resident-for-sure tokens (r = 1) and the shortlist, sketch
   weights exp(q_32 . k_hat) for the rest of the tail.
7. **Certificate** (per row), on those plan weights:
   - **T\*** (truncation set, MSE rule): the tail entries with the largest Horvitz-Thompson jumps are not upweighted
     by 1/r. They then add a known downward bias, and much less variance.
   - **Fetch order "grad"**: entries sorted by how much fetching each one lowers the error bound (bias plus
     variance times the bound's sensitivity), not simply by weight.
   - **k\***: the smallest k such that the exact two-point Chernoff bound on the first k entries of that order gives
     P(|B_hat - B| > eps0 B) <= delta. The first k\* entries are fetched (full K+V).
   - The plan uses only the sketch, the shortlist and r, never the draw. That is what makes the per-decision bound
     valid for the random draw.
8. **Attention**: an additive bias on the scores:
   - 0 on resident-for-sure, fetched, and sampled T\* tokens;
   - log(1/r) on the other sampled tail tokens (the Horvitz-Thompson reweighting);
   - -inf on everything else.

   Softmax over the biased scores then has denominator B_hat.
9. **No persistence**: fetched entries serve that forward only; the next forward starts again from the draw.

## 3. Where each step is in the code

| step | file | function |
|---|---|---|
| 1 KVzip scores | `kvpress/kvpress/presses/cached_kvzip_press.py` | `CachedKVzipPress` (scores cached under `$KVPRESS_SCORE_CACHE`) |
| 2 budget | `phase4/vkv_final/press_final.py` | `VkvFinalPress.draw_frac`; sketch cost `NeedleFetchPress.sketch_frac` -> `sketch_bytes_per_entry` in `kvpress/.../needle_fetch_press.py` |
| 3 residency and draw | `press_final.py` | `VkvFinalPress._compress_post` -> `phase4/vkv1/algo.py: residency`; the draw uses `BernoulliPress`'s generator, seeded by the context tokens |
| 4 sketch | `kvpress/kvpress/presses/needle_fetch_press.py` | `NeedleFetchPress._attach_sketch` -> `top_energy_pairs` (channels), `quant_sketch` (2-bit) |
| 5 shortlist | `press_final.py` | `vkv_final_bias`, block "(a) shortlist" |
| 6 plan weights | `press_final.py` | `vkv_final_bias`, block "(b) plan weights" |
| 7 T\*, order | `phase4/vkv1/algo.py` | `prepare` (calls `truncation`; `order="grad"`) |
| 7 k\* | `phase4/vkv_final/chernoff.py` | `kstar_fast` (default) or `chernoff_curves` + `first_pass` (exact) |
| 7 fetch set | `phase4/vkv1/algo.py` | `fetch_mask` |
| 8 bias | `phase4/vkv1/algo.py` | `attention_bias` |
| hook | `kvpress/kvpress/attention_patch.py` | calls `needle_fetch_press.needle_decode_bias`, which `press_final` routes to `vkv_final_bias` for states carrying `"vkv_final"` |
| statistics | `press_final.py` | `new_stats` (field meanings in its docstring); realised error via `algo.estimate` |

### Dependencies
```
phase4/vkv_final/press_final.py
 ├── kvpress (this repo, kvpress/)
 │    ├── NeedleFetchPress   parent class: model hooks, _attach_sketch (2-bit sketch), sketch_frac
 │    ├── BernoulliPress     the draw's generator; CachedKVzipPress the scores
 │    └── attention_patch    calls needle_decode_bias(query, key, state, scaling) -> our bias
 ├── phase4/vkv1/algo.py     residency, truncation (T*), prepare (order + curves), fetch_mask, estimate,
 │                           attention_bias  (unchanged since vkv_draft0; tests in phase4/tests/test_algo.py)
 └── phase4/vkv_final/chernoff.py   kstar_fast / chernoff_curves (from vkv1_diag/diag_lib.py and the validated
                                    fast k*, carried over unchanged)
```
Nothing else is needed: no `sys.path` entries outside the repository, and the PQ-sketch branch is not used.

### The fast certificate (`chernoff.kstar_fast`)
It computes the same bound, faster, and every k it returns is still certified:
1. **Bernstein screen.** Rows whose one-sided Bernstein bound already passes at k = 0 get k = 0.
2. **Head + Bennett.** Exact two-point terms for the first 16384 entries of the order, and Bennett's upper bound on
   the log-MGF of the rest. Rows that don't pass within those 16384 entries get the full curve.
3. **float32 per-token terms.** The sums and the minimum over lambda stay in float64, and the pass test uses
   delta (1 - 1e-3) to cover the rounding.

Validated on real extracts: 0 decisions where the float64 bound fails at the chosen k, and 0.6-1.1% more fetching
than the exact curve. `fast=False` uses the exact float64 curve.

## 4. How to run

Tests (CPU, ~6 min; from `phase4/vkv_final`):
```
PYTHONPATH=../../kvpress python -m pytest -q test_press_final.py
```
They check:
- the press against a row-by-row reference built from `algo.py`;
- the shortlist (one list per KV head, capped key reads, and a full shortlist equals the perfect plan);
- that the fast k\* is always certified by the exact bound;
- the budget arithmetic and the defaults;
- the hook routing.

In Python:
```python
import press_final                      # routes kvpress's attention hook; put phase4/vkv_final on sys.path
press = press_final.vkv_final_algo()    # 16x, eps0 0.3, delta 0.05, shortlist 128
# use as any kvpress press, e.g. pipe(context, question=..., press=press); press.stats has the counters
```

Downstream (RULER, real generation; submit from `phase4/vkv_final`):
```
mkdir -p logs
KVPRESS_SCORE_CACHE=<score cache dir> \
ARGS="--rows 152:154 --tasks niah_single_3 qa_1 --arms full vkv_final:0.9375:0.3:0.05 --run smoke" \
  sbatch -p a40 --qos=a40 job_down.sbatch
python summarize.py out/smoke     # score, paired differences vs full / perfect, fetching, failure rates
```
Arm flags:
- `:slh=<M>`: shortlist size (0 turns it off);
- `:perfect`: plan on exact weights (the reference);
- `:exact`: the exact float64 certificate instead of the fast k\*.

Use A40 for paired comparisons with the stored reference arms: the Bernoulli draw depends on the GPU model.

## 5. Results (Qwen3-4B-Instruct-2507, RULER-32k, 16x)

All downstream numbers are on 26 contexts (13 tasks x rows 152-153). "fail" is the realised rate of
|B_hat - B| > eps0 B over all decisions; the promise is <= delta. The results were produced with the experiment
copies of this press (`press_fix.py` on the kk/pq-sketch kvpress). The 2-bit path of that kvpress is byte-identical to
this branch's (`quant_sketch`, `top_energy_pairs`, the byte formula, `_attach_sketch`, `attention_patch`).

### Downstream score and guarantee
| configuration | score | vs perfect (eps0 0.3) | fail Q / D at its eps0 | MB per answer |
|---|---|---|---|---|
| full cache | 95.00 | | | |
| perfect sketch (exact weights), eps0 0.3 | 95.00 | | 0.15 / 0.18% | 1082 |
| perfect sketch, eps0 0.5 | 93.85 | | 0.06 / 0.10% | 154 |
| **vkv_final_algo (2-bit + shortlist 128), eps0 0.3, delta 0.05** | **92.18** | **-2.8 +- 1.7** | **3.08 / 2.26%** | 1630 |
| 2-bit, no shortlist, eps0 0.3 | 91.22 | | 5.90 / 7.38% (promise broken) | 1718 |
| 2-bit + shortlist 128, eps0 0.5, delta 0.05 (L40S) | 88.85 | | 1.13 / 0.68% | 505 |
| 2-bit, no shortlist, eps0 0.5 | 77.95 | | 2.01 / 2.84% | 265 |
| PQ-4 sketch (vkv_chern_pq), eps0 0.5 | 46.79 | | 3.39 / 4.47% | 484 |

- vkv_final_algo scores 100 on every niah task, qa and vt, like perfect. Its only losses against perfect are cwe and
  fwe.
- Per task at eps0 0.3, the worst realised failure is cwe at 5.06% in the question pass. All other tasks are below
  4.2% (question) and 3.2% (decode). Measured at 0.5, the same runs fail 0.82 / 0.54%.

### Fetching per answer (vkv_final_algo, eps0 0.3)
| phase | rows that fetch | tokens per fetching row | context read per forward (per layer and KV head) | certified K+V | shortlist keys |
|---|---|---|---|---|---|
| question pass | 28.8% | 272 | 9.25% | 474 MB | 4 MB |
| decode (34 steps) | 34.6% | 164 | 0.49% per step | 891 MB | 262 MB |

eps0 0.3 is what drives the volume: perfect at eps0 0.3 fetches on 31 / 35% of rows too (1082 MB). The shortlist reads
are a fixed ~8 MB per decode step.

### eps0 x delta (upstream, 13 contexts, every second layer; failure at the plan's own eps0)
| eps0 | delta | fail Q / D | worst task | needle lost* | MB per answer (these 18 layers) |
|---|---|---|---|---|---|
| 0.3 | 2% | 2.64 / 2.12% | 4.5% | 0.2% | 1132 |
| 0.3 | 5% | 3.09 / 2.42% | 5.1% | 0.2% | 827 |
| 0.3 | 10% | 3.57 / 2.79% | 5.8% | 0.2% | 600 |
| 0.3 | 20% | 4.30 / 3.35% | 6.8% | 0.2% | 411 |
| 0.5 | 2% | 0.98 / 0.67% | 1.6% | 0.6% | 378 |
| 0.5 | 5% | 1.11 / 0.77% | 1.9% | 0.9% | 263 |
| 0.5 | 10% | 1.27 / 0.91% | 2.2% | 1.3% | 197 |
| 0.5 | 20% | 1.49 / 1.10% | 2.8% | 1.7% | 155 |

\* Rows where the gold answer's tokens hold >= 50% of the attention but end up with less than half their true share
in the estimate (exact weights: 0.4%).

- At eps0 0.3 the margin is thin: about 3% realised against 5% promised, and one task can sit just above 5%.
  delta 2% restores the margin at about 35% more transfer.
- At eps0 0.5 every setting holds with about 4x headroom, and delta mainly trades transfer.
- Downstream at eps0 0.5, delta 10% gave the same answers as delta 5% on all 26 contexts, and delta 20% lost one.

## 6. Why this configuration (history in brief)

- **PQ-4 (`vkv_chern_pq`, the previous final) failed.** Product quantisation snaps rare keys, needles included, to
  typical centroids. Heavy tokens lose about 8 logits, so the plan sees them at about 1/10,000 of their share and
  almost never fetches.
  - PQ-4 chern was therefore plain Bernoulli at the smaller draw it left: 46.8, vs 45.6 for no-fetch Horvitz-Thompson
    at the same draw.
  - A better k-means fit, correcting the total tail mass, or tightening delta did not help. Giving the plan the exact
    weights of the 10 heaviest tokens did.
- **The 2-bit sketch keeps each token's own values.** The true heaviest tail token stays near the top of its ranking
  (median rank 8, vs 196 for PQ-4). But it is still underestimated by about 1 logit, so on its own it loses the needle
  on 20% of the rows that matter.
- **The shortlist turns "near the top" into "exact".** Re-scoring the sketch's top 128 per KV head with the real keys
  brings needle loss to about 1-2%, and the score from 78 to 89 (eps0 0.5) or 92 (eps0 0.3).

## 7. Caveats

- **The guarantee is proven for the plan's own weights.** With a sketch, the plan's weights differ from the true ones
  for unlisted tokens, so "fail <= delta" is shown empirically: 2-3% at eps0 0.3, about 1% at 0.5. It is not proven.
- **The shortlist is an unconditional read**: 128 keys per KV head per forward, about 9 MB per decode step for
  Qwen3-4B. Certified fetches are the adaptive part.
- **The decode guarantee is per decision.** Exact for the given draw, not over the adaptive choice of later queries.
- **Simulation.** The full cache stays on the GPU. Dropping is masking and fetching is unmasking; transfers are
  counted, not timed.
- **Use one GPU model for paired comparisons.** The draw depends on the GPU model (A40 and L40S differ).
