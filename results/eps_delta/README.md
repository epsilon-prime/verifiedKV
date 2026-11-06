# Tolerance and failure probability sweep (RULER, 16x)

This folder holds the sweep of the certificate's error target ε and failure probability δ. VerifiedKV runs with
KVzip + 2-bit sketch at 16× (compression_ratio 0.9375) and a 128-token shortlist, for
ε ∈ {0.01, 0.05, 0.1, 0.2, 0.3} × δ ∈ {0.02, 0.05, 0.1}, plus dense: 16 arms per model.

The models are Qwen3-4B-Instruct-2507 (cdbee75), Ministral-3-8B-Instruct-2512-BF16 (f6fae97) and Qwen3.5-9B (c202236). The
tasks are RULER cwe, qa_1, niah_multikey_3 and niah_multivalue. The rows are the seed-42 permutation, so rows 0-29 are the
first 30 of the 50 test rows.

| model (GPU) | RULER-32K contexts | RULER-128K contexts (sweep stopped early) |
|---|---|---|
| Qwen3-4B (H100) | 120 (30 per task) | 48 (cwe 15, mk3 10, mv 9, qa1 14) |
| Ministral-3-8B (H200) | 120 (30 per task) | 40 (10 per task) |
| Qwen3.5-9B (H200) | 120 (30 per task) | 34 (cwe 10, mk3 9, mv 5, qa1 10) |

A context counts only when all 16 arms finished it, so every arm is compared on the same contexts.

## Files

`eps_delta_cells.csv` has one row per model × context length × arm × task. Its columns are:

- `score` and `n`: the RULER score (string_match) and the number of contexts.
- `diff_vs_dense` and `se_diff`: the paired difference to dense on the same contexts, and its standard error.
- `read_pct`: the share of the KV read from the host per forward.
- `gb_per_answer`: the bytes moved CPU → GPU per answer, in GB.
- `sec_per_answer`: the measured seconds per answer.
- `gpu`: the GPU type the model ran on.

`eps_delta_rows.csv` has one row per (arm, task, row). It includes the 32K rows past 29 that finished before the sweep
stopped; these have `in_cells` = 0.

`experiments/tolerance_sweep/plot_eps_delta.py` draws the figure from the rows file alone:
`python experiments/tolerance_sweep/plot_eps_delta.py --context-k 32` (or `--context-k 128`).

## Traffic

Traffic comes from per-row counters in the press. `read_pct` is (2 × fetched K/V entries + shortlist keys read) divided
by (2 × (decode steps + 1) × layers × KV heads × context tokens). Here an entry is one token of one KV head in one layer.
`gb_per_answer` is the same count in bf16 bytes.

## Code

The runner is `experiments/tolerance_sweep/run_downstream_grid.py`, with per-row traffic counters added. Each model ran on
one GPU type, as listed above.
