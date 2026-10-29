# Final results for the paper

The data for the paper's quality figures and tables (no figures here; make them from these CSVs): three models x four datasets x all methods
(VerifiedKV final arms and the phase 3 baselines), at 8x, 16x and 32x. Scores are 0-100.

## Protocol (what every number means)

| dataset | tasks | questions per task |
|---|---|---|
| RULER-32K, RULER-128K | niah_mk3, niah_mv, cwe, qa1 (RULER string_match) | 50 test rows (seed-42 sample, rows 0-49). Exception: Ministral-3 RULER-128K phase 3 baselines (Dense, KVzip, Fast KVzip, vKV no fetch, vKV fetch phase 3) have 20 rows |
| LOFT-32K, LOFT-128K | nq, hotpotqa, musique, qampari, quest (LOFT official subspan_em) | the first 25 test questions, the same 25 for every method and model (checked) |

LOFT uses 25 questions because the Ministral-3 and Qwen3-4B vkv_final runs are partial. The first 25 are easier
than the full set (Dense scores 4-9 points higher on them), so never compare these LOFT numbers with full-set ones.

## Layout

```
all_models.csv          every score: model, dataset, context_k, rate, method, task, score, n, label, cell
methods.csv             method id -> paper label, origin (phase 3 / final grid), what it is
<model>/                Qwen3-4B, Qwen3.5-9B, Ministral-3-8B
  RULER_32k.csv ...     one per dataset: rows = rate x task (+ mean), columns = methods (paper labels)
  summary.csv           mean over tasks per dataset, context, rate, method; n_per_task
  results_long.csv      this model's rows of all_models.csv
build.py                rebuilds the CSVs from ../final_results/combined_loft25.csv
```

`n` is the number of questions per task in that cell. `cell` is the result folder the score came from: phase 3
cells under `~/phase3/experiments/final_eval/grid/results/`, vkv_final cells under `../final_results/raw/`.

## Methods

| id | label | notes |
|---|---|---|
| dense | Dense | full cache |
| kvzip, fastkvzip | KVzip, Fast KVzip | eviction baselines |
| vkv_nofetch_kvzip | vKV (no fetch)+KVzip | Bernoulli sample, reweighted, no verification or fetch |
| vkv_final_kvzip, vkv_final_fastkvzip | vKV+KVzip, vKV+Fast KVzip | VerifiedKV: eps0 0.05, delta 0.1, 2-bit sketch + 128 shortlist; sketch charged to memory |
| vkv_final_perfect | vKV, exact sketch | reference: exact weights, 2% memory charged |
| vkv_fetch_kvzip, vkv_fetch_fastkvzip | vKV fetch (phase 3) | earlier VerifiedKV version; superseded, kept in the CSVs; leave out of the figures |

## Rebuild

```bash
source ../env.sh            # or any env with pandas + matplotlib
python build.py
```

`../analysis/plot_quality.py --csv all_models.csv --ruler-tasks niah_mk3 niah_mv cwe qa1` draws phase 3's figure
style from this data if wanted (`--models Qwen3.5` for one model).

Chain: phase 3 cells + `../final_results/raw/` -> `../make_loft_firstn.py` -> `../final_results/combined_loft25.csv`
-> `build.py` -> this folder. `combined_loft25.csv` also holds the phase 3 baselines on the 9 other RULER tasks
(972 rows), which `build.py` drops: the vkv_final arms ran only the 4 tasks. The per-model CSVs in `../final_results/` (all LOFT questions run) and
`../baselines/phase3_results.csv` are not used for the paper numbers and are left unchanged.
