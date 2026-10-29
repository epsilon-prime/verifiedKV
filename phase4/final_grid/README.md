# Final grid: vkv_final on RULER and LOFT

Everything needed to run the final evaluation of `vkv_final_algo` (`../VKV_FINAL_ALGO.md`) and score it against the
phase 3 baselines. Every person runs the same committed code, the same sweep files and the same environment; only
paths differ. Do not edit sweeps or arm settings by hand. Change `make_sweeps.py` or `arms.py`, commit, and tell
everyone.

## 1. What runs

| | |
|---|---|
| Models | Qwen3-4B-Instruct-2507, Qwen3.5-9B, Ministral-3-8B-Instruct-2512 (BF16) |
| RULER-32k, RULER-128k | niah_multikey_3, qa_1, niah_multivalue, cwe; 50 test rows each (rows 0-49 of the seed-42 permutation, as phase 3) |
| LOFT-32k, LOFT-128k | nq, hotpotqa, musique, qampari, quest; all test questions (60-100), one shared context per task |
| Ratios | 8x, 16x, 32x (compression_ratio 0.875, 0.9375, 0.96875) |
| Arms | `vkv_final_kvzip`, `vkv_final_fastkvzip`, `vkv_final_perfect` (below) |
| Settings | eps0 0.05, delta 0.1, shortlist 128 per KV head, 2-bit sketch, 64 sinks + 64 window, fast vectorised certificate |

All settings live in `arms.py`. The sweeps (`sweeps/`, 162 files, one per model x length x ratio x task) are written by
`make_sweeps.py`.

**The arms.** In all three, the Bernoulli draw keeps `1 - compression_ratio - charge` of all KV entries, sinks and
window included.
- `vkv_final_kvzip`: the draw is over KVzip scores. The plan uses the 2-bit sketch, and the charge is the sketch's
  memory, 1.76% of the cache (the same share for all three models).
- `vkv_final_fastkvzip`: the same, over Fast KVzip gate scores (no local window, as phase 3's `fast_needle_final`).
- `vkv_final_perfect`: the plan uses the exact attention weights, an ideal sketch. 2% of the cache is charged as a
  stand-in for that sketch's memory.

At 32x the draw keeps 1.37% (2-bit) or 1.125% (perfect).

**Who runs what (2026-10-28).** Krishna: `sweeps/ministral3-8b_*` and `sweeps/qwen3-4b_*`. Tanvi: `sweeps/qwen3.5-9b_*`
(Prajna for RULER-32k, LOFT-32k and RULER-128k; RunPod for LOFT-128k).

## 2. Files

| file | what |
|---|---|
| `arms.py` | the three arms and their settings (the only place they are set) |
| `run_grid.py` | the harness: a copy of phase 3's `run_check.py` (same loading, sampling, prompts, scoring) |
| `make_sweeps.py` | writes `sweeps/` (the grid) and `sweeps_smoke/` (smoke tests on dev rows) |
| `env.sh` | the one environment of every run (conda env, caches, gates, fixed flags) |
| `run_sweeps.sh` | runs sweeps one after another on the visible GPU, one log per sweep in `logs/` |
| `job_grid.sbatch` | Prajna (SLURM) wrapper around `run_sweeps.sh` |
| `launch_local.sh` | no SLURM: spreads a list of sweeps over GPUs through a shared queue |
| `download.sh` | models, datasets and gates (run once, with internet) |
| `gates/` | Fast KVzip gates, with `SHA256SUMS` and `fetch_gates.sh` |
| `env/requirements.txt` | pinned packages of the run environment |
| `score_grid.py` | finished cells -> `results.csv` (phase 3 schema), `--plot` for the figure and tables |
| `scoring.py`, `loft_official/` | scoring: RULER string_match; LOFT with LOFT's official parser (phase 3's `score_loft.py`) |
| `baselines/` | phase 3's baselines (`phase3_results.csv`, rebuilt from every finished phase 3 cell), its figure and tables, and `make_baseline_csv.py` |
| `analysis/plot_quality.py` | phase 3's plot and table builder, extended for the vkv_final arms |
| `test_arms.py`, `check_multiq.py` | tests: arm budgets and registration; LOFT answers do not depend on earlier questions |

## 3. Setup (once per machine)

```bash
git clone -b vkv_final_algo git@github.com:skylight-org/vkv.git && cd vkv/phase4/final_grid
# environment: Python 3.11, torch 2.11.0 (CUDA 12.8 build), then the pinned packages
conda create -n kvpress-final python=3.11 -y && conda activate kvpress-final
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r env/requirements.txt
# paths: set these before `source env.sh` if yours differ (e.g. in ~/.bashrc)
export CONDA_SH=$HOME/miniforge3/etc/profile.d/conda.sh HF_HOME=$HOME/hf
export KVPRESS_SCORE_CACHE=/fast/disk/kvzip_scores   # any folder; KVzip scores are computed on first use
./download.sh ministral3-8b                          # models + datasets + gates (needs internet)
```

**KVzip score cache.** The first arm to see a context computes its KVzip scores and saves them. Every other arm and
ratio reuses them, so scoring is paid once per context. A Ministral cache is about 0.1 GB per 128k context. The cache
is not shared through GitHub: recomputing it costs about 3 GPU-hours for all of Ministral, and runs on different GPU
types draw differently anyway.

## 4. Before the grid: tests and smoke

```bash
source env.sh
python -m pytest -q test_arms.py ../vkv_final/test_press_grid.py      # 18 passed
(cd ../vkv_final && python test_chernoff_vec.py)                       # GPU: vectorised certificate == loop, 0 mismatches
VKV_GRID_RESULTS=$PWD/results_smoke CUDA_VISIBLE_DEVICES=0 ./run_sweeps.sh sweeps_smoke/ministral3-8b_32k_ruler.yaml sweeps_smoke/ministral3-8b_32k_loft.yaml
VKV_GRID_RESULTS=$PWD/results_smoke CUDA_VISIBLE_DEVICES=0 ./run_sweeps.sh sweeps_smoke/ministral3-8b_128k_ruler.yaml
python check_multiq.py --model mistralai/Ministral-3-8B-Instruct-2512-BF16      # must print IDENTICAL
```

The smoke sweeps use dev rows and LOFT dev questions only, never test data. Compare with the reference smoke results
in section 7.

## 5. Running the grid

Commit nothing in between: the final sweeps have `require_clean: true` and refuse to run from a modified checkout.
`cost.json` in every cell records the commit, GPU, package versions and settings.

**Without SLURM** (inside tmux):
```bash
cd phase4/final_grid
./launch_local.sh "0 1 2 3" sweeps/ministral3-8b_128k_*.yaml sweeps/ministral3-8b_32k_*.yaml   # one worker per GPU
```
Each GPU takes the next unclaimed sweep. The same command, rerun, continues: finished cells are skipped, and a sweep
that failed is retried after `rm -r logs/claims/<sweep>`.

**Prajna (SLURM)**, from `phase4/final_grid`:
```bash
SWEEPS="$(ls sweeps/qwen3-4b_32k_r16_ruler_*.yaml | tr '\n' ' ')" sbatch -p a40 --qos=a40 job_grid.sbatch
SWEEPS="sweeps/qwen3-4b_128k_r16_ruler_cwe.yaml" sbatch -p dgx --qos=dgx job_grid.sbatch   # 128k: A100 80 GB
```

**GPU memory.** 128k needs about 50-70 GB (A100/H100 80 GB). 32k fits a 48 GB GPU (A40, L40S, A6000).

**One job per GPU.** Two jobs sharing a GPU gave no speed-up. Measured on an L40S with Qwen3-4B at 32k (3 dev contexts):
one job alone took 582 s; two jobs at once took 1,235 s each, against 1,164 s for the two run one after the other
(-6% throughput). The answers were identical. A single job already keeps the GPU busy, because the vectorised
certificate removed most launch gaps. `job_grid.sbatch` (SWEEPS2) and `launch_local.sh` can still run two per GPU,
but do not.

**Which GPU type.** Keep each (model, length) on one GPU type: the random draw differs between GPU types. Different
types still give valid results, but they are different draws.

## 6. Scoring

```bash
python score_grid.py --results results [other_results_roots...] --plot
```

This writes `results.csv` (columns `model,dataset,context_k,rate,method,task,score`, the same as
`baselines/phase3_results.csv`) and `analysis/final_*` (phase 3's figure and tables with the vkv_final arms, every
method averaged over the four RULER tasks). LOFT is scored only with LOFT's official parser.

To merge two people's results, copy (or commit) each `results/final_vkv_<model>/` folder into one `results/` and
score once.

## 7. Timing and smoke reference

Prajna smoke tests, 2026-10-28: eps0 0.05, delta 0.1, 16x, dev rows (RULER rows 150 / 400, LOFT nq dev
questions). Seconds per RULER answer or per LOFT question, and peak GPU memory (GB, 1e9 bytes):

| model | RULER-32k (A40) | LOFT-32k (A40) | RULER-128k (A100) | LOFT-128k (A100) | peak 32k / 128k |
|---|---|---|---|---|---|
| Qwen3-4B | qa_1 65, multivalue 98, multikey_3 150-180, cwe 170-213 | 76-79 | 214-271 | 59-66 | 19 / 52 |
| Qwen3.5-9B | 15-61 (mean about 28) | 11-23 | 55-85 | 10-16 | 27 / 58 |
| Ministral-3-8B | qa_1 60, multivalue 160, multikey_3 165-209, cwe 158-264 | (running) | 244-382 | (running) | 29 / 68 |

- Checks: all smoke cells finished without failures. Each draw kept its intended fraction: 4.49% for the 2-bit arms
  and 4.25% for the perfect arm at 16x.
- LOFT check (`check_multiq.py`): answers were identical whether asked together or one at a time, for all three
  models.
- Expect some spread from run to run: the first cell of a job also pays one-off warm-up.

## 8. Results

`final_results/` holds the finished results, one model at a time.
- `<model>.csv`: one row per (dataset, length, ratio, arm, task), with the same columns as
  `baselines/phase3_results.csv` plus `n` (the number of answers) and `cell` (the source folder).
- `raw/final_vkv_<model>/`: the cell folders behind the CSV (`predictions.csv`, `metrics.json`, `config.yaml`,
  `cost.json`).
- `logs/`: the sweep logs.

To rescore: `python score_grid.py --results final_results/raw --out x.csv`. It must reproduce `<model>.csv` exactly.

**Qwen3.5-9B: complete, 162 of 162 cells, 2026-10-29.** All cells used eps0 0.05 and delta 0.1, and none ran from a
modified checkout.

| part | GPU | commit |
|---|---|---|
| RULER-32k | Prajna L40S | 833da10 |
| LOFT-32k | Prajna A40 | 833da10 |
| RULER-128k, 9 sweeps | Prajna A100 | 833da10 |
| RULER-128k, 3 sweeps (8x cwe, 16x niah_multikey_3, 32x qa_1) | RunPod H100 | 79becfd |
| LOFT-128k | RunPod H100 | 833da10 |

Commit 79becfd differs from 833da10 only in setup (`env.sh`, `requirements.txt`). The H100 sweeps used the same
cached KVzip scores as the Prajna ones.

Figures: `analysis/qwen35_quality_vs_compression.{png,pdf}` (2x2: RULER and LOFT at 32k and 128k, against the phase 3
baselines including Bernoulli = "vKV (no fetch)+KVzip"), the `_relative` version (% of Dense), and
`analysis/qwen35_quality_per_task.tex`. Every method is averaged over the same 4 RULER tasks. The `analysis/final_*`
figure has all three models; its Qwen3 and Ministral columns show only the baselines until their results are added.
The single-model figure:

    python analysis/plot_quality.py --allow-partial --models Qwen3.5 \
        --csv baselines/phase3_results.csv final_results/qwen3.5-9b.csv --ruler-tasks niah_mk3 niah_mv cwe qa1 \
        --out analysis/qwen35_quality_vs_compression.png --out-relative analysis/qwen35_quality_vs_compression_relative.png \
        --appendix-table analysis/qwen35_quality_per_task.tex
