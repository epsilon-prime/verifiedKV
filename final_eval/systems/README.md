# Systems evaluation (vLLM)

vkv against exact dense recall and FastKVzip eviction as a serving system: accuracy, decode throughput, TPOT,
TTFT, bytes moved over PCIe, GPU memory, and batch scaling, on RULER 32k / 128k and LOFT 128k at 16x compression
(Qwen3-4B-Instruct-2507, H200). Unlike the rest of `final_eval/` (the kvpress / Hugging Face grid), these runs
use the vLLM implementation in the separate `KV_offload` repository; this folder holds only their launchers.
Results and figures are in `../../phase4/final_grid/ablations/systems/`.

## Setup

1. Clone `KV_offload` and build its environment (vLLM 0.29 with the offload connector; `.venv` inside the
   checkout). The results here were produced at `KV_offload` commit `f30a179` (branch `vkv_backend`).
2. Point the launchers at it, if it is not a sibling of this repository:
   ```bash
   export KV_OFFLOAD=/path/to/KV_offload        # default: ../KV_offload next to the vkv checkout
   export PYTHON=/path/to/python                # default: $KV_OFFLOAD/.venv/bin/python
   export LOG_DIR=/path/to/logs                 # default: $KV_OFFLOAD/logs/systems
   ```
   `VKV_ROOT` (this repository) is found from the launchers' location; the runners use it for the harness's
   RULER 128k split (`kvpress/evaluation/benchmarks/ruler/prepare_dataframe.py`) and LOFT scorer.
3. FastKVzip gates: Qwen3-4B uses the authors' released gate, downloaded on first use.
4. `numactl` must be installed: every run binds CPU and host memory to its GPU's NUMA node (pass the node).

Each run is resumable (rows already in `out/<run>/<config>/rows.jsonl` are skipped); a finished arm appends
`<task> <arm> <exit code>` to `$LOG_DIR/<run>.done`. Launch long runs inside `tmux`.

## Launchers (`launchers/`)

| Script | What it runs |
|---|---|
| `run_ruler_sweep.sh` | RULER, one request at a time: vkv-fastkvzip over the (eps, delta) grid, dense recall, FastKVzip evict, dense |
| `run_batch_scaling.sh` | RULER batch scaling, B = 1, 2, 4, 8, one arm per GPU |
| `run_loft.sh` | LOFT RAG 128k |
| `env.sh` | shared paths and checks, sourced by the others |

## Reproducing the results (one task or arm per GPU; `<gpu> <node>` pairs as on our machine: 2 0, 3 0, 5 2, 6 2)

**RULER 32k, 50 rows per task, all 22 vkv points + recall + evict (+ dense):**
```bash
L=final_eval/systems/launchers
for spec in "2 0 niah_multikey_3" "3 0 qa_1" "5 2 niah_multivalue" "6 2 cwe"; do
  DATASET=xAlg-AI/att-hub-ruler-32k MAXLEN=46000 bash $L/run_ruler_sweep.sh r32sweep $spec 0:50 &
done; wait
DATASET=xAlg-AI/att-hub-ruler-32k MAXLEN=46000 ARMS=dense bash $L/run_ruler_sweep.sh r32sweep 2 0 qa_1 0:50
```
Table: `python tools/sweep_table.py out/r32sweep` (from `$KV_OFFLOAD`).

**RULER 128k, 10 rows per task** (niah_multikey_3 and cwe need the longer context and staging ring):
```bash
bash $L/run_ruler_sweep.sh r128sweep 3 0 qa_1 &
bash $L/run_ruler_sweep.sh r128sweep 5 2 niah_multivalue &
MAXLEN=200000 STAGING_SLOTS=400 bash $L/run_ruler_sweep.sh r128sweep 2 0 niah_multikey_3 &
MAXLEN=200000 STAGING_SLOTS=400 bash $L/run_ruler_sweep.sh r128sweep 6 2 cwe &
wait
```

**System metrics, RULER 32k, delta 0.05 points (3 rows per task):**
```bash
SW="0.01:0.05 0.03:0.05 0.05:0.05 0.07:0.05 0.1:0.05 0.15:0.05 0.25:0.05 0.3:0.05 0.35:0.05 0.4:0.05 0.5:0.05"
DATASET=xAlg-AI/att-hub-ruler-32k MAXLEN=46000 EXTRA=--sysmon SWEEP="$SW" ARMS="vkv recall evict dense" \
  bash $L/run_ruler_sweep.sh sys32k 2 0 niah_multikey_3 0:3      # likewise for the other three tasks
```
Tables: `python tools/sys_table.py out/sys32k --csv <file>`; available PCIe / HBM bandwidth:
`CUDA_VISIBLE_DEVICES=2 numactl --cpunodebind=0 --membind=0 python tools/bw_bench.py`.

**Batch scaling** (`CONTEXT=128k` for RULER 128k):
```bash
bash $L/run_batch_scaling.sh bscale 2 0 vkv 0.05 & bash $L/run_batch_scaling.sh bscale 3 0 vkv 0.5 &
bash $L/run_batch_scaling.sh bscale 5 2 recall &   bash $L/run_batch_scaling.sh bscale 6 2 evict & wait
```
CSV: `python tools/batch_scaling_csv.py out/bscale <dest dir>`.

**LOFT 128k:**
```bash
bash $L/run_loft.sh loft128k 0:20 2 0 dense vkv-compact kvzip-evict offload-recall-compact
python tools/score_loft_vllm.py out/loft128k
```
