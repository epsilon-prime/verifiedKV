#!/bin/bash
# Run final-grid sweeps on a machine without SLURM (a shared GPU box), inside tmux or screen. Same environment
# as job_grid.sbatch; only the GPU choice differs.
#
# usage (from final_eval/):  bash launchers/run_local.sh <gpu> sweeps/a.yaml [sweeps/b.yaml ...]
#   <gpu>  nvidia-smi index. Results go to final_eval/results/<group>, exactly as under SLURM.
# Refuses 128k sweeps on cards under 80 GB, and a GPU with less than MIN_FREE_MIB free (default 36000,
# which covers a 32k cell's ~31 GB peak) since other users may share the card.
# Resumable like job_grid.sbatch: cells that already hold metrics.json are skipped.
# Optional environment: ENV (conda env name or path, default kvpress-final), CONDA_SH, HF_HOME,
# KVPRESS_SCORE_CACHE, GATES, THREADS (CPU thread cap, default 16).
set -euo pipefail
GPU=$1; shift
[ -f run_sweep.py ] || { echo "run from final_eval/ (no run_sweep.py in $PWD)" >&2; exit 1; }
TOTAL=$(nvidia-smi -i "$GPU" --query-gpu=memory.total --format=csv,noheader,nounits)
case " $* " in *128k*) [ "$TOTAL" -ge 79000 ] || { echo "refusing: 128k sweeps need an 80 GB card" >&2; exit 1; } ;; esac
FREE=$(nvidia-smi -i "$GPU" --query-gpu=memory.free --format=csv,noheader,nounits)
[ "$FREE" -ge "${MIN_FREE_MIB:-36000}" ] || { echo "refusing: GPU $GPU has only $FREE MiB free" >&2; exit 1; }

source "${CONDA_SH:-$HOME/miniforge3/etc/profile.d/conda.sh}"
conda activate "${ENV:-kvpress-final}"
# machines that mix GPU models: CUDA's default order is fastest-first, so pin it to nvidia-smi's numbering
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU"
# SLURM jobs get 16 CPUs; without SLURM nothing caps the thread pools, so cap them the same way
export OMP_NUM_THREADS="${THREADS:-16}" MKL_NUM_THREADS="${THREADS:-16}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}" \
       TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
       KVPRESS_FASTKVZIP_GATES="${GATES:-$PWD/fastkvzip_gates/gates}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

mkdir -p logs
exec > >(tee -a "logs/local-gpu$GPU-$(date +%m%d-%H%M).out") 2>&1
nvidia-smi -i "$GPU" --query-gpu=name,memory.total --format=csv,noheader
for SW in "$@"; do
  stdbuf -oL -eL python -u run_sweep.py --sweep "$PWD/$SW" 2>&1 \
    | grep --line-buffered -v "Loading weights"
done
