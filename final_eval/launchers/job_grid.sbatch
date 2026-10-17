#!/bin/bash
#SBATCH --job-name=vkv-final
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=16:00:00
#SBATCH --output=logs/%x-%j.out
# Run sweeps of the final grid on one GPU. Submit from final_eval/ (the log path above is relative to it):
#     cd final_eval && SWEEPS="sweeps/<model>_<len>_r<x>[_suffix].yaml ..." sbatch -p <partition> launchers/job_grid.sbatch
# GPU memory: 128k sweeps need an 80 GB card (cells peak at 52-68 GB; Qwen3.5-9B ~50 GB on a 174k-token row);
# 32k sweeps fit a 48 GB card (peak <= 31 GB). Set --time from README.md's cost table: a job stopped at its
# limit keeps its finished cells, and a resubmit runs only what is left (cells holding metrics.json are skipped).
# Optional environment: ENV (conda env, default kvpress-final), CONDA_SH, HF_HOME, KVPRESS_SCORE_CACHE, GATES.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
[ -f run_sweep.py ] || { echo "submit from final_eval/ (no run_sweep.py in $PWD)" >&2; exit 1; }

source "${CONDA_SH:-$HOME/miniforge3/etc/profile.d/conda.sh}"
conda activate "${ENV:-kvpress-final}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}" \
       TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
       KVPRESS_FASTKVZIP_GATES="${GATES:-$PWD/fastkvzip_gates/gates}"
# RULER and SCBench scorers import nltk -> sqlite3, which needs a newer CXXABI than some system libstdc++ have
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for SW in $SWEEPS; do
  stdbuf -oL -eL python -u run_sweep.py --sweep "$PWD/$SW" 2>&1 \
    | grep --line-buffered -v "Loading weights"
done
