#!/bin/bash
#SBATCH --job-name=vkv-grid
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
# Submit FROM phase4/vkv_final (logs/ and out/ are created here):
#   mkdir -p logs && ARGS="--rows 152:154 --tasks qa_1 --arms vkv_final:0.9375:0.3:0.05 --run smoke" sbatch job_down.sbatch
# Environment (override as needed): CONDA_ENV (default kvpress-final), HF_HOME, KVPRESS_SCORE_CACHE.
set -euo pipefail
source "${CONDA_SH:-$HOME/miniforge3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-kvpress-final}"
export HF_HOME="${HF_HOME:-$HOME/hf}" TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" \
       HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd "$SLURM_SUBMIT_DIR"
stdbuf -oL -eL python -u run_downstream_grid.py $ARGS 2>&1 | grep --line-buffered -v "Loading weights"
