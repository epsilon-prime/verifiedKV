#!/bin/bash
#SBATCH --job-name=p3grid
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=16:00:00
#SBATCH --output=/home/rethinkingai-self/23b1835/phase3/experiments/final_eval/grid/logs/%x-%j.out
# usage: SWEEPS="sweeps/<model>_<len>_r<x>.yaml ..." sbatch -p dgx --qos=dgx job_grid.sbatch   (FINAL GRID: test rows; submit only on an explicit go)
# 128k sweeps need an 80 GB A100 (dgx) for every model: Qwen3.5-9B peaks at ~50 GB on a 174k-token row
# and runs out of memory on an A40. 32k sweeps fit an A40.
# Resumable: run_check skips cells that already hold metrics.json, so a resubmit only runs what is left
# (e.g. the fastkvzip cells once a model's gates exist).
set -euo pipefail

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate "${ENV:-kvpress-final}"
export PYTHONPATH="$HOME/src/kvpress-e5"
export CONDA_OVERRIDE_CUDA=12.9 HF_HOME="$HOME/hf" TMPDIR="$HOME/tmp" \
       TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
       KVPRESS_FASTKVZIP_GATES="${GATES:-$HOME/phase3/experiments/final_eval/fastkvzip_gates/gates}"
# RULER and SCBench scorers import nltk -> sqlite3, which needs a CXXABI the node's libstdc++ lacks
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd "$HOME/phase3/experiments/final_eval/grid"
for SW in $SWEEPS; do
  stdbuf -oL -eL python -u "$HOME/phase3/harness_check/run_check.py" --sweep "$PWD/$SW" 2>&1 \
    | grep --line-buffered -v "Loading weights"
done
