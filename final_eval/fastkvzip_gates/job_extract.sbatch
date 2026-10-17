#!/bin/bash
#SBATCH --job-name=p3gate-extract
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=/home/rethinkingai-self/23b1835/phase3/experiments/final_eval/fastkvzip_gates/logs/%x-%j.out
# usage: MODEL=<hf id> N_SINGLE=29 N_CAT=5 sbatch -p dgx --qos=dgx job_extract.sbatch
set -euo pipefail
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate kvpress-final
export PYTHONPATH="$HOME/src/kvpress-e5"
export CONDA_OVERRIDE_CUDA=12.9 HF_HOME="$HOME/hf" TMPDIR="$HOME/tmp" TOKENIZERS_PARALLELISM=false \
       HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd "$HOME/phase3/experiments/final_eval/fastkvzip_gates"
stdbuf -oL -eL python -u extract_features.py --model "$MODEL" --n_single "${N_SINGLE:-29}" --n_cat "${N_CAT:-5}" 2>&1 \
  | grep --line-buffered -v "Loading weights"
