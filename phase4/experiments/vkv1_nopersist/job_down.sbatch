#!/bin/bash
#SBATCH --job-name=p4-npdown
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/home/rethinkingai-self/23b1835/phase4/experiments/vkv1_nopersist/logs/%x-%j.out
# usage: ARGS="--rows 152:153 --tasks qa_1 --arms full vkv1:0.0625:0.3:0.05 --run smoke" \
#        sbatch -p a40 --qos=a40 job_down.sbatch      (32k only; a40 / l40, never dgx)
set -euo pipefail
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate kvpress-final
export PYTHONPATH="$HOME/src/kvpress-e5"
export CONDA_OVERRIDE_CUDA=12.9 HF_HOME="$HOME/hf" TMPDIR="$HOME/tmp" \
       TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd "$HOME/phase4/experiments/vkv1_nopersist"
stdbuf -oL -eL python -u ../vkv1_chern/run_downstream.py $ARGS 2>&1 | grep --line-buffered -v "Loading weights"
