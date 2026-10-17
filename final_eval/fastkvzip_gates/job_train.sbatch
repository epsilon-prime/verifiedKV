#!/bin/bash
#SBATCH --job-name=p3gate-train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=700G
#SBATCH --time=08:00:00
#SBATCH --output=/home/rethinkingai-self/23b1835/phase3/experiments/final_eval/fastkvzip_gates/logs/%x-%j.out
# usage: MODEL=<hf id> NGROUP=4 sbatch -p dgx --qos=dgx [--mem=...] job_train.sbatch
set -euo pipefail
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate kvpress-final
export CONDA_OVERRIDE_CUDA=12.9 HF_HOME="$HOME/hf" TMPDIR="$HOME/tmp" HF_HUB_OFFLINE=1 MPLBACKEND=Agg
nvidia-smi --query-gpu=name --format=csv,noheader
cd "$HOME/phase3/experiments/final_eval/fastkvzip_gates"
stdbuf -oL -eL python -u train_gates.py --model "$MODEL" --ngroup "$NGROUP" ${OUT:+--out "$OUT"} 2>&1 | grep --line-buffered -v "Reshuffled"
