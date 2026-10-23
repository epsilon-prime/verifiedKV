#!/bin/bash
#SBATCH --job-name=p4-sim
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/home/rethinkingai-self/23b1835/phase4/experiments/vkv1_sim/logs/%x-%j.out
# usage: ARGS="<xdir> --out out/<run> [...]" sbatch -p l40 --qos=l40 job_sim.sbatch      (a40 / l40 only)
set -euo pipefail
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate kvpress-final
export PYTHONPATH="$HOME/src/kvpress-e5"
export CONDA_OVERRIDE_CUDA=12.9 TMPDIR="$HOME/tmp" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd "$HOME/phase4/experiments/vkv1_sim"
stdbuf -oL -eL python -u sim.py $ARGS
python summarize.py "$(echo $ARGS | sed -E 's/.*--out ([^ ]+).*/\1/')"
