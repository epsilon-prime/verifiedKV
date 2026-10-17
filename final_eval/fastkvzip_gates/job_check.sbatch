#!/bin/bash
#SBATCH --job-name=p3gate-check
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/home/rethinkingai-self/23b1835/phase3/experiments/final_eval/fastkvzip_gates/logs/%x-%j.out
# usage: NAME=qwen3.5-9b sbatch -p l40 --qos=l40 job_check.sbatch   (CPU work; GPU partitions only)
set -euo pipefail
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate kvpress-final
cd "$HOME/phase3/experiments/final_eval/fastkvzip_gates"
CUDA_VISIBLE_DEVICES="" python -u check_data.py "$NAME"
