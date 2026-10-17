#!/bin/bash
#SBATCH --job-name=p3gate-eval
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=/home/rethinkingai-self/23b1835/phase3/experiments/final_eval/fastkvzip_gates/logs/%x-%j.out
# usage: NAME=qwen3-4b-instruct-2507 GATES="gates_validation/... released.pt" sbatch -p l40 --qos=l40 job_eval.sbatch
set -euo pipefail
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate kvpress-final
cd "$HOME/phase3/experiments/final_eval/fastkvzip_gates"
export HF_HOME="$HOME/hf" HF_HUB_OFFLINE=1
python -u eval_gate.py "$NAME" $GATES
