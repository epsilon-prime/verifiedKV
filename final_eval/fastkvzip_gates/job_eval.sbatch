#!/bin/bash
#SBATCH --job-name=gate-eval
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
# Score gate files on held-out tokens (needs vendor/). Submit from final_eval/fastkvzip_gates/:
#     NAME=<model dir name> GATES="gates/<name>/q4_dim16_sink16.pt [other.pt ...]" sbatch -p <partition> job_eval.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
[ -f train_gates.py ] || { echo "submit from final_eval/fastkvzip_gates/" >&2; exit 1; }
source "${CONDA_SH:-$HOME/miniforge3/etc/profile.d/conda.sh}"
conda activate "${ENV:-kvpress-final}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}" HF_HUB_OFFLINE=1
python -u eval_gate.py "$NAME" $GATES
