#!/bin/bash
#SBATCH --job-name=gate-check
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
# Scan extracted features for values the trainer cannot take (CPU work). Submit from final_eval/fastkvzip_gates/:
#     NAME=<model dir name> sbatch -p <partition> job_check.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
[ -f train_gates.py ] || { echo "submit from final_eval/fastkvzip_gates/" >&2; exit 1; }
source "${CONDA_SH:-$HOME/miniforge3/etc/profile.d/conda.sh}"
conda activate "${ENV:-kvpress-final}"
CUDA_VISIBLE_DEVICES="" python -u check_data.py "$NAME"
