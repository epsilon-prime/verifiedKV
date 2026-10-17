#!/bin/bash
#SBATCH --job-name=gate-train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=700G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
# Train the gates for one model (needs vendor/, see fetch_vendor.sh). Submit from final_eval/fastkvzip_gates/:
#     MODEL=<hf id> NGROUP=4 sbatch -p <partition> [--mem=...] job_train.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
[ -f train_gates.py ] || { echo "submit from final_eval/fastkvzip_gates/" >&2; exit 1; }
source "${CONDA_SH:-$HOME/miniforge3/etc/profile.d/conda.sh}"
conda activate "${ENV:-kvpress-final}"
[ -f vendor/optim.py ] || { echo "vendor/ missing: run bash fetch_vendor.sh first" >&2; exit 1; }
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}" HF_HUB_OFFLINE=1 MPLBACKEND=Agg
nvidia-smi --query-gpu=name --format=csv,noheader
stdbuf -oL -eL python -u train_gates.py --model "$MODEL" --ngroup "$NGROUP" ${OUT:+--out "$OUT"} 2>&1 | grep --line-buffered -v "Reshuffled"
