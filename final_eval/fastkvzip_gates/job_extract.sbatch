#!/bin/bash
#SBATCH --job-name=gate-extract
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
# Training features for one model (80 GB GPU). Submit from final_eval/fastkvzip_gates/:
#     MODEL=<hf id> N_SINGLE=29 N_CAT=5 sbatch -p <partition> job_extract.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
[ -f train_gates.py ] || { echo "submit from final_eval/fastkvzip_gates/" >&2; exit 1; }
source "${CONDA_SH:-$HOME/miniforge3/etc/profile.d/conda.sh}"
conda activate "${ENV:-kvpress-final}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}" TOKENIZERS_PARALLELISM=false \
       HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
stdbuf -oL -eL python -u extract_features.py --model "$MODEL" --n_single "${N_SINGLE:-29}" --n_cat "${N_CAT:-5}" 2>&1 \
  | grep --line-buffered -v "Loading weights"
