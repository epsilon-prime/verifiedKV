#!/bin/bash
# Run final-eval sweeps on fox.cse.iitb.ac.in (CSE infolab: shared box, no SLURM). Same environment as
# grid/job_grid.sbatch; only the conda install (/opt/anaconda3, env on the NAS) and the GPU choice differ.
#
# usage (on fox, inside tmux):  bash run_fox.sh <gpu> <dir> sweeps/a.yaml [sweeps/b.yaml ...]
#   <gpu>  nvidia-smi index (0-1 RTX 6000 Ada, 2-4 RTX A6000)
#   <dir>  folder under final_eval/ holding the sweeps, e.g. grid or fox/smoke; results go to
#          <dir>/results/<group> (run_check's rule), exactly as on Prajna
# 32k only: 128k cells peak at 52-68 GB, fox's cards have 48 GB. 32k cells peak at <= 30.4 GB, so the
# script refuses a GPU with less than 36 GB free (other users share these cards).
# Resumable like job_grid.sbatch: run_check skips cells that already hold metrics.json.
set -euo pipefail
GPU=$1; DIR=$2; shift 2
case " $* " in *128k*) echo "refusing: 128k sweeps need an 80 GB A100 (Prajna dgx)"; exit 1 ;; esac
FREE=$(nvidia-smi -i "$GPU" --query-gpu=memory.free --format=csv,noheader,nounits)
[ "$FREE" -ge 36000 ] || { echo "refusing: GPU $GPU has only $FREE MiB free"; exit 1; }

source /opt/anaconda3/etc/profile.d/conda.sh
conda activate /mnt/nas/devanshujha/envs/kvpress-final
# fox mixes two GPU models; CUDA's default order is fastest-first, so pin it to nvidia-smi's numbering
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU"
# Prajna jobs get 16 CPUs from SLURM; fox has no limit, so cap the thread pools the same way
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
export PYTHONPATH="$HOME/src/kvpress-e5"
export CONDA_OVERRIDE_CUDA=12.9 HF_HOME="$HOME/hf" TMPDIR="$HOME/tmp" \
       TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
       KVPRESS_FASTKVZIP_GATES="${GATES:-$HOME/phase3/experiments/final_eval/fastkvzip_gates/gates}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

cd "$HOME/phase3/experiments/final_eval/$DIR"
mkdir -p logs
exec > >(tee -a "logs/fox-gpu$GPU-$(date +%m%d-%H%M).out") 2>&1
nvidia-smi -i "$GPU" --query-gpu=name,memory.total --format=csv,noheader
for SW in "$@"; do
  stdbuf -oL -eL python -u "$HOME/phase3/harness_check/run_check.py" --sweep "$PWD/$SW" 2>&1 \
    | grep --line-buffered -v "Loading weights"
done
