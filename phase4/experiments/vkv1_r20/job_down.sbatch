#!/bin/bash
#SBATCH --job-name=p4-r20
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/home/rethinkingai-self/23b1835/phase4/experiments/vkv1_r20/logs/%x-%j.out
# copy of vkv1_nopersist/job_down.sbatch that runs several task groups in sequence (one runner call per group).
# usage: RUN=<name> GROUPS_="--tasks cwe --rows 152:162|--tasks vt --rows 152:172" sbatch -p a40 --qos=a40 job_down.sbatch
set -euo pipefail
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate kvpress-final
export PYTHONPATH="$HOME/src/kvpress-e5"
export CONDA_OVERRIDE_CUDA=12.9 HF_HOME="$HOME/hf" TMPDIR="$HOME/tmp" \
       TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd "$HOME/phase4/experiments/vkv1_r20"
ARMS="full bern:0.0625 needle_final_c:0.9375 vkv1:0.0625:0.3:0.05:chern:grad:nopersist vkv1:0.0625:0.5:0.05:chern:grad:nopersist"
IFS='|' read -ra GS <<< "$GROUPS_"
for g in "${GS[@]}"; do
  echo "=== group: $g"
  stdbuf -oL -eL python -u ../vkv1_chern/run_downstream.py $g --arms $ARMS --run "r20/$RUN" 2>&1 \
    | grep --line-buffered -v "Loading weights"
done
echo "=== all groups done"
