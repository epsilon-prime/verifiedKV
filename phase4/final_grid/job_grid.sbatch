#!/bin/bash
#SBATCH --job-name=vkv-final
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/slurm-%x-%j.out
# Prajna. Submit FROM phase4/final_grid:
#   SWEEPS="sweeps/a.yaml sweeps/b.yaml" sbatch -p a40 --qos=a40 job_grid.sbatch
# Two sweep lists on the SAME GPU at once: possible, but measured 6% SLOWER than one after the other (README):
#   SWEEPS="sweeps/a.yaml" SWEEPS2="sweeps/b.yaml" sbatch -p a40 --qos=a40 job_grid.sbatch
# 128k needs an 80 GB A100: -p dgx --qos=dgx. l40 allows at most 2 days.
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
if [ -n "${SWEEPS2:-}" ]; then
  ./run_sweeps.sh $SWEEPS & p1=$!
  ./run_sweeps.sh $SWEEPS2 & p2=$!
  wait $p1; r1=$?; wait $p2; r2=$?
  exit $(( r1 || r2 ))
fi
./run_sweeps.sh $SWEEPS
