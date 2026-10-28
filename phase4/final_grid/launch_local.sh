#!/bin/bash
# Without SLURM: run a list of sweeps over several GPUs, each GPU taking the next unclaimed sweep (a shared queue).
#   ./launch_local.sh "0 1 2 3" sweeps/ministral3-8b_32k_*.yaml
#   (two workers per GPU, "0 0 1 1", is possible but measured slower: README "One job per GPU")
# Run it inside tmux/screen. Claims are folders in logs/claims/ (atomic mkdir), so several launch_local.sh (or
# machines sharing this folder) never run the same sweep twice. To rerun a sweep (e.g. after a crash), remove its
# claim: rm -r logs/claims/<sweep name>. Finished cells are skipped anyway.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPUS=($1); shift
mkdir -p logs/claims
worker() {
  local gpu=$1; shift
  for SW in "$@"; do
    name=$(basename "$SW" .yaml)
    mkdir "logs/claims/$name" 2>/dev/null || continue
    echo "$(hostname) gpu $gpu $(date '+%F %T')" > "logs/claims/$name/who"
    CUDA_VISIBLE_DEVICES=$gpu ./run_sweeps.sh "$SW" > /dev/null 2>&1 \
      && echo "$(date '+%F %T') done $name (gpu $gpu)" || echo "$(date '+%F %T') FAILED $name (gpu $gpu): logs/$name.log"
  done
}
for g in "${GPUS[@]}"; do worker "$g" "$@" & sleep 5; done
wait
echo "all workers finished"
