#!/bin/bash
# Run sweeps one after another on the GPU(s) in CUDA_VISIBLE_DEVICES; one log per sweep in logs/.
#   ./run_sweeps.sh sweeps/qwen3-4b_32k_r16_ruler_cwe.yaml [more.yaml ...]
# Resumable: finished cells are skipped, so rerunning the same command continues an interrupted sweep.
set -uo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"
cd "$FINAL_GRID"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader -i "${CUDA_VISIBLE_DEVICES:-0}" 2>/dev/null || true
status=0
for SW in "$@"; do
  name=$(basename "$SW" .yaml)
  echo "=== $(date '+%F %T') $name on $(hostname) GPU ${CUDA_VISIBLE_DEVICES:-all}"
  stdbuf -oL -eL python -u run_grid.py --sweep "$SW" 2>&1 | grep --line-buffered -v "Loading weights" \
    | tee -a "logs/$name.log"
  rc=${PIPESTATUS[0]}
  [ "$rc" -ne 0 ] && { echo "=== $name exited with $rc"; status=1; }
done
exit $status
