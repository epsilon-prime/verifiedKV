#!/bin/bash
# two copies of one sweep on the same GPU, separate results roots; waits for both
source "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}/env.sh"
cd "$FINAL_GRID"
nvidia-smi -L
VKV_GRID_RESULTS=$PWD/results_pair/A python -u run_grid.py --sweep tests/pair_qwen3-4b.yaml > logs/pair_A.log 2>&1 &
a=$!
sleep 2
VKV_GRID_RESULTS=$PWD/results_pair/B python -u run_grid.py --sweep tests/pair_qwen3-4b.yaml > logs/pair_B.log 2>&1 &
b=$!
wait $a; wait $b
echo "both done"
