#!/bin/bash
# LOFT RAG at 128k (ruler_vllm.py --bench loft, forced question), arms in turn on one GPU; per row: ttft, tpot,
# latency, bytes fetched. Score with $KV_OFFLOAD/tools/score_loft_vllm.py out/<run> (LOFT's official parser).
# usage: run_loft.sh <run> <rows a:b> <gpu> <numa node> <arm...>
#   arms: dense kvzip-evict vkv vkv-compact offload-recall offload-recall-compact
#   LOFT_TASKS  default: all 5 (nq hotpotqa musique qampari quest) at 128k
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
RUN=$1; ROWS=$2; G=$3; N=$4; shift 4
for A in "$@"; do
  case $A in
    dense) C=dense; X="BENCH_CUDAGRAPH=1";;
    kvzip-evict) C=kvzip-evict; X="BENCH_EVICT_FULL=1 BENCH_VKV_COMPACT=0";;
    vkv) C=vkv; X="BENCH_PIECEWISE=1 VKV_GRAPHS=1 BENCH_VKV_COMPACT=0 VKV_CUDA=1";;
    vkv-compact) C=vkv; X="BENCH_PIECEWISE=1 VKV_GRAPHS=1 BENCH_VKV_COMPACT=1 VKV_CUDA=1";;
    offload-recall) C=offload-recall; X="BENCH_PIECEWISE=1 VKV_GRAPHS=1 BENCH_VKV_COMPACT=0";;
    offload-recall-compact) C=offload-recall; X="BENCH_PIECEWISE=1 VKV_GRAPHS=1 BENCH_VKV_COMPACT=1";;
    *) echo "unknown arm $A" >&2; exit 1;;
  esac
  numactl --cpunodebind=$N --membind=$N env CUDA_VISIBLE_DEVICES=$G BENCH_MODEL=$MODEL \
    BENCH_DTYPE=bfloat16 BENCH_GPU_UTIL=0.5 BENCH_MAX_MODEL_LEN=124000 $X \
    "$PYTHON" ruler_vllm.py --bench loft ${LOFT_TASKS:+--tasks $LOFT_TASKS} --arm $C --tag $A \
    --rows $ROWS --run $RUN > "$LOG_DIR/loft_${RUN}_${A}.log" 2>&1
  echo "$A $?" >> "$LOG_DIR/loft_${RUN}.done"
done
