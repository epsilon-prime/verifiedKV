#!/bin/bash
# RULER per-request sweep (ruler_vllm.py, forced question), one task on one GPU: vkv-fastkvzip over the
# (eps, delta) grid in one engine (0.05:0.1 first), then the arms in ARMS. 16x: kappa = 0.0625.
# usage: run_ruler_sweep.sh <run> <gpu> <numa node> <task> [rows a:b]
#   ARMS     vkv recall evict dense (default: vkv recall evict)
#   SWEEP    "EPS:DELTA ..." (default: eps 0.01 .. 0.5 at delta 0.05 and 0.1)
#   DATASET  default RULER 128k (SaylorTwift/RULER-131072-llama-3.2-tokenizer); 32k: xAlg-AI/att-hub-ruler-32k
#   MAXLEN   max_model_len (128k: 140000, 200000 for niah_multikey_3 / cwe; 32k: 46000)
#   EXTRA    more ruler_vllm.py flags, e.g. --sysmon (system metrics)
#   STAGING_SLOTS  vkv staging ring (needed when 8 x context > 256 x 4608 tokens, e.g. 400 at 193k)
# The model must keep each request's offloaded KV in pinned host memory: at 128k, run at most one
# B >= 4 job or two B = 1 jobs per NUMA node.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
RUN=$1; G=$2; N=$3; T=$4; ROWS=${5:-0:10}
if [ -z "$SWEEP" ]; then
  SWEEP="0.05:0.1"
  for D in 0.05 0.1; do
    for E in 0.01 0.03 0.05 0.07 0.1 0.15 0.25 0.3 0.35 0.4 0.5; do
      [ "$E:$D" = "0.05:0.1" ] || SWEEP="$SWEEP $E:$D"
    done
  done
fi
COMMON="CUDA_VISIBLE_DEVICES=$G BENCH_MODEL=$MODEL BENCH_DTYPE=bfloat16 BENCH_GPU_UTIL=0.6 \
  BENCH_MAX_MODEL_LEN=${MAXLEN:-140000} ${STAGING_SLOTS:+BENCH_STAGING_SLOTS=$STAGING_SLOTS}"
ARGS="--warmup --dataset ${DATASET:-SaylorTwift/RULER-131072-llama-3.2-tokenizer} --tasks $T --rows $ROWS --run $RUN $EXTRA"
run() {  # <tag> <env> <ruler_vllm args...>
  local tag=$1 x=$2; shift 2
  numactl --cpunodebind=$N --membind=$N env $COMMON $x "$PYTHON" ruler_vllm.py $ARGS "$@" \
    > "$LOG_DIR/${RUN}_${T}_${tag}.log" 2>&1
  echo "$T $tag $?" >> "$LOG_DIR/${RUN}.done"
}
for A in ${ARMS:-vkv recall evict}; do
  case $A in
    vkv) run vkv "VKV_GRAPHS=1 BENCH_VKV_COMPACT=1 VKV_CUDA=1" --arm vkv-fkvz --tag vkv-fkvz --sweep $SWEEP;;
    recall) run recall "BENCH_PIECEWISE=1 VKV_GRAPHS=1 BENCH_VKV_COMPACT=1" \
              --arm offload-recall --tag offload-recall-compact;;
    evict) run evict "BENCH_EVICT_FULL=1 BENCH_VKV_COMPACT=1" --arm fkvz-evict --tag fkvz-evict;;
    dense) run dense "BENCH_CUDAGRAPH=1" --arm dense --tag dense;;
    *) echo "unknown arm $A" >&2; exit 1;;
  esac
done
