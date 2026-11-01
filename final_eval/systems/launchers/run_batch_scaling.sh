#!/bin/bash
# Batch-size scaling (ruler_batch.py): one arm on one GPU, B = 1 2 4 8 in turn; every request decodes the
# forced question + 128 answer tokens (EOS ignored). Results: $KV_OFFLOAD/out/<run>/<tag>_b<B>/summary.json.
# usage: run_batch_scaling.sh <run> <gpu> <numa node> <arm: vkv|recall|evict> [eps]     (vkv: delta 0.05)
#   CONTEXT  32k (default: rows 0:4 of 4 tasks = 16 requests, max_model_len 46000) or
#            128k (rows 0:2 = 8 requests, prompts 133k-193k, max_model_len 200000)
# At 128k each B >= 4 job pins ~120-230 GB of host memory, so those take a per-NUMA-node lock
# ($LOG_DIR/<run>_node<N>.lock): one at a time per node; two such pools on one node can trigger the OOM killer.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
RUN=$1; G=$2; N=$3; A=$4; E=$5
TASKS="cwe niah_multikey_3 niah_multivalue qa_1"
if [ "${CONTEXT:-32k}" = 128k ]; then
  DS=SaylorTwift/RULER-131072-llama-3.2-tokenizer; ROWS=0:2; MAXLEN=200000; PER_REQ=360
else
  DS=xAlg-AI/att-hub-ruler-32k; ROWS=0:4; MAXLEN=46000; PER_REQ=80
fi
case $A in
  vkv) C=vkv-fkvz; TAG=vkv-fkvz_e${E:?vkv needs eps}_d0.05
       X="VKV_GRAPHS=1 BENCH_VKV_COMPACT=1 VKV_CUDA=1 BENCH_VKV_EPS0=$E BENCH_VKV_DELTA=0.05";;
  recall) C=offload-recall; TAG=offload-recall-compact; X="BENCH_PIECEWISE=1 VKV_GRAPHS=1 BENCH_VKV_COMPACT=1";;
  evict) C=fkvz-evict; TAG=fkvz-evict; X="BENCH_EVICT_FULL=1 BENCH_VKV_COMPACT=1";;
  *) echo "unknown arm $A" >&2; exit 1;;
esac
for BS in 1 2 4 8; do
  SS=$(( BS * PER_REQ > 256 ? BS * PER_REQ : 256 ))   # vkv's staging ring must cover B x context / 576
  cmd="numactl --cpunodebind=$N --membind=$N env CUDA_VISIBLE_DEVICES=$G BENCH_MODEL=$MODEL BENCH_DTYPE=bfloat16 \
    BENCH_GPU_UTIL=0.6 BENCH_MAX_MODEL_LEN=$MAXLEN BENCH_STAGING_SLOTS=$SS $X \
    '$PYTHON' ruler_batch.py --arm $C --tag $TAG --batch $BS --rows $ROWS --dataset $DS --tasks $TASKS \
    --new 128 --run $RUN"
  log="$LOG_DIR/${RUN}_${TAG}_b${BS}.log"
  if [ "${CONTEXT:-32k}" = 128k ] && [ "$BS" -ge 4 ]; then
    flock "$LOG_DIR/${RUN}_node$N.lock" bash -c "$cmd" > "$log" 2>&1
  else
    bash -c "$cmd" > "$log" 2>&1
  fi
  echo "$TAG b$BS $?" >> "$LOG_DIR/${RUN}.done"
done
