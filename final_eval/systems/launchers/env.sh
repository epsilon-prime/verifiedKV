# Sourced by every launcher here. Paths come from the environment, with defaults:
#   VKV_ROOT    this vkv checkout (found from this file's location)
#   KV_OFFLOAD  the KV_offload checkout: the vLLM implementation (ruler_vllm.py, ruler_batch.py,
#               offload_package); default: a KV_offload directory next to VKV_ROOT
#   PYTHON      its interpreter; default: $KV_OFFLOAD/.venv/bin/python
#   LOG_DIR     launcher logs and <run>.done files; default: $KV_OFFLOAD/logs/systems
#   MODEL       default: Qwen/Qwen3-4B-Instruct-2507
# Results go to $KV_OFFLOAD/out/<run>/ (the runners' own layout).
VKV_ROOT="${VKV_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
KV_OFFLOAD="${KV_OFFLOAD:-$(dirname "$VKV_ROOT")/KV_offload}"
PYTHON="${PYTHON:-$KV_OFFLOAD/.venv/bin/python}"
LOG_DIR="${LOG_DIR:-$KV_OFFLOAD/logs/systems}"
MODEL="${MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
export VKV_ROOT
for p in "$KV_OFFLOAD/ruler_vllm.py" "$PYTHON" "$VKV_ROOT/kvpress"; do
  [ -e "$p" ] || { echo "missing $p: set KV_OFFLOAD / PYTHON / VKV_ROOT (see launchers/env.sh)" >&2; exit 1; }
done
command -v numactl >/dev/null || { echo "numactl not found (runs bind CPU and host memory to the GPU's NUMA node)" >&2; exit 1; }
mkdir -p "$LOG_DIR"
cd "$KV_OFFLOAD"
