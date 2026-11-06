# Sourced by every launcher here. Override these before running, if needed:
#   PYTHON   interpreter of the serving environment  [python]
#   LOG_DIR  launcher logs and <run>.done files      [serving/logs]
#   MODEL    model name                              [Qwen/Qwen3-4B-Instruct-2507]
# Results go to serving/benchmarks/out/<run>/.
VKV_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SERVING="$VKV_ROOT/serving"
PYTHON="${PYTHON:-python}"
LOG_DIR="${LOG_DIR:-$SERVING/logs}"
MODEL="${MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
export VKV_ROOT
command -v numactl >/dev/null || { echo "numactl not found (runs bind CPU and host memory to the NUMA node of the GPU)" >&2; exit 1; }
mkdir -p "$LOG_DIR"
cd "$SERVING/benchmarks"
