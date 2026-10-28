#!/bin/bash
# Download what the grid reads, on a machine WITH internet (on Prajna: the login node). Run once.
#   ./download.sh ministral3-8b            # one model (qwen3-4b | qwen3.5-9b | ministral3-8b), or several
# - the model weights into $HF_HOME (Ministral: the BF16 repo without consolidated.safetensors);
# - every dataset the grid uses (RULER-32k/128k: 4 tasks; LOFT 32k/128k: 5 tasks), loaded once through kvpress's
#   own loaders so they resolve offline later (HF_DATASETS_OFFLINE=1 in env.sh);
# - the Fast KVzip gates that are not in git (see gates/README.md) and their checksums.
set -euo pipefail
export HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"
export HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0
cd "$FINAL_GRID"
declare -A REPO=([qwen3-4b]=Qwen/Qwen3-4B-Instruct-2507 [qwen3.5-9b]=Qwen/Qwen3.5-9B
                 [ministral3-8b]=mistralai/Ministral-3-8B-Instruct-2512-BF16)
for m in "$@"; do
  r=${REPO[$m]:?unknown model $m}
  echo "== model $r"
  hf download "$r" --exclude "consolidated.safetensors" >/dev/null
done
echo "== datasets"
python - <<'PY'
import sys
sys.path.insert(0, "../../kvpress/evaluation")
from benchmarks.dataset_loaders import load_ruler_length_dataset, load_loft_dataset
for ds in ("ruler32k", "ruler128k"):
    for t in ("niah_multikey_3", "qa_1", "niah_multivalue", "cwe"):
        print(ds, t, len(load_ruler_length_dataset(ds, t)))
for L in ("32k", "128k"):
    for t in ("nq", "hotpotqa", "musique", "qampari", "quest"):
        print("loft", t, L, len(load_loft_dataset(f"{t}_{L}")))
PY
echo "== gates"
./gates/fetch_gates.sh
