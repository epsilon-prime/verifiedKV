#!/bin/bash
# Fast KVzip gates, one file per model: <model>/q4_dim16_sink16.pt.
# - qwen3.5-9b: in this repository. Trained by the authors of VerifiedKV.
# - ministral-3-8b-instruct-2512-bf16: trained by the authors of VerifiedKV, published as a release asset.
# - qwen3-4b-instruct-2507: the gate released by the Fast KVzip authors (Hugging Face Jang-Hyun/Fast-KVzip).
# Every file is checked against SHA256SUMS.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
GATE_URL="${MINISTRAL_GATE_URL:-https://github.com/skylight-org/vkv/releases/download/final-grid-gates-v1/ministral-3-8b-instruct-2512-bf16_q4_dim16_sink16.pt}"
f=ministral-3-8b-instruct-2512-bf16/q4_dim16_sink16.pt
if [ ! -f "$f" ]; then
  mkdir -p "$(dirname "$f")"
  curl -fL -o "$f" "$GATE_URL"
fi
f=qwen3-4b-instruct-2507/q4_dim16_sink16.pt
if [ ! -f "$f" ]; then
  mkdir -p "$(dirname "$f")"
  python -c "import shutil, sys; from huggingface_hub import hf_hub_download as d; shutil.copy(d('Jang-Hyun/Fast-KVzip', sys.argv[1]), sys.argv[1])" "$f"
fi
sha256sum -c SHA256SUMS
