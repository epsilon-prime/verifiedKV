#!/bin/bash
# Fast KVzip gates, one file per model: <model>/q4_dim16_sink16.pt (the file name kvpress's get_gate_id expects).
# - qwen3.5-9b: in git (21 MB). Trained by us (phase 3, final_eval/fastkvzip_gates).
# - ministral-3-8b-instruct-2512-bf16: trained by us; 179 MB, over GitHub's 100 MB file limit, so it is an asset
#   of the GitHub release named below.
# - qwen3-4b-instruct-2507: the authors' released gate (Hugging Face Jang-Hyun/Fast-KVzip).
# Every file is checked against SHA256SUMS.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
RELEASE_URL="https://github.com/skylight-org/vkv/releases/download/final-grid-gates-v1"
f=ministral-3-8b-instruct-2512-bf16/q4_dim16_sink16.pt
[ -f "$f" ] || { mkdir -p "$(dirname "$f")"; curl -fL -o "$f" "$RELEASE_URL/ministral-3-8b-instruct-2512-bf16_q4_dim16_sink16.pt"; }
f=qwen3-4b-instruct-2507/q4_dim16_sink16.pt
[ -f "$f" ] || { mkdir -p "$(dirname "$f")"; cp -L "$(hf download Jang-Hyun/Fast-KVzip "$f")" "$f"; }
sha256sum -c SHA256SUMS
