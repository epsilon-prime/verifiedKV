#!/bin/bash
# Fast KVzip gates, one file per model: <model>/q4_dim16_sink16.pt (the file name kvpress's get_gate_id expects).
# - qwen3.5-9b: in git (21 MB). Trained by us (phase 3, final_eval/fastkvzip_gates).
# - ministral-3-8b-instruct-2512-bf16: trained by us; 179 MB, over GitHub's 100 MB file limit, so it is an asset
#   of the GitHub release named below.
# - qwen3-4b-instruct-2507: the authors' released gate (Hugging Face Jang-Hyun/Fast-KVzip).
# Every file is checked against SHA256SUMS.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
# The repository is private: the release asset needs `gh` (logged in) or a token with read access in GITHUB_TOKEN.
REPO=skylight-org/vkv TAG=final-grid-gates-v1 ASSET=ministral-3-8b-instruct-2512-bf16_q4_dim16_sink16.pt
f=ministral-3-8b-instruct-2512-bf16/q4_dim16_sink16.pt
if [ ! -f "$f" ]; then
  mkdir -p "$(dirname "$f")"
  if command -v gh >/dev/null && gh auth status >/dev/null 2>&1; then
    gh release download "$TAG" -R "$REPO" -p "$ASSET" -O "$f"
  else
    : "${GITHUB_TOKEN:?install and log in to gh, or export GITHUB_TOKEN (read access to $REPO)}"
    auth=(-H "Authorization: Bearer $GITHUB_TOKEN")
    id=$(curl -fsS "${auth[@]}" "https://api.github.com/repos/$REPO/releases/tags/$TAG" \
         | python3 -c "import json,sys; print(next(a['id'] for a in json.load(sys.stdin)['assets'] if a['name']=='$ASSET'))")
    curl -fL "${auth[@]}" -H "Accept: application/octet-stream" -o "$f" "https://api.github.com/repos/$REPO/releases/assets/$id"
  fi
fi
f=qwen3-4b-instruct-2507/q4_dim16_sink16.pt
[ -f "$f" ] || { mkdir -p "$(dirname "$f")"; cp -L "$(hf download Jang-Hyun/Fast-KVzip "$f")" "$f"; }
sha256sum -c SHA256SUMS
