# The environment of every run. Source it, and do not set these variables by hand.
#   source env.sh            (from any folder)
# Paths you may override BEFORE sourcing (defaults in brackets):
#   CONDA_SH       conda's profile script     [$HOME/miniforge3/etc/profile.d/conda.sh]
#   CONDA_ENV      the conda env               [kvpress-final]
#   VENV           a virtualenv folder to use instead of conda          [unset]
#   HF_HOME        Hugging Face cache (models, datasets)   [$HOME/hf]
#   KVPRESS_SCORE_CACHE   KVzip score cache, filled on first use and shared by all arms and ratios
#                                              [<this folder>/score_cache]
# Everything else is fixed here so that every person and job runs the same thing.
FINAL_GRID="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -n "${VENV:-}" ]; then             # a plain virtualenv instead of conda (e.g. a cloud pod)
  source "$VENV/bin/activate"
else
  source "${CONDA_SH:-$HOME/miniforge3/etc/profile.d/conda.sh}"
  conda activate "${CONDA_ENV:-kvpress-final}"
fi
export FINAL_GRID
export HF_HOME="${HF_HOME:-$HOME/hf}"
export KVPRESS_SCORE_CACHE="${KVPRESS_SCORE_CACHE:-$FINAL_GRID/score_cache}"
export KVPRESS_FASTKVZIP_GATES="$FINAL_GRID/gates"
mkdir -p "$KVPRESS_SCORE_CACHE" "$FINAL_GRID/logs"
export VKV_VEC=1 VKV_LAMBDA_BLOCK=8              # the vectorised certificate (identical k*, fewer kernel launches)
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"  # 0 only for download.sh
# RULER and LOFT scorers import nltk -> sqlite3, which needs the env's libstdc++
if [ -n "${CONDA_PREFIX:-}" ]; then export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"; fi
true  # sourcing must succeed (callers use set -e)
