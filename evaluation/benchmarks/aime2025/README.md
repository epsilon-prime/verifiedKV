# AIME 2025

Problems from the American Invitational Mathematics Examination (AIME) 2025.

Hugging Face dataset: [xAlg-AI/att-hub-aime2025](https://huggingface.co/datasets/xAlg-AI/att-hub-aime2025)
(built from [yentinglin/aime_2025](https://huggingface.co/datasets/yentinglin/aime_2025)).

The older `aime25` entry (`alessiodevoto/aime25`) remains registered for compatibility.

## Usage

```bash
python evaluate.py --dataset aime2025 --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --press_name decoding_knorm --compression_ratio 0.5
```

No `--data_dir` is needed. Gold answers may be a list; scoring extracts an integer in `0..999`.
