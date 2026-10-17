# AIME 2024

Problems from the American Invitational Mathematics Examination (AIME) 2024.

Hugging Face dataset: [xAlg-AI/att-hub-aime2024](https://huggingface.co/datasets/xAlg-AI/att-hub-aime2024)
(built from [Maxwell-Jia/AIME_2024](https://huggingface.co/datasets/Maxwell-Jia/AIME_2024)).

## Usage

```bash
python evaluate.py --dataset aime2024 --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --press_name decoding_knorm --compression_ratio 0.5
```

No `--data_dir` is needed. Scoring extracts the final integer from `\boxed{...}` (with a short fallback).
