# RULER 32k

RULER at 32,768 tokens. Hugging Face: [xAlg-AI/att-hub-ruler-32k](https://huggingface.co/datasets/xAlg-AI/att-hub-ruler-32k).

## Usage

```bash
python evaluate.py --dataset ruler32k --data_dir niah_single_1 \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct --press_name knorm --compression_ratio 0.5
```

`--data_dir` is the RULER task. Metrics reuse `benchmarks/ruler/calculate_metrics.py`.
