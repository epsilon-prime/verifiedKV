# RULER 16k

RULER at 16,384 tokens. Hugging Face: [xAlg-AI/att-hub-ruler-16k](https://huggingface.co/datasets/xAlg-AI/att-hub-ruler-16k).

## Usage

```bash
python evaluate.py --dataset ruler16k --data_dir niah_single_1 \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct --press_name knorm --compression_ratio 0.5
```

`--data_dir` is the RULER task (`niah_single_1`, `vt`, `qa_1`, …). Metrics reuse `benchmarks/ruler/calculate_metrics.py`.
