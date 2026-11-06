# RULER 128k

RULER at 131,072 tokens from the raw
[SaylorTwift/RULER-131072-llama-3.2-tokenizer](https://huggingface.co/datasets/SaylorTwift/RULER-131072-llama-3.2-tokenizer)
dataset. Each sample is a single `input` string; `prepare_dataframe` in `benchmarks/ruler/`
splits it into `context` / `question` / `answer_prefix` at load time.

## Usage

```bash
python evaluate.py --dataset ruler128k --data_dir niah_single_1 \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct --press_name knorm --compression_ratio 0.5 \
    --max_context_length 131072
```

`--data_dir` is the RULER task (HF split name). Metrics reuse `benchmarks/ruler/calculate_metrics.py`.
