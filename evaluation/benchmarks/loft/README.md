# LOFT RAG

[LOFT](https://github.com/google-deepmind/loft) long-context retrieval-augmented generation:
single-value tasks (`nq`, `hotpotqa`, `musique`) and multi-value tasks (`qampari`, `quest`),
each at `32k` / `128k` / `1m` context length.

Datasets are loaded from `f20180301/loft-rag-{dataset}-{length}` (dev + test splits).

## Usage

```bash
python evaluate.py --dataset loft --data_dir nq_32k --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --press_name knorm --compression_ratio 0.5
```

`--data_dir` is `{dataset}_{length}`, for example `hotpotqa_128k` or `quest_1m`.

## Metrics

- Single-value: EM, Subspan EM, F1
- Multi-value: EM, Coverage, Subspan EM
