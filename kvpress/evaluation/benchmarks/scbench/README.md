# SCBench dataset

[SCBench](https://arxiv.org/abs/2412.10319) ([project page](https://aka.ms/SCBench)) evaluates long-context
methods in the **shared context** setting: each sample is one long context, roughly 100k to 130k tokens, plus
several follow-up turns that all reuse it.

## Hugging Face dataset

This benchmark reads [microsoft/SCBench](https://huggingface.co/datasets/microsoft/SCBench) directly, so
there is no `create_huggingface_dataset.py` to run. `flatten_multi_turns` in `utils.py` expands the nested
`multi_turns` column into one row per turn at load time, all rows of a sample carrying the same context, so
the runner prefills and compresses each context once and answers every turn of it from that one compressed
cache.

Prompts and generation budgets are SCBench's own, from `multiturn_templates_scdq` and
`DATA_NAME_TO_MAX_NEW_TOKENS` in
[eval_utils.py](https://github.com/microsoft/MInference/blob/main/scbench/eval_utils.py).

## Usage

```bash
python evaluate.py --dataset scbench --data_dir scbench_vt --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --press_name knorm --compression_ratio 0.5 --max_context_length 130000
```

- `--data_dir` takes the task, with or without the `scbench_` prefix (`vt` and `scbench_vt` both work).
- `--fraction` selects whole contexts, not single turns. With 90 `vt` samples, `--fraction 0.1` gives 9
  contexts, that is 45 questions.
- `--max_context_length` **drops** the samples whose context does not fit rather than truncating them, since
  truncating a shared context removes the evidence the later turns ask about. It counts the context alone, so
  leave the turn and its generation room: use 130000 for Llama 3.1, whose limit is 131,072 positions. At that
  limit `qa_eng` keeps 22 of its 69 samples and the other four tasks keep all of theirs.

## Tasks and metrics

| task | family | answer | metric | `max_new_tokens` |
|---|---|---|---|---|
| `scbench_kv` | retrieval | one uuid | substring match | 150 |
| `scbench_prefix_suffix` | retrieval | one word | substring match | 150 |
| `scbench_vt` | multi-hop tracing | 5 variable names | fraction recovered (RULER's `string_match_all`) | 30 |
| `scbench_qa_eng` | question answering | a short span | SQuAD token F1, and substring match alongside | 40 |
| `scbench_summary` | summarization | a sentence | ROUGE-L F1 | 200 |

The `answer` column holds a list of gold strings, as it does for RULER: one entry for most tasks, and the
five variables to recover for `vt`.

To add one of the other SCBench tasks, give it an entry in `CONTEXT_COLUMN`, `CONTEXT_PREFIX` and
`MAX_NEW_TOKENS` in `utils.py`, and one in `SCORERS` in `calculate_metrics.py`. Note that `mf` keeps its
context as a list of integers rather than a string, so it needs a formatting step too.

## Running without internet access

Prefetch the task once from a machine that can reach the hub, which fills the usual `HF_HOME` cache, then run
with `HF_HUB_OFFLINE=1`:

```bash
python -c "from datasets import load_dataset; load_dataset('microsoft/SCBench', 'scbench_vt', split='test')"
```

The prefetch has to go through `load_dataset`: copying the `.jsonl` files by hand leaves out the repository
metadata that `datasets` needs.
