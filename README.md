# VerifiedKV

Code for the paper "Fetch What You Need, When You Need It: Verified KV Cache Compression for LLM Inference".

VerifiedKV keeps a Poisson sample of the KV cache on the GPU and the full cache in host memory. At each decoding step, a
certificate bounds the attention error of the sample. Tokens are fetched from host memory only while the bound exceeds
a user-specified tolerance.

## Repository layout

| Path | Contents |
|---|---|
| `verifiedkv/` | The method: sampling, certificate, fetch selection and press |
| `kvpress/`, `evaluation/` | Fork of NVIDIA kvpress and its evaluation harness |
| `experiments/` | Scripts for the quality, adherence, shrinkage and host traffic experiments |
| `serving/` | vLLM prototype used for the latency and throughput experiments |
| `results/` | Result tables reported in the paper |
| `figures/` | Scripts and data for the figures and tables in the paper |
| `tests/` | Unit tests |
| `examples/` | A short usage example |

## Installation

Python 3.11 and a CUDA GPU are required.

```bash
git clone https://github.com/skylight-org/vkv.git && cd vkv
conda create -n verifiedkv python=3.11 -y && conda activate verifiedkv
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install -e . --no-deps
```

## Usage

```python
from transformers import pipeline
from verifiedkv import make_press

pipe = pipeline("kv-press-text-generation", model="Qwen/Qwen3-4B-Instruct-2507", device_map="auto",
                dtype="auto", model_kwargs={"attn_implementation": "sdpa"})
press = make_press(compression_ratio=0.9375, scorer="kvzip", sketch="2-bit", eps0=0.05, delta=0.1)
answer = pipe(context, question=question, press=press)["answer"]
```

| Argument | Meaning | Default |
|---|---|---|
| `compression_ratio` | Fraction of the cache removed from the GPU. 0.9375 is 16x. | 0.9375 |
| `scorer` | `kvzip` or `fastkvzip` | `kvzip` |
| `sketch` | `2-bit`, or `perfect`, which plans on the exact attention weights | `2-bit` |
| `eps0`, `delta` | Relative error tolerance and failure probability | 0.05, 0.1 |

The press supports batch size 1. The `fastkvzip` scorer needs the gate files described in `experiments/quality/README.md`.

## Tests

```bash
python -m pytest -q tests
```

## Reproducing the paper

- Quality results: `experiments/quality/README.md`
- Other experiments: `experiments/README.md`
- Latency and throughput: `serving/README.md`
- Figures and tables: `figures/README.md`
- Reported values: `results/README.md`

## License

Apache License 2.0. See `LICENSE` and `NOTICE`.
