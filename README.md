# vKV: final evaluation

KV cache compression with Bernoulli sampling and needle fetch, evaluated against KVzip and Fast KVzip on
RULER and LOFT (32k and 128k context, 8x / 16x / 32x compression; Qwen3-4B-Instruct-2507,
Ministral-3-8B-Instruct-2512, Qwen3.5-9B).

| folder | contents |
|---|---|
| [`final_eval/`](final_eval/) | runner, the 78 sweeps of the final grid, launchers, LOFT scorer, gate training: **start with its README** |
| [`kvpress/`](kvpress/) | our fork of [NVIDIA/kvpress](https://github.com/NVIDIA/kvpress) (via skylight-org/kvpress) with the presses and model support the evaluation uses |
| [`environment/`](environment/) | the exact Python environment |

Tag `final-eval-as-run` marks the code exactly as it ran for the paper.

kvpress is Apache-2.0 (`kvpress/LICENSE`). The Fast KVzip authors' training code is not included (it has no
license); `final_eval/fastkvzip_gates/fetch_vendor.sh` downloads it at the pinned commit.
