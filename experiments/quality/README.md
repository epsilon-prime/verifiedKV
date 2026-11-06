# Quality experiments

## Setup

```bash
cd experiments/quality
export CONDA_ENV=verifiedkv HF_HOME=/path/to/hf
./download.sh qwen3-4b qwen3.5-9b ministral3-8b    # models, datasets and gates; needs internet access
source env.sh
```

`env.sh` sets the environment of every run. It runs offline after the download.

## Grid

| Item | Values |
|---|---|
| Models | Qwen3-4B-Instruct-2507, Qwen3.5-9B, Ministral-3-8B-Instruct-2512 (BF16) |
| RULER-32K, RULER-128K | `niah_multikey_3`, `qa_1`, `niah_multivalue`, `cwe`; 50 test rows per task |
| LOFT-32K, LOFT-128K | `nq`, `hotpotqa`, `musique`, `qampari`, `quest`; first 25 test questions per task in the paper |
| Compression | 8x, 16x, 32x |
| Settings | eps0 0.05, delta 0.1, 128-token shortlist per KV head, 64 sink and 64 local tokens |

| Arm | Paper name |
|---|---|
| `vkv_final_kvzip` | VerifiedKV(C=KVzip, A=2-bit sketch) |
| `vkv_final_fastkvzip` | VerifiedKV(C=Fast KVzip, A=2-bit sketch) |
| `vkv_final_perfect` | VerifiedKV(C=KVzip, A=perfect sketch) |
| `no_press` | Dense |
| `kvzip_c` | KVzip |
| `fastkvzip` | Fast KVzip |
| `bernoulli_kvzip_c` | PoissonComp(C=KVzip) |

## Running

```bash
CUDA_VISIBLE_DEVICES=0 ./run_sweeps.sh sweeps_smoke/qwen3-4b_32k_ruler.yaml   # smoke test on development rows
CUDA_VISIBLE_DEVICES=0 ./run_sweeps.sh sweeps/qwen3-4b_32k_r16_ruler_cwe.yaml # one VerifiedKV sweep
./launch_local.sh "0 1 2 3" sweeps/qwen3-4b_32k_*.yaml sweeps_baselines/qwen3-4b_32k_*.yaml  # several GPUs
python score_grid.py                                                          # writes results.csv
```

Finished cells are skipped, so a run can be resumed with the same command. Sweep files come from `make_sweeps.py`.

## Notes

- KVzip scores are computed on first use and cached in `$KVPRESS_SCORE_CACHE`. All arms and ratios share them.
- Fast KVzip gates are placed in `gates/` by `download.sh` and checked against `gates/SHA256SUMS`.
- LOFT is scored with the official parser in `loft_official/`. `score_grid.py` scores the first 25 test questions of each
  task, as in the paper. Use `--loft-first 0` to score all of them.
