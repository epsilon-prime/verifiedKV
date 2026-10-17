# Final evaluation

Bernoulli sampling and needle fetch against KVzip and Fast KVzip, on RULER and LOFT at 32k and 128k context,
8x / 16x / 32x compression, three models. This folder holds the runner, the 78 sweeps of the final grid, the
launchers, the official LOFT scorer and the Fast KVzip gate training. The press code is in `../kvpress/`.

## Layout

```
final_eval/
  run_sweep.py          runs one sweep (one model load, every cell x arm in it); resumable
  make_grid.py          writes sweeps/ (the grid definition: models, tasks, ratios, arms)
  sweeps/               the 78 sweeps of the final grid, exactly as run
  download.py           fetches models, gates and datasets into HF_HOME and checks their revisions
  launchers/
    job_grid.sbatch     SLURM job: SWEEPS="..." sbatch launchers/job_grid.sbatch
    run_local.sh        the same without SLURM (shared GPU box, run inside tmux)
    drain.sh            submission queue for clusters with per-user job caps
    grid_jobs.txt       the 39 jobs of the final grid: sweeps per job, GPU class, time limit
  scoring/score_loft.py LOFT scores with LOFT's official answer parser (use it for every LOFT number)
  fastkvzip_gates/      feature extraction and training of Fast KVzip gates for Ministral 3 and Qwen3.5
  docs/                 SETUP.md (the setup log and every decision) and FINAL_EVAL_AUDIT.md
```

## Setup

1. **Environment.** Python 3.11, torch 2.11 (CUDA 12.8), transformers 5.2.0, flash-linear-attention 0.5.2
   (Qwen3.5's DeltaNet layers). See `../environment/README.md`. kvpress itself is not installed: `run_sweep.py`
   puts `../kvpress` first on the import path and stops if another kvpress gets imported.
2. **Models, gates and data.** On a machine with internet that shares `HF_HOME` with the GPU nodes:
   ```bash
   export HF_HOME=/path/to/hf_cache
   python final_eval/download.py            # ~50 GB; add --gate-training only to retrain gates
   python final_eval/download.py --check    # all "ok" = the revisions the final grid used
   ```
   Jobs run with `HF_HUB_OFFLINE=1`, so anything missing fails at load time, not silently.
3. **Our Fast KVzip gates** for Ministral 3 and Qwen3.5 (the authors released none) are not in git (200 MB).
   They are attached to the
   [`final-eval-as-run` release](https://github.com/skylight-org/vkv/releases/tag/final-eval-as-run):
   ```bash
   cd final_eval/fastkvzip_gates
   gh release download final-eval-as-run -R skylight-org/vkv -p '*.pt' -p gates.md5   # or from the release page
   md5sum -c gates.md5
   for m in ministral-3-8b-instruct-2512-bf16 qwen3.5-9b; do
     mkdir -p gates/$m && mv ${m}__q4_dim16_sink16.pt gates/$m/q4_dim16_sink16.pt
   done
   ```
   Or retrain them (`fastkvzip_gates/`). Qwen3-4B uses the authors' released gate, which `download.py` fetches.
   Only the `fastkvzip` and `fast_needle_final` arms need gates.

## Running

One sweep, from `final_eval/`:
```bash
cd final_eval
SWEEPS="sweeps/qwen3-4b_32k_r16_ruler.yaml" sbatch -p <partition> --time=2-00:00:00 launchers/job_grid.sbatch
# or, without SLURM, on GPU 0:
bash launchers/run_local.sh 0 sweeps/qwen3-4b_32k_r16_ruler.yaml
# or directly (environment variables as in job_grid.sbatch):
python run_sweep.py --sweep sweeps/qwen3-4b_32k_r16_ruler.yaml
```
The whole grid is 2,076 runs, about 970 GPU-hours (128k: ~730 on A100 80 GB). `launchers/grid_jobs.txt`
lists how it was split into 39 jobs and the time limits that worked; fill in the partitions and feed it to
`launchers/drain.sh`. A job stopped at its time limit keeps its finished cells: resubmit the same line and it
runs only what is left.

**GPU memory.** 32k cells peak at <= 31 GB (48 GB cards are fine). 128k cells peak at 52-68 GB and need an
80 GB card, for every model. Contexts are longer than the names say: RULER is tokenized for Llama, so
"128k" is up to 174k tokens under these models' tokenizers, and LOFT "32k" / "128k" are ~45-50k / ~111-123k.

### Sweep names

`<model>_<length>_r<ratio>_<part>.yaml`, where ratio r8 / r16 / r32 is `compression_ratio` 0.875 / 0.9375 /
0.96875 (the fraction of the KV cache removed) and part is:

| part | cells | arms |
|---|---|---|
| `ruler` | 13 RULER tasks x 50 test rows | main arms (`no_press` in r8 only; `needle_uncharged_c` in r16 only) |
| `loft` | 5 LOFT tasks, every test question | same |
| `fastkvzip`, `fastkvzip_loft` | as above | `fastkvzip`, `fast_needle_final` |
| `seeds_loft` (r16 only) | 5 LOFT tasks | `_s1` / `_s2` draws of the stochastic arms |

Models: `qwen3-4b` = Qwen/Qwen3-4B-Instruct-2507, `ministral3-8b` = mistralai/Ministral-3-8B-Instruct-2512-BF16,
`qwen3.5-9b` = Qwen/Qwen3.5-9B. Test rows: RULER takes `df.sample(50, random_state=42)` per task; LOFT keeps
the test split only (`loft_split: test`; kvpress's loader returns dev and test together).

### Arms

Defined in `../kvpress/evaluation/evaluate_registry.py`.

| arm | what it is |
|---|---|
| `no_press` | full KV cache |
| `kvzip_c` | KVzip, global top-k on its scores (`_c`: scores cached on disk per model and context) |
| `fastkvzip` | Fast KVzip with the authors' 2% recent window at every length (`window_size=None`) |
| `bernoulli_kvzip_c` | Bernoulli sampling on KVzip scores, Horvitz-Thompson reweighted |
| `needle_final_c` | needle fetch: Bernoulli sample + 2-bit key sketch + fetch at decode time; the sketch and fetch reserve are charged to the same memory budget |
| `needle_uncharged_c` | the same Bernoulli draw as `bernoulli_kvzip_c`, with sketch and reserve on top (ablation: what fetching adds at equal resident cache) |
| `fast_needle_final` | `needle_final_c` on Fast KVzip's gate scores instead of KVzip's |
| `<arm>_s1`, `<arm>_s2` | the same arm with another random draw (LOFT has one context per task, so one draw per cell) |

## Outputs and scoring

Each run writes `results/<group>/<dataset>__<task>__<model>__<arm>__<ratio>__.../` holding
`predictions.csv`, `metrics.json` (kvpress's scorer), `config.yaml` and `cost.json` (wall time, peak memory,
GPU, context lengths, the press's kept fraction and, for needle arms, fetch counts). A failed cell is
logged to `results/<group>/failures.json` and the sweep continues.

- **RULER:** `metrics.json` is the official RULER metric (string_match_all / string_match_part, as in
  NVIDIA/RULER); report the mean over the 13 tasks.
- **LOFT:** do NOT use `metrics.json`. kvpress's answer parser is more lenient than LOFT's (it reads past the
  first `[...]` line and falls back to the "Final Answer" text), which raised some compressed cells by 10-20
  points. Score saved predictions with LOFT's own parser:
  `python scoring/score_loft.py results/final_qwen3-4b --out final_qwen3-4b_loft.csv` (report subspan_em).

## Reproducibility notes

- **KVzip score cache.** `kvzip_c` and every `_c` arm reuse KVzip scores from `$KVPRESS_SCORE_CACHE`
  (default `~/.cache/kvpress/kvzip_scores`). The key is model + context tokens, not code version or GPU, and
  the first writer wins. Clear it after changing scoring code; keep one cache per GPU type when comparing
  runs bit for bit.
- **GPU type.** Expect bit-identical results only on the same GPU type. A40 and RTX A6000 were checked and
  matched answer for answer; A100 KVzip scores differed from A40 ones, and other types were not checked. Keep
  each model's table on one GPU type per context length (the final grid: 32k on A40 / A6000, 128k on A100).
- **Qwen3.5** runs fla Triton kernels whose autotune choice depends on sequence length and is cached in
  `~/.triton`; runs on different machines can pick different kernels and differ slightly. The final grid ran
  all of Qwen3.5 on one cluster. Its 128k cells need 80 GB (~50 GB peak on a 174k-token row).
- **Model quirks handled in `../kvpress`:** Ministral 3 is a VLM checkpoint, loaded text-only
  (`kvpress/model_loading.py`); Qwen3.5 (hybrid DeltaNet) gets its recurrent state continued correctly across
  the context / question split, and `<think>` suppressed. See `docs/SETUP.md` section 3.
- **Provenance.** Tag `final-eval-as-run` is the code byte-for-byte as it ran (with the original machine's
  paths); the commit after it changes only paths, defaults and documentation. kvpress base:
  skylight-org/kvpress `compression-experiments` at 9ca8502, plus the final-eval changes.

## Tests

```bash
cd kvpress && python -m pytest tests -q     # GPU tests need a GPU and flash-attn; CPU-only skips them
```
