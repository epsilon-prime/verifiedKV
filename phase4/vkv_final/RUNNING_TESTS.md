# Guide: running the vkv_final downstream tests on Prajna (self-contained)

Written 2026-10-27 for someone (or some model) who has NO access to the files and cannot look anything up: every
path, command and script needed is in this document. Where the text says "paste", the block is meant to be pasted
into a terminal on the Prajna login node (`login1` or `login2`, one at a time) as the user `23b1835`.

## 1. Context in five sentences
- **What is tested:** `vkv_final`, a KV-cache compression method (Qwen3-4B-Instruct-2507, RULER-32k benchmark). It keeps a
  random 1/16 (at 16x) of the KV cache on the GPU, and at every attention row decides, with a probabilistic certificate
  (Chernoff bound), how many missing tokens to fetch so the softmax denominator is estimated within a factor `eps0`
  with failure probability `delta`. It plans using a 2-bit sketch of every key plus 128 exactly-scored "shortlist" keys
  per KV head.
- **Knobs:** compression ratio, `eps0`, `delta`, shortlist size. Current candidate: 16x, `eps0 0.3`, `delta 0.05` or `0.1`,
  shortlist 128.
- **The "sped-up algo":** the same algorithm with a faster certificate computation (the 32-point lambda grid is
  computed as one batch instead of a Python loop). It returns identical k* (checked bit for bit), and is switched on
  with the environment variable `VKV_VEC=1` (the default; `VKV_VEC=0` = old loop version). Measured on L40S: about 10-18% less end-to-end time per context.
- **What a test produces:** for every (task, row) the model's answer. Score = RULER string match: for `qa_*` tasks 100 if any
  gold string is in the answer, else 0; for all other tasks the fraction of gold strings found, times 100.
- **Reference results to compare with** (A40, 26 contexts = 13 tasks x rows 152-153, 16x): full cache 95.00;
  perfect sketch (plans on exact weights, not deployable) 93.85 at eps0 0.5 and 95.00 at eps0 0.3; vkv_final
  eps0 0.3 delta 0.05: 92.18; delta 0.1: 92.18; eps0 0.5 delta 0.05 (L40S): 88.85.

## 2. Cluster rules (they matter; the cluster is not standard)
- **Never run compute on the login node.** It is shared and loaded. Everything that touches torch or a GPU goes through `sbatch`.
  Plain Python on text/JSON files (the scoring script in section 6) is fine on the login node.
- **Partitions:** use only `a40` (4x A40 per node) and `l40` (8x L40S per node). Never `dgx` or `dgx-mpi`.
- **`--qos` must equal the partition name:** `-p a40 --qos=a40`, or `-p l40 --qos=l40`.
- **Per-user limits:** a40 = 3 running / 6 queued; l40 = 4 running / 5 queued. Extra jobs show reason `QOSMaxJobsPerUserLimit` and
  wait; that is normal.
- **Use one GPU type for any comparison.** The random draw (which tokens stay on the GPU) depends on the GPU model, so
  A40 and L40S runs of the same arm give different answers. **Accuracy tests: always `a40`.** Use `l40` only for timing
  or when A40 is full and you do not need to pair with older A40 results.
- **Queue check:** `squeue -A 23b1835` (the usual `squeue --me` does not work on compute nodes).
  Cancel: `scancel <jobid>`. Job history: `sacct -j <jobid> --format=JobID,State,Elapsed,NodeList -X`.
- **Do not edit or delete** anything under `~/vkv-repo` or older `out/` folders.

## 3. File locations (all on the shared home, visible from both login nodes)
| what | path |
|---|---|
| repository, branch `vkv_final_algo` (kvpress + the algorithm code) | `~/vkv-repo` |
| experiment folder (runner, job script, tests): the working directory for everything below | `~/vkv-repo/phase4/vkv_final` |
| runner (no statistics, used for the grid) | `~/vkv-repo/phase4/vkv_final/run_downstream_grid.py` |
| job script | `~/vkv-repo/phase4/vkv_final/job_down_grid.sbatch` |
| press with the speed-up switch | `~/vkv-repo/phase4/vkv_final/press_grid.py`, `chernoff_vec.py` |
| shared KVzip score cache | `~/phase3/cache/kvzip_scores` |
| model + datasets (offline) | `~/hf` |
| stored reference results (full / perfect-sketch arms, A40, RULER-32k rows 152-171) | `~/vkv-repo/phase4/experiments/vkv1_r20/out/r20_all/<arm>/rows.jsonl` |
| conda environment | `kvpress-final` (conda at `~/miniforge3`) |

The job script `job_down_grid.sbatch` (already exists; shown so it can be recreated if it is missing):

```bash
#!/bin/bash
#SBATCH --job-name=vkv-grid
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
source "${CONDA_SH:-$HOME/miniforge3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-kvpress-final}"
export HF_HOME="${HF_HOME:-$HOME/hf}" TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" \
       HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd "$SLURM_SUBMIT_DIR"
stdbuf -oL -eL python -u run_downstream_grid.py $ARGS 2>&1 | grep --line-buffered -v "Loading weights"
```
It reads the runner's arguments from the environment variable `ARGS`, and is submitted from `~/vkv-repo/phase4/vkv_final`
(logs go to `./logs/`, results to `./out/`).

## 4. Setup (paste at the start of every shell session)
```bash
cd ~/vkv-repo/phase4/vkv_final
mkdir -p logs out
export KVPRESS_SCORE_CACHE=$HOME/phase3/cache/kvzip_scores   # REQUIRED. Without it KVzip scores are recomputed into another
                                                              # folder and the random draws no longer match stored results.
```

## 5. The runner: arms, arguments
`run_downstream_grid.py` arguments (all go inside `ARGS="..."`):

| argument | meaning |
|---|---|
| `--tasks T1 T2 ...` | RULER tasks: `niah_single_1 niah_single_2 niah_single_3 niah_multikey_1 niah_multikey_2 niah_multikey_3 niah_multiquery niah_multivalue vt cwe fwe qa_1 qa_2` |
| `--rows a:b` | rows a..b-1 of the (seed-42 shuffled) task. **Any row is allowed.** Rows 0-149 are the final-test rows (do not tune settings on them), 150-151 are calibration contexts, and 152+ are the dev rows used so far (stored full/perfect results exist for 152-171). |
| `--arms A1 A2 ...` | one or more arms (below); every arm runs on every row inside one model load |
| `--run NAME` | output folder: `out/NAME/<arm>/rows.jsonl`. Re-using a NAME resumes: rows already done are skipped. |
| `--max-new-tokens N` | optional cap on generated tokens (default: the task's own limit) |
| `--dataset` | default `ruler32k` (the only one validated here); `--model` default `Qwen/Qwen3-4B-Instruct-2507` |

Arms:
| arm | meaning |
|---|---|
| `full` | no compression (reference) |
| `bern:0.0625` | plain Bernoulli sampling, 1/16 kept (baseline) |
| `vkv_final:<ratio>:<eps0>:<delta>` | the algorithm. `ratio` = compression ratio: `0.875` = 8x, `0.9375` = 16x, `0.96875` = 32x. |
| ...`:slh=<M>` | shortlist size per KV head, default 128; `slh=0` turns the shortlist off, `slh=32` is a smaller buffer |
| ...`:perfect` | plan on exact weights (upper-bound reference, not deployable) |
| ...`:exact` | exact float64 Chernoff curve instead of the fast one (slow; for checking) |

Examples: `vkv_final:0.9375:0.3:0.05` (default), `vkv_final:0.9375:0.3:0.1`, `vkv_final:0.9375:0.5:0.1:slh=32`,
`vkv_final:0.875:0.3:0.1` (8x), `vkv_final:0.96875:0.3:0.1` (32x).
Notes: at 32x the sketch (1.76% of the cache) is more than half the 3.125% budget; if the draw would be empty the runner
stops with an assertion, which is a real limit, not a bug.

**Speed-up switch:** the vectorised certificate is ON by default. Put `VKV_VEC=0` in front of `sbatch` (the environment is
passed to the job) to run the older loop version instead. Both give the same answers; the `VKV_VEC=1` prefixes below are
harmless (they state the default explicitly).

## 6. Downstream accuracy tests: commands to paste

### 6a. Smoke test (about 5 minutes, one context): do this first
```bash
VKV_VEC=1 ARGS="--rows 152:153 --tasks niah_single_3 --arms vkv_final:0.9375:0.3:0.1 --run smoke_vec" \
  sbatch -p a40 --qos=a40 --time=1:00:00 job_down_grid.sbatch
```
When it finishes (`squeue -A 23b1835` empty; log in `logs/vkv-grid-<jobid>.out`), the log must contain a line like
`niah_single_3_152 vkv_final:0.9375:0.3:0.1: 40s pred='...' gold=[...]` and `done`. The gold string
`f1afdb65-b289-4224-8ac9-778d8da8eee4` appears in the correct answer for this row.

### 6b. One task, several arms (typical test)
```bash
# 25 rows of one task, two delta settings, sped-up algo, A40
VKV_VEC=1 ARGS="--rows 152:177 --tasks cwe --arms vkv_final:0.9375:0.3:0.05 vkv_final:0.9375:0.3:0.1 --run mytest/cwe" \
  sbatch -p a40 --qos=a40 --time=8:00:00 job_down_grid.sbatch
```
Cost to plan with: about 90 s per context per arm for eps0 0.3 (60 s with eps0 0.5) on A40 for 32k contexts; a task with 25
rows and 2 arms is about 1.5-3 hours. `cwe` (long answers) is the slowest task, `niah_*` and `vt` the fastest.

### 6c. All 13 tasks, 2 rows each (the standard 26-context comparison), one job per group of tasks
```bash
ARMS="vkv_final:0.9375:0.3:0.05 vkv_final:0.9375:0.3:0.1"; RUN=cmp26   # change both to what you are testing
for T in "cwe niah_single_1 niah_single_2 fwe vt" "niah_multikey_3 niah_multivalue qa_1 niah_single_3" \
         "niah_multiquery niah_multikey_1 niah_multikey_2 qa_2"; do
  n=$(echo $T | tr -d ' ' | md5sum | cut -c1-6)
  VKV_VEC=1 ARGS="--rows 152:154 --tasks $T --arms $ARMS --run $RUN/g$n" sbatch -p a40 --qos=a40 --time=6:00:00 job_down_grid.sbatch
done
```
(3 jobs, about 25 minutes each once running.)

### 6d. A sweep over settings: one job per setting
```bash
for A in vkv_final:0.9375:0.3:0.1 vkv_final:0.9375:0.5:0.1 vkv_final:0.9375:0.3:0.1:slh=32; do
  VKV_VEC=1 ARGS="--rows 152:177 --tasks qa_1 --arms $A --run sweep/$(echo $A | tr ':=' '__')" \
    sbatch -p a40 --qos=a40 --time=8:00:00 job_down_grid.sbatch
done
```
With the 3-running limit, extra jobs queue automatically.

### 6e. Compare against full cache and plain Bernoulli on the same rows
Add `full` and/or `bern:0.0625` to `--arms` (they cost less than 15 s per context). Stored full-cache and perfect-sketch
results already exist for rows 152-171 (section 3), so you only need `full` yourself for rows 172+.

### 6f. Re-checking that the speed-up changes nothing (do once)
Run the same small test twice, with `VKV_VEC=0` and `VKV_VEC=1`, under different `--run` names, on A40:
```bash
VKV_VEC=0 ARGS="--rows 152:156 --tasks qa_1 niah_single_3 --arms vkv_final:0.9375:0.3:0.1 --run chk_loop" sbatch -p a40 --qos=a40 --time=2:00:00 job_down_grid.sbatch
VKV_VEC=1 ARGS="--rows 152:156 --tasks qa_1 niah_single_3 --arms vkv_final:0.9375:0.3:0.1 --run chk_vec" sbatch -p a40 --qos=a40 --time=2:00:00 job_down_grid.sbatch
```
(They may land on different A40 nodes; that is fine, A40s are identical.) Then compare predictions with the script in 7b.
Expect identical `pred` for every context.

## 7. Reading the results (plain Python, fine on the login node)

### 7a. Score one or more runs and compare arms (the file exists as `~/vkv-repo/phase4/vkv_final/score_runs.py`; recreate it from this text if missing)
```python
"""python3 score_runs.py out/RUN1 [out/RUN2 ...]   -> per-arm score, per-task scores, sec/context, and paired diffs."""
import glob, json, math, re, sys
from collections import defaultdict
from pathlib import Path

ctrl = re.compile(r"[\x00-\x1f]")
def score(task, pred, gold):
    pred = ctrl.sub("", pred.strip()).strip().lower()
    hits = [1.0 if g.lower() in pred else 0.0 for g in gold]
    return 100 * (max(hits) if task.startswith("qa") else sum(hits) / len(hits))

arms = defaultdict(dict)                       # arm -> {(task,row): record}
for d in sys.argv[1:]:
    for f in sorted(glob.glob(f"{d}/**/rows.jsonl", recursive=True)):
        arm = Path(f).parent.name
        for line in open(f):
            r = json.loads(line)
            arms[arm][(r["task"], r["row"])] = r
names = sorted(arms)
for a in names:
    S = {k: score(k[0], r["pred"], r["gold"]) for k, r in arms[a].items()}
    sec = sum(r["sec"] for r in arms[a].values()) / len(S)
    gpus = sorted({r.get("gpu", "?") for r in arms[a].values()})
    print(f"\n== {a}: {len(S)} contexts, mean score {sum(S.values()) / len(S):.2f}, {sec:.0f} s/context, GPU {gpus}")
    per = defaultdict(list)
    for k, v in S.items():
        per[k[0]].append(v)
    print("   " + "  ".join(f"{t}:{sum(v) / len(v):.0f}({len(v)})" for t, v in sorted(per.items())))
print("\npaired differences (A - B) on the contexts both have: mean +- SE (A better / A worse)")
for i, a in enumerate(names):
    for b in names[i + 1:]:
        K = sorted(set(arms[a]) & set(arms[b]))
        if len(K) < 2:
            continue
        d = [score(k[0], arms[a][k]["pred"], arms[a][k]["gold"]) - score(k[0], arms[b][k]["pred"], arms[b][k]["gold"]) for k in K]
        mu = sum(d) / len(d)
        se = math.sqrt(sum((x - mu) ** 2 for x in d) / (len(d) - 1) / len(d))
        same = sum(arms[a][k]["pred"] == arms[b][k]["pred"] for k in K)
        print(f"  {a} - {b}: {mu:+.2f} +- {se:.2f}  ({sum(x > 0 for x in d)} / {sum(x < 0 for x in d)}), n={len(K)}, identical answers {same}")
```
Interpretation: a paired difference smaller than about 2 standard errors (SE) is noise. Contexts are paired by (task, row), and
arms run in the same job use the same random draw, so pairing is exact. Pairing across runs is only valid if both ran
on the same GPU model (the `GPU [...]` column).

### 7b. Compare two runs' answers exactly (the speed-up check)
```bash
python3 - <<'EOF'
import glob, json
def load(p):
    d = {}
    for f in glob.glob(p + "/*/rows.jsonl"):
        for l in open(f):
            r = json.loads(l); d[(r["task"], r["row"])] = r["pred"]
    return d
a, b = load("out/chk_loop"), load("out/chk_vec")
K = set(a) & set(b)
print(len(K), "contexts; identical answers:", sum(a[k] == b[k] for k in K))
EOF
```

### 7c. Time per context of a run
`score_runs.py` prints `s/context`; raw numbers are the `sec` field of each row (prefill + KVzip scoring + generation).

## 8. Timing / profiling test (L40S is fine; accuracy is not needed)
```bash
sed -e 's/run_downstream_grid.py/profile_grid.py/' -e 's/vkv-grid/vkv-prof/' job_down_grid.sbatch > job_prof.sbatch   # once
C="--cases ruler32k:niah_single_3:152 ruler32k:cwe:152 ruler64k:niah_single_3:400 --eps0 0.3 --delta 0.1"
VKV_VEC=0 ARGS="$C --out out/prof_loop.json"  sbatch -p l40 --qos=l40 --time=1:00:00 job_prof.sbatch
VKV_VEC=1 ARGS="$C --out out/prof_vec.json"   sbatch -p l40 --qos=l40 --time=1:00:00 job_prof.sbatch
```
Each JSON is a list of records `{case, arm ("full" or "vkv_grid"), total_s, timers{...}, counts{...}}`; compare `total_s`
of the `vkv_grid` records. Reference (L40S, same contexts): loop 40.4 / 93.0 / 59.4 s; vec 32.7 / 76.3 / 53.2 s.
(`ruler64k` needs row indices that exist in that dataset; `400` works.)

## 9. Unit tests (optional, run as GPU-node jobs)
```bash
sbatch -p l40 --qos=l40 --time=0:30:00 --cpus-per-task=4 --mem=32G -J vkv-test -o logs/test-%j.out --wrap \
 'source ~/miniforge3/etc/profile.d/conda.sh && conda activate kvpress-final && cd ~/vkv-repo/phase4/vkv_final && PYTHONPATH=$HOME/vkv-repo/kvpress python -m pytest -q test_press_grid.py'
sbatch -p l40 --qos=l40 --time=0:30:00 --gres=gpu:1 --cpus-per-task=4 --mem=32G -J vkv-vec -o logs/vec-%j.out --wrap \
 'source ~/miniforge3/etc/profile.d/conda.sh && conda activate kvpress-final && cd ~/vkv-repo/phase4/vkv_final && python -u test_chernoff_vec.py'
```
Expected: `8 passed` (about 6 min) and a last line `ALL IDENTICAL`. Output files: `logs/test-<jobid>.out`, `logs/vec-<jobid>.out`.

## 10. What a good test plan looks like (suggestions)
1. Smoke (6a), then the delta/eps comparisons on the hard tasks `cwe qa_1 niah_multikey_3 niah_multivalue`, 25 rows
   each (6b), one task per job. These four tasks are where weaker settings lose accuracy first.
2. Always report: mean score, paired difference with SE against the other arm on the same rows, seconds per context, GPU model.
3. A setting is "better" only if the paired difference exceeds about 2 SE; with 25 rows per task and 4 tasks (100
   contexts) differences under about 2-3 points are noise.
4. Keep `eps0` 0.3 as the default; `eps0 0.5` is about 2.4x cheaper per context but about 3 points less accurate.
   `delta` mainly changes how much is fetched, not compute time.

## 11. Troubleshooting
| symptom | cause / fix |
|---|---|
| job `PENDING` with `QOSMaxJobsPerUserLimit` | you already have the maximum running; wait, it starts by itself |
| `ModuleNotFoundError: press_grid` | not submitted from `~/vkv-repo/phase4/vkv_final` (the job `cd`s to the submit directory) |
| scores far below the reference | not on A40, or `KVPRESS_SCORE_CACHE` not exported before `sbatch` |
| job fails immediately, `sbatch: error: Batch job submission failed` | wrong `--qos` for the partition |
| out of memory | very unlikely at 32k (peak about 17 GB); check you did not request `ruler128k` |
| need to stop a run | `scancel <jobid>`; to continue later, re-submit the identical command (finished rows are skipped) |
| login node very slow | normal (shared); do not run Python with torch there |
