# Vkv1 (draft 0): work in progress

A snapshot of `~/phase4` on the Prajna cluster as of 2026-10-24, pushed as is: no code was changed for the push.
**Start with `VKV1_CHERN_GRAD.md`**, the configuration handed over for testing (Chernoff certificate, grad order, no
persistence).
`README.md` (next to this file) describes the algorithm, the code and the smoke results. The algorithm is in
`theory/SOURCES/vkv_draft0.pdf`.

## How it is wired (still Prajna paths)

- **kvpress:** `vkv1/press.py`, the tests and the experiments import kvpress from `$PYTHONPATH`. On Prajna that is
  `~/src/kvpress-e5`, the same code as this repo's `kvpress/` folder (inherited from the `final-eval` branch).
  Here: `export PYTHONPATH=<repo>/kvpress`.
- **Hard-coded paths:**
  - `experiments/vkv1_down/run_downstream.py` adds `~/src/kvpress-e5/evaluation` and `~/phase4` to `sys.path`;
  - `experiments/vkv1_sim/sim.py` adds `~/phase4` and `~/phase3/experiments/algo_improvements` (for `cert.py`);
  - the two `job_*.sbatch` files use `$HOME/phase4/...` and Prajna's conda setup.
- The 2026-10-24 folders (`experiments/vkv1_diag`, `vkv1_chern`, `vkv1_nopersist`, `vkv1_hx`) import `~/phase4`,
  `~/phase4/experiments/vkv1_diag` (for `diag_lib.py`) and, downstream, `~/src/kvpress-e5/evaluation` by absolute
  path. Their `job_*.sbatch` files `cd` into `$HOME/phase4/experiments/...`. Simplest outside Prajna:
  `ln -s <repo>/phase4 ~/phase4`.
- **`deps/algo_improvements/`:** unchanged copies of the two Phase 3 files Vkv1 depends on. `cert.py` provides
  `HeadSketch` for the simulator. `extract.py` writes the simulator's input (attention extracts; about 64 GB for
  Qwen3-4B, 13 RULER-32k tasks x 2 dev rows, all layers), which is not in git.

## Tests

```bash
cd phase4
CUDA_VISIBLE_DEVICES="" PYTHONPATH=../kvpress python -m pytest -q tests experiments/vkv1_diag/test_diag.py \
    experiments/vkv1_chern/test_press_chern.py experiments/vkv1_chern/test_final_config.py experiments/vkv1_hx/test_hx.py
```

These need `~/phase4` to point at this folder (see above).
