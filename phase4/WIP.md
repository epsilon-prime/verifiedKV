# Vkv1 (draft 0): work in progress

A snapshot of `~/phase4` on the Prajna cluster as of 2026-10-23, pushed as is: no code was changed for the push.
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
- **`deps/algo_improvements/`:** unchanged copies of the two Phase 3 files Vkv1 depends on. `cert.py` provides
  `HeadSketch` for the simulator. `extract.py` writes the simulator's input (attention extracts; about 64 GB for
  Qwen3-4B, 13 RULER-32k tasks x 2 dev rows, all layers), which is not in git.

## Tests

```bash
cd phase4
CUDA_VISIBLE_DEVICES="" PYTHONPATH=../kvpress python -m pytest -q tests
```
