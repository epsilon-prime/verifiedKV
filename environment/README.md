# Environment

The final grid ran in one conda environment: Python 3.11.16 on Linux x86_64, NVIDIA driver 570 (CUDA 12.8).
`requirements-frozen.txt` is its exact package set.

```bash
conda create -n kvpress-final python=3.11.16
conda activate kvpress-final
pip install -r environment/requirements-frozen.txt
```

- **torch** is the CUDA 12.8 build (`torch==2.11.0+cu128`); the file points pip at the PyTorch cu128 index.
- **flash-linear-attention 0.5.2** (`fla-core`) provides Qwen3.5's DeltaNet kernels. Without it transformers
  falls back to its own, slower torch implementation; the Qwen3.5 numbers were produced with fla.
- **flash-attn** is optional: the final grid runs every model under `sdpa` (the runner asserts it, because
  flash attention would ignore the additive bias of the sampling presses). kvpress's tests use it. It needs a
  wheel built for torch 2.11 + CUDA 12.8 (`flash_attn-2.8.3+cu128torch2.11`), installed with
  `pip install --no-deps <wheel>`.
- **kvpress** is not installed; the runner imports `../kvpress` from this repo directly.
- If the RULER or LOFT scorers fail importing `sqlite3` with a `CXXABI` error, put the environment's
  libstdc++ first: `export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH` (the launchers do this).
