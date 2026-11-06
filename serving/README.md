# vLLM prototype

Implementation of VerifiedKV in vLLM 0.29, used for the latency and throughput results in the paper.

| Path | Contents |
|---|---|
| `offload_package/` | Host offload, staging, scoring and the VerifiedKV decision kernels |
| `skylight/` | vLLM sparse attention backend |
| `vllm-0.29.0/` | Modified copy of vLLM 0.29.0 |
| `benchmarks/` | Benchmark runners |
| `launchers/` | Scripts for the RULER, LOFT and batch scaling runs |
| `tools/` | LOFT scoring for vLLM outputs |

Results are in `../results/systems/`. Hardware for the paper: NVIDIA H200, Qwen3-4B-Instruct-2507, 16x.

## Status

This folder is being completed. Build and reproduction instructions will be added here.
