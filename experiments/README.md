# Experiments

| Folder | Paper content | Entry point |
|---|---|---|
| `quality/` | Quality comparison | See `quality/README.md` |
| `adherence/` | Error adherence and fetch statistics | `run_adh.py`, then `make_csv.py` |
| `tolerance_sweep/` | Tolerance and failure probability sweep | `run_downstream_grid.py` |
| `shrinkage/` | Effect of binary shrinkage | `collect.py` |
| `host_traffic/` | Host traffic on Qwen3.5-9B | `run_grid_traffic.py` |

Run `source quality/env.sh` before any of these. Each script lists its options with `--help`.
