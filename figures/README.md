# Figures and tables

Each folder holds a script and the data it reads. Run a script from the repository root, for example:

```bash
python figures/quality/make_quality.py
```

| Script | Paper content |
|---|---|
| `punchline_table/make_table.py` | Table 1 |
| `quality/make_quality.py` | Figure 3 and the appendix quality tables |
| `fetch_stats/make_fetch_table.py` | Table 2 |
| `ablation_eps_delta/make_eps_delta.py` | Tolerance and failure probability figure |
| `ablation_adherence/make_adherence.py` | Error adherence figure |
| `ablation_lambda/make_lambda.py` | Binary shrinkage figure |
| `systems/make_systems.py` | Quality and speed figure |
| `systems/decode_step_profile/make_appendix_systems.py` | Appendix systems figures |

The scripts need pandas and matplotlib. They write their output next to the script.
