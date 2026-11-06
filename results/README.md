# Results

| Path | Contents |
|---|---|
| `quality/all_models.csv` | Every quality score: model, dataset, context length, rate, method, task, score, questions |
| `quality/methods.csv` | Method identifiers and paper names |
| `eps_delta/` | Sweep of the tolerance and failure probability |
| `systems/` | vLLM measurements for RULER-32K and RULER-128K at 16x |

Scores range from 0 to 100. RULER uses 4 tasks and 50 test rows per task, except the Ministral-3-8B RULER-128K baselines,
which use 20 rows. LOFT uses 5 tasks and the first 25 test questions per task.
