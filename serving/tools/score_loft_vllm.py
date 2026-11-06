"""LOFT scores of ruler_vllm.py --bench loft runs, with LOFT's official answer parser (the harness's
final_eval/scoring/score_loft.py: the first line holding "[...]" only, no fallback) and kvpress's
calculate_metrics. Reports subspan_em (the harness's reported metric) per task and arm, plus
the timing / fetch means of the rows.

    python tools/score_loft_vllm.py out/loft128k [--paired]

--paired: score every arm only on the (task, row) pairs that all arms finished.
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

VKV = Path(__file__).resolve().parents[2]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cm = _load("loft_metrics", VKV / "evaluation/benchmarks/loft/calculate_metrics.py")
sys.path.insert(0, str(VKV / "evaluation"))                # score_loft imports benchmarks.loft
official_extract = _load("score_loft", VKV / "experiments/quality/loft_official/score_loft.py").official_extract

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--paired", action="store_true")
args = ap.parse_args()

arms = {d.name: pd.DataFrame(map(json.loads, open(d / "rows.jsonl")))
        for d in sorted(Path(args.run).iterdir()) if (d / "rows.jsonl").exists()}
if args.paired:
    common = set.intersection(*({(t, r) for t, r in zip(df.task, df.row)} for df in arms.values()))
    arms = {a: df[[(t, r) in common for t, r in zip(df.task, df.row)]] for a, df in arms.items()}

cm.extract_prediction = official_extract
out = []
for arm, df in arms.items():
    for task, g in df.groupby("task"):
        m = cm.calculate_metrics(pd.DataFrame({"predicted_answer": g.pred, "answers": g.gold,
                                               "task": task, "answer_prefix": g.answer_prefix}))
        out.append({"arm": arm, "task": task, "n": len(g), "subspan_em": 100 * m["subspan_em"],
                    "ttft_s": g.ttft.mean(), "tpot_ms": 1e3 * g.tpot.median(),
                    "fetch_MB": g.fetch_bytes.mean() / 1e6 if "fetch_bytes" in g else None})
t = pd.DataFrame(out)
print(t.pivot(index="task", columns="arm", values="subspan_em").round(1).to_string())
avg = t.groupby("arm").agg(n=("n", "sum"), subspan_em=("subspan_em", "mean"), ttft_s=("ttft_s", "mean"),
                           tpot_ms=("tpot_ms", "mean"), fetch_MB=("fetch_MB", "mean"))
print("\nmean over tasks:\n" + avg.round(2).to_string())
