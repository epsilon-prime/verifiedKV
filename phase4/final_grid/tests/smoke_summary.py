"""Per-cell time, memory, draw and answers of smoke (or any) results: python tests/smoke_summary.py results_smoke"""
import glob
import json
import os
import sys

import pandas as pd

root = sys.argv[1] if len(sys.argv) > 1 else "results_smoke"
rows = []
for c in sorted(glob.glob(os.path.join(root, "*", "*", "cost.json")) + glob.glob(os.path.join(root, "*", "*", "*", "cost.json"))):
    j = json.load(open(c))
    d = os.path.dirname(c)
    p = pd.read_csv(os.path.join(d, "predictions.csv")) if os.path.isfile(os.path.join(d, "predictions.csv")) else None
    m = json.load(open(os.path.join(d, "metrics.json"))) if os.path.isfile(os.path.join(d, "metrics.json")) else {}
    n = len(p) if p is not None else 0
    score = next(iter(m.values()))["string_match"] if m and "string_match" in str(m) else m.get("subspan_em")
    rows.append({"model": j["model"].split("/")[-1][:14], "dataset": j["dataset"], "task": j["task"],
                 "arm": j["press"].replace("vkv_final_", ""), "gpu": j["gpu"].replace("NVIDIA ", "")[:12],
                 "n": n, "s/answer": round(j["seconds"] / max(n, 1), 1), "peak_gb": round(j["peak_gb"], 1),
                 "draw": round(j.get("kept_frac_mean") or 0, 4), "score": score,
                 "ctx": j.get("ctx_len_median")})
df = pd.DataFrame(rows)
pd.set_option("display.width", 220)
print(df.to_string(index=False))
