"""One results table for the paper with LOFT scored on the same first N test questions for every method.

    python make_loft_firstn.py --phase3 ~/phase3/experiments/final_eval/grid/results [--n 25]

The Ministral-3 and Qwen3-4B vkv_final LOFT cells hold only the first 25 test questions of each task, and those
questions are easier than the full set (Dense scores 4-9 points higher on them). So every LOFT cell, baselines and
all three models' vkv_final arms alike, is rescored on its first N questions. RULER is left as it is.

Before scoring, the first N gold answers of every LOFT cell are checked against one reference cell per (model, task,
length), so every method is scored on the same questions. Writes final_results/combined_loft<N>.csv in phase 3's
schema plus n and cell.
"""

import argparse
import glob
import os

import pandas as pd

import scoring
from loft_official import score_loft as sl

BASELINES = {"no_press": "dense", "kvzip_c": "kvzip", "fastkvzip": "fastkvzip",
             "bernoulli_kvzip_c": "vkv_nofetch_kvzip", "needle_final_c": "vkv_fetch_kvzip",
             "fast_needle_final": "vkv_fetch_fastkvzip"}  # as baselines/make_baseline_csv.py
OURS = {m: m for m in ("vkv_final_kvzip", "vkv_final_fastkvzip", "vkv_final_perfect")}
KEY = ["model", "dataset", "context_k", "rate", "method", "task"]

ap = argparse.ArgumentParser()
ap.add_argument("--phase3", required=True, help="phase 3 grid/results (holding final_<model>/)")
ap.add_argument("--n", type=int, default=25)
ap.add_argument("--out", default=None)
args = ap.parse_args()
here = os.path.dirname(os.path.abspath(__file__))
out = args.out or os.path.join(here, "final_results", f"combined_loft{args.n}.csv")

ref, _orig = {}, scoring.cell_score


def cell_score_firstn(d, dataset):
    if dataset != "loft":
        return _orig(d, dataset)
    df = pd.read_csv(os.path.join(d, "predictions.csv"))
    if len(df) < args.n:
        raise SystemExit(f"{d}: only {len(df)} questions, fewer than {args.n}")
    df = df.head(args.n).copy()
    df["answers"] = df["answers"].apply(sl.gold_list)
    task_l = df["task"].iloc[0]  # e.g. nq_32k
    model = os.path.basename(os.path.normpath(d if "__" in os.path.basename(os.path.normpath(d))
                                              else os.path.dirname(os.path.normpath(d)))).split("__")[2]
    gold = [tuple(map(str, a)) for a in df["answers"]]
    if ref.setdefault((model, task_l), gold) != gold:
        raise SystemExit(f"{d}: its first {args.n} questions differ from the other cells of {model} {task_l}")
    return task_l.rsplit("_", 1)[0], 100 * sl.score(df, sl.official_extract)["subspan_em"], len(df)


scoring.cell_score = cell_score_firstn
parts = [scoring.scan(g, OURS) for g in sorted(glob.glob(os.path.join(here, "final_results", "raw", "final_vkv_*")))]
parts += [scoring.scan(g, BASELINES) for g in sorted(glob.glob(os.path.join(args.phase3, "final_*")))]
df = pd.concat(parts).sort_values(KEY).reset_index(drop=True)
df["score"] = df["score"].round(2)
dup = df.duplicated(KEY, keep=False)
if dup.any():
    raise SystemExit(f"duplicate cells:\n{df[dup]}")
df.to_csv(out, index=False)
print(f"wrote {out}: {len(df)} rows; LOFT n per cell: {sorted(df[df.dataset == 'LOFT'].n.unique())}; "
      f"{len(ref)} (model, task, length) question sets checked")
