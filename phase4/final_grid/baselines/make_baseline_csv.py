"""Rebuild the phase 3 baseline table (phase3_results.csv) from phase 3's final-grid result folders.

    python make_baseline_csv.py --results ~/phase3/experiments/final_eval/grid/results [--check old.csv]

Arms -> methods, as in phase 3's scores_so_far.csv (the table behind the phase 3 figure):
    no_press -> dense, kvzip_c -> kvzip, fastkvzip -> fastkvzip, bernoulli_kvzip_c -> vkv_nofetch_kvzip,
    needle_final_c -> vkv_fetch_kvzip, fast_needle_final -> vkv_fetch_fastkvzip
The LOFT extra seeds (_s1, _s2) and needle_uncharged_c are not in that table and are left out here too.
--check compares every row the old table has against the rebuilt one (and lists the rows that are new).
"""

import argparse
import glob
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import scoring  # noqa: E402

METHODS = {"no_press": "dense", "kvzip_c": "kvzip", "fastkvzip": "fastkvzip",
           "bernoulli_kvzip_c": "vkv_nofetch_kvzip", "needle_final_c": "vkv_fetch_kvzip",
           "fast_needle_final": "vkv_fetch_fastkvzip"}
KEY = ["model", "dataset", "context_k", "rate", "method", "task"]

ap = argparse.ArgumentParser()
ap.add_argument("--results", required=True, help="phase 3 grid/results (holding final_<model>/)")
ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "phase3_results.csv"))
ap.add_argument("--check", help="an older table to compare against")
args = ap.parse_args()

df = pd.concat([scoring.scan(g, METHODS) for g in sorted(glob.glob(os.path.join(args.results, "final_*")))])
df = df.sort_values(KEY).reset_index(drop=True)
df["score"] = df["score"].round(2)  # as phase 3's table
assert not df.duplicated(KEY).any(), df[df.duplicated(KEY, keep=False)]
df.to_csv(args.out, index=False)
print(f"wrote {args.out}: {len(df)} rows")
print(df.groupby(["model", "dataset", "context_k", "method"]).size().unstack("method").to_string())
if args.check:
    old = pd.read_csv(args.check)
    m = old.merge(df, on=KEY, how="outer", suffixes=("_old", ""), indicator=True)
    both = m[m._merge == "both"]
    diff = both[(both.score_old - both.score).abs() > 1e-6]
    print(f"\n{len(both)} rows in both tables, {len(diff)} with a different score; "
          f"{(m._merge == 'right_only').sum()} new rows; {(m._merge == 'left_only').sum()} rows only in the old table")
    if len(diff):
        print(diff[KEY + ["score_old", "score"]].to_string(index=False))
    new = m[m._merge == "right_only"]
    if len(new):
        print("new:", new.groupby(["model", "dataset", "context_k", "method"]).size().to_string())
