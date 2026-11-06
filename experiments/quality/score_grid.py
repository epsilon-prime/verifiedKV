"""Score finished cells into one CSV, and print a summary table (no torch needed).

    python score_grid.py [--results results [more ...]] [--out results.csv]
"""

import argparse
import glob
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_sweeps  # noqa: E402
import scoring  # noqa: E402

METHODS = {"vkv_final_kvzip": "vkv_final_kvzip", "vkv_final_fastkvzip": "vkv_final_fastkvzip",
           "vkv_final_perfect": "vkv_final_perfect", "no_press": "dense", "kvzip_c": "kvzip",
           "fastkvzip": "fastkvzip", "bernoulli_kvzip_c": "vkv_nofetch_kvzip"}
KEY = ["model", "dataset", "context_k", "rate", "method", "task"]

ap = argparse.ArgumentParser()
ap.add_argument("--results", nargs="+", default=[os.path.join(HERE, "results")], help="results roots")
ap.add_argument("--out", default=os.path.join(HERE, "results.csv"))
ap.add_argument("--loft-first", type=int, default=scoring.LOFT_FIRST, help="LOFT questions scored per task; 0 for all")
args = ap.parse_args()
scoring.LOFT_FIRST = args.loft_first or None

groups = [g for r in args.results for g in sorted(glob.glob(os.path.join(r, "final_*")))]
df = pd.concat([scoring.scan(g, METHODS) for g in groups]) if groups else pd.DataFrame(columns=KEY + ["score"])
if len(df):
    dup = df.duplicated(KEY, keep=False)
    assert not dup.any(), f"the same cell in two results roots:\n{df[dup]}"
    df = df.sort_values(KEY).reset_index(drop=True)
    df["score"] = df["score"].round(2)
df.to_csv(args.out, index=False)
print(f"wrote {args.out}: {len(df)} cells")
if len(df):
    print(df.pivot_table(index=["model", "dataset", "context_k", "rate"], columns="method", values="score",
                         aggfunc="mean").round(1).to_string())
    short = df[(df.dataset == "RULER") & (df.n != make_sweeps.N_RULER)]
    if len(short):
        print(f"\nWARNING: {len(short)} RULER cells without {make_sweeps.N_RULER} rows:\n{short[KEY + ['n']]}")
