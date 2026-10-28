"""Score the final grid's finished cells into one CSV in phase 3's schema (baselines/phase3_results.csv), plus a
progress table, and optionally the figure and tables with the phase 3 baselines (no torch, runs anywhere).

    python score_grid.py                       # results/ -> results.csv
    python score_grid.py --results results results_krishna --plot

RULER = RULER's string_match (metrics.json); LOFT = the OFFICIAL subspan_em (loft_official/score_loft.py).
--plot writes analysis/final_*.png|pdf|tex: phase 3's figure with the vkv_final arms, every method averaged over
the final grid's four RULER tasks (phase 3's own figure, all 13 tasks, is baselines/quality_vs_compression.png).
"""

import argparse
import glob
import os
import subprocess
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_sweeps  # noqa: E402
import scoring  # noqa: E402

METHODS = {"vkv_final_kvzip": "vkv_final_kvzip", "vkv_final_fastkvzip": "vkv_final_fastkvzip",
           "vkv_final_perfect": "vkv_final_perfect"}
KEY = ["model", "dataset", "context_k", "rate", "method", "task"]

ap = argparse.ArgumentParser()
ap.add_argument("--results", nargs="+", default=[os.path.join(HERE, "results")], help="results roots")
ap.add_argument("--out", default=os.path.join(HERE, "results.csv"))
ap.add_argument("--plot", action="store_true")
args = ap.parse_args()

groups = [g for r in args.results for g in sorted(glob.glob(os.path.join(r, "final_vkv_*")))]
df = pd.concat([scoring.scan(g, METHODS) for g in groups]) if groups else pd.DataFrame(columns=KEY + ["score"])
if len(df):
    dup = df.duplicated(KEY, keep=False)
    assert not dup.any(), f"the same cell in two results roots:\n{df[dup]}"
    df = df.sort_values(KEY).reset_index(drop=True)
    df["score"] = df["score"].round(2)
df.to_csv(args.out, index=False)
expected = len(make_sweeps.MODELS) * 2 * len(make_sweeps.RATIOS) * len(make_sweeps.ARMS) * (
    len(make_sweeps.RULER_TASKS) + len(make_sweeps.LOFT_TASKS))
print(f"wrote {args.out}: {len(df)} of {expected} cells")
if len(df):
    print(df.pivot_table(index=["model", "dataset", "context_k", "rate"], columns="method", values="score",
                         aggfunc="mean").round(1).to_string())
    short = df[(df.dataset == "RULER") & (df.n != make_sweeps.N_RULER)]
    if len(short):
        print(f"\nWARNING: {len(short)} RULER cells without {make_sweeps.N_RULER} rows:\n{short[KEY + ['n']]}")
if args.plot and len(df):
    a = os.path.join(HERE, "analysis")
    subprocess.run([sys.executable, os.path.join(a, "plot_quality.py"), "--allow-partial",
                    "--csv", os.path.join(HERE, "baselines", "phase3_results.csv"), args.out,
                    "--ruler-tasks", "niah_mk3", "niah_mv", "cwe", "qa1",
                    "--out", os.path.join(a, "final_quality_vs_compression.png"),
                    "--out-relative", os.path.join(a, "final_quality_vs_compression_relative.png"),
                    "--main-table", os.path.join(a, "final_quality_per_task_main.tex"),
                    "--appendix-table", os.path.join(a, "final_quality_per_task_appendix.tex")], check=True)
