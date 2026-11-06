#!/usr/bin/env python3
"""Binary shrinkage figure for one query row: MSE of the denominator estimator when lambda_i in {1, r_i}.

Input (next to this script), extracted from experiments/shrinkage/out/sketch_ab_16x.npz
(Qwen3-4B, RULER-32K, 16x; row index 8810 = RULER cwe context 152, layer 21, decode step 21, |L| = 44,549):
  lambda_row_curve.csv  this row's MSE / MSE(all lambda_i = 1) when the first fraction of L, in decreasing jump
                        J_i = w_i (1 - r_i) / r_i, is set to lambda_i = r_i; order from exact w_i and from the deployed
                        2-bit sketch w_hat_i (the MSE is always computed with exact weights)
  lambda_row.json       the row's jump-rule stopping points (tokens set to r_i, fraction of L, MSE ratio)
  lambda_points.json    the same over all 44,640 rows (medians, share of rows not worse), quoted in the text

Output: lambda_binary_shrinkage.{png,pdf}, one column wide.

Usage (from the paper root, any env with matplotlib):
  python figures/ablation_lambda/make_lambda.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "STIXGeneral"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.linewidth": 0.6,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.minor.size": 1.5,
    "pdf.fonttype": 42,
})

INK = "#000000"
GRID = "#d9d9d9"
# Exact weights are the ideal reference (gray); the deployed sketch is what VerifiedKV runs (its blue).
STYLE = {
    "exact": dict(color="#6b6b6b", linestyle="-", label=r"exact weights $w_i$"),
    "sketch": dict(color="#4a78b0", linestyle=(0, (4, 1.5)), label=r"2-bit sketch $\hat w_i$ (deployed)"),
}


def main() -> None:
    here = Path(__file__).resolve().parent
    rows = list(csv.DictReader(open(here / "lambda_row_curve.csv")))
    row = json.load(open(here / "lambda_row.json"))
    fig, ax = plt.subplots(figsize=(3.3, 2.1))
    ax.axhline(1.0, color=INK, linewidth=0.7, linestyle=(0, (1, 1.5)), zorder=1)
    handles = []
    for key, col in (("exact", "mse_rel_exact_order"), ("sketch", "mse_rel_sketch_order")):
        st, stop = STYLE[key], row["jump_rule"][key]
        # The stopping point is a point of this row's curve: add it to the grid so the circle sits on the line.
        pts = sorted([(float(r["frac_set_to_r"]), float(r[col])) for r in rows] + [(stop["frac"], stop["mse_rel"])])
        (h,) = ax.plot([p[0] for p in pts], [p[1] for p in pts], color=st["color"], linestyle=st["linestyle"],
                       linewidth=1.3, zorder=3)
        handles.append(h)
        # Dotted line at the MSE the jump rule attains, so the reduction from 1.0 reads off the y axis.
        ax.axhline(stop["mse_rel"], color=st["color"], linewidth=0.7, linestyle=(0, (1, 1.5)), zorder=2)
        ax.text(1.0, stop["mse_rel"], f"{stop['mse_rel']:.2f}", transform=ax.get_yaxis_transform(), ha="right",
                va="bottom", fontsize=6.5, color=INK)
        ax.plot(stop["frac"], stop["mse_rel"], marker="o", markersize=4.5, markerfacecolor="white",
                markeredgecolor=st["color"], markeredgewidth=1.1, linestyle="none", zorder=4)
        # exact: below its circle; sketch: left of and below its circle, clear of both curves
        off, ha, va = ((0, -9), "center", "top") if key == "exact" else ((-5, -5), "right", "top")
        ax.annotate(f"{stop['m']} tokens", xy=(stop["frac"], stop["mse_rel"]), xytext=off, textcoords="offset points",
                    ha=ha, va=va, fontsize=6, color=INK)
    handles.append(plt.Line2D([], [], marker="o", markersize=4.5, markerfacecolor="white", markeredgecolor="#333333",
                              markeredgewidth=1.1, linestyle="none"))
    labels = [STYLE["exact"]["label"], STYLE["sketch"]["label"], "minimum"]

    x0 = float(rows[0]["frac_set_to_r"])
    ax.set_xscale("log")
    ax.set_xlim(x0, 1.0)
    ax.set_ylim(0.3, 1.85)
    ax.set_xlabel(r"Fraction of $L$ with $\lambda_i=r_i$ (sorted by $\widehat{w}_i(1-r_i)/r_i$)")
    ax.set_ylabel(r"MSE at this fraction / MSE of unbiased", fontsize=7.5)
    ax.text(x0 * 1.3, 1.025, r"all $\lambda_i=1$ (unbiased)", fontsize=6.5, va="bottom", color=INK)
    ax.annotate(r"all $\lambda_i=r_i$", xy=(1.0, row["mse_evict_rel"]), xytext=(0.30, 1.72), fontsize=6.5,
                ha="right", va="center", arrowprops=dict(arrowstyle="-", linewidth=0.5, color="#555555"))
    ax.grid(True, axis="y", color=GRID, linewidth=0.4, linestyle=":")
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=7, top=True, right=True, which="both")
    ax.legend(handles, labels, loc="upper left", frameon=False, fontsize=6.5, handlelength=2.4)
    out = here / "lambda_binary_shrinkage"
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}.png and .pdf")
    print(json.dumps(row["jump_rule"], indent=1))


if __name__ == "__main__":
    main()
