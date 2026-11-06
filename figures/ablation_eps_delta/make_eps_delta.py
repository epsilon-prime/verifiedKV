#!/usr/bin/env python3
"""eps x delta ablation figure: how fetching and quality change with the certificate's tolerance and failure probability.

Input: eps_delta_rows.csv, a copy of results/eps_delta/eps_delta_rows.csv (commit 15d5169):
one row per (model, context, arm, task, RULER row). VerifiedKV (KVzip) at 16x, 2-bit sketch, 128-token shortlist;
eps in {0.01, 0.05, 0.1, 0.2, 0.3} x delta in {0.02, 0.05, 0.1}, plus Dense. Only rows every arm finished
(in_cells = 1) are used, so every arm is compared on the same contexts.

Output (next to this script): eps_delta_qwen3_32k.{png,pdf}, Qwen3-4B on RULER-32K (120 contexts, 30 per task):
  (a) share of the KV cache read from host memory per decode step vs eps, one line per delta (log-log);
  (b) score change vs Dense on the same contexts (mean, +-1 standard error) vs eps.
It also prints the numbers used in the paper text, including the other two models at the running configuration.

Usage (from the paper root, any env with pandas + matplotlib):
  python figures/ablation_eps_delta/make_eps_delta.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
    "ytick.minor.size": 1.5,
    "pdf.fonttype": 42,
})

EPS = [0.01, 0.05, 0.1, 0.2, 0.3]
DELTAS = [0.02, 0.05, 0.1]
RUNNING = (0.05, 0.1)  # eps0, delta of the main results
# delta is ordered: three steps of the VerifiedKV blue, darker = stricter (smaller delta); marker shape repeats it.
DCOLOR = {0.02: "#1f3f6e", 0.05: "#4a78b0", 0.1: "#7fa6d6"}
DMARKER = {0.02: "s", 0.05: "o", 0.1: "^"}
INK = "#000000"
GRID = "#d9d9d9"
MODELS = {"Qwen3": "Qwen3-4B", "Ministral-3": "Ministral-3-8B", "Qwen3.5": "Qwen3.5-9B"}


def paired(rows: pd.DataFrame) -> pd.DataFrame:
    """Per (model, eps, delta): mean paired score change vs Dense on the same contexts, its SE, and the traffic."""
    dense = rows[rows["method"] == "dense"][["model", "task", "row", "score"]].rename(columns={"score": "dense"})
    v = rows[rows["method"] != "dense"].merge(dense, on=["model", "task", "row"])
    v["d"] = v["score"] - v["dense"]
    return v.groupby(["model", "eps", "delta"]).agg(
        n=("d", "size"), diff=("d", "mean"), se=("d", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
        read=("read_pct", "mean"), gb=("gb_per_answer", "mean")).reset_index()


def plot(g: pd.DataFrame, out: Path) -> None:
    fig, (ax_r, ax_q) = plt.subplots(1, 2, figsize=(3.4, 1.75))
    fig.subplots_adjust(wspace=0.45, left=0.11, right=0.99, bottom=0.22, top=0.78)
    off = {0.02: 0.94, 0.05: 1.0, 0.1: 1.06}  # small multiplicative x-offsets so error bars do not overlap
    handles = []
    for dl in DELTAS:
        s = g[g["delta"] == dl].sort_values("eps")
        kw = dict(color=DCOLOR[dl], marker=DMARKER[dl], markersize=3.2, markeredgewidth=0.6, linewidth=1.0)
        (h,) = ax_r.plot(s["eps"], s["read"], **kw)
        handles.append(h)
        ax_q.errorbar(s["eps"] * off[dl], s["diff"], yerr=s["se"], capsize=1.5, elinewidth=0.6, **kw)
    # Running configuration of the main results.
    run = g[(g["eps"] == RUNNING[0]) & (g["delta"] == RUNNING[1])].iloc[0]
    for ax, y in ((ax_r, run["read"]), (ax_q, run["diff"])):
        x = RUNNING[0] * (off[RUNNING[1]] if ax is ax_q else 1.0)
        ax.plot(x, y, marker="o", markersize=7.5, markerfacecolor="none", markeredgecolor=INK, markeredgewidth=0.7,
                linestyle="none", zorder=5)
    ax_r.annotate("default", xy=(RUNNING[0], run["read"]), xytext=(0.016, 3.0), fontsize=6.5,
                  arrowprops=dict(arrowstyle="-", linewidth=0.5, color=INK))

    ax_r.set_yscale("log")
    ax_r.set_ylim(0.5, 60)
    ax_r.set_yticks([1, 3, 10, 30])
    ax_r.set_yticklabels(["1", "3", "10", "30"])
    ax_r.set_ylabel("KV read from host (%)", fontsize=7.5)
    ax_q.axhline(0, color=INK, linewidth=0.7, zorder=1)
    ax_q.set_ylim(-11, 3)
    ax_q.set_ylabel("Score change vs Dense", fontsize=7.5)
    for ax, tag in ((ax_r, "(a) Host reads per decode step"), (ax_q, "(b) Quality")):
        ax.set_xscale("log")
        ax.minorticks_off()
        ax.set_xticks(EPS)
        ax.set_xticklabels(["0.01", "0.05", "0.1", "", "0.3"])  # 0.2 keeps its tick; its label would hit 0.3's
        ax.set_xlim(0.0085, 0.36)
        ax.set_xlabel(r"Tolerance $\varepsilon_0$")
        ax.grid(True, axis="y", color=GRID, linewidth=0.4, linestyle=":")
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=7, top=True, right=True)
        ax.set_title(tag, fontsize=7.5, pad=3)
    fig.legend(handles, [rf"$\delta={d}$" for d in DELTAS], loc="upper center", ncol=3, frameon=False, fontsize=7,
               handlelength=2.0, columnspacing=1.0, bbox_to_anchor=(0.55, 0.995))
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out.with_suffix('.png')} and .pdf")


def main() -> None:
    here = Path(__file__).resolve().parent
    rows = pd.read_csv(here / "eps_delta_rows.csv")
    rows = rows[(rows["context_k"] == 32) & (rows["in_cells"] == 1)]
    g = paired(rows)
    assert (g["n"] == 120).all(), g[g["n"] != 120]
    plot(g[g["model"] == "Qwen3"], here / "eps_delta_qwen3_32k.png")
    pd.set_option("display.width", 160)
    print("Qwen3-4B, RULER-32K:")
    print(g[g["model"] == "Qwen3"].drop(columns="model").round(2).to_string(index=False))
    print("All models at the running configuration (eps0 = 0.05, delta = 0.1):")
    run = g[(g["eps"] == RUNNING[0]) & (g["delta"] == RUNNING[1])].copy()
    run["model"] = run["model"].map(MODELS)
    print(run.round(2).to_string(index=False))
    stats = rows[rows["method"] != "dense"].groupby("model")[["ctx_tokens", "decode_steps"]].median()
    print("Median context tokens / decode steps:")
    print(stats.to_string())


if __name__ == "__main__":
    main()
