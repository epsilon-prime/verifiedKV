#!/usr/bin/env python3
"""Systems figures (Section 4.4) from the vLLM runs: Qwen3-4B, RULER-32K, 16x, Fast KVzip scores.

Input (next to this script): ruler32k_qwen3-4b_16x_vllm.csv, a copy of results/systems/ruler32k_qwen3-4b_16x_vllm.csv, 50 requests per task.
Columns: method (dense_recall | vkv | fastkvzip_evict), eps, delta, per-task and mean score, ttft_s, tpot_ms,
decode_tok_s, e2e_tok_s, latency_s, fetch_GB_per_request.

Output (one column wide): quality_throughput_latency.{png,pdf}, RULER-32K score on y against decode throughput
(tokens/s, top x axis, dark blue) and end-to-end latency per request (s, bottom x axis reversed so faster is to the right,
light blue). Dense (full cache recalled from host every step) as a diamond on each curve and a dotted line at its score;
VerifiedKV as its eps0 sweep at delta = 0.1, eps0 = 0.01 / 0.05 / 0.5 labelled, the default (0.05) circled. Fast KVzip is left out (score 20,
far below the plotted range); the text gives its numbers.

Usage (from the paper root, any env with pandas + matplotlib):
  python figures/systems/make_systems.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
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
    "pdf.fonttype": 42,
})

INK, GRID = "#000000", "#d9d9d9"
# Throughput (top axis) dark blue, latency (bottom axis) light blue; Dense diamond black on the throughput curve, gray
# on the latency curve.
C_THR, C_LAT = "#2f5d94", "#7fabd9"
MAIN_DELTA, DEFAULT_EPS = 0.1, 0.05
# eps0 labels per curve: offset in points, ha, va
LABELS = {
    "decode_tok_s": {0.01: ((6, 5), "left", "bottom"), 0.05: ((-5, -5), "right", "top"),
                     0.5: ((-4, 0), "right", "center")},
    "latency_s": {0.01: ((8, -6), "left", "top"), 0.05: ((5, 18), "left", "bottom"),
                  0.5: ((2, -3), "right", "top")},
}
LEADER = {("decode_tok_s", 0.01), ("latency_s", 0.01), ("latency_s", 0.05)}  # labels set off from crowded points,
# joined to them by a thin line


def plot(df: pd.DataFrame, out: Path) -> None:
    """Score on y; latency on the bottom x axis (reversed, so faster is to the right) and decode throughput on the top
    x axis. Each axis gets its own curve in its colour; Dense is a diamond on each curve and a dotted line at its
    score."""
    dense = df[df["method"] == "dense_recall"].iloc[0]
    v = df[(df["method"] == "vkv") & (df["delta"] == MAIN_DELTA)].sort_values("eps")
    fig, ax_lat = plt.subplots(figsize=(3.25, 1.5))
    ax_thr = ax_lat.twiny()
    ax_lat.set_xlim(20.5, 6.5)  # reversed: lower latency to the right
    ax_thr.set_xlim(5.5, 29)
    ax_lat.set_ylim(80.7, 89.6)
    ax_lat.set_yticks([82, 84, 86, 88])
    ax_lat.axhline(dense["score"], color=INK, linewidth=0.6, linestyle=(0, (1, 1.5)), zorder=1)
    ax_lat.text(6.7, dense["score"] - 0.15, f"Dense ({dense['score']:.0f})", fontsize=6.5, ha="right", va="top")
    for ax, x, color, marker, c_dense in ((ax_thr, "decode_tok_s", C_THR, "o", INK),
                                          (ax_lat, "latency_s", C_LAT, "s", "#9a9a9a")):
        ax.plot(v[x], v["score"], color=color, linewidth=0.9, marker=marker, markersize=2.2, zorder=3)
        ax.plot(dense[x], dense["score"], marker="D", markersize=3.6, markerfacecolor=c_dense, markeredgecolor=c_dense,
                markeredgewidth=0.6, linestyle="none", zorder=4)
        d = v[v["eps"] == DEFAULT_EPS].iloc[0]
        ax.plot(d[x], d["score"], marker="o", markersize=6.0, markerfacecolor="none", markeredgecolor=INK,
                markeredgewidth=0.6, linestyle="none", zorder=5)
        for eps, (off, ha, va) in LABELS[x].items():
            r = v[v["eps"] == eps].iloc[0]
            ax.annotate(rf"$\varepsilon_0={eps:g}$", xy=(r[x], r["score"]), xytext=off, textcoords="offset points",
                        ha=ha, va=va, fontsize=6.3, color=color,
                        arrowprops=dict(arrowstyle="-", linewidth=0.4, color=color, shrinkA=0, shrinkB=2)
                        if (x, eps) in LEADER else None)
    ax_lat.set_xlabel(r"End-to-end latency per request (s)")
    ax_thr.set_xlabel(r"Decode throughput (tokens/s)")
    ax_lat.set_ylabel("RULER-32K score")
    ax_lat.grid(True, axis="y", color=GRID, linewidth=0.4, linestyle=":")
    ax_lat.set_axisbelow(True)
    for ax in (ax_lat, ax_thr):
        ax.tick_params(labelsize=7)
    ax_lat.tick_params(right=True)
    hs = [plt.Line2D([], [], color=C_THR, linewidth=0.9, marker="o", markersize=2.2),
          plt.Line2D([], [], color=C_LAT, linewidth=0.9, marker="s", markersize=2.2),
          plt.Line2D([], [], color="none", marker="o", markersize=6.0, markerfacecolor="none", markeredgecolor=INK,
                     markeredgewidth=0.6),
          plt.Line2D([], [], color="none", marker="D", markersize=3.6, markerfacecolor=INK, markeredgecolor=INK),
          plt.Line2D([], [], color="none", marker="D", markersize=3.6, markerfacecolor="#9a9a9a",
                     markeredgecolor="#9a9a9a")]
    labels = ["VerifiedKV, throughput (top axis)", "VerifiedKV, latency (bottom axis)",
              r"default $\varepsilon_0=0.05$", "Dense (offload), throughput", "Dense (offload), latency"]
    fig.legend(hs, labels, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=False, fontsize=6,
               handlelength=1.6, columnspacing=1.0, handletextpad=0.4)
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}.png and .pdf")


def main() -> None:
    here = Path(__file__).resolve().parent
    df = pd.read_csv(here / "ruler32k_qwen3-4b_16x_vllm.csv")
    plot(df, here / "quality_throughput_latency")
    cols = ["method", "eps", "delta", "score", "ttft_s", "tpot_ms", "decode_tok_s", "latency_s", "fetch_GB_per_request"]
    print(df[df["delta"].isna() | (df["delta"] == MAIN_DELTA)][cols].to_string(index=False))


if __name__ == "__main__":
    main()
