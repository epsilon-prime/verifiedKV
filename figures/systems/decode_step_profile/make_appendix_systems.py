#!/usr/bin/env python3
"""Appendix systems figures (vLLM, Qwen3-4B, RULER, 16x, Fast KVzip scores), in the paper's style.

Inputs (next to this script, from the systems runs): batch_scaling.csv (32K), batch_scaling_128k.csv, eps_sweep_timing.csv,
step_time_breakdown.csv, pcie_link.csv. All VerifiedKV runs here use delta = 0.05.

Outputs ({png,pdf}):
  sys_batch_scaling   2x3: rows 32K / 128K; decode throughput, TPOT (median) and TTFT (median) vs batch size, log y
  sys_eps_sweep       decode throughput and TPOT vs eps0 at batch 1, 32K (solid) and 128K (dashed); Dense (offload) lines
  sys_step_breakdown  GPU kernel time per decode step, stacked by phase, 32K, batch 1
  sys_pcie            host-to-GPU PCIe link utilization during decode, 32K, batch 1

Usage (from the paper root): python figures/systems/decode_step_profile/make_appendix_systems.py
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

INK, GRID = "#000000", "#d9d9d9"
# Same palette as Figures 3 and 4: Dense (offload) black, Dense (resident) dark gray, Fast KVzip light gray,
# VerifiedKV eps0 = 0.05 dark blue, eps0 = 0.5 light blue.
C_OFF, C_RES, C_FKZ, C_V05, C_V5 = "#000000", "#6b6b6b", "#b5b5b5", "#2f5d94", "#7fabd9"


def name_color(method: str, eps) -> tuple[str, str, str]:
    if method == "dense_recall":
        return "Dense (offload)", C_OFF, "s"
    if method == "dense":
        return "Dense (resident)", C_RES, "D"
    if method == "fastkvzip_evict":
        return "Fast KVzip", C_FKZ, "^"
    return rf"VerifiedKV ($\varepsilon_0={float(eps):g}$)", C_V05 if float(eps) < 0.2 else C_V5, "o"


def style(ax) -> None:
    ax.grid(True, color=GRID, linewidth=0.4, linestyle=":")
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=7, top=True, right=True, which="both")


def save(fig, out: Path) -> None:
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}.png and .pdf")


def batch_scaling(here: Path) -> None:
    metrics = [("decode_tok_s", "Decode throughput (tokens/s)"), ("tpot_ms_median", "TPOT, median (ms)"),
               ("ttft_s_median", "TTFT, median (s)")]
    fig, axes = plt.subplots(2, 3, figsize=(6.75, 3.3))
    fig.subplots_adjust(wspace=0.38, hspace=0.32)
    for i, (fname, ctx) in enumerate((("batch_scaling.csv", "32K"), ("batch_scaling_128k.csv", "128K"))):
        df = pd.read_csv(here / fname)
        for j, (col, ylabel) in enumerate(metrics):
            ax = axes[i, j]
            for (method, eps), g in df.groupby(["method", "eps"], dropna=False, sort=False):
                label, color, marker = name_color(method, eps)
                g = g.sort_values("batch")
                ax.plot(g["batch"], g[col], color=color, marker=marker, markersize=3, linewidth=1.2, label=label)
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.set_xticks([1, 2, 4, 8])
            ax.set_xticklabels(["1", "2", "4", "8"])
            ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
            ax.set_ylabel(ylabel, fontsize=7.5)
            if i == 1:
                ax.set_xlabel("Batch size", fontsize=7.5)
            if j == 0:
                ax.text(-0.42, 0.5, f"RULER-{ctx}", transform=ax.transAxes, rotation=90, ha="center", va="center",
                        fontsize=8.5)
            style(ax)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, 0.94))
    save(fig, here / "sys_batch_scaling")


def eps_sweep(here: Path) -> None:
    df = pd.read_csv(here / "eps_sweep_timing.csv")
    fig, axes = plt.subplots(1, 2, figsize=(3.25, 1.6))
    fig.subplots_adjust(wspace=0.5)
    for ax, (col, ylabel) in zip(axes, (("decode_tok_s", "Decode throughput (tokens/s)"), ("tpot_ms", "TPOT (ms)"))):
        for ctx, ls in (("32k", "-"), ("128k", (0, (3.5, 1.5)))):
            g = df[df["context"] == ctx]
            v = g[g["method"] == "vkv"].sort_values("eps")
            d = g[g["method"] == "dense_recall"].iloc[0]
            ax.plot(v["eps"], v[col], color=C_V05, linestyle=ls, linewidth=1.2, marker="o", markersize=2.2)
            ax.axhline(d[col], color=C_OFF, linestyle=ls, linewidth=0.8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks([0.01, 0.05, 0.5])
        ax.set_xticklabels(["0.01", "0.05", "0.5"])
        ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
        ax.set_xlabel(r"Tolerance $\varepsilon_0$", fontsize=7.5)
        ax.set_ylabel(ylabel, fontsize=7.5)
        style(ax)
    hs = [plt.Line2D([], [], color=C_V05, linewidth=1.2), plt.Line2D([], [], color=C_OFF, linewidth=0.8),
          plt.Line2D([], [], color="#888888", linewidth=1.0), plt.Line2D([], [], color="#888888", linewidth=1.0,
                                                                    linestyle=(0, (3.5, 1.5)))]
    fig.legend(hs, ["VerifiedKV", "Dense (offload)", "32K", "128K"], loc="lower center", ncol=4, frameon=False,
               fontsize=7, handlelength=1.8, columnspacing=0.9, bbox_to_anchor=(0.5, 0.93))
    save(fig, here / "sys_eps_sweep")


def bar_labels(df: pd.DataFrame) -> list[str]:
    return [name_color(m, e)[0] for m, e in zip(df["method"], df["eps"])]


def step_breakdown(here: Path) -> None:
    df = pd.read_csv(here / "step_time_breakdown.csv")
    phases = [("step_ms_decide", "decide (certificate)", C_V05), ("step_ms_fetch", "fetch", C_OFF),
              ("step_ms_attention", "attention", "#9a9a9a"), ("step_ms_gemm", "GEMMs", "#c8c8c8"),
              ("step_ms_other", "other", "#e4e4e4")]
    y = np.arange(len(df))[::-1]
    fig, ax = plt.subplots(figsize=(3.25, 1.75))
    left = np.zeros(len(df))
    for col, label, color in phases:
        ax.barh(y, df[col], left=left, height=0.62, color=color, edgecolor="white", linewidth=0.3, label=label)
        left += df[col].to_numpy()
    for yi, total in zip(y, left):
        ax.text(total + 2, yi, f"{total:.0f} ms", va="center", fontsize=6.5)
    ax.set_yticks(y)
    ax.set_yticklabels(bar_labels(df), fontsize=7)
    ax.set_xlim(0, 175)
    ax.set_xlabel("GPU kernel time per decode step (ms)", fontsize=7.5)
    ax.grid(True, axis="x", color=GRID, linewidth=0.4, linestyle=":")
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=7, top=True, which="both")
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="lower center", bbox_to_anchor=(0.4, 1.0), ncol=5, frameon=False, fontsize=6.5, handlelength=1.2,
              columnspacing=0.7, handletextpad=0.4)
    save(fig, here / "sys_step_breakdown")


def pcie(here: Path) -> None:
    df = pd.read_csv(here / "pcie_link.csv")
    y = np.arange(len(df))[::-1]
    colors = [C_V05 if m == "vkv" else name_color(m, e)[1] for m, e in zip(df["method"], df["eps"])]  # one blue for VerifiedKV
    fig, ax = plt.subplots(figsize=(3.25, 1.75))
    ax.barh(y, df["pcie_link_util_pct"], height=0.62, color=colors, edgecolor="white", linewidth=0.3)
    for yi, pct, gbs in zip(y, df["pcie_link_util_pct"], df["pcie_h2d_GBs"]):
        inside = pct > 60
        ax.text(pct - 1.5 if inside else pct + 1.5, yi, f"{pct:.1f}% ({gbs:.1f} GB/s)", va="center",
                ha="right" if inside else "left", fontsize=6.5, color="white" if inside else INK)
    ax.axvline(100, color=INK, linewidth=0.6, linestyle=(0, (1, 1.5)))
    ax.set_yticks(y)
    ax.set_yticklabels(bar_labels(df), fontsize=7)
    ax.set_xlim(0, 105)
    ax.set_xlabel(f"Host-to-GPU PCIe utilization (% of {df['link_measured_GBs'].iloc[0]:.1f} GB/s)", fontsize=7.5)
    ax.grid(True, axis="x", color=GRID, linewidth=0.4, linestyle=":")
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=7, top=True, which="both")
    ax.tick_params(axis="y", length=0)
    save(fig, here / "sys_pcie")


def main() -> None:
    here = Path(__file__).resolve().parent
    batch_scaling(here)
    eps_sweep(here)
    step_breakdown(here)
    pcie(here)


if __name__ == "__main__":
    import os
    import sys

    main()
    sys.stdout.flush()
    os._exit(0)  # the interpreter can hang at shutdown on the cluster's shared filesystem
