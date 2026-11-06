#!/usr/bin/env python3
"""Error-adherence ablation: does VerifiedKV keep P(relative error > eps0) <= delta per attention decision?

Data: experiments/adherence, VerifiedKV (KVzip scores) at 16x on Qwen3-4B, RULER-32K
(cwe, niah_multikey_3, niah_multivalue, qa_1; contexts 152-153), eps0 in {0.01, 0.05, 0.1, 0.3} x delta in {0.05, 0.1},
run twice with the same sampled tokens: planning on the deployed 2-bit sketch (csv/) and on the exact attention
weights (csv_perfect/, an ideal sketch). Every attention decision (query row x layer, question pass and decode steps)
logs its realised relative error of the denominator after fetching, before fetching, and whether it fetched.

    python make_adherence.py --source experiments/adherence   # rebuild the two extracts, then plot
    python make_adherence.py                                               # plot from the extracts

Extracts (next to this script):
  adherence_by_setting.csv  sketch, eps0, delta, decisions, fetch (share of decisions that fetch), fail (share with
                            error > eps0 after fetching), fail_nofetch (the same estimator without fetching), read_pct
                            (KV read from host per forward, % of the cache)
  adherence_error_hist.csv  at eps0 = 0.05, delta = 0.1: histogram of log10(relative error) after fetching ("post")
                            and without fetching ("nofetch") for both sketches, summed over all decisions
Outputs: adherence_bars.{png,pdf} (used in the paper): failure rate per eps0 at delta = 0.1 for no fetch, the 2-bit sketch
and the perfect sketch, against the delta line. adherence.{png,pdf} (kept, unused), one column wide: the share of decisions whose relative error exceeds a multiple of eps0
(at eps0 = 0.05, delta = 0.1) for the exact sketch, the 2-bit sketch and no fetch. (Earlier three-panel version: failure
rate vs eps0 and fetching vs eps0 as well; panel_fail / panel_fetch kept below, unused.)
"""

from __future__ import annotations

import argparse
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
# Exact sketch = ideal reference (gray, as in the binary-shrinkage figure); 2-bit sketch = deployed (VerifiedKV blue).
C_EXACT, C_2BIT, C_NOFETCH = "#6b6b6b", "#4a78b0", "#b5b5b5"
EPS = [0.01, 0.05, 0.1, 0.3]
DEFAULT = (0.05, 0.1)


def extract(src: Path, here: Path) -> None:
    rows = []
    hists = []
    for sketch, sub in (("2bit", "csv"), ("exact", "csv_perfect")):
        r = pd.read_csv(src / sub / "rows.csv")
        r = r[r["method"] == "vkv_final"].copy()
        r["n"] = r["q_decisions"] + r["d_decisions"]
        for (e, d), g in r.groupby(["eps0", "delta"]):
            w = g["n"] / g["n"].sum()  # decision-weighted over the 8 answers
            rows.append(dict(sketch=sketch, eps0=e, delta=d, answers=len(g), decisions=int(g["n"].sum()),
                             fetch=(w * g["t_fetch_rate"]).sum(), fail=(w * g["t_fail_rate"]).sum(),
                             fail_nofetch=(w * g["t_fail_T0_rate"]).sum(), read_pct=g["t_read_pct"].mean()))
        h = pd.read_csv(src / sub / "histograms.csv")
        h = h[(h["eps0"] == DEFAULT[0]) & (h["delta"] == DEFAULT[1]) & h["hist"].isin(["post", "T0"])]
        h = h.groupby(["hist", "bin", "lo", "hi"], as_index=False)["count"].sum()
        h["hist"] = h["hist"].map({"post": "post", "T0": "nofetch"})
        h.insert(0, "sketch", sketch)
        hists.append(h)
    pd.DataFrame(rows).to_csv(here / "adherence_by_setting.csv", index=False)
    pd.concat(hists).to_csv(here / "adherence_error_hist.csv", index=False)
    print("Wrote adherence_by_setting.csv and adherence_error_hist.csv")


def style(ax) -> None:
    ax.grid(True, color=GRID, linewidth=0.4, linestyle=":")
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=7, top=True, right=True, which="both")


def eps_axis(ax) -> None:
    ax.set_xscale("log")
    ax.set_xticks(EPS)
    ax.set_xticklabels([f"{e:g}" for e in EPS])
    ax.minorticks_off()
    ax.set_xlim(0.008, 0.38)
    ax.set_xlabel(r"Tolerance $\varepsilon_0$")


def panel_fail(ax, s: pd.DataFrame) -> None:
    for d, val in ((0.1, "0.1"), (0.05, "0.05")):
        ax.axhline(100 * d, color=INK, linewidth=0.6, linestyle=(0, (1, 1.5)), zorder=1)
        ax.text(0.0095, 100 * d * 1.08, rf"$\delta={val}$", fontsize=6.3, ha="left", va="bottom")
    nf = s[(s["sketch"] == "2bit") & (s["delta"] == 0.1)].sort_values("eps0")
    ax.plot(nf["eps0"], 100 * nf["fail_nofetch"], color=C_NOFETCH, linewidth=1.1, marker="^", markersize=3,
            label="no fetch")
    for sk, color, name in (("2bit", C_2BIT, "2-bit sketch"), ("exact", C_EXACT, "exact sketch")):
        for d, ls in ((0.1, "-"), (0.05, (0, (3.5, 1.5)))):
            g = s[(s["sketch"] == sk) & (s["delta"] == d)].sort_values("eps0")
            ax.plot(g["eps0"], 100 * g["fail"], color=color, linestyle=ls, linewidth=1.3, marker="o", markersize=3,
                    label=f"{name}" if d == 0.1 else None)
    eps_axis(ax)
    ax.set_yscale("log")
    ax.set_ylim(0.1, 100)
    ax.set_yticks([0.1, 1, 10, 100])
    ax.set_yticklabels(["0.1", "1", "10", "100"])
    ax.set_ylabel(r"Decisions with error $>\varepsilon_0$ (%)")
    ax.set_title("(a) Failure rate", fontsize=8, pad=4)
    style(ax)


def panel_ccdf(ax, h: pd.DataFrame) -> None:
    eps0, delta = DEFAULT
    for sk, hist, color, ls, label in (("2bit", "nofetch", C_NOFETCH, "-", "no fetch"),
                                       ("2bit", "post", C_2BIT, "-", "2-bit sketch"),
                                       ("exact", "post", C_EXACT, "-", "exact sketch")):
        g = h[(h["sketch"] == sk) & (h["hist"] == hist)].sort_values("bin")
        tot = g["count"].sum()
        # P(error > 10^lo) at each finite bin edge, as error / eps0
        lo = g["lo"].to_numpy()
        above = g["count"].to_numpy()[::-1].cumsum()[::-1] / tot
        ok = np.isfinite(lo)
        x = 10 ** lo[ok] / eps0
        ax.plot(x, above[ok], color=color, linestyle=ls, linewidth=1.3, label=label)
    ax.axvline(1.0, color=INK, linewidth=0.6, zorder=1)
    ax.axhline(delta, color=INK, linewidth=0.6, linestyle=(0, (1, 1.5)), zorder=1)
    ax.text(1.15, 1.6, r"error $=\varepsilon_0$", fontsize=6.3, ha="left", va="top")
    ax.text(0.0012, delta * 1.15, rf"$\delta={delta}$", fontsize=6.3, ha="left", va="bottom")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-3, 30)
    ax.set_ylim(1e-4, 2)
    ax.set_xticks([1e-3, 1e-2, 1e-1, 1, 10])
    ax.set_xticklabels(["0.001", "0.01", "0.1", "1", "10"])
    ax.set_yticks([1e-4, 1e-3, 1e-2, 1e-1, 1])
    ax.set_xlabel(r"Error / $\varepsilon_0$")
    ax.set_ylabel(r"Share of decisions above error / $\varepsilon_0$", fontsize=7)
    ax.set_title(rf"Error at $\varepsilon_0={eps0}$, $\delta={delta}$", fontsize=8, pad=4)
    style(ax)


def panel_fetch(ax, s: pd.DataFrame) -> None:
    for sk, ls in (("2bit", "-"), ("exact", (0, (3.5, 1.5)))):
        g = s[(s["sketch"] == sk) & (s["delta"] == 0.1)].sort_values("eps0")
        ax.plot(g["eps0"], 100 * g["fetch"], color=C_2BIT if sk == "2bit" else C_EXACT, linestyle=ls, linewidth=1.3,
                marker="o", markersize=3)
        ax.plot(g["eps0"], g["read_pct"], color=C_2BIT if sk == "2bit" else C_EXACT, linestyle=ls, linewidth=1.3,
                marker="s", markersize=3)
    g = s[(s["sketch"] == "2bit") & (s["delta"] == 0.1)].sort_values("eps0")
    ax.text(0.25, 22, "decisions that fetch", fontsize=6.3, ha="right", va="top")
    ax.text(0.03, 7, "KV read per step", fontsize=6.3, ha="center", va="top")
    eps_axis(ax)
    ax.set_yscale("log")
    ax.set_ylim(0.4, 150)
    ax.set_yticks([1, 10, 100])
    ax.set_yticklabels(["1", "10", "100"])
    ax.set_ylabel("Share (%)")
    ax.set_title(r"(c) Fetching ($\delta=0.1$)", fontsize=8, pad=4)
    style(ax)


def plot_bars(s: pd.DataFrame, out: Path) -> None:
    """Failure rate per tolerance at delta = 0.1: no fetch, 2-bit sketch, perfect sketch, against the delta line."""
    s = s[s["delta"] == 0.1]
    two = s[s["sketch"] == "2bit"].set_index("eps0").loc[EPS]
    exact = s[s["sketch"] == "exact"].set_index("eps0").loc[EPS]
    series = ((100 * two["fail_nofetch"], C_NOFETCH, "no fetch"),
              (100 * two["fail"], C_2BIT, r"$\mathcal{A}$ = 2-bit sketch"),
              (100 * exact["fail"], C_EXACT, r"$\mathcal{A}$ = perfect sketch"))
    fig, ax = plt.subplots(figsize=(3.25, 1.75))
    w = 0.27
    for k, (ys, color, label) in enumerate(series):
        xs = [g + (k - 1) * w for g in range(len(EPS))]
        ax.bar(xs, ys, width=w, color=color, edgecolor="white", linewidth=0.4, zorder=2, label=label)
        for x, y in zip(xs, ys):
            ax.text(x, y + 1.5, f"{y:.1f}" if y < 10 else f"{y:.0f}", ha="center", va="bottom", fontsize=5.5, zorder=4,
                    bbox=dict(boxstyle="square,pad=0.05", facecolor="white", edgecolor="none"))
    ax.axhline(10, color=INK, linewidth=0.8, linestyle=(0, (3, 1.6)), zorder=3)
    ax.text(len(EPS) - 0.52, 11.5, r"$\delta=10\%$", ha="right", va="bottom", fontsize=6.5, zorder=4,
            bbox=dict(boxstyle="square,pad=0.05", facecolor="white", edgecolor="none"))
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_xticks(range(len(EPS)))
    ax.set_xticklabels([f"{e:g}" for e in EPS])
    ax.set_xlim(-0.5, len(EPS) - 0.5)
    ax.set_xlabel(r"Tolerance $\varepsilon_0$")
    ax.set_ylabel(r"Decisions with error $>\varepsilon_0$ (%)", fontsize=7.5)
    ax.grid(True, axis="y", color=GRID, linewidth=0.4, linestyle=":")
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=7, right=True, which="both")
    ax.tick_params(axis="x", bottom=False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, fontsize=7, handlelength=1.6,
              columnspacing=1.0, handletextpad=0.4)
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}.png and .pdf")


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, help="experiments/adherence: rebuild the extracts from it")
    args = ap.parse_args()
    if args.source:
        extract(args.source.expanduser(), here)
    s = pd.read_csv(here / "adherence_by_setting.csv")
    h = pd.read_csv(here / "adherence_error_hist.csv")
    plot_bars(s, here / "adherence_bars")
    fig, ax = plt.subplots(figsize=(3.25, 1.9))
    panel_ccdf(ax, h)
    ax.set_title("")  # eps0 and delta are given in the caption
    hs = [plt.Line2D([], [], color=c, linewidth=1.3) for c in (C_NOFETCH, C_2BIT, C_EXACT)]
    fig.legend(hs, ["no fetch", "2-bit sketch", "exact sketch"], loc="lower center", ncol=3,
               frameon=False, fontsize=7, bbox_to_anchor=(0.55, 0.85), handlelength=1.8, columnspacing=1.0)
    out = here / "adherence"
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}.png and .pdf")
    print(s.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
