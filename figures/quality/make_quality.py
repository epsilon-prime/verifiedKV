#!/usr/bin/env python3
"""Quality figure and per-task tables from the final results.

Input: final_scores.csv, a copy of results/quality/all_models.csv
(model,dataset,context_k,rate,method,task,score,n,label,cell). RULER = 4 tasks x 50 rows; LOFT = 5 tasks x the
first 25 test questions.

Outputs (next to this script):
  quality_vs_compression_relative.{png,pdf}  4x3: rows RULER/LOFT x 32K/128K, columns models; y = task mean as % of
                                             Dense; Dense's absolute task mean printed in each panel
  quality_vs_compression.{png,pdf}           the same with absolute task means (0-100), Dense as a line
  quality_bars.{png,pdf}                     4x3 grouped bars (absolute task means per rate and method), Dense dashed
  quality_per_task_main.tex                  per-task scores at 16x, 128K, all three models (main paper)
  quality_per_task_appendix.tex              one column-width table per model x benchmark x context (appendix)

Usage (from the paper root, any env with pandas + matplotlib):
  python figures/quality/make_quality.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

# The perfect-sketch arm (vkv_final_perfect) is left out.
# Labels follow the paper: vkv_nofetch_kvzip = PoissonComp (Poisson sample, unbiased 1/r weights, no fetch);
# vkv_final_* = VerifiedKV with KVzip / Fast KVzip scores.
METHODS = ["kvzip", "fastkvzip", "vkv_nofetch_kvzip", "vkv_final_kvzip", "vkv_final_fastkvzip"]
TABLE_METHODS = ["dense"] + METHODS
LABELS = {
    "dense": "Dense",
    "kvzip": "KVzip",
    "fastkvzip": "Fast KVzip",
    "vkv_nofetch_kvzip": "PoissonComp (KVzip)",
    "vkv_final_kvzip": "VerifiedKV (KVzip)",
    "vkv_final_fastkvzip": "VerifiedKV (Fast KVzip)",
}
SHORT = {
    "dense": "Dense",
    "kvzip": "KVz",
    "fastkvzip": "F-KVz",
    "vkv_nofetch_kvzip": "PC+K",
    "vkv_final_kvzip": "VKV+K",
    "vkv_final_fastkvzip": "VKV+F",
}

# Times to match the paper body (Times New Roman if installed, else its metric clone Nimbus Roman), STIX maths.
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

# Encoding (user's choice): eviction baselines gray (KVzip solid, Fast KVzip dashed); PoissonComp (no fetch) green;
# the two VerifiedKV arms muted blue (KVzip scores, solid) and light blue (Fast KVzip scores, dashed), so they stay apart when they
# overlap. Line style repeats the scorer (solid = KVzip, dashed = Fast KVzip); marker shape repeats the family.
INK = "#000000"
GRID = "#d9d9d9"
# (the two blues differ in lightness and line style; marker shape separates the families;
# gray is a deliberate neutral). Tables give the exact numbers.
COLOR = {"kvzip": "#6b6b6b", "fastkvzip": "#6b6b6b", "vkv_nofetch_kvzip": "#2bb04a",
         "vkv_final_kvzip": "#4a78b0", "vkv_final_fastkvzip": "#9fc4e6"}
MARKER = {"kvzip": "s", "fastkvzip": "s", "vkv_nofetch_kvzip": "^", "vkv_final_kvzip": "o", "vkv_final_fastkvzip": "o"}


def style(method: str) -> dict:
    c, vkv = COLOR[method], method.startswith("vkv_final")
    return dict(color=c, linestyle=(0, (3.5, 1.5)) if "fast" in method else "-", linewidth=1.5 if vkv else 1.1,
                marker=MARKER[method], markersize=3.4 if vkv else 2.4, markeredgewidth=0.7, markeredgecolor=c,
                markerfacecolor=c, zorder=4 if vkv else 3, clip_on=False)


MODELS = [("Qwen3", "Qwen3-4B"), ("Qwen3.5", "Qwen3.5-9B"), ("Ministral-3", "Ministral-3-8B")]
ROWS = [("RULER", 32), ("RULER", 128), ("LOFT", 32), ("LOFT", 128)]
RATES = [8, 16, 32]
TASKS = {
    "RULER": [("niah_mk3", "MK3"), ("niah_mv", "MV"), ("cwe", "CWE"), ("qa1", "QA1")],
    "LOFT": [("nq", "NQ"), ("hotpotqa", "HotpotQA"), ("musique", "MuSiQue"), ("qampari", "QAMPARI"),
             ("quest", "QUEST")],
}
KEY = ["model", "dataset", "context_k", "rate"]


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["method"].isin(TABLE_METHODS)]
    for dataset, tasks in TASKS.items():  # every (model, dataset, context, rate, method) must have every task
        sub = df[df["dataset"] == dataset]
        assert set(sub["task"]) == {t for t, _ in tasks}, (dataset, sorted(set(sub["task"])))
        n = sub.groupby(KEY + ["method"])["task"].nunique()
        assert (n == len(tasks)).all(), n[n != len(tasks)]
    assert len(df.groupby(KEY + ["method"])) == len(MODELS) * len(ROWS) * len(RATES) * len(TABLE_METHODS)
    return df


def plot_quality(df: pd.DataFrame, out: Path, relative: bool) -> None:
    """relative: y = % of Dense, Dense's score printed per panel; else y = absolute task mean, Dense drawn as a line."""
    agg = df.groupby(KEY + ["method"], as_index=False)["score"].mean()
    dense = agg[agg["method"] == "dense"][KEY + ["score"]].rename(columns={"score": "dense"})
    agg = agg.merge(dense, on=KEY)
    agg["rel"] = 100 * agg["score"] / agg["dense"]  # ratio of task means (a per-task ratio is undefined at Dense 0)

    # Drawn at print size (\textwidth = 6.75 in), so 8 pt here is 8 pt on the page.
    # Full page: about 8.1 in tall leaves room for the caption on the same float page.
    fig, axes = plt.subplots(len(ROWS), len(MODELS), figsize=(6.75, 8.1), sharex=True, sharey=True)
    fig.subplots_adjust(hspace=0.22, wspace=0.08, bottom=0.05, top=0.925, left=0.09, right=0.995)
    handles = {}
    for i, (dataset, ctx) in enumerate(ROWS):
        for j, (model, model_name) in enumerate(MODELS):
            ax = axes[i, j]
            panel = agg[(agg["model"] == model) & (agg["dataset"] == dataset) & (agg["context_k"] == ctx)]
            # Dense is the same at every rate: the 100% line, or its absolute task mean.
            ax.axhline(100 if relative else panel["dense"].iloc[0], color=INK, linewidth=0.8, zorder=2)
            for m in METHODS:
                s = panel[panel["method"] == m].sort_values("rate")
                (h,) = ax.plot(s["rate"], s["rel" if relative else "score"], **style(m))
                handles[m] = h
            if relative:  # Dense's absolute score sits just above the panel's top-right corner, clear of the data.
                ax.text(1.0, 1.02, f"Dense = {panel['dense'].iloc[0]:.1f}", transform=ax.transAxes, ha="right",
                        va="bottom", fontsize=6.5, color=INK)
            if i == 0:
                ax.set_title(model_name, fontsize=9, pad=11)
            if j == 0:
                ax.set_ylabel(f"{dataset} {ctx}K\n" + ("% of Dense" if relative else "Score"), fontsize=8)
            if i == len(ROWS) - 1:
                ax.set_xlabel("Compression rate", fontsize=8)
            ax.set_xscale("log", base=2)
            ax.set_xticks(RATES)
            ax.set_xticklabels([f"{r}$\\times$" for r in RATES])
            ax.minorticks_off()
            ax.set_xlim(6.6, 38.5)
            ax.set_ylim(0, 115 if relative else 105)
            ax.set_yticks([0, 25, 50, 75, 100])
            ax.grid(True, axis="y", color=GRID, linewidth=0.4, linestyle=":")
            ax.set_axisbelow(True)
            ax.tick_params(labelsize=7, top=True, right=True)

    dense_handle = plt.Line2D([], [], color=INK, linewidth=0.8)
    order = ["kvzip", "fastkvzip", "vkv_nofetch_kvzip", "vkv_final_kvzip", "vkv_final_fastkvzip"]
    fig.legend([handles[m] for m in order] + [dense_handle], [LABELS[m] for m in order] + ["Dense (100%)" if relative else "Dense"],
               loc="lower center", ncol=6, fontsize=7.5, handlelength=2.2, columnspacing=0.8, handletextpad=0.5,
               bbox_to_anchor=(0.54, 0.955), frameon=False)
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out.with_suffix('.png')} and .pdf")


def write_main_table(df: pd.DataFrame, out: Path, rate: int = 16, ctx: int = 128) -> None:
    """Two stacked parts (RULER, then LOFT), each scaled to \\textwidth: task rows + mean; a column group per model."""
    sub = df[(df["rate"] == rate) & (df["context_k"] == ctx)]
    score = sub.set_index(["model", "dataset", "task", "method"])["score"]
    k = len(TABLE_METHODS)

    def cells(model: str, dataset: str, task: str | None) -> list[str]:
        if task is None:
            return [f"{score.loc[model, dataset, :, m].mean():.1f}" for m in TABLE_METHODS]
        return [f"{score.loc[model, dataset, task, m]:.1f}" for m in TABLE_METHODS]

    lines = [
        "% Generated by make_quality.py from final_scores.csv; do not edit by hand.",
        f"% Per-task scores at {rate}x, {ctx}K context; all three models; RULER part, then LOFT part.",
        "\\centering",
    ]
    for i, (dataset, tasks) in enumerate(TASKS.items()):
        if i:
            lines.append("\\vspace{0.8em}")
        lines += [
            f"% --- {dataset}-{ctx}K ---",
            "\\resizebox{\\textwidth}{!}{%",
            "\\begin{tabular}{l" + ("|" + "r" * k) * len(MODELS) + "}",
            "\\toprule",
            f"\\textbf{{{dataset}-{ctx}K}} & "
            + " & ".join(f"\\multicolumn{{{k}}}{{c{'|' if j < len(MODELS) - 1 else ''}}}{{\\textbf{{{name}}}}}"
                         for j, (_, name) in enumerate(MODELS)) + " \\\\",
            "Task & " + " & ".join(" & ".join(SHORT[m] for m in TABLE_METHODS) for _ in MODELS) + " \\\\",
            "\\midrule",
        ]
        for t, lab in tasks:
            lines.append(lab + " & " + " & ".join(" & ".join(cells(m, dataset, t)) for m, _ in MODELS) + " \\\\")
        lines += ["\\midrule",
                  "Mean & " + " & ".join(" & ".join(cells(m, dataset, None)) for m, _ in MODELS) + " \\\\",
                  "\\bottomrule", "\\end{tabular}}"]
    lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")


def write_appendix_tables(df: pd.DataFrame, out: Path) -> None:
    """One small tabular per (model, benchmark, context), scaled to \\columnwidth so they flow through the two columns.
    Rows: rate x task (+ mean); columns: methods."""
    score = df.set_index(["model", "dataset", "context_k", "rate", "task", "method"])["score"]
    k = len(TABLE_METHODS)
    lines = ["% Generated by make_quality.py from final_scores.csv; do not edit by hand.",
             "% One table per (model, benchmark, context): every task and compression rate."]
    for model, name in MODELS:
        for dataset, ctx in ROWS:
            tasks = TASKS[dataset]
            lines += [
                f"% --- {name}, {dataset}-{ctx}K ---",
                "\\noindent\\resizebox{\\columnwidth}{!}{%",
                "\\begin{tabular}{ll" + "r" * k + "}",
                "\\toprule",
                f"\\multicolumn{{{k + 2}}}{{c}}{{\\textbf{{{name}, {dataset}-{ctx}K}}}} \\\\",
                "\\midrule",
                "Rate & Task & " + " & ".join(SHORT[m] for m in TABLE_METHODS) + " \\\\",
            ]
            for rate in RATES:
                lines.append("\\midrule")
                for t_i, (t, lab) in enumerate(tasks):
                    vals = [f"{score.loc[model, dataset, ctx, rate, t, m]:.1f}" for m in TABLE_METHODS]
                    lines.append((f"{rate}$\\times$" if t_i == 0 else "") + f" & {lab} & " + " & ".join(vals) + " \\\\")
                means = [f"{score.loc[model, dataset, ctx, rate, :, m].mean():.1f}" for m in TABLE_METHODS]
                lines.append(" & \\textit{Mean} & " + " & ".join(means) + " \\\\")
            lines += ["\\bottomrule", "\\end{tabular}}", "\\par\\vspace{1.2em}", ""]
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")


BAR_MODELS = [MODELS[1], MODELS[0], MODELS[2]]  # bar grid: Qwen3.5-9B first
# Bar grid also shows the exact-sketch reference (plans on exact weights; an oracle), hatched light blue.
BAR_METHODS = METHODS + ["vkv_final_perfect"]
# Legend: VerifiedKV (C = scorer, A = sketch), as in the paper's notation.
BAR_LABELS = {**LABELS,
              "vkv_nofetch_kvzip": r"PoissonComp ($\mathcal{C}$ = KVzip)",
              "vkv_final_kvzip": r"VerifiedKV ($\mathcal{C}$ = KVzip, $\mathcal{A}$ = 2-bit sketch)",
              "vkv_final_fastkvzip": r"VerifiedKV ($\mathcal{C}$ = Fast KVzip, $\mathcal{A}$ = 2-bit sketch)",
              "vkv_final_perfect": r"VerifiedKV ($\mathcal{C}$ = KVzip, $\mathcal{A}$ = perfect sketch)"}
BAR_COLOR = {"kvzip": "#6b6b6b", "fastkvzip": "#b5b5b5", "vkv_nofetch_kvzip": "#55a868",
             "vkv_final_kvzip": "#4a78b0", "vkv_final_fastkvzip": "#9fc4e6"}


def plot_bars(df: pd.DataFrame, out: Path, labels: bool = False) -> None:
    """Grouped bars: rows RULER/LOFT x 32K/128K, columns models; one group per rate, one bar per method (absolute task
    mean); Dense as a dashed line. RULER rows share 0-100, LOFT rows 0-60."""
    agg = df.groupby(KEY + ["method"])["score"].mean()
    fig, axes = plt.subplots(len(ROWS), len(MODELS), figsize=(6.75, 5.2), sharex=True)
    fig.subplots_adjust(hspace=0.16, wspace=0.07, bottom=0.05, top=0.925, left=0.09, right=0.995)
    w = 0.135
    for i, (dataset, ctx) in enumerate(ROWS):
        top = 100 if dataset == "RULER" else 60
        for j, (model, model_name) in enumerate(BAR_MODELS):
            ax = axes[i, j]
            for k, m in enumerate(BAR_METHODS):
                xs = [g + (k - 2.5) * w for g in range(len(RATES))]
                ys = [agg[model, dataset, ctx, r, m] for r in RATES]
                if m == "vkv_final_perfect":
                    ax.bar(xs, ys, width=w, facecolor="white", edgecolor="#7fabd9", hatch="//////", linewidth=0.6,
                           zorder=2, label=BAR_LABELS[m])
                else:
                    ax.bar(xs, ys, width=w, color=BAR_COLOR[m], edgecolor="white", linewidth=0.4, zorder=2,
                           label=BAR_LABELS[m])
                if labels:  # integer score above each bar
                    for x, y in zip(xs, ys):
                        ax.text(x, y + top * 0.012, f"{y:.0f}", ha="center", va="bottom", fontsize=4.2, color=INK,
                                zorder=4, bbox=dict(boxstyle="square,pad=0.08", facecolor="white", edgecolor="none"))
            ax.axhline(agg[model, dataset, ctx, 8, "dense"], color=INK, linewidth=0.9, linestyle=(0, (3, 1.6)),
                       zorder=3, label="Dense")
            if i == 0:
                ax.set_title(model_name, fontsize=9, pad=5)
            if j == 0:
                ax.set_ylabel(f"{dataset}-{ctx}K score", fontsize=8)
            else:
                ax.tick_params(labelleft=False)
            if i == len(ROWS) - 1 and j == 1:
                ax.set_xlabel("Compression rate", fontsize=8)
            ax.set_xticks(range(len(RATES)))
            ax.set_xticklabels([f"{r}$\\times$" for r in RATES])
            ax.set_xlim(-0.5, len(RATES) - 0.5)
            ax.set_ylim(0, top * (1.07 if labels else 1))
            ax.set_yticks(range(0, top + 1, 20))
            ax.grid(True, axis="y", color=GRID, linewidth=0.4, linestyle=":")
            ax.set_axisbelow(True)
            ax.tick_params(labelsize=7, right=True, bottom=False)
    h, l = axes[0, 0].get_legend_handles_labels()
    order = [l.index("Dense")] + [l.index(BAR_LABELS[m]) for m in BAR_METHODS]
    fig.legend([h[o] for o in order], [l[o] for o in order], loc="lower center", ncol=4, fontsize=7,
               handlelength=1.6, columnspacing=0.8, handletextpad=0.4, bbox_to_anchor=(0.54, 0.97), frameon=False)
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out.with_suffix('.png')} and .pdf")


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=here / "final_scores.csv")
    args = ap.parse_args()
    df = load(args.csv)
    plot_quality(df, here / "quality_vs_compression_relative.png", relative=True)
    plot_quality(df, here / "quality_vs_compression.png", relative=False)
    plot_bars(pd.read_csv(args.csv), here / "quality_bars.png")  # raw: includes vkv_final_perfect
    plot_bars(pd.read_csv(args.csv), here / "quality_bars_labeled.png", labels=True)  # variant with scores on bars
    write_main_table(df, here / "quality_per_task_main.tex")
    write_appendix_tables(df, here / "quality_per_task_appendix.tex")


if __name__ == "__main__":
    main()
