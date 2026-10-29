#!/usr/bin/env python3
"""Quality vs. compression: aggregate plot + main/appendix per-task tables.

CSV schema:
  model,dataset,context_k,rate,method,task,score

Figure: 4×3 — rows = (RULER|LOFT)×(32K|128K); cols = Qwen3 | Qwen3.5 | Ministral-3.

Usage:
  python plot_quality.py --csv scores_so_far.csv
  python plot_quality.py --csv scores_so_far.csv --allow-partial   # also plot/average incomplete task sets
  python plot_quality.py --csv dummy.csv --dummy
  python plot_quality.py --make-dummy   # rewrite dummy.csv with placeholder scores

Final grid (phase4/final_grid): several CSVs at once and the final grid's RULER tasks, e.g.
  python plot_quality.py --csv ../baselines/phase3_results.csv ../results.csv \
      --ruler-tasks niah_mk3 niah_mv cwe qa1 --out final.png ...
(copied from phase3/experiments/final_eval/analysis/plot_quality.py; additions: --csv takes several files,
--methods, --ruler-tasks, the vkv_final methods.)

Changes for the real results (scores_so_far.csv):
- LOFT tasks are the five RAG sets the grid runs (nq, hotpotqa, musique, qampari, quest), not retrieval/rag/sql/icl.
- A point / average is shown only when every task of that (model, dataset, context, rate, method) is present,
  so partially finished cells never mix task subsets; --allow-partial turns this off. Skipped groups are printed.
- y-axis: RULER rows 0-100; LOFT rows 0 to just above that row's highest score (LOFT scores are much lower).
- Legend collects every method that appears in any panel (methods absent from the CSV are simply not drawn).
"""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

METHOD_ORDER = [
    "dense",
    "kvzip",
    "fastkvzip",
    "vkv_nofetch_kvzip",
    "vkv_nofetch_fastkvzip",
    "vkv_fetch_kvzip",
    "vkv_fetch_fastkvzip",
    "vkv_final_kvzip",
    "vkv_final_fastkvzip",
    "vkv_final_perfect",
]
# vkv_final (final_grid/) supersedes phase 3's needle fetch (vkv_fetch_*): both are the "fetch" family and share its
# hue, so a figure shows one of the two. When the CSVs hold vkv_final rows, vkv_fetch_* are left out unless --methods
# names them. (A fourth hue next to the three in use failed the palette validator on white.)
SUPERSEDED = {"vkv_fetch_kvzip", "vkv_fetch_fastkvzip"}

METHOD_LABELS = {
    "dense": "Dense",
    "kvzip": "KVzip",
    "fastkvzip": "Fast KVzip",
    "vkv_nofetch_kvzip": "vKV (no fetch)+KVzip",
    "vkv_nofetch_fastkvzip": "vKV (no fetch)+Fast",
    "vkv_fetch_kvzip": "vKV (fetch)+KVzip",
    "vkv_fetch_fastkvzip": "vKV (fetch)+Fast",
    "vkv_final_kvzip": "vKV final+KVzip",
    "vkv_final_fastkvzip": "vKV final+Fast",
    "vkv_final_perfect": "vKV final, exact sketch (ref.)",
}

# Shorter headers for wide per-task tables.
METHOD_SHORT = {
    "dense": "Dense",
    "kvzip": "KVzip",
    "fastkvzip": "F-KVzip",
    "vkv_nofetch_kvzip": "vKV$\\emptyset$+K",
    "vkv_nofetch_fastkvzip": "vKV$\\emptyset$+F",
    "vkv_fetch_kvzip": "vKV$\\checkmark$+K",
    "vkv_fetch_fastkvzip": "vKV$\\checkmark$+F",
    "vkv_final_kvzip": "vKV+K",
    "vkv_final_fastkvzip": "vKV+F",
    "vkv_final_perfect": "vKV$^\\ast$",
}

# Encoding: hue + marker shape = method family (top-k baseline / vKV no fetch / vKV fetch);
# scorer = line + marker fill (KVzip: solid line, filled marker; Fast KVzip: dashed line, hollow marker).
# Dense is the neutral reference (gray, no marker). The three hues are slots 1-3 of the dataviz reference palette,
# validated all-pairs on white (worst CVD dE 9.2, normal-vision dE 24.0); aqua is < 3:1 on white, so identity is
# also carried by marker shape and the per-task tables are the table view.
INK = "#0b0b0b"
INK_2 = "#52514e"
FAMILY_COLOR = {"topk": "#eb6834", "nofetch": "#1baf7a", "fetch": "#2a78d6"}
FAMILY_MARKER = {"topk": "s", "nofetch": "^", "fetch": "o"}
FAMILY = {"kvzip": "topk", "fastkvzip": "topk", "vkv_nofetch_kvzip": "nofetch", "vkv_nofetch_fastkvzip": "nofetch",
          "vkv_fetch_kvzip": "fetch", "vkv_fetch_fastkvzip": "fetch",
          "vkv_final_kvzip": "fetch", "vkv_final_fastkvzip": "fetch"}


def _style(method: str) -> dict:
    if method == "dense":
        return dict(color=INK_2, linestyle="-", linewidth=1.4, marker=None, zorder=2)
    if method == "vkv_final_perfect":  # a reference, like Dense: fetch hue, dotted, hollow diamond
        c = FAMILY_COLOR["fetch"]
        return dict(color=c, linestyle=(0, (1, 1.5)), linewidth=1.6, marker="D", markersize=5, markeredgewidth=1.4,
                    markeredgecolor=c, markerfacecolor="white", zorder=3)
    fam, fast = FAMILY[method], "fast" in method
    c = FAMILY_COLOR[fam]
    return dict(color=c, linestyle=(0, (4, 2)) if fast else "-", linewidth=2.2 if fam == "fetch" else 1.6,
                marker=FAMILY_MARKER[fam], markersize=6, markeredgewidth=1.4, markeredgecolor=c,
                markerfacecolor="white" if fast else c, zorder=4 if fam == "fetch" else 3)


STYLE = {m: _style(m) for m in METHOD_ORDER}
# Legend columns: one per family (KVzip-scored above, Fast-scored below), then Dense.
LEGEND_ORDER = ["kvzip", "fastkvzip", "vkv_nofetch_kvzip", "vkv_nofetch_fastkvzip",
                "vkv_fetch_kvzip", "vkv_fetch_fastkvzip", "dense"]
# with vkv_final in the figure: its two scorers take the fetch column, the exact-sketch reference sits under Dense
LEGEND_ORDER_FINAL = ["kvzip", "fastkvzip", "vkv_nofetch_kvzip", "vkv_nofetch_fastkvzip",
                      "vkv_final_kvzip", "vkv_final_fastkvzip", "dense", "vkv_final_perfect"]

MODELS = ["Qwen3", "Qwen3.5", "Ministral-3"]
DATASETS = ["RULER", "LOFT"]
CONTEXTS = [32, 128]
RATES = [8, 16, 32]

# Row order for the 4×3 figure: dataset × context.
ROW_PANELS = [
    ("RULER", 32),
    ("RULER", 128),
    ("LOFT", 32),
    ("LOFT", 128),
]

# Canonical task lists (13 RULER, 5 LOFT).
RULER_TASKS = [
    ("niah_s1", "S-NIAH1"),
    ("niah_s2", "S-NIAH2"),
    ("niah_s3", "S-NIAH3"),
    ("niah_mk1", "MK-NIAH1"),
    ("niah_mk2", "MK-NIAH2"),
    ("niah_mk3", "MK-NIAH3"),
    ("niah_mv", "MV-NIAH"),
    ("niah_mq", "MQ-NIAH"),
    ("vt", "VT"),
    ("cwe", "CWE"),
    ("fwe", "FWE"),
    ("qa1", "QA1"),
    ("qa2", "QA2"),
]
LOFT_TASKS = [
    ("nq", "NQ"),
    ("hotpotqa", "HotpotQA"),
    ("musique", "MuSiQue"),
    ("qampari", "QAMPARI"),
    ("quest", "QUEST"),
]
TASKS = {"RULER": RULER_TASKS, "LOFT": LOFT_TASKS}
ALL_TASKS = dict(TASKS)  # every known task (TASKS may be narrowed by --ruler-tasks)

# Two-letter task codes for the compact main-paper table.
TASK_SHORT = {
    "niah_s1": "S1",
    "niah_s2": "S2",
    "niah_s3": "S3",
    "niah_mk1": "M1",
    "niah_mk2": "M2",
    "niah_mk3": "M3",
    "niah_mv": "MV",
    "niah_mq": "MQ",
    "vt": "VT",
    "cwe": "CW",
    "fwe": "FW",
    "qa1": "Q1",
    "qa2": "Q2",
    "nq": "NQ",
    "hotpotqa": "HP",
    "musique": "MS",
    "qampari": "QM",
    "quest": "QS",
}

# Mean offset vs Dense at 8× / 32K (higher = worse).
METHOD_GAP = {
    "dense": 0.0,
    "vkv_fetch_kvzip": 3.0,
    "vkv_fetch_fastkvzip": 6.0,
    "vkv_nofetch_kvzip": 12.0,
    "kvzip": 18.0,
    "vkv_nofetch_fastkvzip": 22.0,
    "fastkvzip": 28.0,
}
RATE_PENALTY = {8: 0.0, 16: 10.0, 32: 25.0}  # non-dense only
MODEL_SHIFT = {"Qwen3": 0.0, "Qwen3.5": -1.5, "Ministral-3": 2.0}

GROUP = ["model", "dataset", "context_k", "rate", "method"]


def _noise(key: str, scale: float = 4.0) -> float:
    h = int(hashlib.md5(key.encode()).hexdigest()[:8], 16)
    return ((h % 1000) / 1000.0 - 0.5) * 2.0 * scale


def make_dummy_rows() -> list[dict]:
    rows = []
    for model in MODELS:
        for dataset, task_list in TASKS.items():
            base = 94.0 if dataset == "RULER" else 90.0
            base += MODEL_SHIFT[model]
            for ctx in CONTEXTS:
                ctx_pen = 0.0 if ctx == 32 else 4.0
                for rate in RATES:
                    for method in METHOD_ORDER:
                        for task_id, _ in task_list:
                            gap = METHOD_GAP[method]
                            if method != "dense":
                                gap += RATE_PENALTY[rate]
                            score = (
                                base
                                - ctx_pen
                                - gap
                                + _noise(f"{model}|{dataset}|{ctx}|{rate}|{method}|{task_id}")
                            )
                            score = max(5.0, min(99.5, score))
                            rows.append(
                                {
                                    "model": model,
                                    "dataset": dataset,
                                    "context_k": ctx,
                                    "rate": rate,
                                    "method": method,
                                    "task": task_id,
                                    "score": round(score, 1),
                                }
                            )
    return rows


def write_dummy_csv(path: Path) -> pd.DataFrame:
    df = pd.DataFrame(make_dummy_rows())
    path.write_text(df.to_csv(index=False), encoding="utf-8")
    print(f"Wrote {path} ({len(df)} rows)")
    return df


def load_df(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"model", "dataset", "context_k", "rate", "method", "task", "score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}")
    unknown = {(d, t) for d, t in zip(df["dataset"], df["task"]) if t not in {k for k, _ in ALL_TASKS.get(d, [])}}
    if unknown:
        raise ValueError(f"CSV has tasks this script does not know: {sorted(unknown)}")
    return df


def complete_only(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only (model, dataset, context, rate, method) groups that have every task of their dataset."""
    n_tasks = df.groupby(GROUP)["task"].nunique()
    need = pd.Series([len(TASKS[k[1]]) for k in n_tasks.index], index=n_tasks.index)
    partial = n_tasks[n_tasks < need]
    if len(partial):
        print(f"Skipping {len(partial)} incomplete group(s) (use --allow-partial to include them):")
        for k, n in partial.items():
            print(f"  {k[0]:12s} {k[1]:5s} {k[2]:>3}K {k[3]:>2}x {k[4]:22s} {n}/{len(TASKS[k[1]])} tasks")
    keep = n_tasks[n_tasks >= need].index
    return df.set_index(GROUP).loc[lambda x: x.index.isin(keep)].reset_index()


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(GROUP, as_index=False)["score"].mean()


def relative_to_dense(df_agg: pd.DataFrame) -> pd.DataFrame:
    """Each method's task-mean as % of Dense's task-mean, per (model, dataset, context, rate).

    Ratio of means, not mean of per-task ratios: Dense scores 0 on some LOFT tasks, where a per-task ratio is
    undefined. Groups without a (complete) Dense mean are dropped; Dense itself becomes the 100% line."""
    key = ["model", "dataset", "context_k", "rate"]
    dense = df_agg[df_agg["method"] == "dense"][key + ["score"]].rename(columns={"score": "dense"})
    out = df_agg.merge(dense, on=key, how="inner")
    out = out[out["dense"] > 0]
    out["score"] = 100 * out["score"] / out["dense"]
    return out.drop(columns="dense")


def plot_figure(df_agg: pd.DataFrame, out_path: Path, dummy: bool, relative: bool = False) -> None:
    """4×3: rows = dataset×context; columns = models."""
    n_rows, n_cols = len(ROW_PANELS), len(MODELS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols + (1.0 if n_cols == 1 else 0.0), 10.2),
                             sharex=True, sharey=False, squeeze=False)
    fig.subplots_adjust(hspace=0.32, wspace=0.18, bottom=0.10, top=0.94)

    legend = {}
    for i, (dataset, ctx) in enumerate(ROW_PANELS):
        row = df_agg[(df_agg["dataset"] == dataset) & (df_agg["context_k"] == ctx)]
        if relative:  # % of Dense: one scale for every row, opened up if a method beats Dense
            top = max(100, 10 * math.ceil((df_agg["score"].max() + 3) / 10)) if len(df_agg) else 100
        else:
            top = 100 if dataset == "RULER" or row.empty else min(100, 10 * math.ceil((row["score"].max() + 3) / 10))
        for j, model in enumerate(MODELS):
            ax = axes[i, j]
            sub = row[row["model"] == model]
            for method in METHOD_ORDER:
                m = sub[sub["method"] == method].sort_values("rate")
                if m.empty:
                    continue
                (line,) = ax.plot(m["rate"], m["score"], label=METHOD_LABELS[method], **STYLE[method])
                legend.setdefault(method, line)
            if sub.empty:
                ax.text(0.5, 0.5, "no full-cache baseline yet" if relative else "not run yet", transform=ax.transAxes, ha="center", va="center",
                        fontsize=8, color=INK_2)
            ax.set_xticks(RATES)
            ax.set_xticklabels([f"{r}x" for r in RATES])
            if i == 0:
                ax.set_title(model, fontsize=11, fontweight="bold", color=INK)
            if j == 0:
                ax.set_ylabel(f"{dataset}, {ctx}K\n" + ("% of full cache" if relative else "Quality"), fontsize=8,
                              color=INK)
            if i == n_rows - 1:
                ax.set_xlabel("Compression rate", fontsize=8, color=INK)
            ax.grid(True, color="#e4e3df", linewidth=0.6)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color("#b9b8b2")
            ax.set_ylim(0, top)
            ax.set_xlim(RATES[0] - 1.5, RATES[-1] + 1.5)
            ax.tick_params(labelsize=7, colors=INK_2)

    # Fixed slots keep each family in its own legend column; a method absent from the data leaves a blank slot.
    blank = plt.Line2D([], [], alpha=0)
    order = LEGEND_ORDER_FINAL if any(m.startswith("vkv_final") for m in legend) else LEGEND_ORDER
    fig.legend(
        [legend.get(m, blank) for m in order],
        [METHOD_LABELS[m] if m in legend else "" for m in order],
        loc="lower center",
        ncol=4 if n_cols > 1 else 2,
        frameon=False,
        fontsize=7.5,
        labelcolor=INK,
        handlelength=3.2,
        bbox_to_anchor=(0.5, 0.01),
    )

    if dummy:
        fig.text(
            0.5,
            0.52,
            "[DUMMY]",
            color="red",
            fontsize=48,
            ha="center",
            va="center",
            alpha=0.35,
            fontweight="bold",
            rotation=20,
            zorder=1000,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")
    print(f"Wrote {out_path.with_suffix('.pdf')}")


def plot_single_model(df_agg: pd.DataFrame, model: str, out_path: Path, relative: bool = False) -> None:
    """One model: 2x2 panels (rows RULER / LOFT, columns 32K / 128K), legend below."""
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 7.2), sharex=True)
    fig.subplots_adjust(hspace=0.30, wspace=0.22, bottom=0.20, top=0.90)
    sub = df_agg[df_agg["model"] == model]
    legend = {}
    for i, dataset in enumerate(DATASETS):
        for j, ctx in enumerate(CONTEXTS):
            ax = axes[i, j]
            panel = sub[(sub["dataset"] == dataset) & (sub["context_k"] == ctx)]
            for method in METHOD_ORDER:
                m = panel[panel["method"] == method].sort_values("rate")
                if m.empty:
                    continue
                (line,) = ax.plot(m["rate"], m["score"], label=METHOD_LABELS[method], **STYLE[method])
                legend.setdefault(method, line)
            if relative:
                top = max(100, 10 * math.ceil((sub["score"].max() + 3) / 10)) if len(sub) else 100
            else:
                top = 100 if dataset == "RULER" or panel.empty else min(100, 10 * math.ceil((panel["score"].max() + 3) / 10))
            ax.set_ylim(0, top)
            ax.set_xticks(RATES)
            ax.set_xticklabels([f"{r}x" for r in RATES])
            ax.set_xlim(RATES[0] - 1.5, RATES[-1] + 1.5)
            ax.set_title(f"{dataset}, {ctx}K", fontsize=10, color=INK)
            if j == 0:
                ax.set_ylabel("% of full cache" if relative else "Quality", fontsize=8, color=INK)
            if i == 1:
                ax.set_xlabel("Compression rate", fontsize=8, color=INK)
            ax.grid(True, color="#e4e3df", linewidth=0.6)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color("#b9b8b2")
            ax.tick_params(labelsize=7, colors=INK_2)
    fig.suptitle(model, fontsize=12, fontweight="bold", color=INK)
    blank = plt.Line2D([], [], alpha=0)
    order = LEGEND_ORDER_FINAL if any(m.startswith("vkv_final") for m in legend) else LEGEND_ORDER
    fig.legend([legend.get(m, blank) for m in order], [METHOD_LABELS[m] if m in legend else "" for m in order],
               loc="lower center", ncol=4, frameon=False, fontsize=7.5, labelcolor=INK, handlelength=3.2,
               bbox_to_anchor=(0.5, 0.01))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def _score(
    df: pd.DataFrame, model: str, dataset: str, ctx: int, rate: int, method: str, task: str
) -> str:
    hit = df[
        (df["model"] == model)
        & (df["dataset"] == dataset)
        & (df["context_k"] == ctx)
        & (df["rate"] == rate)
        & (df["method"] == method)
        & (df["task"] == task)
    ]
    return f"{hit['score'].iloc[0]:.1f}" if len(hit) else "--"


def _mean_score(
    df: pd.DataFrame, model: str, dataset: str, ctx: int, rate: int, method: str
) -> str:
    hit = df[
        (df["model"] == model)
        & (df["dataset"] == dataset)
        & (df["context_k"] == ctx)
        & (df["rate"] == rate)
        & (df["method"] == method)
    ]
    return f"{hit['score'].mean():.1f}" if len(hit) else "--"


def write_main_table(
    df: pd.DataFrame, out_path: Path, dummy: bool, rate: int = 16, ctx: int = 128
) -> None:
    """Main-paper table: one shared task col; Qwen3 methods | Qwen3.5 methods."""
    methods = [m for m in METHOD_ORDER if m in set(df["method"])]
    # Model 1 and model 2 only (Ministral-3 deferred to appendix).
    models = [m for m in MODELS[:2] if m in set(df["model"])]
    if len(models) != 2:
        raise ValueError(f"Main table expects exactly 2 models, got {models}")
    m1, m2 = models
    n = len(methods)
    hdr = " & ".join(METHOD_SHORT[m] for m in methods)

    parts: list[str] = []
    if dummy:
        parts.append("% [DUMMY] placeholder --- replace with real per-task results.\n")
    parts.append(
        f"% Main-paper: {rate}$\\times$, {ctx}K; {m1} | {m2}, shared task column.\n"
    )
    parts.append("\\footnotesize\n")
    parts.append("\\setlength{\\tabcolsep}{2.0pt}\n")
    parts.append("\\centering\n")

    for dataset in DATASETS:
        task_list = TASKS[dataset]
        # l | c...c | c...c  — single | after model 1
        parts.append(f"% --- {dataset}: {m1} | {m2} ---\n")
        parts.append("\\begin{tabular}{l|" + ("c" * n) + "|" + ("c" * n) + "}\n")
        parts.append("\\toprule\n")
        parts.append(
            f" & \\multicolumn{{{n}}}{{c|}}{{\\textbf{{{m1}}}}} "
            f"& \\multicolumn{{{n}}}{{c}}{{\\textbf{{{m2}}}}} \\\\\n"
        )
        parts.append(f" & {hdr} & {hdr} \\\\\n")
        parts.append("\\midrule\n")
        for task_id, _ in task_list:
            lab = TASK_SHORT.get(task_id, task_id[:2].upper())
            s1 = [_score(df, m1, dataset, ctx, rate, m, task_id) for m in methods]
            s2 = [_score(df, m2, dataset, ctx, rate, m, task_id) for m in methods]
            parts.append(" & ".join([lab] + s1 + s2) + " \\\\\n")
        parts.append("\\midrule\n")
        a1 = [_mean_score(df, m1, dataset, ctx, rate, m) for m in methods]
        a2 = [_mean_score(df, m2, dataset, ctx, rate, m) for m in methods]
        parts.append(" & ".join(["Av"] + a1 + a2) + " \\\\\n")
        parts.append("\\bottomrule\n\\end{tabular}\n")
        if dataset != DATASETS[-1]:
            parts.append("\\vspace{0.9em}\n\n")

    out_path.write_text("".join(parts), encoding="utf-8")
    print(f"Wrote {out_path}")


def write_appendix_table(df: pd.DataFrame, out_path: Path, dummy: bool) -> None:
    """Full per-task tables: all models × rates × contexts × datasets."""
    methods = [m for m in METHOD_ORDER if m in set(df["method"])]
    models = [m for m in MODELS if m in set(df["model"])]
    parts: list[str] = []
    if dummy:
        parts.append("% [DUMMY] full per-task placeholder tables.\n")

    for model in models:
        for dataset in DATASETS:
            task_list = TASKS[dataset]
            for ctx in CONTEXTS:
                for rate in RATES:
                    n = len(methods)
                    parts.append(f"% --- {model}, {dataset}, {ctx}K, {rate}x ---\n")
                    parts.append("\\begin{tabular}{l" + "c" * n + "}\n")
                    parts.append("\\toprule\n")
                    parts.append(
                        f"\\multicolumn{{{n + 1}}}{{c}}{{{model}, {dataset}, {ctx}K, {rate}$\\times$}} \\\\\n"
                    )
                    parts.append("\\midrule\n")
                    parts.append(
                        "Task & " + " & ".join(METHOD_SHORT[m] for m in methods) + " \\\\\n"
                    )
                    parts.append("\\midrule\n")
                    for task_id, task_lab in task_list:
                        row = [task_lab] + [
                            _score(df, model, dataset, ctx, rate, m, task_id)
                            for m in methods
                        ]
                        parts.append(" & ".join(row) + " \\\\\n")
                    parts.append("\\midrule\n")
                    avg = ["Avg"] + [
                        _mean_score(df, model, dataset, ctx, rate, m) for m in methods
                    ]
                    parts.append(" & ".join(avg) + " \\\\\n")
                    parts.append("\\bottomrule\n\\end{tabular}\n")
                    parts.append("\\vspace{0.8em}\n\n")

    out_path.write_text("".join(parts), encoding="utf-8")
    print(f"Wrote {out_path}")


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, nargs="+", default=[here / "dummy.csv"],
                    help="one or more CSVs in this schema (e.g. baselines/phase3_results.csv and results.csv)")
    ap.add_argument("--methods", nargs="+", help="methods to show (default: all; vkv_fetch_* dropped when "
                    "vkv_final rows are present)")
    ap.add_argument("--models", nargs="+", help="model columns to plot, e.g. Qwen3.5 (default: all three)")
    ap.add_argument("--ruler-tasks", nargs="+", help="RULER task ids to keep, e.g. niah_mk3 niah_mv cwe qa1 "
                    "(default: all 13)")
    ap.add_argument("--make-dummy", action="store_true", help="Rewrite dummy.csv.")
    ap.add_argument("--dummy", action="store_true", help="Red [DUMMY] watermark on figure.")
    ap.add_argument("--allow-partial", action="store_true",
                    help="Also plot / average (model, dataset, context, rate, method) groups missing some tasks.")
    ap.add_argument("--out", type=Path, default=here / "quality_vs_compression.png")
    ap.add_argument("--out-relative", type=Path, default=here / "quality_vs_compression_relative.png",
                    help="Same figure with each score as %% of Dense (full cache).")
    ap.add_argument(
        "--main-table",
        type=Path,
        default=here / "quality_per_task_main.tex",
        help="Compact per-task table for main paper (default: 16x, 128K).",
    )
    ap.add_argument(
        "--appendix-table",
        type=Path,
        default=here / "quality_per_task_appendix.tex",
        help="Full per-task tables for appendix.",
    )
    ap.add_argument("--main-rate", type=int, default=16)
    ap.add_argument("--main-context", type=int, default=128)
    args = ap.parse_args()

    if args.make_dummy:
        write_dummy_csv(here / "dummy.csv")

    if args.models:
        assert all(m in MODELS for m in args.models), args.models
        MODELS[:] = args.models
    if args.ruler_tasks:  # e.g. the final grid's four tasks: every method is then averaged over the same four
        known = dict(RULER_TASKS)
        assert all(t in known for t in args.ruler_tasks), f"unknown RULER task in {args.ruler_tasks}"
        TASKS["RULER"] = [(t, known[t]) for t in args.ruler_tasks]
    dummy = args.dummy or args.csv[0].name.lower().startswith("dummy")
    df = pd.concat([load_df(p) for p in args.csv], ignore_index=True)
    df = df[df["task"].isin([t for d in TASKS.values() for t, _ in d])]
    dup = df.duplicated(["model", "dataset", "context_k", "rate", "method", "task"], keep=False)
    if dup.any():
        raise ValueError(f"the same (model, dataset, context, rate, method, task) in more than one row:\n{df[dup]}")
    if args.methods:
        df = df[df["method"].isin(args.methods)]
    elif df["method"].str.startswith("vkv_final").any():
        df = df[~df["method"].isin(SUPERSEDED)]
    if not args.allow_partial:
        df = complete_only(df)
    df_agg = aggregate(df)
    if len(MODELS) == 1:  # one model: a 2x2 grid reads better than one tall column
        plot_single_model(df_agg, MODELS[0], args.out)
        plot_single_model(relative_to_dense(df_agg), MODELS[0], args.out_relative, relative=True)
    else:
        plot_figure(df_agg, args.out, dummy=dummy)
        plot_figure(relative_to_dense(df_agg), args.out_relative, dummy=dummy, relative=True)
    if len(MODELS) >= 2:  # the main-paper table pairs two models
        write_main_table(df, args.main_table, dummy=dummy, rate=args.main_rate, ctx=args.main_context)
    write_appendix_table(df, args.appendix_table, dummy=dummy)


if __name__ == "__main__":
    main()
