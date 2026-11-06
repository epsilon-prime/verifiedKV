"""The eps x delta figure from eps_delta_rows.csv alone (no torch). Only the contexts every arm finished
(in_cells = 1) are used, so all arms are compared on the same contexts.

Top row: RULER score against the share of the KV read from the host per forward, one point per (eps, delta) arm, with
the Pareto frontier and the dense (full cache) score. Bottom row: score against eps for each delta; error bars are the
standard error of the mean over contexts.

    python plot_eps_delta.py --context-k 32 --out eps_delta_32k.png
"""

import argparse
import csv
import math
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
EPS = [0.01, 0.05, 0.1, 0.2, 0.3]
DEL = [0.02, 0.05, 0.1]
MODELS = {"Qwen3": "Qwen3-4B", "Ministral-3": "Ministral-3-8B", "Qwen3.5": "Qwen3.5-9B"}
ECOL = {0.01: "#08306b", 0.05: "#2171b5", 0.1: "#41b6c4", 0.2: "#fd8d3c", 0.3: "#d7301f"}
DMARK = {0.02: "s", 0.05: "o", 0.1: "^"}
DCOL = {0.02: "#1b9e77", 0.05: "#d95f02", 0.1: "#7570b3"}

ap = argparse.ArgumentParser()
ap.add_argument("--context-k", type=int, choices=[32, 128], default=32)
ap.add_argument("--csv", default=os.path.join(HERE, "..", "..", "results", "eps_delta", "eps_delta_rows.csv"))
ap.add_argument("--out", default=None)
args = ap.parse_args()
out = args.out or os.path.join(HERE, f"eps_delta_{args.context_k}k.png")

recs = defaultdict(lambda: defaultdict(dict))  # model -> (eps, delta) or "dense" -> (task, row) -> (score, frac)
gpu = {}
for r in csv.DictReader(open(args.csv)):
    if int(r["context_k"]) != args.context_k or r["in_cells"] != "1":
        continue
    arm = "dense" if r["method"] == "dense" else (float(r["eps"]), float(r["delta"]))
    frac = float(r["read_pct"]) if r["read_pct"] else float("nan")
    recs[r["model"]][arm][(r["task"], int(r["row"]))] = (float(r["score"]), frac)
    gpu[r["model"]] = r["gpu"]

fig, axes = plt.subplots(2, 3, figsize=(14, 8.2), sharex="row", sharey=True)
for j, m in enumerate(MODELS):
    arms = recs[m]
    common = set.intersection(*(set(v) for v in arms.values())) if arms else set()
    n = len(common)
    axp, axe = axes[0][j], axes[1][j]
    axp.set_title(f"{MODELS[m]} ({gpu.get(m, '?')}): {n} contexts with all 16 arms", fontsize=10)
    if not n:
        continue
    dense = sum(arms["dense"][k][0] for k in common) / n
    pts = []
    for e in EPS:
        for d in DEL:
            v = [arms[(e, d)][k] for k in common]
            s, fr = sum(x[0] for x in v) / n, sum(x[1] for x in v) / n
            pts.append((fr, s))
            axp.scatter(fr, s, color=ECOL[e], marker=DMARK[d], s=46, edgecolor="black", linewidth=0.4, zorder=3)
    front, best = [], -1.0
    for fr, s in sorted(p for p in pts if not math.isnan(p[0])):
        if s > best:
            front.append((fr, s))
            best = s
    axp.step([p[0] for p in front], [p[1] for p in front], where="post", color="gray", lw=1, zorder=2,
             label="Pareto frontier")
    axp.axhline(dense, color="black", ls="--", lw=1, label=f"full cache ({dense:.1f})")
    for d in DEL:
        ys, es = [], []
        for e in EPS:
            s = [arms[(e, d)][k][0] for k in common]
            mu = sum(s) / n
            ys.append(mu)
            es.append(math.sqrt(sum((x - mu) ** 2 for x in s) / (n - 1) / n) if n > 1 else 0.0)
        axe.errorbar(EPS, ys, yerr=es, color=DCOL[d], marker=DMARK[d], ms=4, lw=1.4, capsize=2, label=f"δ = {d}")
    axe.axhline(dense, color="black", ls="--", lw=1)
    axp.legend(fontsize=8, loc="lower left")
    axe.legend(fontsize=8, loc="lower left")
    print(f"{MODELS[m]}: {n} contexts, dense {dense:.1f}; " +
          " ".join(f"e{e}/d{d}:{s:.1f}@{fr:.2f}%" for (fr, s), (e, d) in zip(pts, [(e, d) for e in EPS for d in DEL])))
for j in range(3):
    axp, axe = axes[0][j], axes[1][j]
    axp.set_xscale("log")
    axp.set_xlabel("KV read from host per forward (% of cache, log)")
    axp.grid(alpha=0.3, which="both")
    axe.set_xscale("log")
    axe.set_xticks(EPS)
    axe.set_xticklabels([str(e) for e in EPS])
    axe.set_xlabel("ε")
    axe.grid(alpha=0.3)
axes[0][0].set_ylabel("RULER score")
axes[1][0].set_ylabel("RULER score")
keys = [Line2D([], [], ls="", marker="o", color=ECOL[e], mec="black", mew=0.4, label=f"ε = {e}") for e in EPS]
keys += [Line2D([], [], ls="", marker=DMARK[d], color="white", mec="black", label=f"δ = {d}") for d in DEL]
fig.legend(handles=keys, loc="upper center", ncol=8, fontsize=8.5, bbox_to_anchor=(0.5, 0.955), frameon=False)
fig.suptitle(f"VerifiedKV ε × δ grid at 16× · RULER-{args.context_k}K (cwe, qa_1, niah_multikey_3, niah_multivalue)",
             fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(out, dpi=150)
print("saved", out)
