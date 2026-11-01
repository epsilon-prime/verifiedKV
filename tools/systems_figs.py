"""Paper tables and figures of the 32k systems run: joins the system metrics (sys_table.py's CSV) with
the RULER scores of the 50-row sweep, writes a main and an appendix CSV, and draws three figures.

    python tools/systems_figs.py <systems dir>
      reads  ruler32k_qwen3-4b_16x_vllm_systems_delta0.05.csv, ruler32k_qwen3-4b_16x_vllm.csv
      writes systems_main_ruler32k.csv, systems_appendix_ruler32k.csv,
             fig_step_breakdown.{png,pdf}, fig_score_vs_speed.{png,pdf}, fig_pcie_link.{png,pdf}
"""
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

D = Path(sys.argv[1])
SYS = list(csv.DictReader(open(D / "ruler32k_qwen3-4b_16x_vllm_systems_delta0.05.csv")))
SCORES = {(r["method"], r["eps"]): r for r in csv.DictReader(open(D / "ruler32k_qwen3-4b_16x_vllm.csv"))
          if r["method"] != "vkv" or r["delta"] == "0.05"}
PCIE_AVAIL = 55.5                         # GB/s per direction, measured (tools/bw_bench.py)
f = lambda r, k: float(r[k]) if r.get(k) not in (None, "") else None  # noqa: E731


def name(r):
    return {"dense": "Dense (no offload)", "dense_recall": "Dense recall (exact)",
            "fastkvzip_evict": "FastKVzip evict"}.get(
        r["method"], f"vkv ε={r['eps']}")


recall = next(r for r in SYS if r["method"] == "dense_recall")
main, appx = [], []
for r in SYS:
    s = SCORES[(r["method"], r["eps"])]
    step = {k: f(r, f"step_ms_{k}") or 0.0 for k in ("decide", "fetch", "attention", "gemm", "other")}
    main.append({
        "config": name(r), "method": r["method"], "eps": r["eps"], "delta": r["delta"],
        "ruler_score": f(s, "score"), "score_cwe": f(s, "score_cwe"),
        "tpot_ms": f(r, "tpot_ms"), "decode_tok_s": f(r, "decode_tok_s"),
        "decode_speedup_vs_recall": round(f(r, "decode_tok_s") / f(recall, "decode_tok_s"), 2),
        "energy_J_per_token": f(r, "energy_J_per_answer_token"),
        "pcie_h2d_GBs": f(r, "answer_pcie_h2d_GBs"),
        "pcie_link_util_pct": f(r, "answer_pcie_h2d_util_pct"),
        "fetched_GB_per_request": f(r, "fetch_GB_per_request"),
        "kv_gpu_GB": f(r, "kv_gpu_GB"),
        "kv_gpu_pct_of_full": round(100 * f(r, "kv_gpu_GB") / f(r, "kv_full_GB"), 2),
        **{f"step_ms_{k}": round(v, 2) for k, v in step.items()}})
    appx.append({
        "config": name(r), "method": r["method"], "eps": r["eps"], "delta": r["delta"],
        "hbm_GBs": f(r, "answer_hbm_GBs"), "hbm_util_pct": f(r, "answer_hbm_util_pct"),
        "sm_active_pct": f(r, "answer_sm_active_pct"), "sm_occupancy_pct": f(r, "answer_sm_occupancy_pct"),
        "tensor_pipe_pct": f(r, "answer_tensor_pct"), "fp64_pipe_pct": f(r, "answer_fp64_pct"),
        "fp32_pipe_pct": f(r, "answer_fp32_pct"), "power_W": f(r, "answer_power_W"),
        "pcie_d2h_GBs": f(r, "answer_pcie_d2h_GBs"),
        "ttft_s": f(r, "ttft_s"), "prefill_s": f(r, "prefill_s"),
        "prefill_pcie_d2h_GBs": f(r, "prefill_pcie_d2h_GBs"), "prefill_power_W": f(r, "prefill_power_W"),
        "kv_full_GB": f(r, "kv_full_GB"), "kv_host_GB": f(r, "kv_host_GB"),
        "k_star_mean": f(r, "k_star_mean"), "rows_fetching_pct": f(r, "rows_fetching_pct"),
        "fallback_pct": f(r, "fallback_pct"), "screened_pct": f(r, "screened_pct"),
        "n_rows": int(r["n_rows"]), "n_profiled": int(r["n_profiled"])})
for rows, fn in ((main, "systems_main_ruler32k.csv"), (appx, "systems_appendix_ruler32k.csv")):
    with open(D / fn, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

# ---- figures: the reference palette (dataviz skill), light surface, recessive axes ----------------------
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e2dc", "#fcfcfb"
CAT = {"decide": "#2a78d6", "fetch": "#eb6834", "attention": "#1baf7a", "gemm": "#eda100",
       "other": "#b9b7b0"}
LAB = {"decide": "vkv decide", "fetch": "Fetch / copy-back", "attention": "Attention",
       "gemm": "GEMMs (weights)", "other": "Other"}
plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.titlecolor": INK,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})
KEEP = ["dense", "dense_recall", "0.01", "0.03", "0.05", "0.1", "0.25", "0.5", "fastkvzip_evict"]
pick = lambda rows: [next(r for r in rows if r["method"] == k or r["eps"] == k)  # noqa: E731
                     for k in KEEP]


def clean(ax, grid_axis="x"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def save(fig, stem):
    for ext in ("png", "pdf"):
        fig.savefig(D / f"{stem}.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)


# 1) where a decode step goes: horizontal stacked bars, total TPOT at the end
rows = pick(main)[::-1]
fig, ax = plt.subplots(figsize=(6.4, 3.2))
y = range(len(rows))
left = [0.0] * len(rows)
for k in CAT:
    v = [r[f"step_ms_{k}"] for r in rows]
    ax.barh(y, v, left=left, height=0.62, color=CAT[k], label=LAB[k], edgecolor=SURF, linewidth=1)
    left = [a + b for a, b in zip(left, v)]
for i, r in enumerate(rows):
    ax.text(left[i] + 2, i, f"{r['tpot_ms']:.0f} ms", va="center", color=INK2, fontsize=8)
ax.set_yticks(list(y), [r["config"] for r in rows])
ax.set_xlabel("GPU kernel time per decode step (ms)")
ax.set_title("Where a decode step goes (RULER 32k, 16×, batch 1)", loc="left", fontsize=10)
ax.legend(ncol=5, frameon=False, fontsize=7.5, loc="lower left", bbox_to_anchor=(0, 1.06),
          handlelength=1, columnspacing=0.9)
clean(ax)
save(fig, "fig_step_breakdown")

# 2) the trade-off: RULER score vs decode throughput (log x), vkv's eps curve + the two baselines
v = sorted((r for r in main if r["method"] == "vkv"), key=lambda r: float(r["eps"]))
fig, ax = plt.subplots(figsize=(5.2, 3.2))
ax.plot([r["decode_tok_s"] for r in v], [r["ruler_score"] for r in v], color=CAT["decide"],
        linewidth=2, marker="o", markersize=5, markeredgecolor=SURF, markeredgewidth=1.5,
        label="vkv (ε sweep, δ=0.05)")
for r in v:
    if r["eps"] in ("0.01", "0.1", "0.5"):
        ax.annotate(f"ε={r['eps']}", (r["decode_tok_s"], r["ruler_score"]), textcoords="offset points",
                    xytext=(0, -13), ha="center", fontsize=7.5, color=INK2)
for r, mk, col in ((next(m for m in main if m["method"] == "dense"), "o", INK2), (recall, "s", CAT["fetch"]),
                   (next(m for m in main if m["method"] == "fastkvzip_evict"), "D", CAT["attention"])):
    m = next(x for x in main if x["method"] == r["method"]) if "config" not in r else r
    ax.plot([m["decode_tok_s"]], [m["ruler_score"]], marker=mk, markersize=7, color=col, linestyle="",
            markeredgecolor=SURF, markeredgewidth=1.5, label=m["config"])
ax.set_xscale("log")
ax.set_ylim(0, 100)
ax.set_xlabel("Decode throughput, tokens/s (log)")
ax.set_ylabel("RULER 32k score")
ax.set_title("Accuracy vs decode speed", loc="left", fontsize=10)
ax.legend(frameon=False, fontsize=7.5, loc="lower center")
clean(ax, "both")
save(fig, "fig_score_vs_speed")

# 3) PCIe link: share of the measured host->GPU bandwidth used while decoding
rows = pick(main)[::-1]
fig, ax = plt.subplots(figsize=(5.2, 3.0))
vals = [r["pcie_link_util_pct"] for r in rows]
ax.barh(range(len(rows)), vals, height=0.62,
        color=[CAT["fetch"] if r["method"] == "dense_recall" else CAT["decide"] if r["method"] == "vkv"
               else INK2 if r["method"] == "dense" else CAT["attention"] for r in rows], edgecolor=SURF, linewidth=1)
for i, (r, val) in enumerate(zip(rows, vals)):
    txt = f"{val:.1f}%  ({r['pcie_h2d_GBs']:.1f} GB/s)"
    if val > 60:                                      # inside the bar, clear of the link line
        ax.text(val - 1.5, i, txt, va="center", ha="right", color="white", fontsize=7.5)
    else:
        ax.text(val + 1.5, i, txt, va="center", color=INK2, fontsize=7.5)
ax.axvline(100, color=INK2, linewidth=0.8, linestyle=(0, (3, 2)))
ax.text(99, -0.75, f"measured link\n{PCIE_AVAIL} GB/s", color=INK2, fontsize=7, ha="right", va="bottom")
ax.set_yticks(range(len(rows)), [r["config"] for r in rows])
ax.set_xlim(0, 105)
ax.set_ylim(-0.8, len(rows) - 0.4)
ax.set_xlabel("PCIe host→GPU link utilisation during decode (%)")
ax.set_title("Interconnect pressure", loc="left", fontsize=10)
clean(ax)
save(fig, "fig_pcie_link")
print("wrote", *sorted(p.name for p in D.glob("systems_*_ruler32k.csv")), *sorted(p.name for p in D.glob("fig_*.png")))
