#!/usr/bin/env python3
"""Punchline table: quality, GPU memory and fetch for VerifiedKV (Fast KVzip scores), three models, averaged over the
four benchmarks RULER-32K, RULER-128K, LOFT-32K and LOFT-128K (18 tasks).

Rows: Dense resident, Dense offload, KVzip at 8x / 16x / 32x, Fast KVzip at 8x / 16x / 32x, VerifiedKV (eps0 = 0.05,
delta = 0.1) at 8x / 16x / 32x. KVzip and Fast KVzip: same GPU memory, never fetch (fetch 0 %).
Columns per model:
  Quality  per benchmark, the mean score over its tasks (RULER 4, LOFT 5) as % of Dense; then the plain mean of the
           four benchmark percentages. From ../quality/final_scores.csv
  Memory   KV-related GPU memory as % of the full cache: 1/rate (the budget includes sinks, local window and the 2-bit
           sketch); Dense resident 100 %, Dense offload 0 %
  Fetch    KV read from host per generated token as % of the full cache: total GB fetched over the answer (question
           pass included) / answer tokens / full-cache GB of that context, pooled over a benchmark's tasks; then the
           plain mean of the four benchmark percentages. From fetch_cells_fastkvzip.csv, built with --source from
           <results>/*/{ruler32k,ruler128k,loft}__*vkv_final_fastkvzip*/ (cost.json
           fetch_gb_per_answer and ctx_len_median, predictions.csv answers tokenised with each model's tokenizer).
           Qwen3.5-9B's grid runs did not record traffic; its cells come from the instrumented reruns in
           experiments/host_traffic (--q35; same arm, eps0 .05, delta .1, on fewer rows: RULER-32K 2 rows per task
           (rows 152-153), RULER-128K the first 2 test rows, LOFT the first 5 test questions). GB there = (fetched K/V
           entries + shortlist keys / 2) x 2 x head_dim x 2 B, as in results/eps_delta.
           Dense resident 0 %, Dense offload 100 %.
Shading: fixed scale for every block, green -> yellow -> red; quality 100 % green -> 0 % red, memory and fetch 0 % green
-> 100 % red.

Output: punchline_table.tex (the tabular; \\input inside a table* in the intro). Needs \\usepackage{colortbl}.

Usage (from the paper root):
  python figures/punchline_table/make_table.py --source <results>
      --q35 experiments/host_traffic
      # rebuild fetch_cells_fastkvzip.csv (needs transformers and the three tokenizers, offline), then the table
  python figures/punchline_table/make_table.py   # table from the extract (pandas only)
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path

import pandas as pd

MODELS = [("Qwen3", "Qwen3-4B-Instruct-2507", "Qwen3-4B"), ("Qwen3.5", "Qwen3.5-9B", "Qwen3.5-9B"),
          ("Ministral-3", "Ministral-3-8B-Instruct-2512-BF16", "Ministral-3-8B")]
RATES = [8, 16, 32]
RATIO = {8: 0.875, 16: 0.9375, 32: 0.96875}
BENCHES = [("RULER", 32), ("RULER", 128), ("LOFT", 32), ("LOFT", 128)]
KV_BYTES_PER_TOKEN = {"Qwen/Qwen3-4B-Instruct-2507": 147456, "mistralai/Ministral-3-8B-Instruct-2512-BF16": 139264,
                      "Qwen/Qwen3.5-9B": 32768}  # all layers' K and V in bf16 (Qwen3.5: its 8 full-attention layers)
# Green -> yellow -> red (three stops), light enough for black text but with a strong red at the bad end.
GREEN, YELLOW, RED = (0.70, 0.87, 0.66), (0.99, 0.91, 0.58), (0.93, 0.36, 0.33)


def shade(good: float) -> str:
    """good in [0, 1]: 1 = green, 0.5 = yellow, 0 = red; linear between the stops."""
    g = min(1.0, max(0.0, good))
    lo, hi, t = (RED, YELLOW, 2 * g) if g < 0.5 else (YELLOW, GREEN, 2 * g - 1)
    r, gg, b = (a + (c - a) * t for a, c in zip(lo, hi))
    return f"\\cellcolor[rgb]{{{r:.3f},{gg:.3f},{b:.3f}}}"


def cell(value: float | None, kind: str, fmt: str = "{:.1f}") -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "--"
    good = value / 100 if kind == "quality" else 1 - value / 100
    return shade(good) + fmt.format(value)


def extract(raw: Path, out: Path, q35: Path | None = None) -> None:
    """One row per (model, benchmark, ratio, task) cell: GB fetched over all answers, answer tokens, full-cache GB."""
    from transformers import AutoTokenizer  # only needed here

    tok, rows = {}, []
    for f in sorted(glob.glob(str(raw / "*" / "*__*vkv_final_fastkvzip__*" / "cost.json"))):
        j = json.load(open(f))
        if q35 is not None and j["model"] == "Qwen/Qwen3.5-9B":
            continue  # no traffic counters in these cells; taken from the reruns below
        bench = Path(f).parent.name.split("__")[0]
        if bench == "loft":
            dataset, ctx = "LOFT", int(j["task"].rsplit("_", 1)[1].rstrip("k"))
        else:
            dataset, ctx = "RULER", int(bench.removeprefix("ruler").rstrip("k"))
        m = j["model"]
        if m not in tok:
            tok[m] = AutoTokenizer.from_pretrained(m)
        p = pd.read_csv(Path(f).parent / "predictions.csv")
        n_tok = sum(len(tok[m](str(a), add_special_tokens=False)["input_ids"]) for a in p["predicted_answer"])
        gb = j.get("fetch_gb_per_answer")
        rows.append(dict(model=m.split("/")[-1], dataset=dataset, context_k=ctx, ratio=j["ratio"], task=j["task"],
                         n=len(p), tokens=n_tok, gb=float("nan") if gb is None else gb * len(p),
                         ctx=j["ctx_len_median"], cacheGB=j["ctx_len_median"] * KV_BYTES_PER_TOKEN[m] / 1e9))
    if q35 is not None:
        rows += extract_q35(q35, tok)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Wrote {out} ({len(rows)} cells)")


def extract_q35(src: Path, tok: dict) -> list[dict]:
    """Qwen3.5-9B cells from the instrumented reruns: RULER-32K from csv/rows.csv (one line per answer; MB priced at
    head_dim 128 there, so x2), the other benchmarks from each cell's traffic.json + predictions.csv."""
    from transformers import AutoTokenizer

    m = "Qwen/Qwen3.5-9B"
    tok.setdefault(m, AutoTokenizer.from_pretrained(m))
    ntok = lambda answers: sum(len(tok[m](str(a), add_special_tokens=False)["input_ids"]) for a in answers)
    rows = []
    r = pd.read_csv(src / "csv" / "rows.csv")
    r["ratio"] = r["arm"].str.split(":").str[1].astype(float)
    for (task, ratio), g in r.groupby(["task", "ratio"]):
        rows.append(dict(model="Qwen3.5-9B", dataset="RULER", context_k=32, ratio=ratio, task=task, n=len(g),
                         tokens=ntok(g["pred"]), gb=(g["t_MB"] * 2 / 1e3).sum(), ctx=g["ctx"].median(),
                         cacheGB=g["ctx"].median() * KV_BYTES_PER_TOKEN[m] / 1e9))
    for f in sorted(glob.glob(str(src / "results*" / "q35_traffic" / "*" / "traffic.json"))):
        j = json.load(open(f))
        s = j["stats"]
        p = pd.read_csv(Path(f).parent / "predictions.csv")
        assert len(p) == j["n_answers"], f
        gb = (s["question_fwd_xfer"] + s["decode_fwd_xfer"] + 0.5 * (s["question_fwd_keys"] + s["decode_fwd_keys"])) \
            * 2 * j["head_dim"] * 2 / 1e9
        if j["dataset"] == "loft":
            dataset, ctx = "LOFT", int(j["task"].rsplit("_", 1)[1].rstrip("k"))
        else:
            dataset, ctx = "RULER", int(j["dataset"].removeprefix("ruler").rstrip("k"))
        rows.append(dict(model="Qwen3.5-9B", dataset=dataset, context_k=ctx, ratio=j["ratio"], task=j["task"],
                         n=len(p), tokens=ntok(p["predicted_answer"]), gb=gb, ctx=j["ctx_len_median"],
                         cacheGB=j["ctx_len_median"] * KV_BYTES_PER_TOKEN[m] / 1e9))
    return rows


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, help="<results>: rebuild the fetch extract")
    ap.add_argument("--q35", type=Path, help="experiments/host_traffic: Qwen3.5-9B traffic (with --source)")
    args = ap.parse_args()
    if args.source:
        extract(args.source.expanduser(), here / "fetch_cells_fastkvzip.csv", args.q35 and args.q35.expanduser())
    scores = pd.read_csv(here.parent / "quality" / "final_scores.csv")
    # Quality: % of Dense per benchmark (mean over its tasks), then the mean over the four benchmarks.
    per = scores.groupby(["dataset", "context_k", "model", "method", "rate"])["score"].mean()
    pct = {}
    for key, _, _ in MODELS:
        for method in ("vkv_final_fastkvzip", "fastkvzip", "kvzip"):
            for r in RATES:
                pct[key, method, r] = sum(100 * per[d, c, key, method, r] / per[d, c, key, "dense", 8]
                                          for d, c in BENCHES) / len(BENCHES)
    # Fetch: % of the full cache per generated token per benchmark (pooled over its tasks), then the mean over the
    # four benchmarks; NaN when any cell lacks traffic counters.
    fc = pd.read_csv(here / "fetch_cells_fastkvzip.csv")
    fc["tokxcache"] = fc["tokens"] * fc["cacheGB"]
    per_bench = fc.groupby(["model", "ratio", "dataset", "context_k"]).apply(
        lambda g: 100 * g["gb"].sum() / g["tokxcache"].sum() if g["gb"].notna().all() and g["gb"].sum() > 0
        else float("nan"), include_groups=False)
    fetch = per_bench.groupby(["model", "ratio"]).agg(lambda x: x.mean() if len(x) == len(BENCHES) and x.notna().all()
                                                     else float("nan"))
    print("fetch % per token by benchmark:\n", per_bench.unstack(["dataset", "context_k"]).round(1))

    # Blocks: Dense, KVzip, Fast KVzip (evictions: same GPU memory, never fetch), VerifiedKV.
    order = [(m, r) for m in ("kvz", "fkz", "vkv") for r in RATES]
    label = {"vkv": "\\method{{}} ${}\\times$", "fkz": "Fast KVzip ${}\\times$", "kvz": "KVzip ${}\\times$"}
    rows = {"Dense (resident)": [], "Dense (offload)": []}
    rows.update({label[m].format(r): [] for m, r in order})
    for key, fname, _ in MODELS:
        for m, r in order:
            mem = shade(1 - 1 / r) + {8: "12.5", 16: "6.25", 32: "3.13"}[r]  # 100 / r written out (3.125 -> 3.13)
            if m == "vkv":
                q = pct[key, "vkv_final_fastkvzip", r]
                f = cell(fetch.get((fname, RATIO[r]), float("nan")), "fetch")
            else:
                q = pct[key, {"fkz": "fastkvzip", "kvz": "kvzip"}[m], r]
                f = cell(0.0, "fetch", "{:.0f}")
            rows[label[m].format(r)] += [cell(q, "quality"), mem, f]
        rows["Dense (resident)"] += [cell(100.0, "quality", "{:.0f}"), cell(100.0, "memory", "{:.0f}"),
                                     cell(0.0, "fetch", "{:.0f}")]
        rows["Dense (offload)"] += [cell(100.0, "quality", "{:.0f}"), cell(0.0, "memory", "{:.0f}"),
                                    cell(100.0, "fetch", "{:.0f}")]

    k = 3
    lines = [
        "% Generated by make_table.py; do not edit by hand.",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{l" + ("|" + "r" * k) * len(MODELS) + "}",
        "\\toprule",
        " & " + " & ".join(f"\\multicolumn{{{k}}}{{c{'|' if j < len(MODELS) - 1 else ''}}}{{\\textbf{{{name}}}}}"
                           for j, (_, _, name) in enumerate(MODELS)) + " \\\\",
        # headers on two lines so the table fits the page width
        " & " + " & ".join(["Quality \\%", "\\% Memory", "\\% Fetch"] * len(MODELS)) + " \\\\",
        " & " + " & ".join(["", "resident", "/token"] * len(MODELS)) + " \\\\",
        "\\midrule",
    ]
    for i, (name, vals) in enumerate(rows.items()):
        if i in (2, 2 + len(RATES), 2 + 2 * len(RATES)):  # rules between the Dense, KVzip, Fast KVzip, VerifiedKV blocks
            lines.append("\\midrule")
        lines.append(name + " & " + " & ".join(vals) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    (here / "punchline_table.tex").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
