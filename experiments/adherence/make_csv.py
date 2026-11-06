"""Flattens out/<run>/<arm>/rows.jsonl of the adherence ablation into CSVs (plain python + pandas, no torch).

    python make_csv.py out/<run> [out/<run2> ...] --out csv/

Writes (one line per answer = arm x task x row, unless stated):
  rows.csv        score, time, memory and every counter, per phase (q_ = question pass, d_ = decode steps, t_ = both):
                  decisions, fetch rate, mean k, transfer, failure after fetch / before fetch (lambda rule, T0) / before
                  fetch (plain HT) at eps0 and at every eps in the threshold grid, no-fetch misses, bias and |T*|,
                  share of the KV read and MB per answer
  per_layer.csv   arm x task x row x phase x layer counters
  per_step.csv    arm x task x row x decode step counters
  histograms.csv  arm x task x row x phase x histogram x bin: count and the bin's [lo, hi)
                  (error histograms in log10 of the relative error, k histogram in tokens)

Traffic: one entry = one token of one KV head in one layer, K+V = 2 x head_dim x 2 bytes (bf16). read_pct is
(2 x fetched entries + shortlist keys read) / (2 x the cache read by full attention over the same forwards), as in
results/eps_delta.
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd

HEAD_DIM = 128
ENTRY = 2 * HEAD_DIM * 2  # bytes of one K+V entry
ctrl = re.compile(r"[\x00-\x1f]")


def score(task, pred, gold):  # RULER string match (qa: any gold; others: fraction of golds), as kvpress
    pred = ctrl.sub("", pred.strip()).strip().lower()
    hits = [1.0 if g.lower() in pred else 0.0 for g in gold]
    return 100 * (max(hits) if task.startswith("qa") else sum(hits) / len(hits))


ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+")
ap.add_argument("--out", default="csv")
args = ap.parse_args()
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)

R, PL, PS, H = [], [], [], []
for run in args.runs:
    for f in sorted(Path(run).glob("*/rows.jsonl")):
        for line in f.read_text().splitlines():
            j = json.loads(line)
            arm = j.get("arm", f.parent.name)
            base = {"run": Path(run).name, "arm": arm, "method": arm.split(":")[0], "task": j["task"], "row": j["row"]}
            rec = {**base, "eps0": j.get("eps0"), "delta": j.get("delta"), "score": score(j["task"], j["pred"], j["gold"]),
                   "sec": j["sec"], "peak_gb": j["peak_gb"], "gpu": j["gpu"], "max_new_tokens": j.get("max_new_tokens"),
                   "kept_draw": j.get("kept"), "sketch_frac": j.get("sketch_frac"), "pred": j["pred"][:200]}
            st, x = j.get("stats"), j.get("x") or {}
            if st:
                rec["ctx"] = st["question_fwd_ctx"] / max(st["question_fwd"], 1)
                rec["decode_steps"] = st["decode_steps"]
                rec["decode_steps_any"] = st["decode_steps_any"]
                tot = {}
                for ph, p in (("question", "q"), ("decode", "d")):
                    n = max(st[f"{ph}_decisions"], 1)
                    rec[f"{p}_decisions"] = st[f"{ph}_decisions"]
                    rec[f"{p}_fetch_rate"] = st[f"{ph}_fetch_any"] / n
                    rec[f"{p}_k_mean"] = st[f"{ph}_k_sum"] / n
                    rec[f"{p}_k_mean_when_fetch"] = st[f"{ph}_k_sum"] / max(st[f"{ph}_fetch_any"], 1)
                    rec[f"{p}_xfer_per_decision"] = st[f"{ph}_xfer_sum"] / n
                    for m in ("fail", "fail_eval", "fail_T0", "fail_HT0"):
                        rec[f"{p}_{m}_rate"] = st[f"{ph}_{m}"] / n
                    rec[f"{p}_beta_rel_mean"] = st[f"{ph}_beta_rel_sum"] / n
                    rec[f"{p}_T_size_mean"] = st[f"{ph}_T_size_sum"] / n
                    rec[f"{p}_fwd_fetch_rate"] = st[f"{ph}_fwd_any"] / max(st[f"{ph}_fwd"], 1)
                    rec[f"{p}_read_pct"] = 100 * (2 * st[f"{ph}_fwd_xfer"] + st[f"{ph}_fwd_keys"]) / max(2 * st[f"{ph}_fwd_ctx"], 1)
                    rec[f"{p}_MB"] = (st[f"{ph}_fwd_xfer"] + 0.5 * st[f"{ph}_fwd_keys"]) * ENTRY / 1e6
                    for kk in ("decisions", "fetch_any", "k_sum", "xfer_sum", "fail", "fail_T0", "fail_HT0", "fwd_xfer",
                               "fwd_keys", "fwd_ctx"):
                        tot[kk] = tot.get(kk, 0) + st[f"{ph}_{kk}"]
                    X = x.get(ph)
                    if X:
                        for t, e in enumerate(x["eps_grid"]):
                            for nm in ("post", "T0", "HT"):
                                rec[f"{p}_{nm}_fail_at_{e}"] = X[f"thr_{nm}"][t] / n
                        rec[f"{p}_nofetch_share"] = X["nofetch_n"] / n
                        rec[f"{p}_fail_given_nofetch"] = X["nofetch_fail"] / max(X["nofetch_n"], 1)
                        rec[f"{p}_fail_given_fetch"] = X["fetch_fail"] / max(n - X["nofetch_n"], 1)
                        for li, v in X["layer"].items():
                            PL.append({**base, "eps0": j["eps0"], "delta": j["delta"], "phase": ph, "layer": int(li), **v})
                        if ph == "decode":
                            for s, v in enumerate(X.get("step", [])):
                                PS.append({**base, "eps0": j["eps0"], "delta": j["delta"], "step": s, **v})
                        for hname in ("hist_post", "hist_T0", "hist_HT", "hist_sketchB", "hist_k"):
                            edges = x["k_edges"] if hname == "hist_k" else x["log_edges"]
                            lo = [float("-inf")] + edges
                            hi = edges + [float("inf")]
                            for b, c in enumerate(X[hname]):
                                if c:
                                    H.append({**base, "eps0": j["eps0"], "delta": j["delta"], "phase": ph,
                                              "hist": hname[5:], "bin": b, "lo": lo[b], "hi": hi[b], "count": c})
                n = max(tot["decisions"], 1)
                rec["t_fetch_rate"] = tot["fetch_any"] / n
                for m in ("fail", "fail_T0", "fail_HT0"):
                    rec[f"t_{m}_rate"] = tot[m] / n
                rec["t_read_pct"] = 100 * (2 * tot["fwd_xfer"] + tot["fwd_keys"]) / max(2 * tot["fwd_ctx"], 1)
                rec["t_MB"] = (tot["fwd_xfer"] + 0.5 * tot["fwd_keys"]) * ENTRY / 1e6
            R.append(rec)

for name, rows in (("rows", R), ("per_layer", PL), ("per_step", PS), ("histograms", H)):
    pd.DataFrame(rows).to_csv(out / f"{name}.csv", index=False)
    print(f"{out / name}.csv: {len(rows)} lines")
