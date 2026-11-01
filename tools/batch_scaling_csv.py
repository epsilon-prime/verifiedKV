"""batch_scaling.csv from the ruler_batch.py batch-size sweep (out/bscale/<tag>_b<B>/summary.json).

    python tools/batch_scaling_csv.py out/bscale <dest dir>
Per (method, eps, delta, B): decode throughput (all decode-only steps; and at full concurrency), TPOT
(per-request mean of the answer steps after the first token, then median / mean over requests), TTFT,
end-to-end answer throughput, latency, mean decode concurrency.
"""
import csv
import json
import re
import sys
from pathlib import Path

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
out = []
for d in sorted(src.iterdir()):
    m = re.match(r"(.+)_b(\d+)$", d.name)
    f = d / "summary.json"
    if not m or not f.exists():
        continue
    tag, b = m[1], int(m[2])
    s = json.load(open(f))
    v = re.match(r"vkv-fkvz_e([\d.]+)_d([\d.]+)$", tag)
    method = "vkv" if v else "dense_recall" if "recall" in tag else "fastkvzip_evict"
    out.append({
        "config": (f"vkv ε={v[1]}, δ={v[2]}" if v else
                   {"dense_recall": "Dense recall (exact)", "fastkvzip_evict": "FastKVzip evict"}[method]),
        "method": method, "eps": v[1] if v else "", "delta": v[2] if v else "", "batch": b,
        "n_requests": s["n_req"],
        "decode_tok_s": round(s["decode_tput"], 2),
        "decode_tok_s_full_batch": round(s["decode_tput_steady"], 2) if s.get("decode_tput_steady") else "",
        "mean_decode_concurrency": round(s["mean_decode_concurrency"], 2),
        "tpot_ms_median": round(s["tpot_median_ms"], 2), "tpot_ms_mean": round(s["tpot_mean_ms"], 2),
        "ttft_s_median": round(s["ttft_median"], 2), "latency_s_median": round(s["latency_median"], 2),
        "e2e_tok_s": round(s["e2e_tput"], 2), "wall_s": round(s["wall_s"], 1),
        "preemptions": s["preemptions"]})
order = {"dense_recall": 0, "vkv": 1, "fastkvzip_evict": 2}
out.sort(key=lambda r: (order[r["method"]], float(r["eps"] or 0), r["batch"]))
with open(dst / "batch_scaling.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(out[0]))
    w.writeheader()
    w.writerows(out)
for r in out:
    print(r["config"], r["batch"], r["decode_tok_s"], r["decode_tok_s_full_batch"], r["tpot_ms_median"],
          r["mean_decode_concurrency"], r["preemptions"])
