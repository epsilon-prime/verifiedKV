"""Merge out/r20/j*/<arm>/rows.jsonl into out/r20_all and print the final table (scores with the vkv1_down/summarize.py
string match, paired differences +- SE, MB per answer, realised fail, time).   python analyze.py"""

import glob
import json
import math
import re
from collections import defaultdict
from pathlib import Path

ctrl = re.compile(r"[\x00-\x1f]")


def score(task, pred, gold):
    pred = ctrl.sub("", pred.strip()).strip().lower()
    hits = [1.0 if g.lower() in pred else 0.0 for g in gold]
    return 100 * (max(hits) if task.startswith("qa") else sum(hits) / len(hits))


R = defaultdict(dict)
for f in sorted(glob.glob("out/r20/j*/*/rows.jsonl")):
    arm = f.split("/")[-2]
    for line in open(f):
        r = json.loads(line)
        R[arm][(r["task"], r["row"])] = r
out = Path("out/r20_all")
for arm, rows in R.items():
    (out / arm).mkdir(parents=True, exist_ok=True)
    with open(out / arm / "rows.jsonl", "w") as fh:
        for k in sorted(rows):
            fh.write(json.dumps(rows[k]) + "\n")
F, B, N = "full", "bern:0.0625", "needle_final_c:0.9375"
V3, V5 = "vkv1:0.0625:0.3:0.05:chern:grad:nopersist", "vkv1:0.0625:0.5:0.05:chern:grad:nopersist"
ARMS = [F, B, N, V3, V5]
NAME = {F: "full", B: "bern 1/16", N: "needle 16x", V3: "chern-grad .3", V5: "chern-grad .5"}
common = sorted(set.intersection(*(set(R[a]) for a in ARMS)))
S = {a: {k: score(k[0], R[a][k]["pred"], R[a][k]["gold"]) for k in common} for a in ARMS}
tasks = sorted({t for t, _ in common})
print(f"contexts with all arms: {len(common)} ({len(tasks)} tasks); rows per arm: " + ", ".join(f"{NAME[a]} {len(R[a])}" for a in ARMS))
print("\nscore per task (mean over rows)")
print(f"{'task':18} {'n':>3} " + " ".join(f"{NAME[a]:>13}" for a in ARMS))
for t in tasks:
    ks = [k for k in common if k[0] == t]
    print(f"{t:18} {len(ks):3d} " + " ".join(f"{sum(S[a][k] for k in ks) / len(ks):13.1f}" for a in ARMS))
print(f"{'MEAN (all ctx)':18} {len(common):3d} " + " ".join(f"{sum(S[a].values()) / len(common):13.2f}" for a in ARMS))
print(f"{'MEAN of tasks':18} {'':3} " + " ".join(
    f"{sum(sum(S[a][k] for k in common if k[0] == t) / sum(1 for k in common if k[0] == t) for t in tasks) / len(tasks):13.2f}" for a in ARMS))


def paired(a, b):
    d = [S[a][k] - S[b][k] for k in common]
    n = len(d)
    mu = sum(d) / n
    sd = math.sqrt(sum((x - mu) ** 2 for x in d) / (n - 1))
    return mu, sd / math.sqrt(n), sum(x > 0 for x in d), sum(x < 0 for x in d)


print("\npaired differences over all contexts (mean +- SE; contexts better / worse)")
for a, b in [(V3, F), (V5, F), (N, F), (B, F), (V3, N), (V5, N), (V3, V5)]:
    mu, se, up, dn = paired(a, b)
    print(f"  {NAME[a]:>13} - {NAME[b]:<13} {mu:+6.2f} +- {se:4.2f}   ({up} better / {dn} worse)")


def mb(a, r):
    st = r.get("stats") or {}
    if a.startswith("vkv1"):
        return (st["question_fwd_xfer"] + st["decode_fwd_xfer"]) * 512 / 1e6
    ns = r.get("needle_stats") or {}
    return (ns.get("question_fetched_entries", 0) + ns.get("decode_fetched_entries", 0)) * 512 / 1e6 if ns else 0.0


print("\ncost per answer (mean over contexts)")
print(f"{'arm':>13} {'MB/answer':>10} {'Q fail':>7} {'D fail':>7} {'Q fwd%':>7} {'D fwd%':>8} {'sec':>6} {'peak GB':>8}")
for a in ARMS:
    rs = [R[a][k] for k in common]
    m = sum(mb(a, r) for r in rs) / len(rs)
    sec = sum(r["sec"] for r in rs) / len(rs)
    pk = max(r["peak_gb"] for r in rs)
    if a.startswith("vkv1"):
        tot = lambda key: sum(r["stats"][key] for r in rs)  # noqa: E731
        qf, df = tot("question_fail") / tot("question_decisions"), tot("decode_fail") / tot("decode_decisions")
        qx, dx = 100 * tot("question_fwd_xfer") / tot("question_fwd_ctx"), 100 * tot("decode_fwd_xfer") / tot("decode_fwd_ctx")
        print(f"{NAME[a]:>13} {m:10.1f} {100 * qf:6.2f}% {100 * df:6.2f}% {qx:7.3f} {dx:8.4f} {sec:6.0f} {pk:8.2f}")
    else:
        print(f"{NAME[a]:>13} {m:10.1f} {'':>7} {'':>7} {'':>7} {'':>8} {sec:6.0f} {pk:8.2f}")
print("\nchern-grad MB/answer per task (.3 / .5) vs needle")
for t in tasks:
    ks = [k for k in common if k[0] == t]
    print(f"  {t:18} " + " / ".join(f"{sum(mb(a, R[a][k]) for k in ks) / len(ks):7.1f}" for a in (V3, V5, N)))
