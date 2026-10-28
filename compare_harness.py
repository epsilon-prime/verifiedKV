"""Our RULER arms (out/<run>/<arm>/rows.jsonl) against the harness's stored A40 reference arms
(phase4/experiments/vkv1_r20/out/r20_all: full, Bernoulli 1/16, needle, chern-grad on exact
weights), paired on the contexts both have; plus timing and fetch per answer.
    python compare_harness.py out/r32k5 [--rows 152:154]"""
import argparse, glob, json, math, re, statistics as st
from collections import defaultdict
from pathlib import Path

H = "/home/gaurav_kumar/vkv/phase4/experiments/vkv1_r20/out/r20_all"
HN = {"full": "H full", "bern:0.0625": "H bern 1/16", "needle_final_c:0.9375": "H needle 16x",
      "vkv1:0.0625:0.3:0.05:chern:grad:nopersist": "H chern-grad .3 (exact w)",
      "vkv1:0.0625:0.5:0.05:chern:grad:nopersist": "H chern-grad .5 (exact w)"}
ctrl = re.compile(r"[\x00-\x1f]")


def score(task, pred, gold):
    pred = ctrl.sub("", pred.strip()).strip().lower()
    h = [1.0 if g.lower() in pred else 0.0 for g in gold]
    return 100 * (max(h) if task.startswith("qa") else sum(h) / len(h))


ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--rows", default=None)
a = ap.parse_args()
lo, hi = (int(x) for x in a.rows.split(":")) if a.rows else (0, 10**9)
R = {}
for f in sorted(glob.glob(f"{a.run}/*/rows.jsonl")):
    R[Path(f).parent.name] = {(r["task"], r["row"]): r for r in map(json.loads, open(f))
                              if lo <= r["row"] < hi}
for arm, name in HN.items():
    R[name] = {(r["task"], r["row"]): r for r in map(json.loads, open(f"{H}/{arm}/rows.jsonl"))
               if lo <= r["row"] < hi}
ours = [k for k in R if not k.startswith("H ")]
common = set.intersection(*(set(R[k]) for k in ours)) if ours else set()
for k in R:
    common &= set(R[k])
common = sorted(common)
S = {k: {c: score(c[0], R[k][c]["pred"], R[k][c]["gold"]) for c in common} for k in R}
print(f"contexts in every arm: {len(common)}")
print(f"{'arm':28s} {'score':>6s} | per task")
for k in R:
    per = defaultdict(list)
    for c, v in S[k].items():
        per[c[0]].append(v)
    print(f"{k:28s} {st.mean(S[k].values()):6.2f} | " + " ".join(f"{t[:9]}:{st.mean(v):.0f}" for t, v in sorted(per.items())))


def paired(x, y):
    d = [S[x][c] - S[y][c] for c in common]
    mu = st.mean(d)
    se = math.sqrt(sum((v - mu) ** 2 for v in d) / (len(d) - 1) / len(d)) if len(d) > 1 else 0
    same = sum(R[x][c]["pred"].strip() == R[y][c]["pred"].strip() for c in common)
    return f"{mu:+6.2f} +- {se:4.2f} (better {sum(v > 0 for v in d)} / worse {sum(v < 0 for v in d)}, identical {same})"


print("\npaired (A - B):")
for x in ours:
    for y in ("dense", "H full", "H chern-grad .3 (exact w)"):
        if y in R and y != x:
            print(f"  {x:24s} - {y:26s} {paired(x, y)}")
print("\ntiming and fetch per answer (our arms; first context of each arm left out of timing):")
print(f"{'arm':24s} {'prefill s':>9s} {'q-pass s':>8s} {'TTFT s':>7s} {'TPOT ms':>8s} {'tok/s':>6s} "
      f"{'lat s':>6s} {'copied MB':>9s} {'req. MB':>8s} {'shortl MB':>9s}")
for x in ours:
    rows = [R[x][c] for c in common]
    t = rows[1:] if len(rows) > 1 else rows
    g = lambda k: st.mean(r[k] for r in t if r.get(k) is not None) if any(r.get(k) is not None for r in t) else float("nan")
    cp = st.mean(r.get("fetch_bytes", 0) for r in rows) / 1e6
    rq = st.mean(r.get("vkv", {}).get("xfer_sum", 0) * 512 for r in rows) / 1e6
    sl = st.mean(r.get("vkv", {}).get("keys_sum", 0) * 256 for r in rows) / 1e6
    print(f"{x:24s} {g('prefill_s') if 'prefill_s' in t[0] else g('ttft'):9.2f} {g('question_s') if 'question_s' in t[0] else 0:8.2f} "
          f"{g('ttft'):7.2f} {g('tpot') * 1e3:8.1f} {1 / g('tpot') if g('tpot') else 0:6.1f} {g('latency'):6.2f} "
          f"{cp:9.1f} {rq:8.1f} {sl:9.1f}")
