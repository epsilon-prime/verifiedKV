"""Per-arm table of a ruler_vllm.py run (e.g. the 32k / 128k eps-delta sweep): the harness's RULER
score (score_runs.py's ``score``) per task and averaged over tasks, with the rows' timing and fetch
means (TTFT, answer TPOT, decode tok/s after the forced question = mean of the rows' 1 / TPOT,
end-to-end answer tok/s, latency, bytes fetched per request).

    python tools/sweep_table.py out/r32sweep [--csv out/r32sweep/table.csv] [--rows 0:10]
"""
import argparse
import glob
import json
import re
import statistics as S
from collections import defaultdict
from pathlib import Path

ctrl = re.compile(r"[\x00-\x1f]")


def score(task, pred, gold):
    """/home/gaurav_kumar/vkv/phase4/vkv_final/score_runs.py's score, verbatim."""
    pred = ctrl.sub("", pred.strip()).strip().lower()
    hits = [1.0 if g.lower() in pred else 0.0 for g in gold]
    return 100 * (max(hits) if task.startswith("qa") else sum(hits) / len(hits))


ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--csv", default=None)
ap.add_argument("--rows", default=None, help="a:b: only rows a <= row < b (each task)")
ap.add_argument("--match", default=None, help="only arms whose name contains this")
args = ap.parse_args()

rows = defaultdict(list)
for f in sorted(glob.glob(f"{args.run}/*/rows.jsonl")):
    for r in map(json.loads, open(f)):
        if args.rows and not int(args.rows.split(":")[0]) <= r["row"] < int(args.rows.split(":")[1]):
            continue
        if args.match and args.match not in Path(f).parent.name:
            continue
        rows[Path(f).parent.name].append(r)


def order(arm):
    m = re.match(r".*_e([\d.]+)_d([\d.]+)$", arm)
    return (1, float(m[2]), float(m[1])) if m else (0 if "recall" in arm else 2, 0, 0)


tasks = sorted({r["task"] for rs in rows.values() for r in rs})
short = {"cwe": "cwe", "niah_multikey_3": "mk3", "niah_multivalue": "mv", "qa_1": "qa1"}
out = []
for arm in sorted(rows, key=order):
    rs = rows[arm]
    per = {t: [score(t, r["pred"], r["gold"]) for r in rs if r["task"] == t] for t in tasks}
    m = lambda k: S.mean(r[k] for r in rs if r.get(k) is not None)          # noqa: E731
    out.append({"arm": arm, "n": len(rs),
                **{short.get(t, t): S.mean(v) if v else float("nan") for t, v in per.items()},
                "score": S.mean(S.mean(v) for v in per.values() if v),
                "ttft_s": m("ttft"), "tpot_ms": 1e3 * m("tpot"), "dec_tok_s": m("decode_tok_s"),
                "e2e_tok_s": m("tok_s"), "lat_s": m("latency"),
                "fetch_GB": S.mean(r.get("fetch_bytes", 0.0) for r in rs) / 1e9})
cols = list(out[0])
w = {c: max(len(c), 8) for c in cols}
w["arm"] = max(len(o["arm"]) for o in out)
print("  ".join(c.rjust(w[c]) if c != "arm" else c.ljust(w[c]) for c in cols))
for o in out:
    print("  ".join(o[c].ljust(w[c]) if c == "arm" else
                    (f"{o[c]:{w[c]}d}" if isinstance(o[c], int) else f"{o[c]:{w[c]}.2f}")
                    for c in cols))
if args.csv:
    import csv
    with open(args.csv, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=cols)
        wr.writeheader()
        wr.writerows(out)
