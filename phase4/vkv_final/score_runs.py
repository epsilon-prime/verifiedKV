"""python3 score_runs.py out/RUN1 [out/RUN2 ...]   -> per-arm score, per-task scores, sec/context, and paired diffs."""
import glob, json, math, re, sys
from collections import defaultdict
from pathlib import Path

ctrl = re.compile(r"[\x00-\x1f]")
def score(task, pred, gold):
    pred = ctrl.sub("", pred.strip()).strip().lower()
    hits = [1.0 if g.lower() in pred else 0.0 for g in gold]
    return 100 * (max(hits) if task.startswith("qa") else sum(hits) / len(hits))

arms = defaultdict(dict)                       # arm -> {(task,row): record}
for d in sys.argv[1:]:
    for f in sorted(glob.glob(f"{d}/**/rows.jsonl", recursive=True)):
        arm = Path(f).parent.name
        for line in open(f):
            r = json.loads(line)
            arms[arm][(r["task"], r["row"])] = r
names = sorted(arms)
for a in names:
    S = {k: score(k[0], r["pred"], r["gold"]) for k, r in arms[a].items()}
    sec = sum(r["sec"] for r in arms[a].values()) / len(S)
    gpus = sorted({r.get("gpu", "?") for r in arms[a].values()})
    print(f"\n== {a}: {len(S)} contexts, mean score {sum(S.values()) / len(S):.2f}, {sec:.0f} s/context, GPU {gpus}")
    per = defaultdict(list)
    for k, v in S.items():
        per[k[0]].append(v)
    print("   " + "  ".join(f"{t}:{sum(v) / len(v):.0f}({len(v)})" for t, v in sorted(per.items())))
print("\npaired differences (A - B) on the contexts both have: mean +- SE (A better / A worse)")
for i, a in enumerate(names):
    for b in names[i + 1:]:
        K = sorted(set(arms[a]) & set(arms[b]))
        if len(K) < 2:
            continue
        d = [score(k[0], arms[a][k]["pred"], arms[a][k]["gold"]) - score(k[0], arms[b][k]["pred"], arms[b][k]["gold"]) for k in K]
        mu = sum(d) / len(d)
        se = math.sqrt(sum((x - mu) ** 2 for x in d) / (len(d) - 1) / len(d))
        same = sum(arms[a][k]["pred"] == arms[b][k]["pred"] for k in K)
        print(f"  {a} - {b}: {mu:+.2f} +- {se:.2f}  ({sum(x > 0 for x in d)} / {sum(x < 0 for x in d)}), n={len(K)}, identical answers {same}")
