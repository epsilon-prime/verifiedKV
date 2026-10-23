"""Summarise run_downstream.py: RULER score per task and arm (official string-match metrics), Vkv1's online
statistics, and optional prediction-identity checks between arms.

    python summarize.py out/<run> [--arms REGEX] [--same ARM_A ARM_B ...]
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--arms", default=None, help="regex: only arms matching")
ap.add_argument("--same", nargs="+", default=[], help="pairs of arms whose predictions should be identical")
args = ap.parse_args()
run = Path(args.run)
ctrl = re.compile(r"[\x00-\x1f]")


def score(task, pred, gold):
    pred = ctrl.sub("", pred.strip()).strip().lower()
    hits = [1.0 if g.lower() in pred else 0.0 for g in gold]
    return 100 * (max(hits) if task.startswith("qa") else sum(hits) / len(hits))


pat = re.compile(args.arms) if args.arms else None
arms = sorted(p.name for p in run.iterdir() if (p / "rows.jsonl").exists() and (pat is None or pat.search(p.name)))
data = {}
for a in arms:
    seen, recs = set(), []
    for x in (run / a / "rows.jsonl").read_text().splitlines():
        r = json.loads(x)
        if (r["task"], r["row"]) not in seen:
            seen.add((r["task"], r["row"]))
            recs.append(r)
    data[a] = recs
common = set.intersection(*[{(r["task"], r["row"]) for r in rs} for rs in data.values()])
tasks = sorted({t for t, _ in common})
print(f"{run}: {len(common)} contexts common to all arms ({len(tasks)} tasks)\n")
w = max(len(a) for a in arms) + 1
print(f"{'arm':{w}s} {'mean':>6s} {'sec':>6s} | " + " ".join(f"{t.replace('niah_', 'n_')[:11]:>11s}" for t in tasks))
for a in arms:
    per = defaultdict(list)
    secs = []
    for r in data[a]:
        if (r["task"], r["row"]) in common:
            per[r["task"]].append(score(r["task"], r["pred"], r["gold"]))
            secs.append(r["sec"])
    means = {t: sum(v) / len(v) for t, v in per.items()}
    print(f"{a:{w}s} {sum(means.values()) / len(means):6.1f} {sum(secs) / len(secs):6.1f} | "
          + " ".join(f"{means[t]:11.1f}" for t in tasks))

rows = [a for a in arms if any("stats" in r for r in data[a])]
if rows:
    print("\nVkv1 online statistics (pooled; decisions = (layer, query head, position); fwd = (forward, KV head);"
          " xfer = entries read from the host)")
    print(f"{'arm':{w}s} {'kept':>6s} | {'ph':>2s} {'fetch':>6s} {'k*|f':>7s} {'xfer|f':>7s} {'fwd_any':>7s} "
          f"{'fwd%':>7s} {'fail':>7s} {'T0':>7s} {'HT0':>7s} {'beta%':>6s} {'|T*|':>6s} | {'steps_any':>9s} "
          f"{'pers%':>6s} {'MB/ans':>7s}")
    for a in rows:
        rs = [r for r in data[a] if "stats" in r and (r["task"], r["row"]) in common]
        S = defaultdict(float)
        for r in rs:
            for k, v in r["stats"].items():
                S[k] += v
        kept = sum(r["kept"] for r in rs) / len(rs)
        # bytes moved: entries x (K + V) x head_dim x 2 bytes; head_dim 128 (Qwen3-4B)
        mb = (S["question_fwd_xfer"] + S["decode_fwd_xfer"]) * 2 * 128 * 2 / 1e6 / len(rs)
        for k, ph in (("question", "Q"), ("decode", "D")):
            n, nf = max(S[f"{k}_decisions"], 1), max(S[f"{k}_fetch_any"], 1)
            tail = (f"{S['decode_steps_any'] / max(S['decode_steps'], 1):9.3f} "
                    f"{100 * S['persisted_extra'] / max(S['persisted_slots'], 1):6.3f} {mb:7.1f}") if ph == "D" else ""
            print(f"{a if ph == 'Q' else '':{w}s} {f'{kept:.4f}' if ph == 'Q' else '':>6s} | {ph:>2s} "
                  f"{S[f'{k}_fetch_any'] / n:6.3f} {S[f'{k}_k_sum'] / nf:7.1f} {S[f'{k}_xfer_sum'] / nf:7.1f} "
                  f"{S[f'{k}_fwd_any'] / max(S[f'{k}_fwd'], 1):7.3f} "
                  f"{100 * S[f'{k}_fwd_xfer'] / max(S[f'{k}_fwd_ctx'], 1):7.4f} {S[f'{k}_fail'] / n:7.4f} "
                  f"{S[f'{k}_fail_T0'] / n:7.4f} {S[f'{k}_fail_HT0'] / n:7.4f} "
                  f"{100 * S[f'{k}_beta_rel_sum'] / n:6.2f} {S[f'{k}_T_size_sum'] / n:6.1f} | {tail}")

for i in range(0, len(args.same) - 1, 2):
    A, B = args.same[i], args.same[i + 1]
    pa = {(r["task"], r["row"]): r["pred"] for r in data.get(A, [])}
    pb = {(r["task"], r["row"]): r["pred"] for r in data.get(B, [])}
    keys = sorted(set(pa) & set(pb))
    same = sum(pa[k] == pb[k] for k in keys)
    print(f"\nidentical predictions {A} vs {B}: {same}/{len(keys)}")
    for k in keys:
        if pa[k] != pb[k]:
            print(f"  {k}: {pa[k][:80]!r} | {pb[k][:80]!r}")
