"""Score a vkv_final downstream run and compare it with the full cache and the perfect-sketch Vkv on the same contexts.

    python summarize.py out/<run> [out/<run2> ...]

Reference arms come from ../experiments/vkv1_r20/out/r20_all (A40, rows 152-171 of RULER-32k, Qwen3-4B, 16x):
full cache, and chern-grad planned on the exact weights ("perfect sketch") at eps0 0.3 and 0.5. Run on A40 for paired
comparisons: the Bernoulli draw depends on the GPU model.

Score: RULER's string match (qa: any gold in the answer; others: fraction of golds found). Paired differences are
mean +- standard error over the shared contexts. Transfer is in MB per answer: certified fetches count a full K+V
entry (512 B at head_dim 128, bf16), shortlist key reads half of one.
"""

import glob
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
R20 = HERE.parent / "experiments" / "vkv1_r20" / "out" / "r20_all"
REFS = {"full": "full cache", "vkv1:0.0625:0.3:0.05:chern:grad:nopersist": "perfect eps0 0.3",
        "vkv1:0.0625:0.5:0.05:chern:grad:nopersist": "perfect eps0 0.5"}
ctrl = re.compile(r"[\x00-\x1f]")
ENTRY = 512  # bytes per K+V entry per KV head (2 x 128 dims x bf16)


def score(task, pred, gold):
    pred = ctrl.sub("", pred.strip()).strip().lower()
    hits = [1.0 if g.lower() in pred else 0.0 for g in gold]
    return 100 * (max(hits) if task.startswith("qa") else sum(hits) / len(hits))


def load(path):
    return {(r["task"], r["row"]): r for r in map(json.loads, Path(path).read_text().splitlines())}


arms = {}
for d in sys.argv[1:]:
    for f in sorted(glob.glob(f"{d}/**/rows.jsonl", recursive=True)):
        name = f"{Path(d).name}/{Path(f).parent.name}"
        arms.setdefault(name, {}).update(load(f))
refs = {label: load(R20 / arm / "rows.jsonl") for arm, label in REFS.items() if (R20 / arm / "rows.jsonl").exists()}
for name, rows in arms.items():
    K = sorted(k for k in rows if all(k in ref for ref in refs.values()))
    S = {k: score(k[0], rows[k]["pred"], rows[k]["gold"]) for k in K}
    print(f"\n== {name}: {len(K)} contexts ({len({k[0] for k in K})} tasks), GPU {rows[K[0]].get('gpu', '?')}")
    print(f"  score {sum(S.values()) / len(K):.2f}")
    for label, ref in refs.items():
        d = [S[k] - score(k[0], ref[k]["pred"], ref[k]["gold"]) for k in K]
        mu = sum(d) / len(d)
        se = math.sqrt(sum((x - mu) ** 2 for x in d) / max(len(d) - 1, 1) / len(d))
        ref_mean = sum(score(k[0], ref[k]["pred"], ref[k]["gold"]) for k in K) / len(K)
        print(f"  vs {label:17} ({ref_mean:6.2f}): {mu:+6.2f} +- {se:4.2f}  (better on {sum(x > 0 for x in d)}, "
              f"worse on {sum(x < 0 for x in d)})")
    st = [rows[k]["stats"] for k in K if "stats" in rows[k]]
    if st:
        t = lambda key: sum(s.get(key, 0) for s in st)  # noqa: E731
        n = len(st)
        print(f"  draw keeps {sum(rows[k].get('kept_draw') or 0 for k in K) / len(K):.4f} of the full cache "
              f"(sketch {sum(rows[k].get('sketch_frac') or 0 for k in K) / len(K):.4f})")
        print(f"  {'phase':8} {'decisions':>10} {'fetching':>9} {'k/fetch':>8} {'fail@eps0':>9} {'fail@0.5':>8} "
              f"{'ctx read/fwd':>12} {'fetch MB':>9} {'key MB':>7}")
        for p in ("question", "decode"):
            dec = max(t(f"{p}_decisions"), 1)
            fa = t(f"{p}_fetch_any")
            print(f"  {p:8} {dec / n:10.0f} {100 * fa / dec:8.1f}% {t(f'{p}_k_sum') / max(fa, 1):8.0f} "
                  f"{100 * t(f'{p}_fail') / dec:8.2f}% {100 * t(f'{p}_fail_eval') / dec:7.2f}% "
                  f"{100 * t(f'{p}_fwd_xfer') / max(t(f'{p}_fwd_ctx'), 1):11.3f}% "
                  f"{t(f'{p}_fwd_xfer') * ENTRY / 1e6 / n:9.1f} {0.5 * t(f'{p}_fwd_keys') * ENTRY / 1e6 / n:7.1f}")
        mb = sum((s["question_fwd_xfer"] + s["decode_fwd_xfer"] + 0.5 * (s["question_fwd_keys"] + s["decode_fwd_keys"]))
                 * ENTRY / 1e6 for s in st) / n
        print(f"  total MB per answer {mb:.1f}; decode steps per answer {t('decode_steps') / n:.0f}")
        per = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
        for k in K:
            s = rows[k]["stats"]
            v = per[k[0]]
            v[0] += s["question_fail"]
            v[1] += s["question_decisions"]
            v[2] += s["decode_fail"]
            v[3] += s["decode_decisions"]
        worst = sorted(per.items(), key=lambda kv: -kv[1][0] / max(kv[1][1], 1))[:3]
        print("  worst tasks, fail@eps0 Q / D: " + ", ".join(
            f"{tk} {100 * v[0] / max(v[1], 1):.2f} / {100 * v[2] / max(v[3], 1):.2f}%" for tk, v in worst))
    print(f"  {'task':16} {'score':>6} " + " ".join(f"{lab[:13]:>13}" for lab in refs))
    for task in sorted({k[0] for k in K}):
        ks = [k for k in K if k[0] == task]
        print(f"  {task:16} {sum(S[k] for k in ks) / len(ks):6.1f} " + " ".join(
            f"{sum(score(k[0], ref[k]['pred'], ref[k]['gold']) for k in ks) / len(ks):13.1f}" for ref in refs.values()))
