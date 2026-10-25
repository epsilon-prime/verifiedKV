"""Add Vkv1 chern-grad to the Phase 3 final-eval scores and check that the shared arms match across the two row sets.

Final eval (phase3 scores_so_far.csv): Qwen3-4B-Instruct-2507, RULER-32k, 16x, TEST rows 0-49 (50 per task).
This sweep (../out/r20_all): same model / length / ratio, DEV rows 152-171 (20 per task). Shared arms:
    dense <-> full, vkv_nofetch_kvzip <-> bern:0.0625, vkv_fetch_kvzip <-> needle_final_c:0.9375
Writes scores_with_vkv1.csv (Phase 3 rows + the two chern-grad arms) and prints the per-task / mean comparison.
"""

import json
import math
import re
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
P3 = Path.home() / "phase3/experiments/final_eval/analysis/scores_so_far.csv"
R20 = HERE.parent / "out/r20_all"
TASK = {"niah_single_1": "niah_s1", "niah_single_2": "niah_s2", "niah_single_3": "niah_s3", "niah_multikey_1": "niah_mk1",
        "niah_multikey_2": "niah_mk2", "niah_multikey_3": "niah_mk3", "niah_multivalue": "niah_mv",
        "niah_multiquery": "niah_mq", "vt": "vt", "cwe": "cwe", "fwe": "fwe", "qa_1": "qa1", "qa_2": "qa2"}
ARM = {"full": "dense", "bern:0.0625": "vkv_nofetch_kvzip", "needle_final_c:0.9375": "vkv_fetch_kvzip",
       "vkv1:0.0625:0.3:0.05:chern:grad:nopersist": "vkv1_chern_03",
       "vkv1:0.0625:0.5:0.05:chern:grad:nopersist": "vkv1_chern_05"}
ctrl = re.compile(r"[\x00-\x1f]")


def score(task, pred, gold):  # RULER string_match_all / string_match_part (qa), as vkv1_down/summarize.py
    pred = ctrl.sub("", pred.strip()).strip().lower()
    hits = [1.0 if g.lower() in pred else 0.0 for g in gold]
    return 100 * (max(hits) if task.startswith("qa") else sum(hits) / len(hits))


rows = []
for arm, meth in ARM.items():
    for line in open(R20 / arm / "rows.jsonl"):
        r = json.loads(line)
        rows.append({"method": meth, "task": TASK[r["task"]], "row": r["row"], "score": score(r["task"], r["pred"], r["gold"])})
r20 = pd.DataFrame(rows)
p3 = pd.read_csv(P3)
q = p3[(p3.model == "Qwen3") & (p3.dataset == "RULER") & (p3.context_k == 32)]
# merged CSV for the figure: Phase 3 as is + the chern-grad task means (Qwen3, RULER, 32k, 16x)
# vkv1_devrow_dense = the full cache on the same dev rows (only the relative plot reads it; never drawn)
add = (r20[r20.method.str.startswith("vkv1") | (r20.method == "dense")]
       .assign(method=lambda x: x.method.where(x.method != "dense", "vkv1_devrow_dense"))
       .groupby(["method", "task"], as_index=False)["score"].mean()
       .assign(model="Qwen3", dataset="RULER", context_k=32, rate=16))
pd.concat([p3, add[p3.columns]], ignore_index=True).to_csv(HERE / "scores_with_vkv1.csv", index=False)
print(f"wrote scores_with_vkv1.csv: {len(p3)} Phase 3 rows + {len(add)} rows (2 Vkv1 arms + dev-row full)")

f16 = q[q.rate == 16].pivot_table(index="task", columns="method", values="score")
d20 = r20.groupby(["task", "method"])["score"].mean().unstack()
shared = ["dense", "vkv_nofetch_kvzip", "vkv_fetch_kvzip"]
lab = {"dense": "full", "vkv_nofetch_kvzip": "bern", "vkv_fetch_kvzip": "needle", "vkv1_chern_03": "chern.3",
       "vkv1_chern_05": "chern.5"}
print("\nper task, 16x: final eval (test rows 0-49, n=50) | this sweep (dev rows 152-171, n=20)")
print(f"{'task':9} " + " ".join(f"{lab[m]:>15}" for m in shared) + f" {'chern.3':>8} {'chern.5':>8}")
for t in f16.index:
    print(f"{t:9} " + " ".join(f"{f16.loc[t, m]:6.1f} | {d20.loc[t, m]:6.1f}" for m in shared)
          + f" {d20.loc[t, 'vkv1_chern_03']:8.1f} {d20.loc[t, 'vkv1_chern_05']:8.1f}")
print("\nmean over 13 tasks (approx. SE of the difference from the dev-row spread: sd * sqrt(1/260 + 1/650))")
for m in shared:
    sd = r20[r20.method == m]["score"].std()
    a, b = f16[m].mean(), d20[m].mean()
    se = sd * math.sqrt(1 / 260 + 1 / 650)
    print(f"  {lab[m]:7} final {a:6.2f}  dev {b:6.2f}  dev - final {b - a:+5.2f} (+- {se:4.2f}, {(b - a) / se:+.1f} SE)")
for m in ("vkv1_chern_03", "vkv1_chern_05"):
    print(f"  {lab[m]:7} dev {d20[m].mean():6.2f}")
print("\nother final-eval arms at 16x (test rows):", {m: round(f16[m].mean(), 2) for m in f16.columns if m not in shared})
