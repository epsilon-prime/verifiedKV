"""Builds this folder's CSVs from ../final_results/combined_loft25.csv (read only; nothing else is changed).

    python build.py && ./make_figures.sh

Protocol of every number here: RULER = 4 tasks (niah_mk3, niah_mv, cwe, qa1), 50 test rows (Ministral-3 RULER-128K
phase 3 baselines: 20 rows); LOFT = 5 tasks, the first 25 test questions of each, the same 25 for every method and
model.
"""

import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "final_results", "combined_loft25.csv")

RULER_TASKS = ["niah_mk3", "niah_mv", "cwe", "qa1"]
LOFT_TASKS = ["nq", "hotpotqa", "musique", "qampari", "quest"]
MODELS = {"Qwen3": "Qwen3-4B", "Qwen3.5": "Qwen3.5-9B", "Ministral-3": "Ministral-3-8B"}
# method id -> (paper label, origin, what it is)
METHODS = {
    "dense": ("Dense", "phase 3", "full KV cache, no compression"),
    "kvzip": ("KVzip", "phase 3", "KVzip top-k eviction"),
    "fastkvzip": ("Fast KVzip", "phase 3", "Fast KVzip (learned gates) top-k eviction"),
    "vkv_nofetch_kvzip": ("vKV (no fetch)+KVzip", "phase 3",
                          "Bernoulli/Poisson sample on KVzip scores, reweighted, never fetches"),
    "vkv_final_kvzip": ("vKV+KVzip", "final grid", "VerifiedKV final: KVzip scores, 2-bit sketch + 128 shortlist"),
    "vkv_final_fastkvzip": ("vKV+Fast KVzip", "final grid",
                            "VerifiedKV final: Fast KVzip scores, 2-bit sketch + 128 shortlist"),
    "vkv_final_perfect": ("vKV, exact sketch", "final grid",
                          "VerifiedKV final, KVzip scores, exact weights as sketch (2% memory charged); reference"),
    "vkv_fetch_kvzip": ("vKV fetch (phase 3)+KVzip", "phase 3", "earlier VerifiedKV version, superseded"),
    "vkv_fetch_fastkvzip": ("vKV fetch (phase 3)+Fast KVzip", "phase 3", "earlier VerifiedKV version, superseded"),
}
ORDER = list(METHODS)

df = pd.read_csv(SRC)
df = df[(df.dataset == "LOFT") | df.task.isin(RULER_TASKS)].copy()
assert set(df.method) <= set(METHODS), set(df.method) - set(METHODS)
assert (df[df.dataset == "LOFT"].n == 25).all()
df["model_dir"] = df.model.map(MODELS)
df["label"] = df.method.map(lambda m: METHODS[m][0])
key = ["model", "dataset", "context_k", "rate", "method", "task"]
df = df.sort_values(key).reset_index(drop=True)
df.to_csv(os.path.join(HERE, "all_models.csv"), index=False,
          columns=key + ["score", "n", "label", "cell"])
pd.DataFrame([(m, *v) for m, v in METHODS.items()], columns=["method", "label", "origin", "description"]).to_csv(
    os.path.join(HERE, "methods.csv"), index=False)

for model, d_model in df.groupby("model"):
    out = os.path.join(HERE, MODELS[model])
    os.makedirs(out, exist_ok=True)
    summ = []
    for (ds, ctx), d in d_model.groupby(["dataset", "context_k"]):
        tasks = RULER_TASKS if ds == "RULER" else LOFT_TASKS
        cols = [m for m in ORDER if m in set(d.method)]
        blocks = []
        for rate, dr in d.groupby("rate"):
            w = dr.pivot(index="task", columns="method", values="score").reindex(tasks)[cols]
            w.loc["mean"] = w.mean()
            w.insert(0, "rate", f"{rate}x")
            blocks.append(w)
            n = dr.groupby("method").n.first()
            summ += [{"dataset": ds, "context_k": ctx, "rate": rate, "method": m, "label": METHODS[m][0],
                      "mean_score": round(w.loc["mean", m], 2), "n_tasks": int(w[m].iloc[:-1].notna().sum()),
                      "n_per_task": int(n[m])} for m in cols]
        wide = pd.concat(blocks).reset_index().rename(columns={"index": "task"})
        wide = wide[["rate", "task"] + cols].round(2)
        wide.columns = ["rate", "task"] + [METHODS[m][0] for m in cols]
        wide.to_csv(os.path.join(out, f"{ds}_{ctx}k.csv"), index=False)
    pd.DataFrame(summ).to_csv(os.path.join(out, "summary.csv"), index=False)
    d_model.to_csv(os.path.join(out, "results_long.csv"), index=False, columns=key + ["score", "n", "label", "cell"])
print(f"wrote all_models.csv ({len(df)} rows), methods.csv, and {', '.join(MODELS.values())}/")
