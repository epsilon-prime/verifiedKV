"""Scores of finished cells (no torch needed).

- RULER: string_match from metrics.json.
- LOFT: the official subspan_em (loft_official/score_loft.py).
- If a cell ran more than once, the newest finished run is used.
"""

import glob
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from loft_official import score_loft as sl  # noqa: E402

MODEL = {"Qwen--Qwen3-4B-Instruct-2507": "Qwen3", "Qwen--Qwen3.5-9B": "Qwen3.5",
         "mistralai--Ministral-3-8B-Instruct-2512-BF16": "Ministral-3"}
RATE = {"0.88": 8, "0.94": 16, "0.97": 32}  # folder names keep two decimals: 0.875, 0.9375, 0.96875
RULER_TASK = {"niah_single_1": "niah_s1", "niah_single_2": "niah_s2", "niah_single_3": "niah_s3",
              "niah_multikey_1": "niah_mk1", "niah_multikey_2": "niah_mk2", "niah_multikey_3": "niah_mk3",
              "niah_multiquery": "niah_mq", "niah_multivalue": "niah_mv", "vt": "vt", "cwe": "cwe", "fwe": "fwe",
              "qa_1": "qa1", "qa_2": "qa2"}


LOFT_FIRST = 25  # the paper scores the first 25 test questions of each LOFT task; None scores all of them


def cell_score(d: str, dataset: str) -> tuple[str, float, int]:
    """(task name as in the CSV, score 0-100, number of answers) of one finished run folder."""
    df = pd.read_csv(os.path.join(d, "predictions.csv"))
    saved = json.load(open(os.path.join(d, "metrics.json")))
    if dataset == "loft":
        if LOFT_FIRST is not None:
            if len(df) < LOFT_FIRST:
                raise SystemExit(f"{d}: only {len(df)} questions, fewer than {LOFT_FIRST}")
            df = df.head(LOFT_FIRST).copy()
        df["answers"] = df["answers"].apply(sl.gold_list)
        task = df["task"].iloc[0].rsplit("_", 1)[0]  # nq_32k -> nq
        return task, 100 * sl.score(df, sl.official_extract)["subspan_em"], len(df)
    (task, m), = saved.items()
    return RULER_TASK[task], float(m["string_match"]), len(df)


def scan(root: str, methods: dict[str, str]) -> pd.DataFrame:
    """Every finished cell under ``root`` whose arm is a key of ``methods`` (arm -> CSV method name).
    no_press (ratio 0) is written once per rate."""
    rows = []
    for base in sorted(glob.glob(os.path.join(root, "*__*", ""))):
        parts = os.path.basename(os.path.normpath(base)).split("__")
        dataset, task_dir, model, arm, ratio = parts[:5]
        if arm not in methods:
            continue
        d = sl.newest_finished(base)
        if d is None:
            continue
        task, score, n = cell_score(d, dataset)
        L = task_dir.rsplit("_", 1)[1] if dataset == "loft" else dataset[len("ruler"):]
        rates = RATE.values() if arm == "no_press" else [RATE[ratio]]
        for rate in rates:
            rows.append({"model": MODEL[model], "dataset": "LOFT" if dataset == "loft" else "RULER",
                         "context_k": int(L[:-1]), "rate": rate, "method": methods[arm], "task": task,
                         "score": score, "n": n, "cell": os.path.relpath(d, root)})
    return pd.DataFrame(rows)
