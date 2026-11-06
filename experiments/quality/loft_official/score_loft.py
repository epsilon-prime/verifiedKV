"""LOFT scores with LOFT's OFFICIAL answer parser, from saved predictions (no rerun). Use this for every LOFT number.

    python score_loft.py <results_dir> [<results_dir> ...] [--out scores.csv]

    e.g. python score_loft.py ../grid/results/final_qwen3-4b --out final_qwen3-4b_loft.csv

Why: kvpress's `benchmarks/loft/calculate_metrics.py` scores
exactly as LOFT does, but its `extract_prediction` is more lenient than LOFT's `utils.extract_prediction`
(google-deepmind/loft, main):
- LOFT reads only the FIRST line containing "[...]"; if it does not parse, the answer is empty. kvpress goes on
  to later lines;
- kvpress also falls back to the plain text after "Final Answer"; LOFT has no fallback.
On the 715 rehearsal LOFT cells, 1.6% of answers were accepted only by kvpress, raising 8 cells by 10-20 points,
all of them compressed arms. So the scores in metrics.json are not the official ones; this script is.

Per cell (the newest finished run, as readiness/summarize.py picks it): the official subspan_em (the reported
metric), em and f1 / coverage, plus the kvpress (saved) subspan_em. Every cell is first rescored with kvpress's
own parser and must reproduce its saved metrics.json exactly, or the script stops: that guards the reading of
predictions.csv (the gold answers are saved as numpy reprs without commas, ['Crick' 'Watson'], which a plain
ast.literal_eval silently merges into one wrong answer).
"""

import argparse
import ast
import glob
import json
import os
import re
import sys

import pandas as pd

# this repository's evaluation code
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "evaluation"))
from benchmarks.loft import calculate_metrics as cm  # noqa: E402

KVPRESS_EXTRACT = cm.extract_prediction


def official_extract(model_output: str, answer_prefix: str = "final answer") -> list[str]:
    """LOFT's utils.extract_prediction, same logic: the first line holding "[...]" only, no fallback; then
    LOFT's RAG configs apply convert_to_str (normalisation is done by calculate_metrics, as in LOFT)."""

    def escape_single_quotes(s: str) -> str:
        return re.sub(r"([a-zA-Z0-9])'([a-zA-Z0-9])", r"\1\'\2", s)

    preds = []
    for line in model_output.replace("*", "").replace("`", "").strip().split("\n"):
        if "[" in line and "]" in line:
            pred_as_str = line[line.find("["): line.rfind("]") + 1].strip()
            try:
                preds = ast.literal_eval(escape_single_quotes(pred_as_str))
            except Exception:  # LOFT prints and leaves preds empty
                pass
            break
    if isinstance(preds, (list, tuple)):
        return [str(p) for p in preds]
    return [str(preds)] if preds else []


def gold_list(s: str) -> list[str]:
    """The saved gold answers: a numpy repr without commas, ['a' "b's"], quoted items read one by one."""
    return [ast.literal_eval(t) for t in re.findall(r"'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"", s)]


def score(df: pd.DataFrame, extract) -> dict:
    cm.extract_prediction = extract
    try:
        return cm.calculate_metrics(df)
    finally:
        cm.extract_prediction = KVPRESS_EXTRACT


def newest_finished(base: str) -> str | None:
    """kvpress writes a rerun into a numbered subfolder; take the newest one holding metrics and predictions."""
    runs = [base] + sorted(glob.glob(os.path.join(base, "[0-9]*", "")), key=lambda d: int(os.path.basename(d[:-1])))
    done = [d for d in runs if all(os.path.isfile(os.path.join(d, f)) for f in ("metrics.json", "predictions.csv"))]
    return done[-1] if done else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+", help="result group folders (holding loft__* cell folders)")
    ap.add_argument("--out", help="write the per-cell table here (csv)")
    args = ap.parse_args()

    rows = []
    for root in args.results:
        for base in sorted(glob.glob(os.path.join(root, "loft__*", ""))):
            d = newest_finished(base)
            if d is None:
                continue
            df = pd.read_csv(os.path.join(d, "predictions.csv"))
            df["answers"] = df["answers"].apply(gold_list)
            saved = json.load(open(os.path.join(d, "metrics.json")))
            kv = score(df, KVPRESS_EXTRACT)
            if abs(kv["subspan_em"] - saved["subspan_em"]) > 1e-9:
                sys.exit(f"{d}: kvpress rescoring gives {kv['subspan_em']} but metrics.json has "
                         f"{saved['subspan_em']}; predictions.csv was not read as it was scored. Stopping.")
            off = score(df, official_extract)
            cost = json.load(open(os.path.join(d, "cost.json"))) if os.path.isfile(os.path.join(d, "cost.json")) else {}
            task = df["task"].iloc[0]
            rows.append({
                "group": os.path.basename(os.path.normpath(root)), "task": task, "arm": cost.get("press"),
                "ratio": cost.get("ratio", df["compression_ratio"].iloc[0]), "n": len(df),
                "subspan_em": 100 * off["subspan_em"],  # the official, reported metric
                "em": 100 * off["em"],
                **({"coverage": 100 * off["coverage"]} if "coverage" in off else {"f1": 100 * off["f1"]}),
                "subspan_em_kvpress": 100 * saved["subspan_em"],
                "parsed_differently": int(sum(KVPRESS_EXTRACT(str(o)) != official_extract(str(o))
                                              for o in df["predicted_answer"].fillna(""))),
                "cell": os.path.relpath(d, root),
            })
    out = pd.DataFrame(rows)
    if out.empty:
        sys.exit("no finished LOFT cells found")
    changed = out[(out.subspan_em - out.subspan_em_kvpress).abs() > 1e-9]
    print(f"{len(out)} LOFT cells; kvpress parser reproduces every saved score; official subspan_em differs "
          f"in {len(changed)} cells; answers parsed differently {int(out.parsed_differently.sum())} of {int(out.n.sum())}")
    if len(changed):
        print(changed[["group", "task", "arm", "ratio", "subspan_em_kvpress", "subspan_em"]].to_string(index=False))
    if args.out:
        out.to_csv(args.out, index=False)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
