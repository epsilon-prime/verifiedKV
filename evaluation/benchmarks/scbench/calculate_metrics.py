# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Metrics for SCBench (https://aka.ms/SCBench), one per task family:
#   kv, prefix_suffix   a single opaque string   -> substring match
#   vt                  a list of variables      -> fraction of the gold variables recovered
#   qa_eng              a short free-text span   -> SQuAD-style token F1, with substring match alongside
#   summary             a sentence               -> ROUGE-L F1

import re
import string
from typing import Callable, Dict, List

import numpy as np
import pandas as pd

CONTROL_CHARACTERS = re.compile(r"[\x00-\x1f]")
ARTICLES = re.compile(r"\b(a|an|the)\b")


def normalize(text) -> str:
    """Lowercases and strips control characters, punctuation and articles.

    SCBench gold answers carry stray quoting, for example '"Peyton"' in qa_eng, so punctuation has to go
    before any comparison.
    """
    text = CONTROL_CHARACTERS.sub(" ", str(text)).lower()
    text = "".join(" " if character in string.punctuation else character for character in text)
    return " ".join(ARTICLES.sub(" ", text).split())


def as_list(answer) -> List[str]:
    """The answer column holds a list of golds: a single one for most tasks, several for vt."""
    if isinstance(answer, str):
        return [answer]
    return [str(gold) for gold in answer]


def substring_match(prediction: str, golds: List[str]) -> float:
    """1 if any gold appears in the prediction. Used where the answer is one opaque string."""
    normalized = normalize(prediction)
    return 100.0 * float(any(normalize(gold) in normalized for gold in golds if str(gold).strip()))


def string_match_all(prediction: str, golds: List[str]) -> float:
    """Fraction of the golds that appear in the prediction. Used for vt, which asks for several variables."""
    golds = [gold for gold in golds if str(gold).strip()]
    if not golds:
        return 0.0
    normalized = normalize(prediction)
    return 100.0 * sum(normalize(gold) in normalized for gold in golds) / len(golds)


def token_f1(prediction: str, golds: List[str]) -> float:
    """SQuAD-style token F1, maximized over the golds."""
    predicted_tokens = normalize(prediction).split()
    best = 0.0
    for gold in golds:
        gold_tokens = normalize(gold).split()
        if not gold_tokens or not predicted_tokens:
            best = max(best, float(predicted_tokens == gold_tokens))
            continue
        common = sum(min(predicted_tokens.count(token), gold_tokens.count(token)) for token in set(gold_tokens))
        if common:
            precision, recall = common / len(predicted_tokens), common / len(gold_tokens)
            best = max(best, 2 * precision * recall / (precision + recall))
    return 100.0 * best


def rouge_l(prediction: str, golds: List[str]) -> float:
    """ROUGE-L F1, computed here from the longest common subsequence of the normalized tokens.

    The `rouge` package used for longbench is not applied here: its tokenizer drops the LaTeX and the
    mathematics that fill the physics and computer science abstracts SCBench summarizes, which scores those
    samples on whatever prose is left.
    """
    predicted_tokens = normalize(prediction).split()
    best = 0.0
    for gold in golds:
        gold_tokens = normalize(gold).split()
        if not predicted_tokens or not gold_tokens:
            continue
        # length of the longest common subsequence, with a rolling row
        previous = [0] * (len(gold_tokens) + 1)
        for predicted_token in predicted_tokens:
            current = [0]
            for index, gold_token in enumerate(gold_tokens):
                if predicted_token == gold_token:
                    current.append(previous[index] + 1)
                else:
                    current.append(max(current[index], previous[index + 1]))
            previous = current
        lcs = previous[-1]
        if lcs:
            precision, recall = lcs / len(predicted_tokens), lcs / len(gold_tokens)
            best = max(best, 2 * precision * recall / (precision + recall))
    return 100.0 * best


SCORERS: Dict[str, Dict[str, Callable[[str, List[str]], float]]] = {
    "scbench_kv": {"substring_match": substring_match},
    "scbench_prefix_suffix": {"substring_match": substring_match},
    "scbench_vt": {"string_match_all": string_match_all},
    "scbench_qa_eng": {"f1": token_f1, "substring_match": substring_match},
    "scbench_summary": {"rouge_l": rouge_l},
}


def calculate_metrics(df: pd.DataFrame) -> dict:
    scores = {}
    for task, df_task in df.groupby("task"):
        predictions = df_task["predicted_answer"].fillna("").tolist()
        golds = [as_list(answer) for answer in df_task["answer"].tolist()]
        scores[str(task)] = {
            name: round(float(np.mean([metric_fn(p, g) for p, g in zip(predictions, golds)])), 2)
            for name, metric_fn in SCORERS[str(task)].items()
        }
    return scores
