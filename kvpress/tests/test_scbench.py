# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest

from evaluation.benchmarks.scbench.calculate_metrics import (
    calculate_metrics,
    normalize,
    rouge_l,
    string_match_all,
    substring_match,
    token_f1,
)
from evaluation.benchmarks.scbench.utils import (
    ANSWER_PREFIX,
    CONTEXT_PREFIX,
    MAX_NEW_TOKENS,
    flatten_multi_turns,
    normalize_task,
)


class WordTokenizer:
    """Stands in for a Hugging Face tokenizer: one token per whitespace-separated word."""

    def encode(self, text, add_special_tokens=False, truncation=False, max_length=None):
        tokens = text.split()
        if truncation and max_length is not None:
            tokens = tokens[:max_length]
        return tokens


def raw_samples(task, n_samples=2, n_turns=3):
    """The raw shape microsoft/SCBench is loaded in: one row per sample, turns nested."""
    context_column = "input" if task == "scbench_vt" else "context"
    rows = []
    for sample in range(n_samples):
        rows.append(
            {
                context_column: f"context of sample {sample} " + "filler " * 10,
                "multi_turns": [
                    {"input": f"question {turn} of sample {sample}", "answer": f"answer {turn}-{sample}"}
                    for turn in range(n_turns)
                ],
            }
        )
    return pd.DataFrame(rows)


def test_normalize_task_accepts_both_spellings():
    assert normalize_task("vt") == "scbench_vt"
    assert normalize_task("scbench_vt") == "scbench_vt"


@pytest.mark.parametrize("task", ["unknown", "scbench_unknown", "repoqa"])
def test_normalize_task_rejects_unsupported_tasks(task):
    with pytest.raises(AssertionError):
        normalize_task(task)


def test_normalize_task_requires_a_task():
    with pytest.raises(AssertionError):
        normalize_task(None)


@pytest.mark.parametrize("task", ["scbench_kv", "scbench_prefix_suffix", "scbench_qa_eng", "scbench_summary"])
def test_flatten_multi_turns_emits_one_row_per_turn(task):
    df = flatten_multi_turns(raw_samples(task, n_samples=2, n_turns=3), task=task)

    assert len(df) == 6
    assert set(df.columns) == {
        "context",
        "question",
        "answer_prefix",
        "answer",
        "task",
        "max_new_tokens",
        "context_id",
        "turn",
    }
    assert (df["task"] == task).all()
    assert (df["max_new_tokens"] == MAX_NEW_TOKENS[task]).all()
    assert (df["answer_prefix"] == ANSWER_PREFIX.get(task, "")).all()
    assert df["turn"].tolist() == [0, 1, 2, 0, 1, 2]
    assert df["context_id"].tolist() == [0, 0, 0, 1, 1, 1]
    # a single gold is still wrapped in a list, so every task shares one answer column shape
    assert df["answer"].tolist() == [[f"answer {t}-{s}"] for s in range(2) for t in range(3)]


@pytest.mark.parametrize("task", ["scbench_kv", "scbench_prefix_suffix", "scbench_vt", "scbench_summary"])
def test_flatten_multi_turns_sends_the_turn_through_unwrapped(task):
    """Every task except qa_eng carries its own cue inside the turn, as SCBench's templates do."""
    df = flatten_multi_turns(raw_samples(task, n_samples=1, n_turns=1), task=task)

    assert df.loc[0, "question"] == "question 0 of sample 0"
    assert df.loc[0, "answer_prefix"] == ""


def test_flatten_multi_turns_prepends_the_task_instruction():
    """kv is the one supported task whose instruction SCBench puts in front of the context."""
    df = flatten_multi_turns(raw_samples("scbench_kv", n_samples=1, n_turns=1), task="kv")

    assert df.loc[0, "context"].startswith("Extract the value corresponding to the specified key")
    assert "context of sample 0" in df.loc[0, "context"]
    assert CONTEXT_PREFIX["scbench_prefix_suffix"] == ""


def test_flatten_multi_turns_wraps_the_bare_questions_of_qa_eng():
    df = flatten_multi_turns(raw_samples("scbench_qa_eng", n_samples=1, n_turns=1), task="qa_eng")

    assert df.loc[0, "context"].startswith("Read the book and answer the question.")
    assert df.loc[0, "question"] == "Question: question 0 of sample 0\n"
    assert df.loc[0, "answer_prefix"] == "Answer:"


def test_flatten_multi_turns_keeps_one_identical_context_per_sample():
    """The runner groups rows by the context string: all turns of a sample must prefill exactly once."""
    df = flatten_multi_turns(raw_samples("scbench_qa_eng", n_samples=2, n_turns=3), task="qa_eng")

    assert df["context"].nunique() == 2
    grouped = df.groupby("context")
    assert grouped.size().tolist() == [3, 3]
    assert (grouped["answer_prefix"].nunique() == 1).all()


def test_flatten_multi_turns_reads_the_vt_context_from_its_own_column():
    raw = raw_samples("scbench_vt", n_samples=1, n_turns=1)
    raw.loc[0, "multi_turns"] = [{"input": "which variables", "answer": ["AAA", "BBB"]}]

    df = flatten_multi_turns(raw, task="vt")

    assert df.loc[0, "context"].startswith("context of sample 0")
    # vt asks for several variables at once, so the gold stays a list
    assert df.loc[0, "answer"] == ["AAA", "BBB"]


def test_flatten_multi_turns_drops_contexts_over_the_limit():
    raw = raw_samples("scbench_prefix_suffix", n_samples=2, n_turns=2)
    raw.loc[1, "context"] = "far too many words " * 50

    df = flatten_multi_turns(raw, task="prefix_suffix", tokenizer=WordTokenizer(), max_context_length=20)

    assert df["context"].nunique() == 1
    assert len(df) == 2
    assert df["context"].str.startswith("context of sample 0").all()


def test_flatten_multi_turns_requires_a_tokenizer_to_apply_the_limit():
    with pytest.raises(AssertionError):
        flatten_multi_turns(raw_samples("scbench_qa_eng"), task="qa_eng", max_context_length=20)


def test_flatten_multi_turns_fails_when_nothing_fits():
    with pytest.raises(AssertionError):
        flatten_multi_turns(
            raw_samples("scbench_qa_eng"), task="qa_eng", tokenizer=WordTokenizer(), max_context_length=1
        )


def test_normalize_strips_the_quoting_scbench_golds_carry():
    assert normalize('"Peyton"') == "peyton"
    assert normalize("The  Answer\n") == "answer"


def test_substring_match_is_all_or_nothing():
    assert substring_match("the answer is Peyton, I think", ['"Peyton"']) == 100.0
    assert substring_match("the answer is Seb", ['"Peyton"']) == 0.0


def test_string_match_all_scores_the_fraction_recovered():
    golds = ["CTLFS", "VVUTI", "QBORE", "IHQFY"]

    assert string_match_all("they are: CTLFS, VVUTI, QBORE, IHQFY", golds) == 100.0
    assert string_match_all("they are: CTLFS, VVUTI", golds) == 50.0
    assert string_match_all("no idea", golds) == 0.0


def test_token_f1_rewards_partial_overlap():
    assert token_f1("red bicycle", ["red bicycle"]) == 100.0
    # 2 of the 4 predicted tokens hit both golds, so precision 0.5 and recall 1.0
    assert token_f1("answer is red bicycle", ["red bicycle"]) == pytest.approx(200 / 3)
    # articles are stripped, so a leading "a" costs nothing
    assert token_f1("a red bicycle", ["red bicycle"]) == 100.0
    assert token_f1("green tractor", ["red bicycle"]) == 0.0


def test_rouge_l_uses_the_longest_common_subsequence():
    assert rouge_l("the paper studies order divisor graphs", ["the paper studies order divisor graphs"]) == 100.0
    assert 0.0 < rouge_l("the paper studies graphs", ["the paper studies order divisor graphs"]) < 100.0
    assert rouge_l("", ["anything at all"]) == 0.0


def test_calculate_metrics_reports_one_entry_per_task():
    df = pd.DataFrame(
        [
            {"task": "scbench_vt", "answer": ["AAA", "BBB"], "predicted_answer": "they are AAA and CCC"},
            {"task": "scbench_vt", "answer": ["CCC"], "predicted_answer": "it is CCC"},
        ]
    )

    assert calculate_metrics(df) == {"scbench_vt": {"string_match_all": 75.0}}


def test_calculate_metrics_scores_qa_eng_on_two_metrics():
    df = pd.DataFrame([{"task": "scbench_qa_eng", "answer": ['"Peyton"'], "predicted_answer": "Peyton"}])

    assert calculate_metrics(df) == {"scbench_qa_eng": {"f1": 100.0, "substring_match": 100.0}}


def test_calculate_metrics_treats_a_missing_prediction_as_empty():
    df = pd.DataFrame([{"task": "scbench_kv", "answer": ["cb59052b"], "predicted_answer": None}])

    assert calculate_metrics(df) == {"scbench_kv": {"substring_match": 0.0}}
