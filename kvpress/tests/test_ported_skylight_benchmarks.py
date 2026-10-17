# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pytest

from evaluation.benchmarks.aime2024.calculate_metrics import calculate_metrics as aime2024_metrics
from evaluation.benchmarks.aime2024.calculate_metrics import extract_aime_answer
from evaluation.benchmarks.aime2025.calculate_metrics import calculate_metrics as aime2025_metrics
from evaluation.benchmarks.dataset_loaders import LOFT_SUBSETS, RULER_TASKS, _require_task
from evaluation.benchmarks.loft.calculate_metrics import calculate_metrics as loft_metrics
from evaluation.benchmarks.ruler.prepare_dataframe import (
    OUTPUT_COLUMNS,
    RulerSplitError,
    prepare_dataframe,
    split_context_question,
)


_TEMPLATES = {
    "niah": (
        "One of the special magic numbers for pale-cactus is hidden in the "
        "following text. Make sure to memorize it.\n"
        "{haystack}\n"
        "What is the special magic number for pale-cactus mentioned in the "
        "provided text? "
        "The special magic number for pale-cactus mentioned in the provided "
        "text is"
    ),
    "vt": (
        "Memorize and track the chain(s) of variable assignment hidden in the "
        "following text.\n"
        "{haystack}\n"
        "Question: Find all variables that are assigned the value 64886 in the "
        "text above."
        "Answer: According to the chain(s) of variable assignment in the text "
        "above, 5 variables are assigned the value 64886, they are:  SGM LJP\n"
        "{haystack}\n"
        "Question: Find all variables that are assigned the value 12345 in the "
        "text above."
        "Answer: According to the chain(s) of variable assignment in the text "
        "above, 5 variables are assigned the value 12345, they are: "
    ),
    "cwe": (
        "Below is a numbered list of words. In these words, some appear more "
        "often than others. Memorize the ones that appear most often.\n"
        "Question: What are the 10 most common words in the above list?\n"
        "{haystack}\n"
        "Question: What are the 10 most common words in the above list? "
        "Answer: The top 10 words that appear most often in the list are:"
    ),
    "fwe": (
        "Read the following coded text and track the frequency of each coded "
        "word.\n"
        "{haystack}\n"
        "Question: Do not provide any explanation. Please ignore the dots "
        "'....'. What are the three most frequently appeared words in the "
        "above coded text? "
        "Answer: According to the coded text above, the three most frequently "
        "appeared words are:"
    ),
    "qa": (
        "Answer the question based on the given documents. Only give me the "
        "answer and do not output any other words.\n\n"
        "The following are given documents.\n\n"
        "{haystack}\n\n"
        "Answer the question based on the given documents. Only give me the "
        "answer and do not output any other words.\n\n"
        "Question: What colour is the sky? "
        "Answer:"
    ),
}


def _raw_input(family: str, haystack: str = "HAYSTACK " * 16) -> str:
    return _TEMPLATES[family].format(haystack=haystack)


def test_require_task_validates():
    assert _require_task("vt", RULER_TASKS, "ruler64k") == "vt"
    with pytest.raises(AssertionError):
        _require_task(None, RULER_TASKS, "ruler64k")
    with pytest.raises(AssertionError):
        _require_task("not_a_task", LOFT_SUBSETS, "loft")


@pytest.mark.parametrize("family", ["niah", "vt", "cwe", "fwe", "qa"])
def test_split_context_question_round_trips(family):
    text = _raw_input(family)
    task = f"{family}_1" if family in {"niah", "qa"} else family
    context, question, answer_prefix = split_context_question(text, task)
    assert context + question + answer_prefix == text
    assert question
    assert answer_prefix


@pytest.mark.parametrize("family", ["vt", "cwe", "qa"])
def test_split_uses_last_question_anchor(family):
    text = _raw_input(family)
    task = f"{family}_1" if family == "qa" else family
    context, question, _ = split_context_question(text, task)
    assert "HAYSTACK" in context
    assert context.count("HAYSTACK") >= 1


def test_split_raises_when_anchor_missing():
    with pytest.raises(RulerSplitError):
        split_context_question("no anchors here", "niah_single_1")


def test_prepare_dataframe_schema():
    raw = pd.DataFrame(
        {
            "index": [0, 1],
            "input": [_raw_input("niah"), _raw_input("niah")],
            "outputs": [np.array(["gold"], dtype=object), np.array(["gold"], dtype=object)],
            "length": [100, 100],
        }
    )
    df = prepare_dataframe(raw, "niah_single_1", 65536)
    assert list(df.columns) == OUTPUT_COLUMNS
    assert (df["task"] == "niah_single_1").all()
    assert (df["context_length"] == 65536).all()
    assert (df["max_new_tokens"] == 128).all()


def test_aime2024_metrics_extract_boxed_and_fallback():
    assert extract_aime_answer(r"The answer is \boxed{023}.") == "23"
    df = pd.DataFrame(
        {
            "answer": ["23", "33", "156", "902", "45"],
            "predicted_answer": [
                r"The answer is \boxed{23}.",
                r"After solving, we get \boxed{033}.",
                r"Therefore, the answer is \boxed{156}.",
                "The final answer is 902.",
                r"I think the answer is \boxed{44}.",
            ],
        }
    )
    metrics = aime2024_metrics(df)
    assert metrics["correct"] == 4
    assert metrics["accuracy"] == 0.8


def test_aime2025_metrics_accept_list_answers():
    df = pd.DataFrame(
        {
            "answer": [["12"], ["34"], "['56']"],
            "predicted_answer": [
                r"\boxed{12}",
                "the answer is 99",
                r"Final answer \boxed{56}",
            ],
        }
    )
    metrics = aime2025_metrics(df)
    assert metrics["correct"] == 2
    assert metrics["total"] == 3


def test_loft_single_value_metrics():
    df = pd.DataFrame(
        {
            "task": ["nq_32k", "nq_32k"],
            "answers": [["Paris"], ["London"]],
            "predicted_answer": [
                'Final Answer: ["Paris"]',
                'Final Answer: ["Berlin"]',
            ],
            "answer_prefix": ["Final Answer: ", "Final Answer: "],
        }
    )
    metrics = loft_metrics(df)
    assert metrics["em"] == 0.5
    assert "f1" in metrics
    assert "coverage" not in metrics


def test_loft_multi_value_metrics():
    df = pd.DataFrame(
        {
            "task": ["qampari_32k"],
            "answers": [["a", "b"]],
            "predicted_answer": ['Final Answer: ["a", "b"]'],
            "answer_prefix": ["Final Answer: "],
        }
    )
    metrics = loft_metrics(df)
    assert metrics["em"] == 1.0
    assert metrics["coverage"] == 1.0
    assert "f1" not in metrics
