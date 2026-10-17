# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import re
from typing import Any, List

import pandas as pd

from ..utils import extract_boxed


def extract_numerical_answer(text: str) -> str:
    """Extract an AIME integer (0-999) from model output."""
    if not isinstance(text, str) or not text.strip():
        return ""

    boxed = extract_boxed(text, last=True)
    if boxed is not None:
        numbers = re.findall(r"\d+", boxed)
        if numbers:
            return numbers[-1]

    answer_patterns = [
        r"(?:answer|solution)\s*(?:is|:)\s*(\d+)",
        r"(?:therefore|thus|so)\s*(?:the\s+)?(?:answer|solution)\s*(?:is|:)\s*(\d+)",
    ]
    for pattern in answer_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches and 0 <= int(matches[-1]) <= 999:
            return matches[-1]

    for line in reversed(text.strip().split("\n")):
        line = line.strip()
        if not line:
            continue
        numbers = re.findall(r"\b\d+\b", line)
        if numbers and 0 <= int(numbers[-1]) <= 999:
            return numbers[-1]

    for num_str in reversed(re.findall(r"\b\d+\b", text)):
        if 0 <= int(num_str) <= 999:
            return num_str
    return ""


def as_answer_list(answer) -> List[str]:
    if isinstance(answer, list):
        return [str(a) for a in answer]
    if isinstance(answer, str):
        try:
            parsed = ast.literal_eval(answer)
            if isinstance(parsed, list):
                return [str(a) for a in parsed]
        except (SyntaxError, ValueError):
            pass
        return [answer]
    return [str(answer)]


def calculate_metrics(df: pd.DataFrame) -> dict[str, Any]:
    predictions = df["predicted_answer"].fillna("").tolist()
    references = [as_answer_list(answer) for answer in df["answer"].tolist()]

    correct = 0
    extracted_count = 0
    boxed_count = 0
    for pred, refs in zip(predictions, references):
        extracted = extract_numerical_answer(str(pred))
        if extracted:
            extracted_count += 1
            if extracted in refs:
                correct += 1
        if extract_boxed(str(pred), last=True) is not None:
            boxed_count += 1

    total = len(predictions)
    return {
        "accuracy": correct / total if total else 0.0,
        "exact_match": correct / total if total else 0.0,
        "correct": correct,
        "answered": extracted_count,
        "total": total,
        "extraction_rate": extracted_count / total if total else 0.0,
        "boxed_format_rate": boxed_count / total if total else 0.0,
    }
