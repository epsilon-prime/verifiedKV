# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
from typing import Any

import pandas as pd

from ..utils import extract_boxed


def extract_aime_answer(text: str) -> str:
    """Extract an AIME integer answer from model output, preferring \\boxed{}."""
    if not isinstance(text, str) or not text.strip():
        return ""

    boxed = extract_boxed(text, last=True)
    if boxed is not None:
        number_match = re.search(r"\d+", boxed)
        if number_match:
            return str(int(number_match.group()))
        return boxed.strip()

    for line in reversed(text.strip().split("\n")):
        if not line.strip():
            continue
        number_match = re.search(r"\b(\d{1,3})\b", line)
        if number_match:
            return str(int(number_match.group(1)))
    return ""


def normalize_answer(answer: str) -> str:
    answer = str(answer).strip()
    if answer.isdigit():
        return str(int(answer))
    return answer


def calculate_metrics(df: pd.DataFrame) -> dict[str, Any]:
    total = len(df)
    correct = 0
    extraction_failures = 0

    for _, row in df.iterrows():
        ground_truth = normalize_answer(row["answer"])
        predicted_text = "" if pd.isna(row["predicted_answer"]) else str(row["predicted_answer"])
        extracted = extract_aime_answer(predicted_text)
        if not extracted:
            extraction_failures += 1
            continue
        if normalize_answer(extracted) == ground_truth:
            correct += 1

    return {
        "accuracy": correct / total if total else 0.0,
        "correct": correct,
        "answered": total - extraction_failures,
        "total": total,
        "extraction_success_rate": (total - extraction_failures) / total if total else 0.0,
    }
