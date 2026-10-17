# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dataset loaders for benchmarks that need more than a single Hub config."""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
from datasets import load_dataset

from .ruler.prepare_dataframe import prepare_dataframe

logger = logging.getLogger(__name__)

RULER_TASKS = (
    "cwe",
    "fwe",
    "niah_multikey_1",
    "niah_multikey_2",
    "niah_multikey_3",
    "niah_multiquery",
    "niah_multivalue",
    "niah_single_1",
    "niah_single_2",
    "niah_single_3",
    "qa_1",
    "qa_2",
    "vt",
)

# Pre-split Hub datasets: config name == split name == RULER task.
RULER_CONFIG_SPLIT = {
    "ruler16k": ("xAlg-AI/att-hub-ruler-16k", 16384),
    "ruler32k": ("xAlg-AI/att-hub-ruler-32k", 32768),
}

# Raw Hub datasets: one default config, splits are RULER tasks; need prepare_dataframe.
RULER_RAW_SPLIT = {
    "ruler64k": ("SaylorTwift/RULER-65536-llama-3.2-tokenizer", 65536),
    "ruler128k": ("SaylorTwift/RULER-131072-llama-3.2-tokenizer", 131072),
}

LOFT_SUBSETS = (
    "nq_32k",
    "nq_128k",
    "nq_1m",
    "hotpotqa_32k",
    "hotpotqa_128k",
    "hotpotqa_1m",
    "musique_32k",
    "musique_128k",
    "musique_1m",
    "qampari_32k",
    "qampari_128k",
    "qampari_1m",
    "quest_32k",
    "quest_128k",
    "quest_1m",
)

# Hub configs created by the AIME create_huggingface_dataset scripts.
AIME_CONFIGS = {
    "aime2024": ("xAlg-AI/att-hub-aime2024", "aime2024"),
    "aime2025": ("xAlg-AI/att-hub-aime2025", "aime2025"),
}


def _require_task(data_dir: Optional[str], allowed: tuple[str, ...], label: str) -> str:
    assert data_dir is not None, f"data_dir must be set for {label}, one of {allowed}"
    task = str(data_dir)
    assert task in allowed, f"Unsupported {label} task {task!r}, expected one of {allowed}"
    return task


def load_ruler_length_dataset(dataset_name: str, data_dir: Optional[str]) -> pd.DataFrame:
    """Load one RULER task for ruler16k / 32k / 64k / 128k."""
    task = _require_task(data_dir, RULER_TASKS, dataset_name)

    if dataset_name in RULER_CONFIG_SPLIT:
        repo_id, context_length = RULER_CONFIG_SPLIT[dataset_name]
        logger.info(f"Loading {repo_id} config/split={task}")
        df = load_dataset(repo_id, task, split=task).to_pandas()
        df["context_length"] = context_length
        return df

    if dataset_name in RULER_RAW_SPLIT:
        repo_id, context_length = RULER_RAW_SPLIT[dataset_name]
        logger.info(f"Loading raw {repo_id} split={task}")
        df = load_dataset(repo_id, split=task).to_pandas()
        return prepare_dataframe(df, task, context_length)

    raise ValueError(f"Unknown ruler-length dataset: {dataset_name}")


def load_loft_dataset(data_dir: Optional[str]) -> pd.DataFrame:
    """Load one LOFT RAG subset (dev + test) from f20180301/loft-rag-{dataset}-{length}."""
    subset = _require_task(data_dir, LOFT_SUBSETS, "loft")
    parts = subset.split("_")
    length = parts[-1]
    dataset = "_".join(parts[:-1])
    hf_dataset_id = f"f20180301/loft-rag-{dataset}-{length}"

    logger.info(f"Loading LOFT RAG subset {subset} from {hf_dataset_id}")
    dataset_dict = load_dataset(hf_dataset_id)
    frames = []
    for split_name in ("dev", "test"):
        if split_name in dataset_dict:
            split_df = dataset_dict[split_name].to_pandas()
            split_df["split"] = split_name
            frames.append(split_df)

    assert frames, f"No splits found for {subset} ({hf_dataset_id})"
    df = pd.concat(frames, ignore_index=True)
    df["task"] = subset

    required = ("context", "question", "answers", "task", "answer_prefix", "max_new_tokens")
    missing = [col for col in required if col not in df.columns]
    assert not missing, f"Missing required LOFT columns: {missing}"
    return df


def load_aime_dataset(dataset_name: str) -> pd.DataFrame:
    """Load a preformatted AIME Hub dataset by its config name."""
    assert dataset_name in AIME_CONFIGS, f"Unknown AIME dataset: {dataset_name}"
    repo_id, config_name = AIME_CONFIGS[dataset_name]
    logger.info(f"Loading {repo_id} config={config_name}")
    return load_dataset(repo_id, config_name, split="test").to_pandas()
