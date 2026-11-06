# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Optional

import pandas as pd
from tqdm import tqdm
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)

# SCBench ships one long shared context per sample plus a list of follow-up turns. The column holding the
# shared context is "context" for every task except vt, which calls it "input".
CONTEXT_COLUMN = {
    "scbench_kv": "context",
    "scbench_prefix_suffix": "context",
    "scbench_qa_eng": "context",
    "scbench_summary": "context",
    "scbench_vt": "input",
}

# Instruction that SCBench puts in front of the context, from multiturn_templates_scdq in
# https://github.com/microsoft/MInference/blob/main/scbench/eval_utils.py. Most tasks carry their
# instructions inside the context or the turn already, and prepend nothing.
CONTEXT_PREFIX = {
    "scbench_kv": "Extract the value corresponding to the specified key in the JSON object below.\n\n",
    "scbench_prefix_suffix": "",
    "scbench_qa_eng": "Read the book and answer the question. Be very concise in your answer.\n\n",
    "scbench_summary": "",
    "scbench_vt": "",
}

# How SCBench wraps a turn. Only qa_eng, whose turns are bare questions, needs one; the other tasks send
# the turn through as it is. The trailing cue goes in answer_prefix, as it does for longbench.
QUESTION_TEMPLATE = {"scbench_qa_eng": "Question: {question}\n"}
ANSWER_PREFIX = {"scbench_qa_eng": "Answer:"}

# Generation budget per task, from DATA_NAME_TO_MAX_NEW_TOKENS in the same file, so that scores stay
# comparable to the published SCBench results.
MAX_NEW_TOKENS = {
    "scbench_kv": 150,
    "scbench_prefix_suffix": 150,
    "scbench_qa_eng": 40,
    "scbench_summary": 200,
    "scbench_vt": 30,
}

SUPPORTED_TASKS = tuple(sorted(MAX_NEW_TOKENS))


def normalize_task(task: Optional[str]) -> str:
    """Accept both the Hugging Face config name ("scbench_vt") and the bare task name ("vt")."""
    assert task is not None, f"data_dir must be set for scbench, one of {SUPPORTED_TASKS}"
    task = str(task)
    name = task if task.startswith("scbench_") else f"scbench_{task}"
    assert name in SUPPORTED_TASKS, f"Unsupported scbench task {task!r}, expected one of {SUPPORTED_TASKS}"
    return name


def drop_long_contexts(
    df: pd.DataFrame,
    context_column: str,
    tokenizer: PreTrainedTokenizer,
    max_context_length: int,
) -> pd.DataFrame:
    """
    Drops the samples whose context is longer than max_context_length tokens.

    SCBench contexts run up to ~130k tokens, so a few samples do not fit in the position budget of a given
    model. They are dropped rather than truncated: truncating a shared context silently removes the evidence
    the follow-up turns ask about, which would score the model on an unanswerable question.

    Parameters
    ----------
    df : pd.DataFrame
        The raw SCBench dataframe, one row per sample.
    context_column : str
        The column holding the shared context.
    tokenizer : PreTrainedTokenizer
        The tokenizer used to measure the context.
    max_context_length : int
        The maximum allowed context length, in tokens.

    Returns
    -------
    pd.DataFrame
        The rows of df whose context fits.
    """
    lengths = []
    for context in tqdm(df[context_column], desc="Measuring scbench contexts"):
        # truncation stops the tokenizer one token past the limit, which is all that is needed to compare
        tokens = tokenizer.encode(context, add_special_tokens=False, truncation=True, max_length=max_context_length + 1)
        lengths.append(len(tokens))

    mask = pd.Series(lengths, index=df.index) <= max_context_length
    if not mask.all():
        # a warning, not an info: dropping a sample changes what the reported score is a mean over
        logger.warning(f"Dropped {(~mask).sum()}/{len(df)} scbench samples longer than {max_context_length} tokens.")
    df = df.loc[mask]
    assert len(df) > 0, f"No scbench sample has a context of at most {max_context_length} tokens."
    return df


def flatten_multi_turns(
    df: pd.DataFrame,
    task: Optional[str],
    tokenizer: Optional[PreTrainedTokenizer] = None,
    max_context_length: Optional[int] = None,
) -> pd.DataFrame:
    """
    Turns raw SCBench samples into the row format the evaluation runner expects.

    Each SCBench sample is one long context plus a list of follow-up turns that all share it. This function
    emits one row per turn, every row of a sample carrying the identical context string. The runner groups
    rows by context, so a sample is prefilled and compressed once and all of its turns are answered from that
    single compressed cache, which is the shared-context setting SCBench is built to measure.

    The prompt follows SCBench's own same-context-different-query templates: the task instruction goes in
    front of the context, the turn is wrapped where the task asks for it, and the answer cue goes in
    answer_prefix. SCBench concatenates its context and query prompts with no separator of their own, which
    is what the generation pipeline does too.

    Parameters
    ----------
    df : pd.DataFrame
        The raw dataset, one row per sample, as loaded from microsoft/SCBench.
    task : Optional[str]
        The task, either as the Hugging Face config name ("scbench_vt") or bare ("vt").
    tokenizer : Optional[PreTrainedTokenizer]
        Tokenizer used to measure contexts. Only needed together with max_context_length.
    max_context_length : Optional[int]
        If set, samples whose context exceeds this many tokens are dropped.

    Returns
    -------
    pd.DataFrame
        One row per turn, with the columns context, question, answer_prefix, answer, task, max_new_tokens,
        context_id and turn. The answer column holds a list of strings: the single gold answer for most
        tasks, and the list of variables to recover for vt.
    """
    task = normalize_task(task)
    context_column = CONTEXT_COLUMN[task]
    assert context_column in df.columns, f"{task} is missing its context column {context_column!r}"

    if max_context_length is not None:
        assert tokenizer is not None, "a tokenizer is required to apply max_context_length to scbench"
        df = drop_long_contexts(df, context_column, tokenizer, max_context_length)

    question_template = QUESTION_TEMPLATE.get(task, "{question}")
    rows = []
    for context_id, sample in zip(df.index, df.to_dict("records")):
        for turn_id, turn in enumerate(sample["multi_turns"]):
            answer = turn["answer"]
            # vt asks for several variables at once and is scored on how many of them are recovered, so the
            # gold stays a list. Every other task has a single gold, wrapped here for a uniform column.
            # to_pandas hands back numpy arrays rather than lists, hence the check on str and not on list.
            answers = [answer] if isinstance(answer, str) else list(answer)
            rows.append(
                {
                    "context": CONTEXT_PREFIX[task] + sample[context_column],
                    "question": question_template.format(question=turn["input"]),
                    "answer_prefix": ANSWER_PREFIX.get(task, ""),
                    "answer": [str(a) for a in answers],
                    "task": task,
                    "max_new_tokens": MAX_NEW_TOKENS[task],
                    "context_id": context_id,
                    "turn": turn_id,
                }
            )

    logger.info(f"Prepared {len(df)} {task} contexts as {len(rows)} single-turn rows.")
    return pd.DataFrame(rows)
