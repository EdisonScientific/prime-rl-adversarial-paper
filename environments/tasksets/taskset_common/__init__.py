"""Shared loading for the SFT-baseline eval tasksets.

The canonical datasets share one schema, so loading is uniform and each taskset
module only defines its scoring rule, ported verbatim from its adversarial RL env.
"""

import re
from typing import Callable, Literal

import verifiers.v1 as vf
from datasets import load_dataset

ANSWER_RE = re.compile(r"\s*<answer>(.*?)</answer>\s*\Z", re.DOTALL | re.IGNORECASE)


def extract_answer(reply: str) -> str | None:
    """The answer-block payload, or None unless the reply (after an optional
    reasoning block) is exactly one well-formed answer block."""
    rest = reply.rpartition("</think>")[2]
    match = ANSWER_RE.fullmatch(rest)
    if match is None:
        return None
    answer = match.group(1)
    if not answer.strip() or re.search(r"</?answer\b", answer, re.IGNORECASE):
        return None
    return answer


class AnswerData(vf.TaskData):
    answer: str
    """The reference answer."""


class CanonicalConfig(vf.TasksetConfig):
    dataset: str
    split: Literal["train", "validation"] = "validation"


def question_field(row: dict) -> str:
    return row["info"]["question"]


def answer_field(row: dict) -> str:
    return row["answer"]


def load_tasks(
    taskset: vf.Taskset,
    task_type: type[vf.Task],
    question: Callable[[dict], str] = question_field,
    answer: Callable[[dict], str] = answer_field,
) -> list[vf.Task]:
    config = taskset.config
    rows = load_dataset(config.dataset, split=config.split)
    return [
        task_type(AnswerData(idx=i, prompt=question(row), answer=answer(row)), config.task)
        for i, row in enumerate(rows)
    ]
