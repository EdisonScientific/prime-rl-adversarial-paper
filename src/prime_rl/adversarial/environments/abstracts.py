import json
import os
import random
from typing import Any

import verifiers as vf
from datasets import Dataset

from prime_rl.adversarial.environments.base import GeneratorEnv, make_example
from prime_rl.adversarial.environments.judge import BinaryJudgeEnv

PAPER_ABSTRACT_QUESTION = "Generate a scientifically plausible research paper abstract."

PAPER_ABSTRACT_GENERATOR_SYSTEM_PROMPT = """\
You are writing a scientifically plausible research paper abstract.
Think only as much as needed.

Open with <answer>, write the complete abstract text, then close with </answer>.

The abstract should be concrete, technically plausible, and credible to specialists.
Do not include titles, authors, journal names, dates, citations, markdown, generic
placeholders, or multiple abstracts.
Stop immediately after the closing answer tag.
"""

GRANT_ABSTRACT_QUESTION = "Write the abstract for an NIH grant proposal with the following title.\n{title}"

GRANT_ABSTRACT_GENERATOR_SYSTEM_PROMPT = """\
You are writing a scientifically plausible NIH grant proposal abstract.

Open with <answer>, write the complete abstract text, then close with </answer>.

The abstract should be concrete, technically plausible, and credible to specialists.
Stop immediately after the closing answer tag.
"""


def load_prepared_records(dataset_path: str, num_examples: int, seed: int) -> list[dict[str, Any]]:
    """Prepared abstracts JSONL, one object per line, already curated offline."""
    with open(os.path.expanduser(dataset_path), encoding="utf-8") as file:
        records = [json.loads(line) for line in file if line.strip()]
    if len(records) < num_examples:
        raise ValueError(f"{dataset_path!r} has {len(records)} records, but {num_examples} were requested.")
    random.Random(seed).shuffle(records)
    return records[:num_examples]


class PaperAbstractGeneratorEnv(GeneratorEnv):
    """Unconditional corpus generation: every example shares one fixed instruction,
    so the judge discriminates generated abstracts from corpus samples."""

    judge_task_label = "Abstract-writing task"
    judge_candidate_label = "Candidate abstract"

    def __init__(self, dataset_path: str, num_examples: int = 10_000, seed: int = 0, **_: Any):
        records = load_prepared_records(dataset_path, num_examples, seed)
        super().__init__(
            dataset=Dataset.from_list(
                [
                    make_example(idx, PAPER_ABSTRACT_QUESTION, record["abstract"].strip())
                    for idx, record in enumerate(records)
                ]
            ),
            system_prompt=PAPER_ABSTRACT_GENERATOR_SYSTEM_PROMPT,
        )


class GrantAbstractGeneratorEnv(GeneratorEnv):
    """Title-conditioned generation: the generator and the judge both see the
    grant title, and the reference is that grant's abstract."""

    judge_task_label = "Abstract-writing task"
    judge_candidate_label = "Candidate abstract"

    def __init__(self, dataset_path: str, num_examples: int = 10_000, seed: int = 0, **_: Any):
        records = load_prepared_records(dataset_path, num_examples, seed)
        super().__init__(
            dataset=Dataset.from_list(
                [
                    make_example(
                        idx,
                        GRANT_ABSTRACT_QUESTION.format(title=record["title"].strip()),
                        record["abstract"].strip(),
                    )
                    for idx, record in enumerate(records)
                ]
            ),
            system_prompt=GRANT_ABSTRACT_GENERATOR_SYSTEM_PROMPT,
        )


def load_paper_environment(kind: str = "generator", **kwargs: Any) -> vf.Environment:
    if kind == "generator":
        return PaperAbstractGeneratorEnv(**kwargs)
    if kind == "judge":
        return BinaryJudgeEnv(**kwargs)
    raise ValueError("paper_abstract adversarial env kind must be generator or judge.")


def load_grant_environment(kind: str = "generator", **kwargs: Any) -> vf.Environment:
    if kind == "generator":
        return GrantAbstractGeneratorEnv(**kwargs)
    if kind == "judge":
        return BinaryJudgeEnv(**kwargs)
    raise ValueError("grant_abstract adversarial env kind must be generator or judge.")
