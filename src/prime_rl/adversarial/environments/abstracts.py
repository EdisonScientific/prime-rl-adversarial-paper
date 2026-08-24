import json
import os
import random
from typing import Any

import verifiers as vf
from datasets import Dataset

from prime_rl.adversarial.environments.base import GeneratorEnv, make_example
from prime_rl.adversarial.environments.judge import BinaryJudgeEnv

QUESTION = "Generate a scientifically plausible research paper abstract."

PAPER_ABSTRACT_GENERATOR_SYSTEM_PROMPT = """\
You are writing a scientifically plausible research paper abstract.
Think only as much as needed.

Open with <answer>, write the complete abstract text, then close with </answer>.

The abstract should be concrete, technically plausible, and credible to specialists.
Do not include titles, authors, journal names, dates, citations, markdown, generic
placeholders, or multiple abstracts.
Stop immediately after the closing answer tag.
"""


class PaperAbstractGeneratorEnv(GeneratorEnv):
    """Abstract-corpus generator env: free generation judged against real
    abstracts, with no verifiable reward (and hence no generator eval)."""

    judge_task_label = "Abstract-writing task"
    judge_candidate_label = "Candidate abstract"

    def __init__(self, dataset_path: str, num_examples: int = 10_000, seed: int = 0, **_: Any):
        with open(os.path.expanduser(dataset_path), encoding="utf-8") as file:
            abstracts = [json.loads(line)["abstract"].strip() for line in file if line.strip()]
        if len(abstracts) < num_examples:
            raise ValueError(f"{dataset_path!r} has {len(abstracts)} abstracts, but {num_examples} were requested.")
        random.Random(seed).shuffle(abstracts)
        super().__init__(
            dataset=Dataset.from_list(
                [make_example(idx, QUESTION, abstract) for idx, abstract in enumerate(abstracts[:num_examples])]
            ),
            system_prompt=PAPER_ABSTRACT_GENERATOR_SYSTEM_PROMPT,
        )


def load_environment(kind: str = "generator", **kwargs: Any) -> vf.Environment:
    if kind == "generator":
        return PaperAbstractGeneratorEnv(**kwargs)
    if kind == "judge":
        return BinaryJudgeEnv(**kwargs)
    raise ValueError("abstracts adversarial env kind must be generator or judge.")
