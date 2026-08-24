import json
import os
import random
import re
from typing import Any

import verifiers as vf
from datasets import Dataset

from prime_rl.adversarial.environments.base import GeneratorEnv, make_example
from prime_rl.adversarial.environments.judge import BinaryJudgeEnv
from prime_rl.adversarial.utils import extract_answer

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?")

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


def load_prepared_abstracts(dataset_path: str, num_examples: int, seed: int) -> list[str]:
    """Prepared abstracts JSONL: one ``{"abstract": ...}`` object per line,
    already domain-filtered, length-bounded, and deduplicated offline."""
    with open(os.path.expanduser(dataset_path), encoding="utf-8") as file:
        abstracts = [json.loads(line)["abstract"].strip() for line in file if line.strip()]
    if len(abstracts) < num_examples:
        raise ValueError(f"{dataset_path!r} has {len(abstracts)} abstracts, but {num_examples} were requested.")
    random.Random(seed).shuffle(abstracts)
    return abstracts[:num_examples]


class PaperAbstractGeneratorEnv(GeneratorEnv):
    """Paper-abstract generator env; the adversarial orchestrator assigns rewards."""

    judge_task_label = "Abstract-writing task"
    judge_candidate_label = "Candidate abstract"

    def __init__(
        self,
        dataset_path: str,
        eval_dataset_path: str | None = None,
        num_examples: int = 10_000,
        num_eval_examples: int | None = None,
        seed: int = 0,
        **_: Any,
    ):
        if eval_dataset_path is not None:
            train_abstracts = load_prepared_abstracts(dataset_path, num_examples, seed)
            eval_abstracts = load_prepared_abstracts(eval_dataset_path, num_eval_examples or 0, seed + 1)
        else:
            # Single-source setup: hold the eval split out of the train sample.
            total = num_examples + (num_eval_examples or 0)
            abstracts = load_prepared_abstracts(dataset_path, total, seed)
            train_abstracts, eval_abstracts = abstracts[:num_examples], abstracts[num_examples:]
        super().__init__(
            dataset=Dataset.from_list(
                [make_example(idx, QUESTION, abstract) for idx, abstract in enumerate(train_abstracts)]
            ),
            eval_dataset=Dataset.from_list(
                [make_example(idx, QUESTION, abstract) for idx, abstract in enumerate(eval_abstracts)]
            )
            if eval_abstracts
            else None,
            system_prompt=PAPER_ABSTRACT_GENERATOR_SYSTEM_PROMPT,
        )

    def adversarial_candidate_answer(self, completion: Any) -> str | None:
        # Raw text: the judge should see the real formatting, newlines included.
        return extract_answer(completion)

    def adversarial_generator_result(self, completion: Any, example: dict[str, Any]) -> dict[str, Any]:
        reference = self.adversarial_reference_answer(example)
        candidate = self.adversarial_candidate_answer(completion)
        format_ok = candidate is not None
        # Verifiable reward: word-level equality, ignoring case and punctuation.
        verifiable = float(
            format_ok
            and " ".join(WORD_RE.findall((candidate or "").lower())) == " ".join(WORD_RE.findall(reference.lower()))
        )
        return self.generator_result(candidate, format_ok, verifiable)


def load_environment(kind: str = "generator", **kwargs: Any) -> vf.Environment:
    if kind == "generator":
        return PaperAbstractGeneratorEnv(**kwargs)
    if kind == "judge":
        return BinaryJudgeEnv(**kwargs)
    raise ValueError("abstracts adversarial env kind must be generator or judge.")
