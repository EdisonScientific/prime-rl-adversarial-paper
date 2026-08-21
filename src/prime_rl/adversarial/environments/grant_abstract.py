import json
import os
import random
import re
from typing import Any

import verifiers as vf
from datasets import Dataset

from prime_rl.adversarial.environments.base import GeneratorEnv, make_example
from prime_rl.adversarial.environments.judge import BinaryJudgeEnv
from prime_rl.adversarial.utils import collapse_whitespace, extract_answer

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?")

ABSTRACT_FIELD_NAMES = ("abstract", "abstract_text", "paper_abstract", "description")

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


def normalized_text_key(text: str) -> str:
    return " ".join(WORD_RE.findall(text.lower()))


def build_paper_abstract_dataset(
    dataset_path: str,
    abstract_field: str | None = None,
    num_examples: int | None = 10_000,
    seed: int = 0,
) -> Dataset:
    """Load abstracts from a prepared JSONL file (one object per line), dedupe, and sample."""
    path = os.path.expanduser(dataset_path)
    field_names = (abstract_field,) if abstract_field else ABSTRACT_FIELD_NAMES
    abstracts: list[str] = []
    seen: set[str] = set()
    with open(path, encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            record = json.loads(line)
            abstract = next((str(record[name]) for name in field_names if record.get(name)), None)
            if abstract is None:
                continue
            abstract = collapse_whitespace(abstract.strip().strip('"').strip("'"))
            key = normalized_text_key(abstract)
            if not key or key in seen:
                continue
            seen.add(key)
            abstracts.append(abstract)
    if not abstracts:
        raise ValueError(f"No paper abstract examples found in {path!r}.")
    if num_examples is not None and len(abstracts) > num_examples:
        random.Random(seed).shuffle(abstracts)
        abstracts = abstracts[:num_examples]
    return Dataset.from_list([make_example(idx, QUESTION, abstract) for idx, abstract in enumerate(abstracts)])


class PaperAbstractGeneratorEnv(GeneratorEnv):
    """Paper-abstract generator env; the adversarial orchestrator assigns rewards."""

    judge_task_label = "Abstract-writing task"
    judge_candidate_label = "Candidate abstract"

    def __init__(
        self,
        dataset_path: str,
        eval_dataset_path: str | None = None,
        abstract_field: str | None = None,
        num_examples: int | None = 10_000,
        num_eval_examples: int | None = None,
        seed: int = 0,
        **_: Any,
    ):
        eval_dataset = None
        if eval_dataset_path is not None:
            eval_dataset = build_paper_abstract_dataset(
                eval_dataset_path, abstract_field=abstract_field, num_examples=num_eval_examples, seed=seed + 1
            )
        super().__init__(
            dataset=build_paper_abstract_dataset(
                dataset_path, abstract_field=abstract_field, num_examples=num_examples, seed=seed
            ),
            eval_dataset=eval_dataset,
            system_prompt=PAPER_ABSTRACT_GENERATOR_SYSTEM_PROMPT,
        )

    def adversarial_candidate_answer(self, completion: Any) -> str | None:
        answer = extract_answer(completion, require_think=False)
        if answer is None:
            return None
        abstract = collapse_whitespace(answer).strip().strip('"').strip("'")
        return abstract or None

    def adversarial_generator_result(self, completion: Any, example: dict[str, Any]) -> dict[str, Any]:
        reference = self.adversarial_reference_answer(example)
        candidate = self.adversarial_candidate_answer(completion)
        format_ok = candidate is not None
        verifiable = float(format_ok and normalized_text_key(candidate or "") == normalized_text_key(reference))
        return self.generator_result(candidate, format_ok, verifiable)


def load_environment(kind: str = "generator", **kwargs: Any) -> vf.Environment:
    if kind == "generator":
        return PaperAbstractGeneratorEnv(**kwargs)
    if kind == "judge":
        return BinaryJudgeEnv(**kwargs)
    raise ValueError("grant_abstract adversarial env kind must be generator or judge.")
