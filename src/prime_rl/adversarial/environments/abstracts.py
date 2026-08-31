import json
import os
import random
import re
from typing import Any

import verifiers as vf
from datasets import Dataset, load_dataset

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

DESLOPIFY_GENERATOR_SYSTEM_PROMPT = """\
You are rewriting an AI-rephrased research paper abstract so it reads as human-written.

Open with <answer>, write the complete rewritten abstract, then close with </answer>.

Preserve all of the information and change only wording and style.
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


def hf_grant_examples(hf_dataset: str, split: str, num_examples: int | None = None) -> list[dict[str, Any]]:
    """Examples from an SFT-format hub dataset: the prompt is the title-conditioned
    question and the completion wraps the reference abstract in answer tags."""
    rows = load_dataset(hf_dataset, split=split)
    if num_examples is not None:
        rows = rows.select(range(min(num_examples, len(rows))))
    examples = []
    for idx, row in enumerate(rows):
        question = row["prompt"][0]["content"]
        completion = row["completion"][0]["content"]
        match = re.fullmatch(r"\s*<answer>(.*)</answer>\s*", completion, re.DOTALL)
        if match is None:
            raise ValueError(f"{hf_dataset} row {idx} completion is not a single answer block.")
        examples.append(make_example(idx, question, match.group(1).strip()))
    return examples


def hf_question_answer_examples(hf_dataset: str, split: str, num_examples: int | None = None) -> list[dict[str, Any]]:
    """Examples from a hub dataset whose rows already carry the full generator
    question and the reference answer as plain-text columns."""
    rows = load_dataset(hf_dataset, split=split)
    if num_examples is not None:
        rows = rows.select(range(min(num_examples, len(rows))))
    return [make_example(idx, row["question"], row["answer"]) for idx, row in enumerate(rows)]


class DeslopifyAbstractGeneratorEnv(GeneratorEnv):
    """Rewrite-conditioned generation: the generator sees an AI rephrasing of a
    human-written abstract and must produce text indistinguishable from the
    original, which serves as the reference the judge compares against."""

    judge_task_label = "Abstract-restoration task"
    judge_candidate_label = "Candidate abstract"

    def __init__(self, hf_dataset: str, num_examples: int | None = None, require_reasoning: bool = True, **_: Any):
        super().__init__(
            dataset=Dataset.from_list(hf_question_answer_examples(hf_dataset, "train", num_examples)),
            eval_dataset=Dataset.from_list(hf_question_answer_examples(hf_dataset, "validation")),
            system_prompt=DESLOPIFY_GENERATOR_SYSTEM_PROMPT,
            require_reasoning=require_reasoning,
        )


class PaperAbstractGeneratorEnv(GeneratorEnv):
    """Unconditional corpus generation: every example shares one fixed instruction,
    so the judge discriminates generated abstracts from corpus samples."""

    judge_task_label = "Abstract-writing task"
    judge_candidate_label = "Candidate abstract"

    def __init__(
        self, dataset_path: str, num_examples: int = 10_000, seed: int = 0, require_reasoning: bool = True, **_: Any
    ):
        records = load_prepared_records(dataset_path, num_examples, seed)
        super().__init__(
            dataset=Dataset.from_list(
                [
                    make_example(idx, PAPER_ABSTRACT_QUESTION, record["abstract"].strip())
                    for idx, record in enumerate(records)
                ]
            ),
            system_prompt=PAPER_ABSTRACT_GENERATOR_SYSTEM_PROMPT,
            require_reasoning=require_reasoning,
        )


class GrantAbstractGeneratorEnv(GeneratorEnv):
    """Title-conditioned generation: the generator and the judge both see the
    grant title, and the reference is that grant's abstract."""

    judge_task_label = "Abstract-writing task"
    judge_candidate_label = "Candidate abstract"

    def __init__(
        self,
        dataset_path: str | None = None,
        hf_dataset: str | None = None,
        num_examples: int = 10_000,
        seed: int = 0,
        require_reasoning: bool = True,
        **_: Any,
    ):
        # hf_dataset wins when both are set: layered configs can override the
        # base layer's dataset_path without being able to delete the key.
        if hf_dataset is not None:
            examples = hf_grant_examples(hf_dataset, "train", num_examples)
            eval_examples = hf_grant_examples(hf_dataset, "validation")
        elif dataset_path is not None:
            records = load_prepared_records(dataset_path, num_examples, seed)
            examples = [
                make_example(
                    idx,
                    GRANT_ABSTRACT_QUESTION.format(title=record["title"].strip()),
                    record["abstract"].strip(),
                )
                for idx, record in enumerate(records)
            ]
            eval_examples = []
        else:
            raise ValueError("grant_abstract requires dataset_path or hf_dataset.")
        super().__init__(
            dataset=Dataset.from_list(examples),
            eval_dataset=Dataset.from_list(eval_examples) if eval_examples else None,
            system_prompt=GRANT_ABSTRACT_GENERATOR_SYSTEM_PROMPT,
            require_reasoning=require_reasoning,
        )


def load_paper_environment(kind: str = "generator", **kwargs: Any) -> vf.Environment:
    if kind == "generator":
        return PaperAbstractGeneratorEnv(**kwargs)
    if kind == "judge":
        return BinaryJudgeEnv(**kwargs)
    raise ValueError("paper_abstract adversarial env kind must be generator or judge.")


def load_deslopify_environment(kind: str = "generator", **kwargs: Any) -> vf.Environment:
    if kind == "generator":
        return DeslopifyAbstractGeneratorEnv(**kwargs)
    if kind == "judge":
        return BinaryJudgeEnv(**kwargs)
    raise ValueError("deslopify_abstract adversarial env kind must be generator or judge.")


def load_grant_environment(kind: str = "generator", **kwargs: Any) -> vf.Environment:
    if kind == "generator":
        return GrantAbstractGeneratorEnv(**kwargs)
    if kind == "judge":
        return BinaryJudgeEnv(**kwargs)
    raise ValueError("grant_abstract adversarial env kind must be generator or judge.")
