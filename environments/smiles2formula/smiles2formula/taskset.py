"""smiles2formula: molecular formula prediction scored like the adversarial RL env.

A completion is correct when an optional reasoning block is followed by exactly
one answer block holding a well-formed formula (the element regex tiles it
exactly) with zero L1 atom-count distance to the reference formula.
"""

import re
from collections import Counter
from typing import Literal

import verifiers.v1 as vf
from datasets import load_dataset

ANSWER_RE = re.compile(r"\s*<answer>(.*?)</answer>\s*\Z", re.DOTALL | re.IGNORECASE)
FORMULA_RE = re.compile(r"([A-Z][a-z]?)(\d*)")


def extract_answer(reply: str) -> str | None:
    rest = reply.rpartition("</think>")[2]
    match = ANSWER_RE.fullmatch(rest)
    if match is None:
        return None
    answer = match.group(1)
    if not answer.strip() or re.search(r"</?answer\b", answer, re.IGNORECASE):
        return None
    return answer


def formula_atom_counts(formula: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    for element, count in FORMULA_RE.findall(formula):
        counts[element] += int(count or 1)
    return counts


def formula_matches(candidate: str, reference: str) -> bool:
    parts = list(FORMULA_RE.finditer(candidate))
    if not parts or "".join(match.group(0) for match in parts) != candidate:
        return False
    candidate_counts = formula_atom_counts(candidate)
    reference_counts = formula_atom_counts(reference)
    elements = set(candidate_counts) | set(reference_counts)
    return sum(abs(candidate_counts[el] - reference_counts[el]) for el in elements) == 0


class Smiles2FormulaData(vf.TaskData):
    answer: str
    """The reference molecular formula."""


class Smiles2FormulaTask(vf.Task[Smiles2FormulaData]):
    @vf.reward(weight=1.0)
    async def correct(self, trace: vf.Trace, runtime: vf.Runtime) -> float:
        candidate = extract_answer(trace.last_reply or "")
        return float(candidate is not None and formula_matches(candidate, self.data.answer))


class Smiles2FormulaConfig(vf.TasksetConfig):
    dataset: str = "EdisonScientific/smiles2formula"
    split: Literal["train", "validation"] = "validation"


class Smiles2FormulaTaskset(vf.Taskset[Smiles2FormulaTask, Smiles2FormulaConfig]):
    def load(self) -> list[Smiles2FormulaTask]:
        rows = load_dataset(self.config.dataset, split=self.config.split)
        return [
            Smiles2FormulaTask(
                Smiles2FormulaData(idx=i, prompt=row["info"]["question"], answer=row["answer"]),
                self.config.task,
            )
            for i, row in enumerate(rows)
        ]
