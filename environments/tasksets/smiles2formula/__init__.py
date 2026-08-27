"""smiles2formula: molecular formula prediction scored by zero L1 atom-count
distance to the reference, over a well-formed formula (the element regex must
tile the answer exactly)."""

import re
from collections import Counter

import verifiers.v1 as vf
from taskset_common import AnswerData, CanonicalConfig, extract_answer, load_tasks

FORMULA_RE = re.compile(r"([A-Z][a-z]?)(\d*)")


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


class Smiles2FormulaTask(vf.Task[AnswerData]):
    @vf.reward(weight=1.0)
    async def correct(self, trace: vf.Trace, runtime: vf.Runtime) -> float:
        candidate = extract_answer(trace.last_reply or "")
        return float(candidate is not None and formula_matches(candidate, self.data.answer))


class Smiles2FormulaConfig(CanonicalConfig):
    dataset: str = "EdisonScientific/smiles2formula"


class Smiles2FormulaTaskset(vf.Taskset[Smiles2FormulaTask, Smiles2FormulaConfig]):
    def load(self) -> list[Smiles2FormulaTask]:
        return load_tasks(self, Smiles2FormulaTask)


__all__ = ["Smiles2FormulaTaskset"]
