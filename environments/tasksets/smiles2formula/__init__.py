import re
from collections import Counter

from taskset_common import extract_answer, make_taskset

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


def verifiable_reward(reply: str, reference: str) -> float:
    candidate = extract_answer(reply)
    return float(candidate is not None and formula_matches(candidate, reference))


Smiles2FormulaTaskset = make_taskset("Smiles2Formula", "EdisonScientific/smiles2formula", verifiable_reward)

__all__ = ["Smiles2FormulaTaskset"]
