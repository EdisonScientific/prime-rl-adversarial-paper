import ast
import os
import re
from collections import Counter
from typing import Any

import verifiers as vf
from datasets import Dataset, load_dataset
from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

from prime_rl.adversarial.environments.base import GeneratorEnv, make_example
from prime_rl.adversarial.environments.judge import BinaryJudgeEnv
from prime_rl.adversarial.utils import collapse_whitespace, extract_answer

FORMULA_RE = re.compile(r"([A-Z][a-z]?)(\d*)")

ETHER0_SMILES_TO_FORMULA_SYSTEM_PROMPT = """\
You are a careful chemistry assistant. Given a SMILES string, compute its molecular formula exactly.
Think only as much as needed.

Your final answer must use exactly this format:
<think>thinking</think>
<answer>FORMULA</answer>

Do not write anything before or after these tags.

Self-check your answer before the final answer. Count every atom implied by the
SMILES, including implicit hydrogens, and compare the counts to your formula.
The formula L1 distance is the sum over every element of
abs(candidate_count - target_count). The best possible distance is 0.
If the distance is not 0, revise the formula. Do not show the count table.
"""


def formula_atom_counts(formula: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    for element, count in FORMULA_RE.findall(formula):
        counts[element] += int(count or 1)
    return counts


def parse_formula_counts(formula: str | None) -> Counter[str] | None:
    if formula is None:
        return None
    text = collapse_whitespace(formula)
    if not text:
        return None
    parts = list(FORMULA_RE.finditer(text))
    if not parts or "".join(match.group(0) for match in parts) != text:
        return None
    return formula_atom_counts(text)


def formula_l1_to_target(candidate_formula: str | None, solution: dict[str, Any]) -> float | None:
    """L1 atom-count distance between candidate and target; None when the candidate is not a valid formula."""
    candidate_counts = parse_formula_counts(candidate_formula)
    if candidate_counts is None:
        return None
    target_counts = formula_atom_counts(ether0_target_formula(solution))
    elements = set(candidate_counts) | set(target_counts)
    return float(sum(abs(candidate_counts[element] - target_counts[element]) for element in elements))


def ether0_target_formula(solution: dict[str, Any]) -> str:
    if solution["fxn_name"] == "formula_eval":
        reference_mol = Chem.MolFromSmiles(solution["answer_info"])
        if reference_mol is None:
            raise ValueError("Ether0 formula_eval reference answer is not a valid SMILES string.")
        return CalcMolFormula(reference_mol)

    if solution["fxn_name"] == "functional_group_eval":
        formula, groups = ast.literal_eval(solution["answer_info"])
        if groups:
            raise ValueError("Ether0 adversarial env only supports simple-formula examples with no group constraints.")
        return str(formula)

    raise ValueError(f"Ether0 adversarial env only supports formula rewards for now, got {solution['fxn_name']!r}.")


def build_ether0_smiles_to_formula_dataset(
    dataset_name: str,
    split: str,
    problem_type: str,
    num_examples: int,
    seed: int,
) -> Dataset:
    token = os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    source = load_dataset(dataset_name, split=split, token=token)
    if problem_type:
        source = source.filter(lambda row: row["problem_type"] == problem_type)
    source = source.shuffle(seed=seed).select(range(num_examples))

    rows = []
    for example_id, row in enumerate(source):
        reference_smiles = row["ideal"]
        solution = row["solution"]
        target_formula = ether0_target_formula(solution)
        question = "Compute the molecular formula for this SMILES string.\n\n" f"SMILES:\n{reference_smiles}"
        rows.append(
            make_example(
                example_id,
                question,
                target_formula,
                info={"reference_smiles": reference_smiles, "solution": solution},
            )
        )
    return Dataset.from_list(rows)


class Ether0SmilesToFormulaGeneratorEnv(GeneratorEnv):
    """Ether0 diagnostic env: compute molecular formulas from reference SMILES."""

    judge_task_label = "Chemistry question"
    judge_candidate_label = "Candidate molecular formula"

    def __init__(
        self,
        dataset_name: str = "futurehouse/molrqa2",
        split: str = "train",
        problem_type: str = "simple-formula",
        num_examples: int = 5_000,
        num_eval_examples: int | None = None,
        seed: int = 0,
        **_: Any,
    ):
        total = num_examples + (num_eval_examples or 0)
        dataset = build_ether0_smiles_to_formula_dataset(dataset_name, split, problem_type, total, seed)
        super().__init__(
            dataset=dataset.select(range(num_examples)),
            eval_dataset=dataset.select(range(num_examples, total)) if num_eval_examples else None,
            system_prompt=ETHER0_SMILES_TO_FORMULA_SYSTEM_PROMPT,
        )

    def adversarial_candidate_answer(self, completion: Any) -> str | None:
        answer = extract_answer(completion)
        if answer is None:
            return None
        answer = collapse_whitespace(answer)
        return answer or None

    def adversarial_generator_result(self, completion: Any, example: dict[str, Any]) -> dict[str, Any]:
        info = example.get("info") or {}
        candidate = self.adversarial_candidate_answer(completion)
        formula_l1 = formula_l1_to_target(candidate, info["solution"])
        return self.generator_result(candidate, formula_l1 is not None, float(formula_l1 == 0.0))


def load_environment(kind: str = "generator", **kwargs: Any) -> vf.Environment:
    if kind == "generator":
        return Ether0SmilesToFormulaGeneratorEnv(**kwargs)
    if kind == "judge":
        return BinaryJudgeEnv(**kwargs)
    raise ValueError("smiles_to_formula adversarial env kind must be generator or judge.")
