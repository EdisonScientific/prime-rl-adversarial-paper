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
from prime_rl.adversarial.utils import extract_answer

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


def ether0_target_formula(solution: str) -> str:
    """Target formula from an ether0 solution string (``fxn_name!:!answer_info!:!problem_type``)."""
    fxn_name, answer_info, _ = solution.split("!:!")
    if fxn_name == "formula_eval":
        reference_mol = Chem.MolFromSmiles(answer_info)
        if reference_mol is None:
            raise ValueError("Ether0 formula_eval reference answer is not a valid SMILES string.")
        return CalcMolFormula(reference_mol)

    if fxn_name == "functional_group_eval":
        formula, groups = ast.literal_eval(answer_info)
        if groups:
            raise ValueError("Ether0 adversarial env only supports simple-formula examples with no group constraints.")
        return str(formula)

    raise ValueError(f"Ether0 adversarial env only supports formula rewards for now, got {fxn_name!r}.")


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
        # Sample ether0 rows of the requested problem type and turn each into a
        # SMILES-to-formula question; the eval split is held out of the same draw.
        total = num_examples + (num_eval_examples or 0)
        token = os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
        source = load_dataset(dataset_name, split=split, token=token)
        if problem_type:
            source = source.filter(lambda row: row["problem_type"] == problem_type)
        rows = []
        for example_id, row in enumerate(source.shuffle(seed=seed).select(range(total))):
            question = "Compute the molecular formula for this SMILES string.\n\n" f"SMILES:\n{row['ideal']}"
            rows.append(
                make_example(
                    example_id,
                    question,
                    ether0_target_formula(row["solution"]),
                    info={"reference_smiles": row["ideal"], "solution": row["solution"]},
                )
            )
        dataset = Dataset.from_list(rows)
        super().__init__(
            dataset=dataset.select(range(num_examples)),
            eval_dataset=dataset.select(range(num_examples, total)) if num_eval_examples else None,
            system_prompt=ETHER0_SMILES_TO_FORMULA_SYSTEM_PROMPT,
        )

    def adversarial_candidate_answer(self, completion: Any) -> str | None:
        # Raw answer text: the judge sees exactly what the generator wrote.
        return extract_answer(completion) or None

    def adversarial_generator_result(self, completion: Any, example: dict[str, Any]) -> dict[str, Any]:
        info = example.get("info") or {}
        candidate = self.adversarial_candidate_answer(completion)
        # A candidate is well-formed when the formula regex tiles it exactly; the
        # verifiable reward is zero L1 atom-count distance to the target formula.
        formula_l1 = None
        if candidate:
            parts = list(FORMULA_RE.finditer(candidate))
            if parts and "".join(match.group(0) for match in parts) == candidate:
                candidate_counts = formula_atom_counts(candidate)
                target_counts = formula_atom_counts(ether0_target_formula(info["solution"]))
                elements = set(candidate_counts) | set(target_counts)
                formula_l1 = float(sum(abs(candidate_counts[el] - target_counts[el]) for el in elements))
        return self.generator_result(candidate, formula_l1 is not None, float(formula_l1 == 0.0))


def load_environment(kind: str = "generator", **kwargs: Any) -> vf.Environment:
    if kind == "generator":
        return Ether0SmilesToFormulaGeneratorEnv(**kwargs)
    if kind == "judge":
        return BinaryJudgeEnv(**kwargs)
    raise ValueError("smiles_to_formula adversarial env kind must be generator or judge.")
