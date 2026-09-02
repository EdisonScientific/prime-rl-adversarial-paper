import re
from collections import Counter
from typing import Any

import verifiers as vf
from datasets import Dataset, load_dataset

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
"""


def formula_atom_counts(formula: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    for element, count in FORMULA_RE.findall(formula):
        counts[element] += int(count or 1)
    return counts


class Ether0SmilesToFormulaGeneratorEnv(GeneratorEnv):
    """SMILES-to-molecular-formula over the prepared ether0 dataset. The verifiable
    reward is zero L1 atom-count distance to the target formula."""

    judge_task_label = "Chemistry question"
    judge_candidate_label = "Candidate molecular formula"

    def __init__(
        self,
        dataset_path: str,
        num_examples: int | None = None,
        num_eval_examples: int | None = None,
        require_reasoning: bool = True,
        **_: Any,
    ):
        source = load_dataset("parquet", data_dir=f"{dataset_path}/data")

        def rows(split, limit: int | None) -> list[dict[str, Any]]:
            selected = split if limit is None else split.select(range(limit))
            return [make_example(idx, row["info"]["question"], row["answer"]) for idx, row in enumerate(selected)]

        eval_rows = rows(source["validation"], num_eval_examples)
        super().__init__(
            dataset=Dataset.from_list(rows(source["train"], num_examples)),
            eval_dataset=Dataset.from_list(eval_rows) if eval_rows else None,
            system_prompt=ETHER0_SMILES_TO_FORMULA_SYSTEM_PROMPT,
            require_reasoning=require_reasoning,
        )

    def adversarial_candidate_answer(self, completion: Any) -> str | None:
        if not self.reasoning_ok(completion):
            return None
        return extract_answer(completion) or None

    def adversarial_generator_result(self, completion: Any, example: dict[str, Any]) -> dict[str, Any]:
        candidate = self.adversarial_candidate_answer(completion)
        # Format only requires the answer block, so any tag-valid answer goes to
        # the judge. The verifiable reward additionally requires a well-formed
        # formula (the formula regex tiles it exactly) at zero L1 atom-count
        # distance to the target.
        formula_l1 = None
        if candidate:
            parts = list(FORMULA_RE.finditer(candidate))
            if parts and "".join(match.group(0) for match in parts) == candidate:
                candidate_counts = formula_atom_counts(candidate)
                target_counts = formula_atom_counts(self.adversarial_reference_answer(example))
                elements = set(candidate_counts) | set(target_counts)
                formula_l1 = float(sum(abs(candidate_counts[el] - target_counts[el]) for el in elements))
        result = self.generator_result(candidate, candidate is not None, float(formula_l1 == 0.0))
        if candidate is not None:
            result["metrics"]["well_formed_formula"] = float(formula_l1 is not None)
        if formula_l1 is not None:
            result["metrics"]["verifiable_distance"] = formula_l1
        return result


def load_environment(kind: str = "generator", **kwargs: Any) -> vf.Environment:
    if kind == "generator":
        return Ether0SmilesToFormulaGeneratorEnv(**kwargs)
    if kind == "judge":
        return BinaryJudgeEnv(**kwargs)
    raise ValueError("smiles_to_formula adversarial env kind must be generator or judge.")
