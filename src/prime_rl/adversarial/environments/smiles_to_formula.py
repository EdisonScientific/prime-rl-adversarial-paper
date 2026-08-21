import ast
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.ipc as pa_ipc
import verifiers as vf
from datasets import Dataset, load_dataset

from prime_rl.adversarial.environments.base import GeneratorEnv, make_example, reservoir_sample
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
"""

ETHER0_SMILES_TO_FORMULA_HINT = """\

Self-check your answer before the final answer. Count every atom implied by the
SMILES, including implicit hydrogens, and compare the counts to your formula.
The formula L1 distance is the sum over every element of
abs(candidate_count - target_count). The best possible distance is 0.
If the distance is not 0, revise the formula. Do not show the count table.
"""


def text_field(row: dict[str, Any], *names: str) -> str:
    for name in names:
        value = row.get(name)
        if value is not None and str(value).strip():
            return str(value)
    raise ValueError(f"Missing required text field. Tried: {', '.join(names)}")


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


def formula_l1_to_target(candidate_formula: str | None, solution: Any) -> float | None:
    """L1 atom-count distance between candidate and target; None when the candidate is not a valid formula."""
    candidate_counts = parse_formula_counts(candidate_formula)
    if candidate_counts is None:
        return None
    target_counts = formula_atom_counts(ether0_target_formula(solution))
    elements = set(candidate_counts) | set(target_counts)
    return float(sum(abs(candidate_counts[element] - target_counts[element]) for element in elements))


def build_ether0_smiles_to_formula_dataset(
    dataset_name: str,
    split: str,
    problem_type: str,
    num_examples: int | None,
    seed: int,
) -> Dataset:
    token = os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
    arrow_cache_dir = os.environ.get("ETHER0_MOLRQA2_ARROW_DIR")
    if dataset_name == "futurehouse/molrqa2" and arrow_cache_dir:
        arrow_path = Path(arrow_cache_dir) / f"molrqa2-{split}.arrow"

        def arrow_rows():
            with pa.memory_map(arrow_path.as_posix(), "r") as source:
                for batch in pa_ipc.open_stream(source):
                    if problem_type:
                        batch = batch.filter(pc.equal(batch["problem_type"], problem_type))
                    yield from batch.to_pylist()

        dataset = reservoir_sample(arrow_rows(), num_examples, seed)
    else:
        source = load_dataset(dataset_name, split=split, token=token)
        dataset = reservoir_sample(
            (row for row in source if not problem_type or row.get("problem_type") == problem_type),
            num_examples,
            seed,
        )

    rows = []
    for example_id, row in enumerate(dataset):
        reference_smiles = text_field(row, "ideal", "answer", "reference")
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


def ether0_target_formula(solution: Any) -> str:
    from ether0.models import RewardFunctionInfo
    from rdkit import Chem
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula

    info = RewardFunctionInfo.model_validate(solution)
    if info.fxn_name == "formula_eval":
        reference_mol = Chem.MolFromSmiles(info.answer_info)
        if reference_mol is None:
            raise ValueError("Ether0 formula_eval reference answer is not a valid SMILES string.")
        return CalcMolFormula(reference_mol)

    if info.fxn_name == "functional_group_eval":
        formula, groups = ast.literal_eval(info.answer_info)
        if groups:
            raise ValueError("Ether0 adversarial env only supports simple-formula examples with no group constraints.")
        return str(formula)

    raise ValueError(f"Ether0 adversarial env only supports formula rewards for now, got {info.fxn_name!r}.")


class Ether0SmilesToFormulaGeneratorEnv(GeneratorEnv):
    """Ether0 diagnostic env: compute molecular formulas from reference SMILES."""

    judge_task_label = "Chemistry question"
    judge_candidate_label = "Candidate molecular formula"

    def __init__(
        self,
        dataset_name: str = "futurehouse/molrqa2",
        split: str = "train",
        problem_type: str = "simple-formula",
        num_examples: int | None = 5_000,
        eval_split: str | None = None,
        num_eval_examples: int | None = None,
        holdout_eval_from_train: bool = False,
        seed: int = 0,
        formula_l1_hint: bool = True,
        **_: Any,
    ):
        self._dataset_args = {
            "dataset_name": dataset_name,
            "split": split,
            "problem_type": problem_type,
            "num_examples": num_examples,
            "seed": seed,
        }
        if holdout_eval_from_train and (eval_split != split or num_eval_examples is None):
            raise ValueError(
                "holdout_eval_from_train requires eval_split to match split and num_eval_examples to be set."
            )
        self._holdout_eval_from_train = holdout_eval_from_train
        self._num_eval_examples = num_eval_examples
        self._eval_dataset_args = (
            {
                **self._dataset_args,
                "split": eval_split,
                "num_examples": num_eval_examples,
                "seed": seed + 1,
            }
            if eval_split is not None and num_eval_examples is not None
            else None
        )
        self._formula_l1_hint = formula_l1_hint
        dataset, eval_dataset = self._build_datasets()
        super().__init__(
            dataset=dataset,
            eval_dataset=eval_dataset,
            system_prompt=self._build_system_prompt(),
        )

    def _build_datasets(self) -> tuple[Dataset, Dataset | None]:
        if not self._holdout_eval_from_train:
            dataset = build_ether0_smiles_to_formula_dataset(**self._dataset_args)
            eval_dataset = (
                build_ether0_smiles_to_formula_dataset(**self._eval_dataset_args)
                if self._eval_dataset_args is not None
                else None
            )
            return dataset, eval_dataset

        full_dataset = build_ether0_smiles_to_formula_dataset(
            **{
                **self._dataset_args,
                "num_examples": None,
            }
        ).shuffle(seed=self._dataset_args["seed"])
        eval_count = self._num_eval_examples or 0
        train_count = self._dataset_args["num_examples"]
        train_end = len(full_dataset) if train_count is None else eval_count + train_count
        return full_dataset.select(range(eval_count, train_end)), full_dataset.select(range(eval_count))

    def _build_system_prompt(self) -> str:
        return ETHER0_SMILES_TO_FORMULA_SYSTEM_PROMPT + (
            ETHER0_SMILES_TO_FORMULA_HINT if self._formula_l1_hint else ""
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

