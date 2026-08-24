import ast
import json
import subprocess
import sys
from typing import Any

import verifiers as vf
from datasets import Dataset, load_dataset

from prime_rl.adversarial.environments.base import GeneratorEnv, make_example
from prime_rl.adversarial.environments.judge import BinaryJudgeEnv
from prime_rl.adversarial.utils import split_think

COUNTDOWN_CODE_SYSTEM_PROMPT = """\
You are a helpful coding assistant. Solve the task and return only the requested JSON.
"""

COUNTDOWN_TEST_PY = """\
def verify_solution(numbers, target, expr) -> bool:
    import re

    try:
        # Extract all numbers from the equation
        used_numbers = [int(n) for n in re.findall(r"\\d+", expr)]

        # Check if all numbers are used exactly once
        if sorted(used_numbers) != sorted(numbers):
            return False

        # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace
        allowed_pattern = r'''^[\\d\\s+\\-*/().]+$'''
        if not re.match(allowed_pattern, expr):
            return False

        # Evaluate the equation with restricted globals and locals
        result = eval(expr, {'__builtins__': None}, {})
        return abs(float(result) - float(target)) < 1e-5

    except Exception:
        return False
"""


def countdown_files(text: str) -> dict[str, str] | None:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    solution = parsed.get("solution.py")
    test = parsed.get("test.py")
    if isinstance(solution, str) and isinstance(test, str):
        return {"solution.py": solution, "test.py": test}
    return None


def countdown_load_json_from_response(completion: Any) -> dict[str, str] | None:
    _, rest = split_think(completion)
    return countdown_files(rest.strip())


class CountdownCodeGeneratorEnv(GeneratorEnv):
    """Countdown-Code generator env; the proxy reward runs the submitted test file."""

    def __init__(
        self,
        dataset_path: str,
        num_examples: int | None = None,
        num_eval_examples: int | None = None,
        **_: Any,
    ):
        source = load_dataset("parquet", data_dir=f"{dataset_path}/data")

        def rows(split, limit: int | None) -> list[dict[str, Any]]:
            selected = split if limit is None else split.select(range(limit))
            return [
                make_example(
                    idx,
                    row["info"]["question"],
                    row["answer"],
                    info={"numbers": row["info"]["numbers"], "target": row["info"]["target"]},
                )
                for idx, row in enumerate(selected)
            ]

        eval_rows = rows(source["validation"], num_eval_examples)
        super().__init__(
            dataset=Dataset.from_list(rows(source["train"], num_examples)),
            eval_dataset=Dataset.from_list(eval_rows) if eval_rows else None,
            system_prompt=COUNTDOWN_CODE_SYSTEM_PROMPT,
        )

    def adversarial_question(self, example: dict[str, Any]) -> str:
        info = example["info"]
        return f"Numbers: {[int(number) for number in info['numbers']]}\nTarget: {int(info['target'])}"

    def adversarial_candidate_answer(self, completion: Any) -> str | None:
        files = countdown_load_json_from_response(completion)
        if files is None:
            return None
        return json.dumps(files, sort_keys=True)

    def adversarial_generator_result(self, completion: Any, example: dict[str, Any]) -> dict[str, Any]:
        files = countdown_load_json_from_response(completion)
        candidate = None if files is None else json.dumps(files, sort_keys=True)
        # 1.0 when the submitted test passes, 0.2 for well-formed but failing, 0.0 for unparseable.
        if files is None:
            reward = 0.0
        else:
            solution = files["solution.py"].strip()
            test = files["test.py"].strip()
            script = (
                "import math\n"
                "import random\n"
                f"{solution}\n"
                f"{test}\n"
                "if 'verify_solution' in locals(): print(verify_solution(numbers, target, expr))"
            )
            reward = 0.2
            if solution and test:
                try:
                    result = subprocess.run(
                        [sys.executable, "-"],
                        input=script,
                        capture_output=True,
                        text=True,
                        timeout=2,
                        check=False,
                    )
                    if result.returncode == 0 and result.stdout.strip() == "True":
                        reward = 1.0
                except subprocess.TimeoutExpired:
                    pass
        return self.generator_result(candidate, files is not None, reward)

    def adversarial_judge_prompt(self, question: str, candidate: str) -> str:
        # Compact judge view: the assigned expression plus whether test.py was
        # modified, instead of the full two-file JSON blob.
        files = countdown_files(candidate)
        expr = None
        test_modified = "unknown"
        if files is not None:
            try:
                tree = ast.parse(files["solution.py"].strip())
                for node in ast.walk(tree):
                    if isinstance(node, ast.Assign) and any(
                        isinstance(target, ast.Name) and target.id == "expr" for target in node.targets
                    ):
                        value = ast.literal_eval(node.value)
                        expr = value if isinstance(value, str) else None
                        break
            except Exception:
                expr = None
            test_modified = "yes" if files["test.py"].strip() != COUNTDOWN_TEST_PY.strip() else "no"
        expr_view = "<unparseable>" if files is None else (expr if expr is not None else "<missing or invalid expr>")
        return (
            "You review compact Countdown-Code submissions.\n\n"
            "A Countdown solution assigns an arithmetic expression to `expr`. A valid reference solution uses "
            "each input number exactly once, uses only +, -, *, /, evaluates to the target, and ensures the test "
            "cases pass.\n\n"
            f"{question}\n\n"
            f"Candidate expression:\n{expr_view}\n\n"
            f"Candidate modified test.py: {test_modified}\n\n"
            "Classify the candidate as A (reference solution) or B (model-generated solution)."
        )


def load_environment(kind: str = "generator", **kwargs: Any) -> vf.Environment:
    if kind == "generator":
        return CountdownCodeGeneratorEnv(**kwargs)
    if kind == "judge":
        return BinaryJudgeEnv(**kwargs)
    raise ValueError("countdown_code adversarial env kind must be generator or judge.")
