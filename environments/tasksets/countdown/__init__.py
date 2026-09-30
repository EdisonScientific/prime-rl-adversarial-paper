"""The reward runs the submitted solution.py against the canonical test file
with the reference problem's numbers and target bound after it, so neither a
modified test.py nor a substituted easier problem can game it."""

import ast
import asyncio
import json
import sys

from taskset_common import make_taskset

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


def reference_problem(reference: str) -> tuple[list[int], int]:
    """The problem's numbers and target, read from the gold completion's
    solution.py assignments. Gold completions are trusted, so a reference that
    doesn't declare the problem is a dataset error."""
    files = countdown_files(reference.rpartition("</think>")[2].strip())
    if files is None:
        raise ValueError("countdown reference completion is not a two-file JSON")
    declared: dict = {}
    for node in ast.walk(ast.parse(files["solution.py"])):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in ("numbers", "target") and name not in declared:
                declared[name] = ast.literal_eval(node.value)
    if "numbers" not in declared or "target" not in declared:
        raise ValueError("countdown reference solution declares no numbers/target")
    return sorted(int(n) for n in declared["numbers"]), int(declared["target"])


async def canonical_test_passes(solution: str, numbers: list[int], target: int) -> bool:
    # The reference problem is bound after the solution runs, so the model's
    # own numbers/target assignments cannot reach the verifier.
    script = (
        f"import math\nimport random\n{solution}\nnumbers = {numbers}\ntarget = {target}\n"
        f"{COUNTDOWN_TEST_PY}\nprint(verify_solution(numbers, target, expr))"
    )
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, _ = await asyncio.wait_for(process.communicate(script.encode()), timeout=2)
    except TimeoutError:
        process.kill()
        return False
    return process.returncode == 0 and stdout.decode().strip() == "True"


async def verifiable_reward(reply: str, reference: str) -> float:
    files = countdown_files(reply.rpartition("</think>")[2].strip())
    if files is None or not files["solution.py"].strip():
        return 0.0
    numbers, target = reference_problem(reference)
    return float(await canonical_test_passes(files["solution.py"].strip(), numbers, target))


CountdownTaskset = make_taskset("Countdown", "EdisonScientific/countdown", verifiable_reward)

__all__ = ["CountdownTaskset"]
