"""The reward runs the submitted solution.py against the canonical test file,
so a modified test.py cannot game it."""

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


async def canonical_test_passes(solution: str) -> bool:
    script = (
        f"import math\nimport random\n{solution}\n{COUNTDOWN_TEST_PY}\nprint(verify_solution(numbers, target, expr))"
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
    return float(await canonical_test_passes(files["solution.py"].strip()))


CountdownTaskset = make_taskset("Countdown", "EdisonScientific/countdown", verifiable_reward)

__all__ = ["CountdownTaskset"]
