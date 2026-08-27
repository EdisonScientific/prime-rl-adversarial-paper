"""countdown: two-file Countdown coding tasks scored like the adversarial RL env.

A completion is a JSON object with solution.py and test.py after an optional
reasoning block. The reward runs the submitted solution against the canonical
test file, so a modified test.py cannot game it.
"""

import asyncio
import json
import sys
from typing import Literal

import verifiers.v1 as vf
from datasets import load_dataset

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
        "import math\n"
        "import random\n"
        f"{solution}\n"
        f"{COUNTDOWN_TEST_PY}\n"
        "print(verify_solution(numbers, target, expr))"
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


class CountdownData(vf.TaskData):
    pass


class CountdownTask(vf.Task[CountdownData]):
    @vf.reward(weight=1.0)
    async def correct(self, trace: vf.Trace, runtime: vf.Runtime) -> float:
        reply = trace.last_reply or ""
        files = countdown_files(reply.rpartition("</think>")[2].strip())
        if files is None or not files["solution.py"].strip():
            return 0.0
        return float(await canonical_test_passes(files["solution.py"].strip()))


class CountdownConfig(vf.TasksetConfig):
    dataset: str = "EdisonScientific/countdown"
    split: Literal["train", "validation"] = "validation"


class CountdownTaskset(vf.Taskset[CountdownTask, CountdownConfig]):
    def load(self) -> list[CountdownTask]:
        rows = load_dataset(self.config.dataset, split=self.config.split)
        return [
            CountdownTask(
                CountdownData(idx=i, prompt=row["info"]["question"]),
                self.config.task,
            )
            for i, row in enumerate(rows)
        ]
