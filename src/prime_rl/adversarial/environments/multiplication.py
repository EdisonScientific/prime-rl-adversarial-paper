import random
import re
from typing import Any

import verifiers as vf
from datasets import Dataset

from prime_rl.adversarial.environments.base import GeneratorEnv, make_example
from prime_rl.adversarial.environments.judge import BinaryJudgeEnv
from prime_rl.adversarial.utils import extract_strict_think_answer_text

INTEGER_RE = re.compile(r"[+-]?(?:\d+|\d{1,3}(?:,\d{3})+)")

MULTIPLICATION_GENERATOR_SYSTEM_PROMPT = """\
You are a careful arithmetic assistant. Multiply the two integers exactly.
Think only as much as needed.

Your final response must use exactly this format:
<think>brief reasoning</think>
<answer>integer answer</answer>

Do not write anything before or after these tags.
Stop immediately after the closing answer tag.
"""


def build_multiplication_dataset(n_digits: int, num_examples: int, seed: int) -> Dataset:
    rng = random.Random(seed)
    low = 0 if n_digits == 1 else 10 ** (n_digits - 1)
    high = 10**n_digits - 1
    rows = []
    for example_id in range(num_examples):
        left = rng.randint(low, high)
        right = rng.randint(low, high)
        product = left * right
        question = f"What is {left} multiplied by {right}?"
        rows.append(make_example(example_id, question, str(product), info={"product": product}))
    return Dataset.from_list(rows)


class MultiplicationGeneratorEnv(GeneratorEnv):
    """Four-digit multiplication env used by the adversarial experiments."""

    judge_task_label = "Problem"

    def __init__(self, n_digits: int = 3, num_examples: int = 10_000, seed: int = 0, **_: Any):
        super().__init__(
            dataset=build_multiplication_dataset(n_digits=n_digits, num_examples=num_examples, seed=seed),
            system_prompt=MULTIPLICATION_GENERATOR_SYSTEM_PROMPT,
        )

    def adversarial_candidate_answer(self, completion: Any) -> int | None:
        answer = extract_strict_think_answer_text(completion)
        if answer is None:
            return None
        answer = answer.strip()
        if not INTEGER_RE.fullmatch(answer):
            return None
        return int(answer.replace(",", ""))

    def adversarial_generator_result(self, completion: Any, example: dict[str, Any]) -> dict[str, Any]:
        target = int((example.get("info") or {})["product"])
        parsed = self.adversarial_candidate_answer(completion)
        format_ok = parsed is not None
        return self.generator_result(parsed, format_ok, float(format_ok and parsed == target))


def load_environment(kind: str = "generator", **kwargs: Any) -> vf.Environment:
    if kind == "generator":
        return MultiplicationGeneratorEnv(**kwargs)
    if kind == "judge":
        return BinaryJudgeEnv(**kwargs)
    raise ValueError("multiplication adversarial env kind must be generator or judge.")

