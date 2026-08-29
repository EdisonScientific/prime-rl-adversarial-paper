import random
import re
from typing import Any

import verifiers as vf
from datasets import Dataset

from prime_rl.adversarial.environments.base import GeneratorEnv, make_example
from prime_rl.adversarial.environments.judge import BinaryJudgeEnv
from prime_rl.adversarial.utils import extract_answer, has_reasoning

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


class MultiplicationGeneratorEnv(GeneratorEnv):
    """Seeded n-digit multiplication. The verifiable reward is exact-product match."""

    judge_task_label = "Problem"

    def __init__(
        self,
        n_digits: int = 3,
        num_examples: int = 10_000,
        num_eval_examples: int | None = None,
        seed: int = 0,
        **_: Any,
    ):
        total = num_examples + (num_eval_examples or 0)
        rng = random.Random(seed)
        low = 0 if n_digits == 1 else 10 ** (n_digits - 1)
        high = 10**n_digits - 1
        rows = []
        for example_id in range(total):
            left = rng.randint(low, high)
            right = rng.randint(low, high)
            question = f"What is {left} multiplied by {right}?"
            rows.append(make_example(example_id, question, str(left * right), info={"product": left * right}))
        dataset = Dataset.from_list(rows)
        super().__init__(
            dataset=dataset.select(range(num_examples)),
            eval_dataset=dataset.select(range(num_examples, total)) if num_eval_examples else None,
            system_prompt=MULTIPLICATION_GENERATOR_SYSTEM_PROMPT,
        )

    def adversarial_candidate_answer(self, completion: Any) -> str | None:
        if not has_reasoning(completion):
            return None
        answer = extract_answer(completion)
        if answer is None or not INTEGER_RE.fullmatch(answer.strip()):
            return None
        return answer

    def adversarial_generator_result(self, completion: Any, example: dict[str, Any]) -> dict[str, Any]:
        target = int((example.get("info") or {})["product"])
        candidate = self.adversarial_candidate_answer(completion)
        format_ok = candidate is not None
        verifiable = float(format_ok and int(candidate.strip().replace(",", "")) == target)
        return self.generator_result(candidate, format_ok, verifiable)


def load_environment(kind: str = "generator", **kwargs: Any) -> vf.Environment:
    if kind == "generator":
        return MultiplicationGeneratorEnv(**kwargs)
    if kind == "judge":
        return BinaryJudgeEnv(**kwargs)
    raise ValueError("multiplication adversarial env kind must be generator or judge.")
