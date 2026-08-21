import math
from typing import Any

import verifiers as vf
from datasets import Dataset

from prime_rl.adversarial.utils import extract_choice

JUDGE_SYSTEM_PROMPT = """\
You are an answer discriminator. Given a task and one candidate answer,
decide whether the candidate is more likely to be a reference answer or a model-generated answer.

Answer with exactly one character:
A for reference answer.
B for model-generated answer.
"""

class BinaryJudgeEnv(vf.SingleTurnEnv):
    """One-token A/B judge; training labels are attached by the orchestrator."""

    def __init__(self, **_: Any):
        super().__init__(
            dataset=Dataset.from_list([{"example_id": 0, "prompt": [{"role": "user", "content": "bootstrap"}]}]),
            system_prompt=JUDGE_SYSTEM_PROMPT,
            rubric=vf.Rubric(funcs=[self.judge_reward]),
        )

    async def judge_reward(self, completion, answer, **kwargs) -> float:
        return float(extract_choice(completion) == str(answer).strip().upper())

    def adversarial_choice(self, completion: Any) -> str | None:
        return extract_choice(completion)

    def adversarial_reference_probability_proxy(self, rollout: vf.RolloutOutput) -> float | None:
        choice = self.adversarial_choice(rollout.get("completion"))
        if choice is None:
            return None
        trajectory = rollout.get("trajectory") or []
        for step in reversed(trajectory):
            tokens = step.get("tokens") or {}
            logprobs = tokens.get("completion_logprobs") or []
            if logprobs:
                chosen_probability = min(max(math.exp(float(logprobs[0])), 0.0), 1.0)
                return chosen_probability if choice == "A" else 1.0 - chosen_probability
        return None

