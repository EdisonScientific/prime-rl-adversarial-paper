from typing import Any

import verifiers as vf


def make_example(example_id: int, question: str, answer: str, info: dict[str, Any] | None = None) -> dict[str, Any]:
    """A dataset row in the shape all adversarial envs share."""
    return {
        "example_id": example_id,
        "prompt": [{"role": "user", "content": question}],
        "answer": answer,
        "info": {"question": question, "reference": answer, **(info or {})},
    }


class GeneratorEnv(vf.SingleTurnEnv):
    """Base for adversarial generator envs.

    The adversarial orchestrator assigns rewards after rollout (from the judge
    and/or ``verifiable_reward``), so the rubric is a zero-valued placeholder.
    It drives envs through this protocol:

    - ``adversarial_question(example)``: task text shown to the judge.
    - ``adversarial_reference_answer(example)``: ground-truth answer text.
    - ``adversarial_candidate_answer(completion)``: parsed model answer, or
      ``None`` when the completion doesn't follow the answer format.
    - ``adversarial_generator_result(completion, example)``: dict with
      ``candidate``, ``format_ok``, ``verifiable_reward``, and per-rollout
      ``metrics``.
    - ``adversarial_judge_prompt(question, candidate)``: the A/B classification
      prompt for the judge.
    """

    judge_task_label = "Task"
    judge_candidate_label = "Candidate answer"

    def __init__(self, **kwargs: Any):
        kwargs.setdefault("rubric", vf.Rubric(funcs=[self.placeholder_reward]))
        super().__init__(**kwargs)

    # The plural ``completions`` signature marks this as a group reward func, which
    # routes generator envs through ``Env.run_group`` (one call per GRPO group).
    async def placeholder_reward(self, completions, **kwargs) -> list[float]:
        return [0.0 for _ in completions]

    def generator_result(self, candidate: Any, format_ok: bool, verifiable_reward: float) -> dict[str, Any]:
        return {
            "candidate": "" if candidate is None else str(candidate),
            "format_ok": format_ok,
            "verifiable_reward": verifiable_reward,
            "metrics": {"format_reward": float(format_ok), "verifiable_reward": verifiable_reward},
        }

    def adversarial_question(self, example: dict[str, Any]) -> str:
        info = example.get("info") or {}
        return str(info.get("question") or example["prompt"][-1]["content"])

    def adversarial_reference_answer(self, example: dict[str, Any]) -> str:
        info = example.get("info") or {}
        return str(info.get("reference") or example["answer"])

    def adversarial_candidate_answer(self, completion: Any) -> Any:
        raise NotImplementedError

    def adversarial_generator_result(self, completion: Any, example: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    # Candidates are injected verbatim (newlines and all) between explicit tags so the
    # judge sees the real formatting and multi-line answers cannot bleed into the prompt.
    def adversarial_judge_prompt(self, question: str, candidate: Any) -> str:
        return (
            f"{self.judge_task_label}:\n{question}\n\n"
            f"{self.judge_candidate_label}:\n<candidate>\n{candidate}\n</candidate>\n\n"
            "Classify the candidate as A (reference answer) or B (model-generated answer)."
        )

    def adversarial_pairwise_judge_prompt(self, question: str, candidate_a: Any, candidate_b: Any) -> str:
        return (
            f"{self.judge_task_label}:\n{question}\n\n"
            f"Candidate A:\n<candidate_a>\n{candidate_a}\n</candidate_a>\n\n"
            f"Candidate B:\n<candidate_b>\n{candidate_b}\n</candidate_b>\n\n"
            "Which candidate is more likely to be the reference answer? Answer A or B."
        )
