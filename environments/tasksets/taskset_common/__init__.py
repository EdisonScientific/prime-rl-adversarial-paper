"""Each taskset module defines a verifiable reward ported verbatim from its
adversarial RL env; make_taskset wraps it in the Taskset scaffolding the
verifiers loader expects. The canonical datasets share one schema, so loading
is uniform."""

import inspect
import re
import types
from typing import Awaitable, Callable, Literal

import verifiers.v1 as vf
from datasets import load_dataset

ANSWER_RE = re.compile(r"\s*<answer>(.*?)</answer>\s*\Z", re.DOTALL | re.IGNORECASE)

Reward = Callable[[str, str], float | Awaitable[float]]


def extract_answer(reply: str) -> str | None:
    """The answer-block payload, or None unless the reply (after an optional
    reasoning block) is exactly one well-formed answer block."""
    rest = reply.rpartition("</think>")[2]
    match = ANSWER_RE.fullmatch(rest)
    if match is None:
        return None
    answer = match.group(1)
    if not answer.strip() or re.search(r"</?answer\b", answer, re.IGNORECASE):
        return None
    return answer


class AnswerData(vf.TaskData):
    answer: str
    """The reference answer."""


class CanonicalConfig(vf.TasksetConfig):
    dataset: str
    split: Literal["train", "validation"] = "validation"


def question_field(row: dict) -> str:
    return row["info"]["question"]


def answer_field(row: dict) -> str:
    return row["answer"]


def make_taskset(
    prefix: str,
    dataset: str,
    reward: Reward,
    reward_name: str = "correct",
    question: Callable[[dict], str] = question_field,
    answer: Callable[[dict], str] = answer_field,
) -> type[vf.Taskset]:
    """A Taskset class applying ``reward(reply, reference)`` to every task.

    ``reward_name`` becomes the metric key in eval logs. The classes are built
    dynamically so the loader's generic introspection still resolves the config.
    """

    async def apply_reward(self, trace: vf.Trace, runtime: vf.Runtime) -> float:
        result = reward(trace.last_reply or "", self.data.answer)
        return await result if inspect.isawaitable(result) else result

    apply_reward.__name__ = reward_name
    task_cls = types.new_class(
        f"{prefix}Task",
        (vf.Task[AnswerData],),
        exec_body=lambda ns: ns.update({reward_name: vf.reward(weight=1.0)(apply_reward)}),
    )
    config_cls = type(f"{prefix}Config", (CanonicalConfig,), {"__annotations__": {"dataset": str}, "dataset": dataset})

    def load(self) -> list[vf.Task]:
        rows = load_dataset(self.config.dataset, split=self.config.split)
        return [
            task_cls(AnswerData(idx=i, prompt=question(row), answer=answer(row)), self.config.task)
            for i, row in enumerate(rows)
        ]

    return types.new_class(
        f"{prefix}Taskset",
        (vf.Taskset[task_cls, config_cls],),
        exec_body=lambda ns: ns.update({"load": load}),
    )
