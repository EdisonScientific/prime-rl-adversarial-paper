"""multiplication: n-digit multiplication questions scored by exact answer match.

The scoring rule mirrors the adversarial RL verifiable reward. A completion is
correct when an optional reasoning block is followed by exactly one answer block
whose integer equals the reference product.
"""

import re
from typing import Literal

import verifiers.v1 as vf
from datasets import load_dataset

ANSWER_RE = re.compile(r"\s*<answer>(.*?)</answer>\s*\Z", re.DOTALL | re.IGNORECASE)
INTEGER_RE = re.compile(r"[+-]?(?:\d+|\d{1,3}(?:,\d{3})+)")


def parse_answer(reply: str) -> int | None:
    rest = reply.rpartition("</think>")[2]
    match = ANSWER_RE.fullmatch(rest)
    if match is None:
        return None
    answer = match.group(1).strip()
    if not INTEGER_RE.fullmatch(answer):
        return None
    return int(answer.replace(",", ""))


class MultiplicationData(vf.TaskData):
    answer: str
    """The reference product as a string."""


class MultiplicationTask(vf.Task[MultiplicationData]):
    @vf.reward(weight=1.0)
    async def correct(self, trace: vf.Trace, runtime: vf.Runtime) -> float:
        prediction = parse_answer(trace.last_reply or "")
        return float(prediction is not None and prediction == int(self.data.answer))


class MultiplicationConfig(vf.TasksetConfig):
    dataset: str = "EdisonScientific/multiplication"
    split: Literal["train", "validation"] = "validation"


class MultiplicationTaskset(vf.Taskset[MultiplicationTask, MultiplicationConfig]):
    def load(self) -> list[MultiplicationTask]:
        rows = load_dataset(self.config.dataset, split=self.config.split)
        return [
            MultiplicationTask(
                MultiplicationData(idx=i, prompt=row["question"], answer=row["answer"]),
                self.config.task,
            )
            for i, row in enumerate(rows)
        ]
