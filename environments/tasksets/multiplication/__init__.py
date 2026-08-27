"""multiplication: n-digit multiplication scored by exact integer match."""

import re

import verifiers.v1 as vf
from taskset_common import AnswerData, CanonicalConfig, extract_answer, load_tasks

INTEGER_RE = re.compile(r"[+-]?(?:\d+|\d{1,3}(?:,\d{3})+)")


def parse_answer(reply: str) -> int | None:
    answer = (extract_answer(reply) or "").strip()
    if not INTEGER_RE.fullmatch(answer):
        return None
    return int(answer.replace(",", ""))


class MultiplicationTask(vf.Task[AnswerData]):
    @vf.reward(weight=1.0)
    async def correct(self, trace: vf.Trace, runtime: vf.Runtime) -> float:
        prediction = parse_answer(trace.last_reply or "")
        return float(prediction is not None and prediction == int(self.data.answer))


class MultiplicationConfig(CanonicalConfig):
    dataset: str = "EdisonScientific/multiplication"


class MultiplicationTaskset(vf.Taskset[MultiplicationTask, MultiplicationConfig]):
    def load(self) -> list[MultiplicationTask]:
        return load_tasks(self, MultiplicationTask)


__all__ = ["MultiplicationTaskset"]
