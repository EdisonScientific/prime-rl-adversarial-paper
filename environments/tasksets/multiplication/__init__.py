import re

from taskset_common import extract_answer, make_taskset

INTEGER_RE = re.compile(r"[+-]?(?:\d+|\d{1,3}(?:,\d{3})+)")


def parse_answer(reply: str) -> int | None:
    answer = (extract_answer(reply) or "").strip()
    if not INTEGER_RE.fullmatch(answer):
        return None
    return int(answer.replace(",", ""))


def verifiable_reward(reply: str, reference: str) -> float:
    prediction = parse_answer(reply)
    return float(prediction is not None and prediction == int(reference))


MultiplicationTaskset = make_taskset("Multiplication", "EdisonScientific/multiplication", verifiable_reward)

__all__ = ["MultiplicationTaskset"]
