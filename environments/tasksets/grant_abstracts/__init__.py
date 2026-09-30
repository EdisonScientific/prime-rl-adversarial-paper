"""Abstracts have no verifiable correctness (the RL env scores with a judge), so
the reward only checks format; the file monitor persists each eval round's
generations for external evaluation against the reference in the task data."""

from taskset_common import extract_answer, make_taskset

QUESTION = "Write the abstract for an NIH grant proposal with the following title.\n{title}"


def format_reward(reply: str, reference: str) -> float:
    return float(extract_answer(reply) is not None)


GrantAbstractsTaskset = make_taskset(
    "GrantAbstracts",
    "EdisonScientific/grant-abstracts-NIH-fy2026",
    format_reward,
    reward_name="format",
    question=lambda row: QUESTION.format(title=row["title"].strip()),
    answer=lambda row: row["abstract"].strip(),
)

__all__ = ["GrantAbstractsTaskset"]
