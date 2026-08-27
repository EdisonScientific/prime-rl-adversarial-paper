"""grant_abstracts: title-conditioned NIH grant abstract generation.

There is no verifiable correctness for a generated abstract; in the adversarial
RL env the reward comes from a judge. The reward here only checks that the reply
is exactly one well-formed answer block, so the wandb series measures format
compliance. The generated abstracts are persisted per eval step by the file
monitor (rollouts/step_N/eval/all/traces.jsonl) for external evaluation against
the reference abstract kept in the task data.
"""

import re
from typing import Literal

import verifiers.v1 as vf
from datasets import load_dataset

ANSWER_RE = re.compile(r"\s*<answer>(.*?)</answer>\s*\Z", re.DOTALL | re.IGNORECASE)
QUESTION = "Write the abstract for an NIH grant proposal with the following title.\n{title}"


def extract_answer(reply: str) -> str | None:
    rest = reply.rpartition("</think>")[2]
    match = ANSWER_RE.fullmatch(rest)
    if match is None:
        return None
    answer = match.group(1)
    if not answer.strip() or re.search(r"</?answer\b", answer, re.IGNORECASE):
        return None
    return answer


class GrantAbstractsData(vf.TaskData):
    answer: str
    """The reference abstract, kept in the saved traces for external evaluation."""


class GrantAbstractsTask(vf.Task[GrantAbstractsData]):
    @vf.reward(weight=1.0)
    async def format(self, trace: vf.Trace, runtime: vf.Runtime) -> float:
        return float(extract_answer(trace.last_reply or "") is not None)


class GrantAbstractsConfig(vf.TasksetConfig):
    dataset: str = "EdisonScientific/grant-abstracts-NIH-fy2026"
    split: Literal["train", "validation"] = "validation"


class GrantAbstractsTaskset(vf.Taskset[GrantAbstractsTask, GrantAbstractsConfig]):
    def load(self) -> list[GrantAbstractsTask]:
        rows = load_dataset(self.config.dataset, split=self.config.split)
        return [
            GrantAbstractsTask(
                GrantAbstractsData(
                    idx=i,
                    prompt=QUESTION.format(title=row["title"].strip()),
                    answer=row["abstract"].strip(),
                ),
                self.config.task,
            )
            for i, row in enumerate(rows)
        ]
