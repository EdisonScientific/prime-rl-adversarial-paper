"""grant_abstracts: title-conditioned NIH grant abstract generation. There is no
verifiable correctness (the RL env scores with a judge): the reward only checks
format, and the file monitor persists each eval round's generations for external
evaluation against the reference abstract kept in the task data."""

import verifiers.v1 as vf
from taskset_common import AnswerData, CanonicalConfig, extract_answer, load_tasks

QUESTION = "Write the abstract for an NIH grant proposal with the following title.\n{title}"


class GrantAbstractsTask(vf.Task[AnswerData]):
    @vf.reward(weight=1.0)
    async def format(self, trace: vf.Trace, runtime: vf.Runtime) -> float:
        return float(extract_answer(trace.last_reply or "") is not None)


class GrantAbstractsConfig(CanonicalConfig):
    dataset: str = "EdisonScientific/grant-abstracts-NIH-fy2026"


class GrantAbstractsTaskset(vf.Taskset[GrantAbstractsTask, GrantAbstractsConfig]):
    def load(self) -> list[GrantAbstractsTask]:
        return load_tasks(
            self,
            GrantAbstractsTask,
            question=lambda row: QUESTION.format(title=row["title"].strip()),
            answer=lambda row: row["abstract"].strip(),
        )


__all__ = ["GrantAbstractsTaskset"]
