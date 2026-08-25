from typing import Any

import verifiers as vf

from prime_rl.adversarial.environments.abstracts import load_grant_environment, load_paper_environment
from prime_rl.adversarial.environments.countdown_code import load_environment as load_countdown_code
from prime_rl.adversarial.environments.judge import BinaryJudgeEnv
from prime_rl.adversarial.environments.multiplication import load_environment as load_multiplication
from prime_rl.adversarial.environments.smiles_to_formula import load_environment as load_smiles_to_formula

_GENERATOR_LOADERS = {
    "multiplication": load_multiplication,
    "countdown_code": load_countdown_code,
    "ether0_smiles_to_formula": load_smiles_to_formula,
    "smiles_to_formula": load_smiles_to_formula,
    "grant_abstract": load_grant_environment,
    "paper_abstract": load_paper_environment,
}


def load_environment(kind: str = "generator", task: str | None = None, **kwargs: Any) -> vf.Environment:
    if kind == "judge":
        return BinaryJudgeEnv(**kwargs)
    if kind != "generator":
        raise ValueError("adversarial env kind must be generator or judge.")
    if task is None:
        raise ValueError("legacy adversarial env configs require a task for generator environments.")
    try:
        loader = _GENERATOR_LOADERS[task]
    except KeyError:
        supported_tasks = ", ".join(sorted(_GENERATOR_LOADERS))
        raise ValueError(f"unsupported adversarial generator task {task!r}; expected one of: {supported_tasks}") from None
    return loader(kind=kind, **kwargs)


__all__ = ["load_environment"]
