from pathlib import Path

import pytest

from prime_rl.configs.adversarial import AdversarialRLConfig
from prime_rl.utils.config import cli

CONFIG_DIR = Path("configs/adversarial/test")
ENVS = ["multiplication", "countdown_code", "smiles_to_formula", "grant_abstract", "paper_abstract"]
OVERLAYS = [[], ["pairwise"], ["lora"], ["short"], ["8b14b"], ["pairwise", "lora", "short"]]


@pytest.mark.parametrize("overlays", OVERLAYS, ids=lambda overlays: "+".join(overlays) or "plain")
@pytest.mark.parametrize("env", ENVS)
def test_composed_recipes_validate(env: str, overlays: list[str]):
    """The adversarial configs are layers; only their compositions must validate."""
    args = []
    for name in ["base", env, *overlays]:
        args += ["@", (CONFIG_DIR / f"{name}.toml").as_posix()]
    cli(AdversarialRLConfig, args=args)
