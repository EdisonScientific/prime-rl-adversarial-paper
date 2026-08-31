from pathlib import Path

import pytest

from prime_rl.configs.adversarial import AdversarialRLConfig
from prime_rl.utils.config import cli

CONFIG_DIR = Path("configs/adversarial/test")
EXPERIMENT_DIR = Path("configs/adversarial/experiments")
ENVS = ["multiplication", "countdown_code", "smiles_to_formula", "grant_abstract", "paper_abstract"]
OVERLAYS = [[], ["pairwise"], ["lora"], ["short"], ["8b14b"], ["pairwise", "lora", "short"]]
EXPERIMENTS = [
    ("smiles_to_formula", ["smiles2formula_8b32b"]),
    ("smiles_to_formula", ["smiles2formula_8b32b", "smiles2formula_8b32b_lora"]),
    ("grant_abstract", ["grant_nih_32b235b_lora"]),
    ("grant_abstract", ["grant_nih_32b235b_genlora"]),
    ("smiles_to_formula", ["smiles2formula_8b32b", "smiles2formula_8b_exact"]),
    ("smiles_to_formula", ["smiles2formula_8b32b", "smiles2formula_8b8b"]),
    ("smiles_to_formula", ["smiles2formula_8b32b", "smiles2formula_8b8b", "smiles2formula_8b8b_both"]),
]


@pytest.mark.parametrize("overlays", OVERLAYS, ids=lambda overlays: "+".join(overlays) or "plain")
@pytest.mark.parametrize("env", ENVS)
def test_composed_recipes_validate(env: str, overlays: list[str]):
    """The adversarial configs are layers; only their compositions must validate."""
    args = []
    for name in ["base", env, *overlays]:
        args += ["@", (CONFIG_DIR / f"{name}.toml").as_posix()]
    cli(AdversarialRLConfig, args=args)


@pytest.mark.parametrize(
    ("env", "experiments"), EXPERIMENTS, ids=lambda value: "+".join(value) if isinstance(value, list) else value
)
def test_experiment_recipes_validate(env: str, experiments: list[str]):
    paths = [CONFIG_DIR / "base.toml", CONFIG_DIR / f"{env}.toml"]
    paths += [EXPERIMENT_DIR / f"{name}.toml" for name in experiments]
    args = []
    for path in paths:
        args += ["@", path.as_posix()]
    cli(AdversarialRLConfig, args=args)
