from pathlib import Path

import pytest

from prime_rl.configs.adversarial import AdversarialRLConfig
from prime_rl.utils.config import cli

CONFIG_DIR = Path("configs/adversarial")
EXPERIMENT_DIR = Path("configs/adversarial/experiments")
ENVS = [
    "multiplication",
    "smiles_to_formula",
    "grant_abstract",
    "deslopify_abstract",
]
EXPERIMENTS = [
    ("smiles_to_formula", ["smiles2formula_8b32b"]),
    ("smiles_to_formula", ["smiles2formula_8b32b", "smiles2formula_8b32b_lora"]),
    ("grant_abstract", ["grant_nih_32b235b_lora"]),
    ("grant_abstract", ["grant_nih_32b235b_genlora"]),
    ("smiles_to_formula", ["smiles2formula_8b32b", "smiles2formula_8b_exact"]),
    ("smiles_to_formula", ["smiles2formula_8b32b", "smiles2formula_8b8b"]),
    ("smiles_to_formula", ["smiles2formula_8b32b", "smiles2formula_8b8b", "smiles2formula_8b8b_both"]),
    ("deslopify_abstract", ["deslopify_8b8b"]),
    ("deslopify_abstract", ["deslopify_8b8b", "deslopify_8b8b_lora"]),
    ("deslopify_abstract", ["deslopify_14b14b"]),
    ("multiplication", ["multiplication_ab"]),
]


@pytest.mark.parametrize("env", ENVS)
def test_composed_recipes_validate(env: str):
    """The adversarial configs are layers; only their compositions must validate."""
    args = []
    for name in ["base", env]:
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
