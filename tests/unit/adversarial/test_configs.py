from pathlib import Path

import pytest

from prime_rl.configs.adversarial import AdversarialRLConfig
from prime_rl.utils.config import cli

CONFIG_DIR = Path("configs/adversarial")
EXPERIMENT_DIR = Path("configs/adversarial/experiments")
ENVS = [
    "multiplication",
    "countdown_code",
    "smiles_to_formula",
    "medical_abstract",
    "grant_abstract",
]
EXPERIMENTS = [
    ("multiplication", ["multiplication/06b06b"]),
    ("multiplication", ["multiplication/06b06b", "multiplication/1p7bjudge"]),
    ("multiplication", ["multiplication/06b06b", "multiplication/4bjudge"]),
    ("multiplication", ["multiplication/06b06b", "multiplication/exact"]),
    ("multiplication", ["multiplication/06b06b", "multiplication/judgexexact"]),
    ("countdown_code", ["countdown/1p7b1p7b"]),
    ("countdown_code", ["countdown/1p7b1p7b", "countdown/4bjudge"]),
    ("countdown_code", ["countdown/1p7b1p7b", "countdown/8bjudge"]),
    ("countdown_code", ["countdown/1p7b1p7b", "countdown/14bjudge"]),
    ("countdown_code", ["countdown/1p7b1p7b", "countdown/32bjudge"]),
    ("countdown_code", ["countdown/1p7b1p7b", "countdown/exact"]),
    ("countdown_code", ["countdown/1p7b1p7b", "countdown/judgexexact"]),
    ("smiles_to_formula", ["smiles/8b8b"]),
    ("smiles_to_formula", ["smiles/8b8b", "smiles/14bjudge"]),
    ("smiles_to_formula", ["smiles/8b8b", "smiles/32bjudge"]),
    ("smiles_to_formula", ["smiles/8b8b", "smiles/exact"]),
    ("smiles_to_formula", ["smiles/8b8b", "smiles/judgexexact"]),
    ("medical_abstract", ["medical/8b14b"]),
    ("grant_abstract", ["nih/32b235b"]),
]


@pytest.mark.parametrize("env", ENVS)
def test_composed_recipes_validate(env: str):
    """The adversarial configs are layers. Only their compositions must validate."""
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
