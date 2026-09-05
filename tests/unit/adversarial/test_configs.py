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
    "grant_abstract",
    "deslopify_abstract",
    "medical_abstract",
]
EXPERIMENTS = [
    ("smiles_to_formula", ["smiles_8b8b"]),
    ("smiles_to_formula", ["smiles_8b8b", "smiles_14bjudge"]),
    ("smiles_to_formula", ["smiles_8b8b", "smiles_32bjudge"]),
    ("smiles_to_formula", ["smiles_8b8b", "smiles_32bjudge", "smiles_32bjudge_lora"]),
    ("smiles_to_formula", ["smiles_8b8b", "smiles_exact"]),
    ("smiles_to_formula", ["smiles_8b8b", "smiles_judgexexact"]),
    ("grant_abstract", ["grant_nih_32b235b_lora"]),
    ("grant_abstract", ["grant_nih_32b235b_genlora"]),
    ("grant_abstract", ["grant_nih_32b235b_fullft"]),
    ("grant_abstract", ["grant_nih_32b235b_fullft_noema"]),
    ("deslopify_abstract", ["deslopify_8b8b"]),
    ("deslopify_abstract", ["deslopify_8b8b", "deslopify_8b8b_lora"]),
    ("deslopify_abstract", ["deslopify_14b14b"]),
    ("multiplication", ["multiplication_ab"]),
    ("multiplication", ["multiplication_06b06b"]),
    ("multiplication", ["multiplication_06b06b", "multiplication_1p7bjudge"]),
    ("multiplication", ["multiplication_06b06b", "multiplication_4bjudge"]),
    ("multiplication", ["multiplication_06b06b", "multiplication_exact"]),
    ("multiplication", ["multiplication_06b06b", "multiplication_judgexexact"]),
    ("countdown_code", ["countdown_1p7b1p7b"]),
    ("countdown_code", ["countdown_1p7b1p7b", "countdown_4bjudge"]),
    ("countdown_code", ["countdown_1p7b1p7b", "countdown_8bjudge"]),
    ("countdown_code", ["countdown_1p7b1p7b", "countdown_14bjudge"]),
    ("countdown_code", ["countdown_1p7b1p7b", "countdown_32bjudge"]),
    ("countdown_code", ["countdown_1p7b1p7b", "countdown_exact"]),
    ("countdown_code", ["countdown_1p7b1p7b", "countdown_judgexexact"]),
    ("medical_abstract", ["medical_8b14b"]),
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
