from pathlib import Path

from prime_rl.configs.adversarial import AdversarialRLConfig
from prime_rl.utils.config import cli

CONFIG_DIR = Path("configs/adversarial")


def test_example_recipe_validates():
    """The adversarial configs are layers; only their composition must validate."""
    args = []
    for name in ["base", "multiplication"]:
        args += ["@", (CONFIG_DIR / f"{name}.toml").as_posix()]
    cli(AdversarialRLConfig, args=args)
