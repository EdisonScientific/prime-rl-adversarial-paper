import argparse
import json
from collections.abc import Callable
from typing import Any

import verifiers as vf


def _parse_env_arg(raw: str) -> tuple[str, Any]:
    key, sep, value = raw.partition("=")
    if not sep or not key:
        raise ValueError(f"Expected --env-arg KEY=VALUE, got {raw!r}")
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = value
    return key, parsed


def main(load_environment: Callable[..., vf.Environment]) -> None:
    parser = argparse.ArgumentParser(description="Preview an adversarial environment.")
    parser.add_argument("--kind", choices=["generator", "judge"], default="generator")
    parser.add_argument("--num-examples", type=int, default=3)
    parser.add_argument("--env-arg", action="append", default=[], help="Environment kwarg as KEY=JSON_VALUE.")
    args = parser.parse_args()

    env_kwargs = dict(_parse_env_arg(raw) for raw in args.env_arg)
    env = load_environment(kind=args.kind, **env_kwargs)
    print(f"{env.__class__.__name__}")
    if args.kind == "judge" or not hasattr(env, "get_dataset"):
        return
    for idx, example in enumerate(env.get_dataset()):
        if idx >= args.num_examples:
            break
        print(json.dumps(example, ensure_ascii=False, default=str))

