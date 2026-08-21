import re
from pathlib import Path
from typing import Any

import orjson
import verifiers as vf
from verifiers.utils.save_utils import make_serializable

from prime_rl.transports.rollouts.types import TrainingSample
from prime_rl.utils.chat_template import deserialize_tool_calls, normalize_messages
from prime_rl.utils.logger import get_logger

ANSWER_RE = re.compile(r"\s*<answer>(.*?)</answer>\s*\Z", re.DOTALL | re.IGNORECASE)


def completion_text(completion: Any) -> str:
    """Final message text; reasoning split off by a vllm reasoning parser is
    re-wrapped in <think> tags so parsed and raw completions look identical."""
    if isinstance(completion, str):
        return completion
    if not completion:
        return ""
    message = completion[-1] if isinstance(completion, list) else completion
    content = message.get("content") if isinstance(message, dict) else getattr(message, "content", "")
    reasoning = (
        message.get("reasoning_content") if isinstance(message, dict) else getattr(message, "reasoning_content", None)
    )
    if reasoning:
        return f"<think>{reasoning}</think>\n{content}"
    return str(content or "")


def split_think(completion: Any) -> tuple[str, str]:
    """``(reasoning, rest)`` around the last ``</think>``. The opening ``<think>``
    is optional — chat templates and thinking-budget interventions inject tags."""
    text = completion_text(completion).strip()
    reasoning, sep, rest = text.rpartition("</think>")
    if not sep:
        return "", text
    return reasoning.removeprefix("<think>").strip(), rest


def extract_answer(completion: Any, *, require_think: bool = True) -> str | None:
    """The ``<answer>...</answer>`` payload, or None when the completion is not a
    reasoning block (when required) followed by exactly one answer block."""
    reasoning, rest = split_think(completion)
    if require_think and not reasoning:
        return None
    match = ANSWER_RE.fullmatch(rest)
    if match is None:
        return None
    answer = match.group(1).strip()
    if not answer or re.search(r"</?answer\b", answer, re.IGNORECASE):
        return None
    return answer


def extract_choice(completion: Any) -> str | None:
    """A/B judge choice; judges sample with structured outputs, so the completion is the bare letter."""
    _, rest = split_think(completion)
    choice = rest.strip().upper()
    return choice if choice in {"A", "B"} else None


def collapse_whitespace(text: str) -> str:
    return " ".join(str(text).strip().split())


def get_model_completion_len(output: vf.RolloutOutput) -> int:
    """Sum of model-generated completion tokens across all turns (excludes
    environment-injected tokens between turns)."""
    return sum(len(step["tokens"]["completion_ids"]) for step in output["trajectory"] if step.get("tokens"))


def save_rollouts(rollouts: list[vf.RolloutOutput], path: Path, exclude_keys: set[str] | None = None) -> None:
    """Save rollouts to a JSONL file using verifiers serialization."""
    path.parent.mkdir(parents=True, exist_ok=True)
    opts = orjson.OPT_APPEND_NEWLINE | orjson.OPT_SERIALIZE_NUMPY
    with open(path, "wb") as f:
        for rollout in rollouts:
            row = {k: v for k, v in rollout.items() if k not in exclude_keys} if exclude_keys else rollout
            f.write(orjson.dumps(row, default=make_serializable, option=opts))


def _clean_messages(messages: Any, default_role: str) -> list[dict[str, Any]]:
    messages = deserialize_tool_calls(normalize_messages(messages, default_role))
    return [
        {**message, "content": message["content"].strip()} if isinstance(message.get("content"), str) else message
        for message in messages
    ]


def backfill_rollout_tokens(output: vf.RolloutOutput, renderer) -> None:
    """Populate missing step tokens from prompt/completion messages via the renderer."""
    from renderers.base import build_trajectory_step

    for step in output["trajectory"]:
        if step["tokens"] is None:
            prompt = _clean_messages(step.get("prompt"), "user")
            completion = _clean_messages(step.get("completion"), "assistant")
            step["tokens"] = build_trajectory_step(renderer, prompt, completion)


def rollout_to_sample(output: vf.RolloutOutput, *, env_name: str = "") -> TrainingSample | None:
    """Convert one single-turn rollout into a flat TrainingSample.

    Errored rollouts keep their tokens but train on nothing (fully masked
    completion). Returns None for rollouts with no trajectory or no tokens.
    """
    trajectory = output["trajectory"]
    if not trajectory:
        get_logger().warning(
            f"No trajectory steps for example {output['example_id']} "
            f"(error={output.get('error')}, stop={output.get('stop_condition')}). Skipping rollout."
        )
        return None
    if len(trajectory) != 1:
        raise NotImplementedError(
            f"Adversarial envs are single-turn; got {len(trajectory)} trajectory steps "
            f"for example {output['example_id']}."
        )
    tokens = trajectory[0]["tokens"]
    if tokens is None:
        get_logger().warning(f"Missing rollout tokens for example {output['example_id']}. Skipping rollout.")
        return None
    if tokens.get("routed_experts") is not None or tokens.get("multi_modal_data") is not None:
        raise NotImplementedError(
            "The adversarial port supports text-only rollouts without router replay; "
            "got a routed_experts or multi_modal_data payload."
        )

    prompt_ids = list(tokens["prompt_ids"])
    completion_ids = list(tokens["completion_ids"])
    if output["error"] is not None:
        completion_mask = [False] * len(tokens["completion_mask"])
    else:
        completion_mask = list(map(bool, tokens["completion_mask"]))
    token_ids = prompt_ids + completion_ids
    return TrainingSample(
        token_ids=token_ids,
        mask=[False] * len(prompt_ids) + completion_mask,
        logprobs=[0.0] * len(prompt_ids) + list(tokens["completion_logprobs"]),
        temperatures=[1.0] * len(token_ids),
        env_name=env_name,
    )
