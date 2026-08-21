import re
from pathlib import Path
from typing import Any

import orjson
import verifiers as vf
from verifiers.utils.save_utils import make_serializable

from prime_rl.transports.rollouts.types import TrainingSample
from prime_rl.utils.chat_template import deserialize_tool_calls, normalize_messages
from prime_rl.utils.logger import get_logger

CHOICE_FALLBACK_RE = re.compile(r"\b([AB])\b", re.IGNORECASE)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
STRICT_ANSWER_ONLY_RE = re.compile(r"\s*<answer>(.*?)</answer>\s*\Z", re.DOTALL | re.IGNORECASE)
STRICT_THINK_ANSWER_RE = re.compile(r"\s*<think>(.*?)</think>\s*<answer>(.*?)</answer>\s*\Z", re.DOTALL | re.IGNORECASE)


def to_text(completion: Any) -> str:
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


def strip_think(text: str) -> str:
    end = text.rfind("</think>")
    return text[end + len("</think>") :] if end != -1 else text


def completion_has_reasoning(completion: Any) -> bool:
    text = to_text(completion).strip()
    match = re.match(r"\s*<think>(.*?)</think>", text, re.DOTALL | re.IGNORECASE)
    return bool(match and match.group(1).strip())


def extract_strict_think_answer_text(completion: Any) -> str | None:
    match = STRICT_THINK_ANSWER_RE.fullmatch(to_text(completion).strip())
    if not match:
        return None
    thought = match.group(1).strip()
    answer = match.group(2).strip()
    return answer if thought and answer else None


def extract_strict_answer_only_text(completion: Any) -> str | None:
    text = strip_think(to_text(completion)).strip()
    match = STRICT_ANSWER_ONLY_RE.fullmatch(text)
    if match is None:
        return None
    answer = match.group(1).strip()
    if not answer or re.search(r"</?answer\b", answer, re.IGNORECASE):
        return None
    return answer


def extract_answer_text(completion: Any) -> str | None:
    answers = [answer.strip() for answer in ANSWER_RE.findall(strip_think(to_text(completion)))]
    answers = [answer for answer in answers if answer]
    return answers[-1] if answers else None


def extract_choice(completion: Any) -> str | None:
    answer = extract_answer_text(completion)
    text = answer if answer is not None else strip_think(to_text(completion)).strip()
    choice = text.strip().upper()
    if choice in {"A", "B"}:
        return choice
    matches = CHOICE_FALLBACK_RE.findall(choice)
    return matches[-1].upper() if matches else None


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
