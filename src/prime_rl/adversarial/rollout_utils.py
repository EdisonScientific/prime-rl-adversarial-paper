from typing import Any

import verifiers as vf

from prime_rl.transport import TrainingSample
from prime_rl.utils.chat_template import deserialize_tool_calls, normalize_messages
from prime_rl.utils.logger import get_logger


def _strip_message_content(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {**message, "content": message["content"].strip()} if isinstance(message.get("content"), str) else message
        for message in messages
    ]


def _tokenize_step_with_renderer(step: vf.TrajectoryStep, renderer) -> dict[str, Any]:
    """Tokenize a trajectory step using a Renderer."""
    from renderers.base import build_trajectory_step

    prompt = normalize_messages(step.get("prompt"), default_role="user")
    completion = normalize_messages(step.get("completion"), default_role="assistant")
    prompt = _strip_message_content(deserialize_tool_calls(prompt))
    completion = _strip_message_content(deserialize_tool_calls(completion))
    return build_trajectory_step(renderer, prompt, completion)


def backfill_rollout_tokens(output: vf.RolloutOutput, renderer) -> None:
    """Populate missing step tokens from prompt/completion messages via the renderer."""
    for step in output["trajectory"]:
        if step["tokens"] is None:
            step["tokens"] = _tokenize_step_with_renderer(step, renderer)


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
