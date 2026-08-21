from pathlib import Path

import orjson
import verifiers as vf
from verifiers.utils.save_utils import make_serializable

from prime_rl.configs.adversarial_orchestrator import OrchestratorConfig
from prime_rl.utils.client import InferencePool
from prime_rl.utils.logger import get_logger


async def setup_student_inference_pool(*, config: OrchestratorConfig, tokenizer):
    """Build the student inference pool + matching renderer."""
    from renderers.base import create_renderer

    client_config = config.student.client
    model_name = config.student.model.name
    renderer = create_renderer(tokenizer, config.renderer)
    get_logger().info(f"Initialized {type(renderer).__name__} for {model_name}")
    inference_pool = InferencePool(
        client_config,
        model_name=model_name,
        train_client_type="renderer",
        eval_client_type="openai_chat_completions",
        renderer_config=config.renderer,
    )
    return renderer, inference_pool


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
