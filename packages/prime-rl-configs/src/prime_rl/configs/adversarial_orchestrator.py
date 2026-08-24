from pathlib import Path
from typing import Annotated, TypeAlias

from pydantic import AliasChoices, Field, model_validator
from renderers import AutoRendererConfig, RendererConfig

from prime_rl.configs.monitors import WandbMonitorConfig
from prime_rl.configs.orchestrator import (
    FileSystemWeightBroadcastConfig,
    NCCLWeightBroadcastConfig,
    TrainSamplingConfig,
)
from prime_rl.configs.shared import (
    BaseModelConfig,
    ClientConfig,
    FileSystemTransportConfig,
    LogConfig,
    TransportConfig,
)
from prime_rl.configs.trainer import TokenizerConfig
from prime_rl.utils.config import BaseConfig


class EnvConfig(BaseConfig):
    """Legacy-verifiers env; the v1 migration replaces this family with upstream imports."""

    id: str = "prime_rl.adversarial.envs"

    name: str | None = None

    args: dict = {}
    """Kwargs for ``vf.load_environment``."""

    @property
    def stripped_id(self) -> str:
            return self.id.split("@")[0]

    @property
    def resolved_name(self) -> str:
        return self.name or self.stripped_id

class TrainConfig(BaseConfig):
    env: EnvConfig = EnvConfig()

    sampling: TrainSamplingConfig = TrainSamplingConfig()

    max_retries: int = Field(3, ge=0)

class EvalConfig(BaseConfig):
    """Periodic generator eval on the env's eval dataset, using the train sampling args."""

    num_examples: int = -1
    """``-1`` uses all available examples."""

    group_size: int = Field(1, ge=1, validation_alias=AliasChoices("group_size", "rollouts_per_example"))

    interval: int = Field(100, ge=1)


WeightBroadcastConfig: TypeAlias = Annotated[
    FileSystemWeightBroadcastConfig | NCCLWeightBroadcastConfig, Field(discriminator="type")
]


class OrchestratorConfig(BaseConfig):
    model: BaseModelConfig = BaseModelConfig()

    client: ClientConfig = ClientConfig()

    train: TrainConfig = TrainConfig()

    tokenizer: TokenizerConfig = TokenizerConfig()

    renderer: RendererConfig = AutoRendererConfig()
    """Renderer that tokenizes rollouts; ``auto`` resolves from the tokenizer name."""

    eval: EvalConfig | None = None

    log: LogConfig = LogConfig()

    wandb: WandbMonitorConfig | None = None

    weight_broadcast: WeightBroadcastConfig = FileSystemWeightBroadcastConfig()

    rollout_transport: TransportConfig = FileSystemTransportConfig()

    output_dir: Path = Path("outputs")
    """Root for this role's weights, rollouts, and logs; must match the trainer's."""

    batch_size: int | None = Field(None, ge=1)

    seed: int | None = None

    pad_to_multiple_of: int = Field(1, ge=1)

    group_size: int = Field(1, ge=1, validation_alias=AliasChoices("group_size", "rollouts_per_example"))
    """Rollouts per example (the GRPO group)."""

    seq_len: int = 2048

    num_train_workers: int = Field(1, ge=1)

    max_steps: int | None = None

    @model_validator(mode="after")
    def auto_setup_tokenizer(self):
        if self.tokenizer.name is None:
            self.tokenizer.name = self.model.name
        if self.tokenizer.trust_remote_code is None:
            self.tokenizer.trust_remote_code = self.model.trust_remote_code
        return self

    @model_validator(mode="after")
    def validate_renderer_auto_resolves(self):
        """Reject the silent DefaultRenderer fallback at config time."""
        if self.renderer.name != "auto":
            return self
        from renderers.base import MODEL_RENDERER_MAP

        model_id = self.tokenizer.name or self.model.name
        if model_id not in MODEL_RENDERER_MAP:
            raise ValueError(
                f"renderer 'auto' cannot resolve {model_id!r} (not in renderers.base.MODEL_RENDERER_MAP); "
                f"set [orchestrator.renderer] name='default' or a model-specific renderer explicitly."
            )
        return self

    @model_validator(mode="after")
    def resolve_batching(self):
        if self.batch_size is not None and self.batch_size % self.group_size != 0:
            raise ValueError("Batch size must be divisible by the number of samples per problem")
        return self

    @model_validator(mode="after")
    def resolve_sampling_defaults(self):
        """Populate vLLM sampling defaults (top_k/min_p disabled, token ids returned)."""
        extra_body = self.train.sampling.extra_body
        extra_body.setdefault("top_k", -1)
        extra_body.setdefault("min_p", 0.0)
        extra_body.setdefault("return_token_ids", True)
        return self
