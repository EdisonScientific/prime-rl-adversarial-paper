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
    """A legacy-verifiers env (``id`` + kwargs for ``vf.load_environment``) — not the
    verifiers.v1 ``configs.orchestrator.EnvConfig``. The Train/Eval config family below
    forks with it; migrating the envs to v1 replaces all of them with upstream imports."""

    id: str = "prime_rl.adversarial.envs"
    """Environment ID passed to ``vf.load_environment``; may carry an ``@version`` suffix."""

    name: str | None = None
    """Display name; defaults to the ``id`` without ``@version``."""

    args: dict = {}
    """Keyword arguments forwarded to ``vf.load_environment``."""

    @property
    def stripped_id(self) -> str:
        """Environment ID without the @version suffix."""
        return self.id.split("@")[0]

    @property
    def resolved_name(self) -> str:
        return self.name or self.stripped_id

class TrainConfig(BaseConfig):
    env: EnvConfig = EnvConfig()
    """The role's single training environment."""

    sampling: TrainSamplingConfig = TrainSamplingConfig()
    """Training sampling configuration."""

    max_retries: int = Field(3, ge=0)
    """Retries for a failed rollout."""

class EvalConfig(BaseConfig):
    """Periodic generator eval on the env's held-out eval dataset (rolled out
    with the train sampling args)."""

    num_examples: int = -1
    """Eval examples to sample from the dataset. ``-1`` uses all available examples."""

    group_size: int = Field(1, ge=1, validation_alias=AliasChoices("group_size", "rollouts_per_example"))
    """Rollouts generated per example."""

    interval: int = Field(100, ge=1)
    """Step interval at which to evaluate the model."""


WeightBroadcastConfig: TypeAlias = Annotated[
    FileSystemWeightBroadcastConfig | NCCLWeightBroadcastConfig, Field(discriminator="type")
]


class OrchestratorConfig(BaseConfig):
    model: BaseModelConfig = BaseModelConfig()
    """The model being trained (rollouts sample from its inference server)."""

    client: ClientConfig = ClientConfig()
    """Client for the model's inference server."""

    train: TrainConfig = TrainConfig()

    tokenizer: TokenizerConfig = TokenizerConfig()

    renderer: RendererConfig = AutoRendererConfig()
    """Renderer that tokenizes rollouts; ``auto`` resolves from the tokenizer name."""

    eval: EvalConfig | None = None
    """Evaluation configuration."""

    log: LogConfig = LogConfig()

    wandb: WandbMonitorConfig | None = None

    weight_broadcast: WeightBroadcastConfig = FileSystemWeightBroadcastConfig()
    """Transport used to receive updated weights from the trainer."""

    rollout_transport: TransportConfig = FileSystemTransportConfig()
    """Transport used to ship rollouts from orchestrator to trainer."""

    output_dir: Path = Path("outputs")
    """Root for this role's weights, rollouts, and logs; must match the trainer's."""

    batch_size: int | None = Field(None, ge=1)
    """Samples to train on per step."""

    seed: int | None = None
    """Dataset sampling seed for the synchronous adversarial loop."""

    pad_to_multiple_of: int = Field(1, ge=1)
    """Pad packed microbatches to a multiple of this token count."""

    group_size: int = Field(1, ge=1, validation_alias=AliasChoices("group_size", "rollouts_per_example"))
    """Rollouts per example (the GRPO group)."""

    seq_len: int = 2048
    """Training sequence length."""

    num_train_workers: int = Field(1, ge=1)
    """Training workers to use."""

    max_steps: int | None = None
    """Maximum training steps. If None, runs indefinitely."""

    @model_validator(mode="after")
    def auto_setup_tokenizer(self):
        if self.tokenizer.name is None:
            self.tokenizer.name = self.model.name
        if self.tokenizer.trust_remote_code is None:
            self.tokenizer.trust_remote_code = self.model.trust_remote_code
        return self

    @model_validator(mode="after")
    def auto_setup_session_headers(self):
        """Sticky DP-aware routing at the inference router."""
        self.client.extra_headers_from_state.setdefault("X-Session-ID", "trajectory_id")
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
