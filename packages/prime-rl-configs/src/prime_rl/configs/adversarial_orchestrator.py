from pathlib import Path
from typing import Annotated, Any, TypeAlias

from pydantic import AliasChoices, Field, model_validator
from renderers import AutoRendererConfig, RendererConfig

from prime_rl.configs.monitors import PrimeMonitorConfig, WandbMonitorConfig
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
    verifiers.v1 ``configs.orchestrator.EnvConfig``; migrating the envs to v1 replaces this."""

    id: str = "reverse-text"
    """Registered verifiers environment ID (e.g. ``math-env``, ``primeintellect/math-env``). May include an ``@version`` suffix for installation."""

    name: str | None = None
    """Display name for this environment in logs, metrics, and buffer keys. Defaults to the ``id`` without ``@version``. Must be unique across all envs in the same group."""

    args: dict = {}
    """Keyword arguments forwarded to ``vf.load_environment``. See the environment's docstring for accepted args."""

    max_retries: int = Field(3, ge=0)
    """Times the env server retries a failed rollout before returning an error."""

    @property
    def stripped_id(self) -> str:
        """Environment ID without the @version suffix."""
        return self.id.split("@")[0]

    @property
    def resolved_name(self) -> str:
        return self.name or self.stripped_id

    @model_validator(mode="after")
    def validate_env_name(self):
        if self.resolved_name == "all":
            raise ValueError(
                'Environment name "all" is reserved for global metric aggregation. Use a different name or id.'
            )
        return self

class TrainEnvConfig(EnvConfig):
    sampling: TrainSamplingConfig = TrainSamplingConfig()
    """Per-env sampling overrides. Unset fields inherit from the group-level train sampling config."""

    group_size: int = Field(1, ge=1, validation_alias=AliasChoices("group_size", "rollouts_per_example"))
    """Rollouts generated per example for GRPO group-relative advantages.
    Inherits from ``orchestrator.group_size`` when unset."""


class TrainConfig(BaseConfig):
    env: list[TrainEnvConfig] = [TrainEnvConfig()]
    """Training environments."""

    sampling: TrainSamplingConfig = TrainSamplingConfig()
    """Shared training sampling configuration."""

    max_retries: int = Field(3, ge=0)
    """Default retries for failed rollouts. Can be overridden per env."""

    @model_validator(mode="after")
    def resolve_env_defaults(self):
        """Resolve per-env overrides: inherit group-level sampling and max_retries."""
        group_sampling = self.sampling.model_dump()
        for env in self.env:
            if "sampling" not in env.model_fields_set:
                env.sampling = TrainSamplingConfig(**group_sampling)
            else:
                merged = group_sampling | env.sampling.model_dump(exclude_unset=True)
                env.sampling = TrainSamplingConfig(**merged)
            if "max_retries" not in env.model_fields_set:
                env.max_retries = self.max_retries
        return self

    @model_validator(mode="after")
    def validate_unique_env_names(self):
        env_names = [env.resolved_name for env in self.env]
        duplicates = [n for n in env_names if env_names.count(n) > 1]
        if duplicates:
            raise ValueError(
                f"Duplicate training environment names: {set(duplicates)}. Each env must have a unique name."
            )
        return self

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


class LegacyClientConfig(ClientConfig):
    @model_validator(mode="before")
    @classmethod
    def accept_singleton_base_url_lists(cls, data: Any) -> Any:
        if isinstance(data, dict) and isinstance(data.get("base_url"), list):
            urls = data["base_url"]
            if len(urls) != 1:
                raise ValueError("adversarial client.base_url must contain exactly one URL when passed as a list.")
            data = {**data, "base_url": urls[0]}
        return data


class OrchestratorConfig(BaseConfig):
    model: BaseModelConfig = BaseModelConfig()
    """The model being trained (rollouts sample from its inference server)."""

    client: LegacyClientConfig = LegacyClientConfig()
    """Client for the model's inference server."""

    train: TrainConfig = TrainConfig()

    tokenizer: TokenizerConfig = TokenizerConfig()

    renderer: RendererConfig = AutoRendererConfig()
    """Typed renderer config (``renderers.RendererConfig`` discriminated
    union), required — rollouts are tokenized through the renderer. Defaults
    to ``"auto"``, which resolves from ``tokenizer.name_or_path`` via
    ``MODEL_RENDERER_MAP``."""

    eval: EvalConfig | None = None
    """Evaluation configuration."""

    log: LogConfig = LogConfig()

    wandb: WandbMonitorConfig | None = None

    prime_monitor: PrimeMonitorConfig | None = None

    weight_broadcast: WeightBroadcastConfig = FileSystemWeightBroadcastConfig()
    """Transport used to receive updated weights from the trainer."""

    rollout_transport: TransportConfig = FileSystemTransportConfig()
    """Transport used to ship rollouts from orchestrator to trainer."""

    output_dir: Path = Path("outputs/run_default")
    """Directory to write outputs to — checkpoints, weights, rollouts, and logs are written as subdirectories. Should be a persistent directory with enough disk space and unique per experiment running on a single node."""

    batch_size: int | None = Field(None, ge=1)
    """Samples to train on per step."""

    seed: int | None = None
    """Dataset sampling seed for the synchronous adversarial loop."""

    pad_to_multiple_of: int = Field(1, ge=1)
    """Pad packed microbatches to a multiple of this token count."""

    group_size: int = Field(1, ge=1, validation_alias=AliasChoices("group_size", "rollouts_per_example"))
    """Output sequences returned per example during training."""

    seq_len: int = 2048
    """Training sequence length. Shorter samples are padded; longer samples are truncated."""

    # TODO(Mika): This should be automatic from the number of ZMQ connections
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
        """Ensure X-Session-ID header is always set for sticky DP-aware routing at the inference router."""
        self.client.extra_headers_from_state.setdefault("X-Session-ID", "trajectory_id")
        return self

    @model_validator(mode="after")
    def auto_setup_prime_monitor_run_name(self):
        """Default ``prime_monitor.name`` to the W&B run name when monitoring
        is enabled and the user hasn't named the prime-monitor run explicitly."""
        if self.prime_monitor is None or self.prime_monitor.name is not None:
            return self
        if self.wandb is not None and self.wandb.name:
            self.prime_monitor.name = self.wandb.name
        return self

    @model_validator(mode="after")
    def validate_renderer_auto_resolves(self):
        """Reject the silent DefaultRenderer fallback at config time.

        When ``renderer.name='auto'`` and the model isn't in
        ``MODEL_RENDERER_MAP``, ``create_renderer`` would fall back to
        ``DefaultRenderer``. That fallback doesn't fix the
        position-dependent chat-template bug the renderer client exists
        to solve, and rejects envs that pass tools (the rollout dies
        with "RendererPool does not support tools") unless
        ``DefaultRendererConfig.tool_parser`` is configured. Surface at
        config time so ``--dry-run`` reports the error.
        """
        if self.renderer.name != "auto":
            return self
        from renderers.base import MODEL_RENDERER_MAP

        model_id = self.tokenizer.name or self.model.name
        if model_id in MODEL_RENDERER_MAP:
            return self
        raise ValueError(
            f"orchestrator.renderer.name='auto' but "
            f"{model_id!r} is not in renderers.base.MODEL_RENDERER_MAP, so it "
            f"would silently fall back to DefaultRenderer. Pick one: "
            f"(a) [orchestrator.renderer] name='default' — for fine-tunes / "
            f"vendored mirrors with custom chat templates (DefaultRenderer "
            f"calls apply_chat_template); set tool_parser=<name> if the env "
            f"uses tools. "
            f"(b) [orchestrator.renderer] name=<model-specific renderer> — "
            f"if {model_id!r} is template-identical to a mapped family "
            f"(and ideally also add it upstream to "
            f"renderers.base.MODEL_RENDERER_MAP)."
        )

    @model_validator(mode="after")
    def resolve_batching(self):
        if self.batch_size is None:
            self.batch_size = 128
        if self.batch_size % self.group_size != 0:
            raise ValueError("Batch size must be divisible by the number of samples per problem")

        # Propagate the top-level ``group_size`` into each train env that didn't set its own.
        for env_cfg in self.train.env:
            if "group_size" not in env_cfg.model_fields_set:
                env_cfg.group_size = self.group_size

        return self

    @model_validator(mode="after")
    def resolve_env_config(self):
        """Populate vLLM sampling defaults on each train env."""
        for env in self.train.env:
            env.sampling.extra_body.setdefault("top_k", -1)
            env.sampling.extra_body.setdefault("min_p", 0.0)
            env.sampling.extra_body.setdefault("return_token_ids", True)
        return self
