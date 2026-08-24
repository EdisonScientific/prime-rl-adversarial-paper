from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, model_validator

from prime_rl.configs.adversarial_orchestrator import (
    FileSystemWeightBroadcastConfig as AdversarialOrchestratorFileSystemWeightBroadcastConfig,
)
from prime_rl.configs.adversarial_orchestrator import (
    NCCLWeightBroadcastConfig as AdversarialOrchestratorNCCLWeightBroadcastConfig,
)
from prime_rl.configs.adversarial_orchestrator import OrchestratorConfig as AdversarialOrchestratorConfig
from prime_rl.configs.inference import InferenceConfig
from prime_rl.configs.monitors import WandbMonitorConfig
from prime_rl.configs.rl import (
    DeploymentConfig,
    SharedFileSystemWeightBroadcastConfig,
    SharedLogConfig,
    SharedModelConfig,
    SharedNCCLWeightBroadcastConfig,
    SharedNIXLWeightBroadcastConfig,
    SharedWeightBroadcastConfig,
    SingleNodeDeploymentConfig,
)
from prime_rl.configs.shared import ResumeConfig, TransportConfig
from prime_rl.configs.trainer import CheckpointConfig as TrainerCheckpointConfig
from prime_rl.configs.trainer import (
    FileSystemWeightBroadcastConfig as TrainerFileSystemWeightBroadcastConfig,
)
from prime_rl.configs.trainer import (
    NCCLWeightBroadcastConfig as TrainerNCCLWeightBroadcastConfig,
)
from prime_rl.configs.trainer import (
    NIXLWeightBroadcastConfig as TrainerNIXLWeightBroadcastConfig,
)
from prime_rl.configs.trainer import TokenizerConfig, TrainerConfig
from prime_rl.utils.config import BaseConfig


class JudgeUpdateGateConfig(BaseConfig):
    enabled: bool = False
    """Skip judge updates while the judge is accurate enough."""

    accuracy_threshold: float = Field(0.9, ge=0.0, le=1.0)
    """Skip the judge update when the accuracy EMA is at or above this."""

    ema_decay: float = Field(0.95, ge=0.0, lt=1.0)
    """Decay of the judge-accuracy EMA the gate tracks."""

    warmup_steps: int = Field(0, ge=0)
    """Steps to always train the judge before the gate can skip."""

    max_skip_streak: int | None = Field(20, ge=1)
    """Force a judge update after this many consecutive skips. None never forces."""


class AdversarialRoleConfig(BaseConfig):
    trainer: TrainerConfig = TrainerConfig()
    orchestrator: AdversarialOrchestratorConfig = AdversarialOrchestratorConfig()
    inference: InferenceConfig | None = None
    """vLLM server for this role. None reuses an externally managed server."""

    output_dir: Path = Path("outputs")
    clean: bool = False
    """Wipe this role's output_dir before launch."""

    log: SharedLogConfig = SharedLogConfig()
    model: SharedModelConfig | None = None
    """Shared model name, propagated to trainer, orchestrator, and inference."""

    tokenizer: TokenizerConfig | None = None
    """Shared tokenizer, propagated to trainer and orchestrator."""

    max_steps: int | None = None
    """Shared step budget, propagated to trainer and orchestrator."""

    seq_len: int | None = None
    """Shared sequence length, propagated to trainer and orchestrator."""

    weight_broadcast: SharedWeightBroadcastConfig | None = None
    """Shared weight-broadcast transport, propagated to trainer, orchestrator, and inference."""

    rollout_transport: TransportConfig | None = None
    """Shared rollout transport, propagated to trainer and orchestrator."""

    resume: ResumeConfig | None = None
    deployment: DeploymentConfig = SingleNodeDeploymentConfig()
    """GPU/node split between this role's trainer and inference."""

    wandb: WandbMonitorConfig | None = None
    """Shared W&B run, propagated to trainer and orchestrator monitors."""

    @model_validator(mode="after")
    def resolve_shared_fields(self):
        if self.model is not None:
            self.trainer.model.name = self.model.name
            self.orchestrator.model.name = self.model.name
            if self.inference is not None:
                self.inference.vllm.model = self.model.name
        if self.tokenizer is not None:
            self.trainer.tokenizer = self.tokenizer.model_copy(deep=True)
            self.orchestrator.tokenizer = self.tokenizer.model_copy(deep=True)
        if self.max_steps is not None:
            self.trainer.max_steps = self.max_steps
            self.orchestrator.max_steps = self.max_steps
        if self.seq_len is not None:
            self.trainer.model.seq_len = self.seq_len
            self.orchestrator.seq_len = self.seq_len
        if self.weight_broadcast is not None:
            self._propagate_weight_broadcast()
        if self.rollout_transport is not None:
            self.orchestrator.rollout_transport = self.rollout_transport.model_copy(deep=True)
        # The trainer must consume batches over the same transport the orchestrator
        # sends on (the upstream trainer default is ZMQ, ours is filesystem).
        self.trainer.rollout_transport = self.orchestrator.rollout_transport.model_copy(deep=True)
        if self.resume is not None:
            self.trainer.resume = self.resume.model_copy(deep=True)
        if self.wandb is not None:
            self.trainer.monitors.wandb = self.wandb.model_copy(deep=True)
            self.orchestrator.wandb = self.wandb.model_copy(deep=True)
        return self

    def _propagate_weight_broadcast(self) -> None:
        assert self.weight_broadcast is not None
        data = self.weight_broadcast.model_dump()
        if isinstance(self.weight_broadcast, SharedFileSystemWeightBroadcastConfig):
            self.trainer.weight_broadcast = TrainerFileSystemWeightBroadcastConfig()
            self.orchestrator.weight_broadcast = AdversarialOrchestratorFileSystemWeightBroadcastConfig()
        elif isinstance(self.weight_broadcast, SharedNCCLWeightBroadcastConfig):
            self.trainer.weight_broadcast = TrainerNCCLWeightBroadcastConfig(**data)
            self.orchestrator.weight_broadcast = AdversarialOrchestratorNCCLWeightBroadcastConfig(**data)
        elif isinstance(self.weight_broadcast, SharedNIXLWeightBroadcastConfig):
            self.trainer.weight_broadcast = TrainerNIXLWeightBroadcastConfig(**data)
            raise ValueError("adversarial-rl does not support NIXL weight broadcast yet.")
        else:
            raise ValueError(f"Unsupported weight broadcast config: {self.weight_broadcast!r}")
        if self.inference is not None:
            self.inference.weight_broadcast.type = self.weight_broadcast.type


class AdversarialRLConfig(BaseConfig):
    """Configures one synchronous generator/judge prototype run."""

    generator: Annotated[AdversarialRoleConfig, Field(description="Generator RL run.")]
    judge: Annotated[AdversarialRoleConfig, Field(description="Judge classifier run.")]
    output_dir: Annotated[Path, Field(description="Top-level output directory.")] = Path("outputs/adversarial")
    clean_output_dir: Annotated[bool, Field(description="Clean nested output dirs before launch.")] = False
    generator_reward_mode: Annotated[
        Literal["judge", "exact_only", "judge_times_exact"],
        Field(
            description=(
                "Use judge feedback, sparse exact-only rewards, or judge-gated exact rewards."
            )
        ),
    ] = "judge"
    judge_objective: Literal["single_candidate", "pairwise"] = "single_candidate"
    """``single_candidate``: the judge scores generated and reference answers separately
    (two judge calls per rollout). ``pairwise``: the judge sees both in one prompt and
    picks the reference (one call per rollout; requires ``pairwise = true`` in the judge env args)."""

    resume_step: int | None = Field(None, ge=1)
    judge_update_gate: JudgeUpdateGateConfig = JudgeUpdateGateConfig()
    log: SharedLogConfig = SharedLogConfig()

    @model_validator(mode="after")
    def resolve_nested_runs(self):
        for run in (self.generator, self.judge):
            if run.deployment.type == "multi_node":
                expected_infer_nodes = int(run.inference is not None)
                if run.deployment.total_infer_nodes != expected_infer_nodes:
                    raise ValueError("adversarial-rl multi_node roles support at most one inference node per role.")
            elif run.deployment.type != "single_node":
                raise ValueError("adversarial-rl supports single_node or one-train-one-infer multi_node roles.")
        if self.generator.orchestrator.batch_size is None or self.judge.orchestrator.batch_size is None:
            raise ValueError("adversarial-rl requires fixed generator and judge batch_size.")
        self.generator = self.with_output_dir(self.generator, self.output_dir / "generator")
        self.judge = self.with_output_dir(self.judge, self.output_dir / "judge")
        if self.resume_step is not None:
            for run in (self.generator, self.judge):
                if run.trainer.ckpt is None:
                    run.trainer.ckpt = TrainerCheckpointConfig()
                run.trainer.resume = ResumeConfig(step=self.resume_step)
        return self

    def with_output_dir(self, run: AdversarialRoleConfig, output_dir: Path) -> AdversarialRoleConfig:
        run = run.model_copy(deep=True)
        run.output_dir = output_dir
        run.trainer.output_dir = output_dir
        run.orchestrator.output_dir = output_dir
        if self.clean_output_dir:
            run.clean = True
        return run
