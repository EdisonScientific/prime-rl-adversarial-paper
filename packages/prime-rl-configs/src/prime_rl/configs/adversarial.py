from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, model_validator

from prime_rl.configs.adversarial_orchestrator import (
    FileSystemWeightBroadcastConfig as AdversarialOrchestratorFileSystemWeightBroadcastConfig,
)
from prime_rl.configs.adversarial_orchestrator import CheckpointConfig as AdversarialOrchestratorCheckpointConfig
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
from prime_rl.configs.shared import EnvVars, ResumeConfig, RunConfig, SlurmConfig, TransportConfig
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
from prime_rl.configs.trainer import CheckpointConfig as TrainerCheckpointConfig
from prime_rl.utils.config import BaseConfig


class JudgeUpdateGateConfig(BaseConfig):
    """Optionally skip judge updates when the judge is already too accurate."""

    enabled: bool = False
    accuracy_threshold: float = Field(0.9, ge=0.0, le=1.0)
    ema_decay: float = Field(0.95, ge=0.0, lt=1.0)
    warmup_steps: int = Field(0, ge=0)
    max_skip_streak: int | None = Field(20, ge=1)


class AdversarialCurriculumStageConfig(BaseConfig):
    """Archived curriculum stage shape kept so older disabled TOML blocks parse."""

    reference_formula_prefix_terms_from_end: int | None = None
    no_reference_formula_prefix: bool = False

    @model_validator(mode="after")
    def validate_stage(self):
        if self.no_reference_formula_prefix and self.reference_formula_prefix_terms_from_end is not None:
            raise ValueError(
                "Set either no_reference_formula_prefix or reference_formula_prefix_terms_from_end, not both."
            )
        return self


class AdversarialCurriculumConfig(BaseConfig):
    """Archived curriculum shape; enabled curriculum is not supported by the minimal port."""

    enabled: bool = False
    metric: str = "verifiable_reward"
    threshold: float = 0.3
    window: int = Field(50, ge=1)
    min_steps_per_stage: int = Field(100, ge=0)
    max_steps_per_stage: int | None = Field(None, ge=1)
    min_metric_values: dict[str, float] = Field(default_factory=dict)
    max_metric_values: dict[str, float] = Field(default_factory=dict)
    stages: list[AdversarialCurriculumStageConfig] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_curriculum(self):
        if self.enabled and not self.stages:
            raise ValueError("Enabled adversarial curriculum requires at least one stage.")
        return self


class AdversarialRoleConfig(BaseConfig):
    """One trainable role in a synchronous adversarial run.

    This intentionally preserves the older adversarial TOML shape while reusing
    the current trainer and inference config models.
    """

    trainer: TrainerConfig = TrainerConfig()
    orchestrator: AdversarialOrchestratorConfig = AdversarialOrchestratorConfig()
    inference: InferenceConfig | None = None
    env_vars: EnvVars = {}
    run: RunConfig = RunConfig()
    output_dir: Path = Path("outputs")
    clean: bool = False
    log: SharedLogConfig = SharedLogConfig()
    model: SharedModelConfig | None = None
    tokenizer: TokenizerConfig | None = None
    max_steps: int | None = None
    seq_len: int | None = None
    weight_broadcast: SharedWeightBroadcastConfig | None = None
    rollout_transport: TransportConfig | None = None
    resume: ResumeConfig | None = None
    deployment: DeploymentConfig = SingleNodeDeploymentConfig()
    slurm: SlurmConfig | None = None
    dry_run: bool = False
    wandb: WandbMonitorConfig | None = None

    @model_validator(mode="after")
    def resolve_shared_fields(self):
        if self.model is not None:
            self.trainer.model.name = self.model.name
            self.orchestrator.student.model.name = self.model.name
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
            self.trainer.rollout_transport = self.rollout_transport.model_copy(deep=True)
            self.orchestrator.rollout_transport = self.rollout_transport.model_copy(deep=True)
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
    judge_objective: Literal["single_candidate"] = "single_candidate"
    resume_step: int | None = Field(None, ge=1)
    judge_update_gate: JudgeUpdateGateConfig = JudgeUpdateGateConfig()
    curriculum: AdversarialCurriculumConfig = AdversarialCurriculumConfig()
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
        for run in (self.generator, self.judge):
            if getattr(run, "teacher_inference", None) is not None or (
                getattr(run.deployment, "num_teacher_gpus", None) or 0
            ):
                raise ValueError("adversarial-rl currently does not support teacher inference.")
        self.generator = self.with_output_dir(self.generator, self.output_dir / "generator")
        self.judge = self.with_output_dir(self.judge, self.output_dir / "judge")
        if self.resume_step is not None:
            for run in (self.generator, self.judge):
                if run.trainer.ckpt is None:
                    run.trainer.ckpt = TrainerCheckpointConfig()
                if run.orchestrator.ckpt is None:
                    run.orchestrator.ckpt = AdversarialOrchestratorCheckpointConfig()
                run.trainer.resume = ResumeConfig(step=self.resume_step)
                run.orchestrator.ckpt.resume_step = self.resume_step
        return self

    def with_output_dir(self, run: AdversarialRoleConfig, output_dir: Path) -> AdversarialRoleConfig:
        run = run.model_copy(deep=True)
        run.output_dir = output_dir
        run.trainer.output_dir = output_dir
        run.orchestrator.output_dir = output_dir / "run_default"
        if self.clean_output_dir:
            run.clean = True
        return run
