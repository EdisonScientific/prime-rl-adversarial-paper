from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from prime_rl.configs.adversarial_orchestrator import OrchestratorConfig as AdversarialOrchestratorConfig
from prime_rl.configs.inference import InferenceConfig
from prime_rl.configs.monitors import WandbMonitorConfig
from prime_rl.configs.rl import (
    DeploymentConfig,
    SharedLogConfig,
    SharedModelConfig,
    SingleNodeDeploymentConfig,
)
from prime_rl.configs.shared import TransportConfig
from prime_rl.configs.trainer import TrainerConfig
from prime_rl.utils.config import BaseConfig


class JudgeUpdateGateConfig(BaseConfig):
    enabled: bool = Field(False, description="Skip judge updates while the judge is accurate enough.")
    accuracy_threshold: float = Field(0.9, ge=0.0, le=1.0, description="Skip when the accuracy EMA is at or above this.")
    ema_decay: float = Field(0.95, ge=0.0, lt=1.0, description="Decay of the judge-accuracy EMA the gate tracks.")
    warmup_steps: int = Field(0, ge=0, description="Steps to always train the judge before the gate can skip.")
    max_skip_streak: int | None = Field(
        20, ge=1, description="Force a judge update after this many consecutive skips. None never forces."
    )


class AdversarialRoleConfig(BaseConfig):
    trainer: TrainerConfig = TrainerConfig()
    orchestrator: AdversarialOrchestratorConfig = AdversarialOrchestratorConfig()
    inference: InferenceConfig | None = Field(
        None, description="vLLM server for this role. None reuses an externally managed server."
    )
    output_dir: Path = Path("outputs")
    clean: bool = Field(False, description="Wipe this role's output_dir before launch.")
    log: SharedLogConfig = SharedLogConfig()
    model: SharedModelConfig | None = Field(
        None, description="Shared model name, propagated to trainer, orchestrator, and inference."
    )
    max_steps: int | None = Field(None, description="Shared step budget, propagated to trainer and orchestrator.")
    seq_len: int | None = Field(None, description="Shared sequence length, propagated to trainer and orchestrator.")
    rollout_transport: TransportConfig | None = Field(
        None, description="Shared rollout transport, propagated to trainer and orchestrator."
    )
    deployment: DeploymentConfig = Field(
        SingleNodeDeploymentConfig(), description="GPU/node split between this role's trainer and inference."
    )
    wandb: WandbMonitorConfig | None = Field(
        None, description="Shared W&B run, propagated to trainer and orchestrator monitors."
    )

    @model_validator(mode="after")
    def resolve_shared_fields(self):
        if self.model is not None:
            self.trainer.model.name = self.model.name
            self.orchestrator.model.name = self.model.name
            if self.inference is not None:
                self.inference.vllm.model = self.model.name
        if self.max_steps is not None:
            self.trainer.max_steps = self.max_steps
            self.orchestrator.max_steps = self.max_steps
        if self.seq_len is not None:
            self.trainer.model.seq_len = self.seq_len
            self.orchestrator.seq_len = self.seq_len
        if self.rollout_transport is not None:
            self.orchestrator.rollout_transport = self.rollout_transport.model_copy(deep=True)
        # The trainer's rollout transport must match the orchestrator's because their defaults differ.
        self.trainer.rollout_transport = self.orchestrator.rollout_transport.model_copy(deep=True)
        if self.wandb is not None:
            self.trainer.monitors.wandb = self.wandb.model_copy(deep=True)
            self.orchestrator.wandb = self.wandb.model_copy(deep=True)
        return self

    @model_validator(mode="after")
    def resolve_lora(self):
        lora = self.trainer.model.lora
        if lora is None:
            return self
        if self.trainer.weight_broadcast.type != "filesystem":
            raise ValueError("LoRA training requires filesystem weight broadcast.")
        if self.inference is not None:
            self.inference.vllm.enable_lora = True
            self.inference.vllm.max_lora_rank = lora.rank
        return self


class AdversarialRLConfig(BaseConfig):
    """Configures one synchronous generator/judge prototype run."""

    generator: AdversarialRoleConfig = Field(description="Generator RL run.")
    judge: AdversarialRoleConfig = Field(description="Judge classifier run.")
    output_dir: Path = Field(Path("outputs/adversarial"), description="Top-level output directory.")
    clean_output_dir: bool = Field(False, description="Clean nested output dirs before launch.")
    generator_reward_mode: Literal["judge", "exact_only", "judge_times_exact"] = Field(
        "judge", description="Use judge feedback, sparse exact-only rewards, or judge-gated exact rewards."
    )
    judge_objective: Literal["single_candidate", "pairwise"] = Field(
        "single_candidate",
        description=(
            "single_candidate scores generated and reference answers separately (two judge calls per rollout). "
            "pairwise shows the judge both in one prompt and picks the reference (one call per rollout, "
            "requires pairwise = true in the judge env args)."
        ),
    )
    resume: bool = Field(
        False, description="Resume both roles from the latest step where each has a trainer checkpoint."
    )
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
        return self

    def with_output_dir(self, run: AdversarialRoleConfig, output_dir: Path) -> AdversarialRoleConfig:
        run = run.model_copy(deep=True)
        run.output_dir = output_dir
        run.trainer.output_dir = output_dir
        run.orchestrator.output_dir = output_dir
        if self.clean_output_dir:
            run.clean = True
        return run
