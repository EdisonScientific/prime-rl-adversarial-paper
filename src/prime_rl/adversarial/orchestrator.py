import asyncio
import math
import time
from dataclasses import dataclass
from itertools import cycle
from statistics import pstdev
from typing import Any, Iterable

import tomli_w
import verifiers as vf

import prime_rl._compat  # noqa: F401
import prime_rl.monitors as monitors
from prime_rl.configs.adversarial import AdversarialRLConfig, AdversarialRoleConfig
from prime_rl.adversarial.advantage import compute_advantages
from prime_rl.adversarial.runtime_env import TrainEnv
from prime_rl.adversarial.orchestrator_utils import get_model_completion_len, save_rollouts, setup_student_inference_pool
from prime_rl.adversarial.rollout_utils import backfill_rollout_tokens, interleave_rollout
from prime_rl.orchestrator.packing import BatchPacker
from prime_rl.trainer.model import setup_tokenizer
from prime_rl.transport import TrainingSample, setup_micro_batch_sender
from prime_rl.utils.client import InferencePool, init_nccl_broadcast
from prime_rl.utils.config import cli
from prime_rl.utils.logger import setup_logger
from prime_rl.utils.pathing import get_broadcast_dir, get_log_dir, get_rollout_dir, get_step_path, wait_for_path
from prime_rl.utils.process import set_proc_title
from prime_rl.utils.utils import clean_exit


@dataclass
class RoleRuntime:
    name: str
    config: AdversarialRoleConfig
    env: TrainEnv
    tokenizer: Any
    renderer: Any
    inference_pool: InferencePool
    model_name: str
    client_cycle: Iterable[vf.ClientConfig]
    ckpt_step: int = 0
    train_step: int = 0

    async def next_client(self) -> vf.ClientConfig:
        return next(self.client_cycle)


@dataclass
class ScoredBatch:
    judge_rollouts: list[vf.RolloutOutput]
    generator_rewards: list[float]


@dataclass
class JudgeUpdateGateState:
    accuracy_ema: float | None = None
    skip_streak: int = 0


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def mean_rollout_metric(rollouts: list[vf.RolloutOutput], key: str) -> float | None:
    values = []
    for rollout in rollouts:
        value = rollout.get("metrics", {}).get(key)
        if value is not None:
            values.append(float(value))
    return mean(values) if values else None


def mean_optional_float(values: Iterable[float | None]) -> float:
    present = [float(value) for value in values if value is not None]
    return mean(present)


def finite_rollout_metric(rollout: vf.RolloutOutput, key: str) -> float | None:
    value = rollout.get("metrics", {}).get(key)
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def paired_rollout_metrics(rollouts: list[vf.RolloutOutput], x_key: str, y_key: str) -> tuple[list[float], list[float]]:
    xs: list[float] = []
    ys: list[float] = []
    for rollout in rollouts:
        x_value = finite_rollout_metric(rollout, x_key)
        y_value = finite_rollout_metric(rollout, y_key)
        if x_value is None or y_value is None:
            continue
        xs.append(x_value)
        ys.append(y_value)
    return xs, ys


def pearson_corr(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    x_mean = mean(xs)
    y_mean = mean(ys)
    x_centered = [x - x_mean for x in xs]
    y_centered = [y - y_mean for y in ys]
    x_norm = math.sqrt(sum(x * x for x in x_centered))
    y_norm = math.sqrt(sum(y * y for y in y_centered))
    if x_norm == 0.0 or y_norm == 0.0:
        return None
    return sum(x * y for x, y in zip(x_centered, y_centered)) / (x_norm * y_norm)


def rank_values(values: list[float]) -> list[float]:
    ranks = [0.0] * len(values)
    indexed_values = sorted(enumerate(values), key=lambda item: item[1])
    idx = 0
    while idx < len(indexed_values):
        next_idx = idx + 1
        while next_idx < len(indexed_values) and indexed_values[next_idx][1] == indexed_values[idx][1]:
            next_idx += 1
        rank = (idx + next_idx - 1) / 2.0
        for original_idx, _ in indexed_values[idx:next_idx]:
            ranks[original_idx] = rank
        idx = next_idx
    return ranks


def spearman_corr(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    return pearson_corr(rank_values(xs), rank_values(ys))


def logit_probability(probability: float) -> float:
    probability = min(max(probability, 1e-7), 1.0 - 1e-7)
    return math.log(probability / (1.0 - probability))


def judge_reward_from_probability(probability: float | None) -> float:
    if probability is None:
        return 0.0
    return probability


def mean_judge_probability_where(
    rollouts: list[vf.RolloutOutput],
    predicate: Any,
) -> float | None:
    values = []
    for rollout in rollouts:
        if not predicate(rollout):
            continue
        value = finite_rollout_metric(rollout, "judge_reference_probability_proxy")
        if value is not None:
            values.append(value)
    return mean(values) if values else None


def add_optional_metric(log_values: dict[str, float], key: str, value: float | None) -> None:
    if value is not None and math.isfinite(value):
        log_values[key] = value


def add_judge_alignment_diagnostics(log_values: dict[str, float], generator_rollouts: list[vf.RolloutOutput]) -> None:
    probability_key = "judge_reference_probability_proxy"
    diagnostic_targets = (
        "relative_error",
        "relative_error_clipped",
        "exact_reward",
        "format_reward",
        "valid_formula",
        "formula_l1_distance",
        "formula_l1_normalized",
        "formula_heavy_l1_distance",
        "formula_heavy_l1_normalized",
        "countdown_proxy_reward",
        "countdown_gold_reward",
        "countdown_reward_hacking_gap",
        "countdown_target_error",
        "countdown_target_error_clipped",
        "countdown_test_unchanged",
    )
    for target_key in diagnostic_targets:
        probabilities, targets = paired_rollout_metrics(generator_rollouts, probability_key, target_key)
        if not probabilities:
            continue
        logits = [logit_probability(probability) for probability in probabilities]
        add_optional_metric(
            log_values,
            f"adversarial/diagnostics/judge_probability_correlation/{target_key}",
            pearson_corr(probabilities, targets),
        )
        add_optional_metric(
            log_values,
            f"adversarial/diagnostics/judge_logit_correlation/{target_key}",
            pearson_corr(logits, targets),
        )
        add_optional_metric(
            log_values,
            f"adversarial/diagnostics/judge_probability_spearman/{target_key}",
            spearman_corr(probabilities, targets),
        )

    exact_probability = mean_judge_probability_where(
        generator_rollouts,
        lambda rollout: finite_rollout_metric(rollout, "exact_reward") == 1.0,
    )
    inexact_probability = mean_judge_probability_where(
        generator_rollouts,
        lambda rollout: finite_rollout_metric(rollout, "exact_reward") == 0.0,
    )
    format_ok_probability = mean_judge_probability_where(
        generator_rollouts,
        lambda rollout: finite_rollout_metric(rollout, "format_reward") == 1.0,
    )
    format_bad_probability = mean_judge_probability_where(
        generator_rollouts,
        lambda rollout: finite_rollout_metric(rollout, "format_reward") == 0.0,
    )
    low_error_probability = mean_judge_probability_where(
        generator_rollouts,
        lambda rollout: (
            (error := finite_rollout_metric(rollout, "relative_error_clipped")) is not None and error <= 0.01
        ),
    )
    medium_error_probability = mean_judge_probability_where(
        generator_rollouts,
        lambda rollout: (
            (error := finite_rollout_metric(rollout, "relative_error_clipped")) is not None and 0.01 < error <= 0.1
        ),
    )
    high_error_probability = mean_judge_probability_where(
        generator_rollouts,
        lambda rollout: (
            (error := finite_rollout_metric(rollout, "relative_error_clipped")) is not None and error > 0.1
        ),
    )

    add_optional_metric(log_values, "adversarial/diagnostics/judge_probability_mean/exact", exact_probability)
    add_optional_metric(log_values, "adversarial/diagnostics/judge_probability_mean/inexact", inexact_probability)
    add_optional_metric(log_values, "adversarial/diagnostics/judge_probability_mean/format_ok", format_ok_probability)
    add_optional_metric(log_values, "adversarial/diagnostics/judge_probability_mean/format_bad", format_bad_probability)
    add_optional_metric(log_values, "adversarial/diagnostics/judge_probability_mean/error_le_0.01", low_error_probability)
    add_optional_metric(
        log_values,
        "adversarial/diagnostics/judge_probability_mean/error_0.01_to_0.1",
        medium_error_probability,
    )
    add_optional_metric(log_values, "adversarial/diagnostics/judge_probability_mean/error_gt_0.1", high_error_probability)
    if exact_probability is not None and inexact_probability is not None:
        log_values["adversarial/diagnostics/judge_probability_gap/exact_minus_inexact"] = (
            exact_probability - inexact_probability
        )
    if format_ok_probability is not None and format_bad_probability is not None:
        log_values["adversarial/diagnostics/judge_probability_gap/format_ok_minus_bad"] = (
            format_ok_probability - format_bad_probability
        )
    if low_error_probability is not None and high_error_probability is not None:
        log_values["adversarial/diagnostics/judge_probability_gap/low_error_minus_high_error"] = (
            low_error_probability - high_error_probability
        )


def generator_result_reward(result: dict[str, Any], reward_mode: str) -> float:
    if reward_mode == "exact_only":
        return float(result["exact_reward"])
    return float(result["exact_reward"])


def write_orchestrator_control(run: AdversarialRoleConfig) -> None:
    control_dir = run.orchestrator.output_dir / "control"
    control_dir.mkdir(parents=True, exist_ok=True)
    with open(control_dir / "orch.toml", "wb") as f:
        tomli_w.dump(run.orchestrator.model_dump(exclude_none=True, mode="json"), f)


def get_max_steps(run: AdversarialRoleConfig) -> int:
    max_steps = run.max_steps or run.trainer.max_steps or run.orchestrator.max_steps
    if max_steps is None:
        raise ValueError("adversarial prototype requires max_steps.")
    return max_steps


def require_config(config: AdversarialRLConfig) -> None:
    gen = config.generator.orchestrator
    judge = config.judge.orchestrator
    if get_max_steps(config.generator) != get_max_steps(config.judge):
        raise ValueError("generator and judge max_steps must match for the synchronous prototype.")
    supported_weight_broadcasts = {"filesystem", "nccl"}
    if (
        config.generator.trainer.weight_broadcast.type not in supported_weight_broadcasts
        or config.judge.trainer.weight_broadcast.type not in supported_weight_broadcasts
    ):
        raise ValueError("the synchronous prototype supports filesystem or NCCL weight broadcast.")
    if gen.rollout_transport.type != "filesystem" or judge.rollout_transport.type != "filesystem":
        raise ValueError("the synchronous prototype currently supports only filesystem rollout transport.")
    if len(gen.train.env) != 1 or len(judge.train.env) != 1:
        raise ValueError("configure exactly one generator env and one judge env.")
    if gen.batch_size is None or judge.batch_size is None:
        raise ValueError("generator and judge batch_size are required.")
    if gen.batch_size % gen.group_size != 0:
        raise ValueError("generator batch_size must be divisible by group_size.")
    max_judge_batch_size = 2 * gen.batch_size
    if judge.batch_size > max_judge_batch_size:
        raise ValueError(
            f"judge batch_size must be at most {max_judge_batch_size} for single-candidate A/B training."
        )
    if judge.group_size != 1:
        raise ValueError("judge group_size must be 1 for classifier training.")
    if config.curriculum.enabled:
        raise ValueError("adversarial curriculum is not part of this minimal experiment port.")
    resume_steps = [
        run.trainer.ckpt.resume_step
        for run in (config.generator, config.judge)
        if run.trainer.ckpt is not None and run.trainer.ckpt.resume_step is not None
    ]
    if config.resume_step is None and resume_steps:
        raise ValueError("adversarial resume requires top-level resume_step so generator and judge stay in sync.")
    if config.resume_step is not None:
        if any(step != config.resume_step for step in resume_steps):
            raise ValueError("generator and judge trainer resume_step must match top-level adversarial resume_step.")
        if config.resume_step >= get_max_steps(config.generator):
            raise ValueError("adversarial resume_step must be lower than max_steps.")


async def setup_role(name: str, config: AdversarialRoleConfig) -> RoleRuntime:
    logger = setup_logger(config.log.level or "info", json_logging=config.log.json_logging)
    orch = config.orchestrator
    env = TrainEnv(orch.train.env[0], max_seq_len=orch.seq_len)
    await env.start(
        log_dir=get_log_dir(config.output_dir) / "envs" / "train",
        log_level=orch.log.vf_level,
        json_logging=orch.log.json_logging,
    )
    tokenizer = setup_tokenizer(orch.tokenizer)
    renderer, inference_pool = await setup_student_inference_pool(config=orch, tokenizer=tokenizer)
    rollout_model_name = orch.student.model.name
    logger.info(f"Waiting for {name} inference pool")
    await inference_pool.wait_for_ready(rollout_model_name)
    logger.success(f"{name} inference pool ready")
    if orch.weight_broadcast.type == "nccl":
        logger.info(f"Initializing {name} NCCL weight broadcast ({orch.weight_broadcast})")
        await init_nccl_broadcast(
            inference_pool.admin_clients,
            orch.weight_broadcast.host,
            orch.weight_broadcast.port,
            orch.weight_broadcast.timeout,
            inference_world_size=orch.weight_broadcast.inference_world_size,
            quantize_in_weight_transfer=orch.weight_broadcast.quantize_in_weight_transfer,
        )
    return RoleRuntime(
        name=name,
        config=config,
        env=env,
        tokenizer=tokenizer,
        renderer=renderer,
        inference_pool=inference_pool,
        model_name=rollout_model_name,
        client_cycle=cycle([inference_pool.train_client]),
    )


async def run_generator_group(
    runtime: RoleRuntime,
    example: dict[str, Any],
    group_size: int,
    cache_salt: str,
) -> list[vf.RolloutOutput]:
    if runtime.env.requires_group_scoring:
        outputs = await runtime.env.run_group(
            client=await runtime.next_client(),
            example=example,
            model_name=runtime.model_name,
            group_size=group_size,
            cache_salt=cache_salt,
        )
    else:
        example_id = example.get("example_id", "unknown")
        tasks = []
        for rollout_idx in range(group_size):
            tasks.append(
                runtime.env.run_rollout(
                    client=await runtime.next_client(),
                    example=example,
                    model_name=runtime.model_name,
                    cache_salt=f"{cache_salt}-{example_id}-{rollout_idx}",
                )
            )
        outputs = await asyncio.gather(*tasks)
    for output in outputs:
        output["env_name"] = runtime.env.name
    return outputs


async def run_generator_eval(
    generator: RoleRuntime,
    examples: list[dict[str, Any]],
    *,
    step: int,
    group_size: int,
    reward_mode: str,
) -> list[vf.RolloutOutput]:
    generator_groups = await asyncio.gather(
        *(
            run_generator_group(generator, example, group_size, cache_salt=f"eval-{generator.ckpt_step}")
            for example in examples
        )
    )
    eval_rollouts = []
    for example, rollouts in zip(examples, generator_groups):
        for rollout in rollouts:
            result = generator.env.env.adversarial_generator_result(rollout.get("completion"), example)
            rollout["reward"] = generator_result_reward(result, reward_mode)
            rollout.setdefault("metrics", {}).update(result["metrics"])
            rollout.setdefault("info", {})
            rollout["info"]["kind"] = "eval_generated"
            rollout["info"]["eval_step"] = step
            eval_rollouts.append(rollout)
    return eval_rollouts


async def run_judge_rollout(
    runtime: RoleRuntime,
    *,
    example_id: int,
    prompt: str,
    label: float,
    kind: str,
    cache_salt: str,
    info: dict[str, Any] | None = None,
) -> vf.RolloutOutput:
    answer = "A" if label >= 0.5 else "B"
    rollout_info = {"kind": kind, "label": label}
    if info:
        rollout_info.update(info)
    output = await runtime.env.run_rollout(
        client=await runtime.next_client(),
        example={
            "example_id": example_id,
            "prompt": [{"role": "user", "content": prompt}],
            "answer": answer,
            "info": rollout_info,
        },
        model_name=runtime.model_name,
        cache_salt=cache_salt,
    )
    output["env_name"] = runtime.env.name
    output.setdefault("metrics", {})["binary_classifier_label"] = label
    output["metrics"]["judge_reference_label"] = label
    output["metrics"]["judge_generated_sample"] = float(kind == "generated")
    output["metrics"]["judge_reference_sample"] = float(kind == "reference")
    return output


async def score_and_build_judge_batch(
    generator_env: Any,
    judge: RoleRuntime,
    *,
    step: int,
    generator_groups: list[list[vf.RolloutOutput]],
    examples: list[dict[str, Any]],
    reward_mode: str,
) -> ScoredBatch:
    judge_tasks: list[asyncio.Task[vf.RolloutOutput]] = []
    generated_task_positions: list[tuple[vf.RolloutOutput, dict[str, Any]]] = []
    generator_rewards: list[float] = []

    def assign_generator_reward(
        rollout: vf.RolloutOutput,
        result: dict[str, Any],
        reward: float,
        judge_reference_probability_proxy: float | None,
    ) -> None:
        rollout["reward"] = reward
        rollout.setdefault("metrics", {})
        metrics = dict(result["metrics"])
        metrics["judge_reference_probability_proxy"] = judge_reference_probability_proxy
        metrics["judge_reference_logit_proxy"] = (
            None if judge_reference_probability_proxy is None else logit_probability(judge_reference_probability_proxy)
        )
        rollout["metrics"].update(metrics)
        generator_rewards.append(reward)

    for group_idx, (example, rollouts) in enumerate(zip(examples, generator_groups)):
        question = generator_env.adversarial_question(example)
        reference = generator_env.adversarial_reference_answer(example)
        for rollout_idx, rollout in enumerate(rollouts):
            result = generator_env.adversarial_generator_result(rollout.get("completion"), example)
            if not result["format_ok"]:
                assign_generator_reward(
                    rollout,
                    result,
                    reward=0.0,
                    judge_reference_probability_proxy=None,
                )
                continue

            base_id = step * 1_000_000 + group_idx * 10_000 + rollout_idx * 2
            generated_task_positions.append((rollout, result))
            judge_tasks.append(
                asyncio.create_task(
                    run_judge_rollout(
                        judge,
                        example_id=base_id,
                        prompt=generator_env.adversarial_judge_prompt(question, result["candidate"]),
                        label=0.0,
                        kind="generated",
                        cache_salt=str(judge.ckpt_step),
                    )
                )
            )
            judge_tasks.append(
                asyncio.create_task(
                    run_judge_rollout(
                        judge,
                        example_id=base_id + 1,
                        prompt=generator_env.adversarial_judge_prompt(question, reference),
                        label=1.0,
                        kind="reference",
                        cache_salt=str(judge.ckpt_step),
                    )
                )
            )

    judge_rollouts = await asyncio.gather(*judge_tasks) if judge_tasks else []
    generated_judge_rollouts = judge_rollouts[0::2]
    judge_env = judge.env.env

    for (rollout, result), judge_rollout in zip(generated_task_positions, generated_judge_rollouts):
        judge_reference_probability_proxy = judge_env.adversarial_reference_probability_proxy(judge_rollout)
        if not result["format_ok"]:
            reward = 0.0
        elif reward_mode == "exact_only":
            reward = result["exact_reward"]
        elif reward_mode == "judge_times_exact":
            judge_reward = judge_reward_from_probability(judge_reference_probability_proxy)
            reward = judge_reward * result["exact_reward"]
        else:
            reward = judge_reward_from_probability(judge_reference_probability_proxy)
        assign_generator_reward(
            rollout,
            result,
            reward=reward,
            judge_reference_probability_proxy=judge_reference_probability_proxy,
        )

    return ScoredBatch(
        judge_rollouts=list(judge_rollouts),
        generator_rewards=generator_rewards,
    )


async def rollouts_to_samples(runtime: RoleRuntime, rollouts: list[vf.RolloutOutput]) -> list[TrainingSample]:
    needs_backfill = any(step["tokens"] is None for rollout in rollouts for step in rollout["trajectory"])
    if needs_backfill:
        await asyncio.gather(
            *(
                asyncio.to_thread(
                    backfill_rollout_tokens,
                    rollout,
                    runtime.tokenizer,
                    renderer=runtime.renderer,
                )
                for rollout in rollouts
            )
        )
    train_examples: list[TrainingSample] = []
    temperature = float(runtime.env.sampling_args.get("temperature", 1.0) or 1.0)
    for rollout in rollouts:
        samples = interleave_rollout(rollout)
        if samples is None:
            continue
        classifier_label = (rollout.get("metrics") or {}).get("binary_classifier_label")
        for sample in samples:
            if classifier_label is not None:
                target = "A" if float(classifier_label) >= 0.5 else "B"
                target_ids = runtime.tokenizer.encode(target, add_special_tokens=False)
                if not target_ids:
                    continue
                first_train_idx = next((idx for idx, is_train in enumerate(sample.mask) if is_train), len(sample.token_ids))
                prefix_ids = sample.token_ids[:first_train_idx]
                token_ids = prefix_ids + target_ids
                sample = TrainingSample(
                    token_ids=token_ids,
                    mask=[False] * len(prefix_ids) + [True] * len(target_ids),
                    logprobs=[0.0] * len(token_ids),
                    temperatures=[temperature] * len(token_ids),
                    env_name=sample.env_name,
                    rl_weights=[0.0] * len(token_ids),
                    ce_weights=[0.0] * len(prefix_ids) + [1.0] * len(target_ids),
                    advantages=[0.0] * len(token_ids),
                )
            else:
                advantage = float(rollout["advantage"])
                sample.temperatures = [temperature] * len(sample.token_ids)
                sample.advantages = [advantage if is_train else 0.0 for is_train in sample.mask]
            train_examples.append(sample)
    return train_examples


async def send_role_batch(runtime: RoleRuntime, rollouts: list[vf.RolloutOutput], group_size: int, step: int) -> None:
    if runtime.name == "judge":
        for rollout in rollouts:
            rollout["advantage"] = 0.0
    else:
        for start in range(0, len(rollouts), group_size):
            compute_advantages(
                rollouts[start : start + group_size],
                runtime.config.orchestrator.advantage,
                max_seq_len=runtime.config.orchestrator.seq_len,
            )
    examples = await rollouts_to_samples(runtime, rollouts)
    if not examples:
        raise RuntimeError(f"{runtime.name} produced no trainable examples at step {step}")
    micro_batch_grid = await asyncio.to_thread(BatchPacker(runtime.config.orchestrator).pack, examples)
    sender = setup_micro_batch_sender(
        runtime.config.orchestrator.output_dir,
        runtime.config.orchestrator.num_train_workers,
        step,
        runtime.config.orchestrator.rollout_transport,
    )
    await sender.send(micro_batch_grid)
    sender.close()


async def wait_and_update(runtime: RoleRuntime, next_step: int) -> None:
    weights_path = get_step_path(get_broadcast_dir(runtime.config.orchestrator.output_dir), next_step)
    await wait_for_path(weights_path / "STABLE", interval=1, log_interval=30)
    await runtime.inference_pool.update_weights(weights_path, step=next_step)
    runtime.ckpt_step = next_step
    runtime.train_step = next_step


async def initialize_from_resume(runtime: RoleRuntime, resume_step: int) -> None:
    await wait_and_update(runtime, resume_step)


def should_skip_judge_update(config: AdversarialRLConfig, state: JudgeUpdateGateState, step: int, accuracy: float) -> bool:
    gate = config.judge_update_gate
    if not gate.enabled:
        return False

    if state.accuracy_ema is None:
        state.accuracy_ema = accuracy
    else:
        state.accuracy_ema = gate.ema_decay * state.accuracy_ema + (1.0 - gate.ema_decay) * accuracy

    skip = step >= gate.warmup_steps and state.accuracy_ema >= gate.accuracy_threshold
    if gate.max_skip_streak is not None:
        skip = skip and state.skip_streak < gate.max_skip_streak
    if skip:
        state.skip_streak += 1
    else:
        state.skip_streak = 0
    return skip


@clean_exit
async def adversarial_orchestrate(config: AdversarialRLConfig) -> None:
    require_config(config)
    logger = setup_logger(config.log.level or "info", json_logging=config.log.json_logging)
    logger.info("Starting synchronous adversarial prototype")
    write_orchestrator_control(config.generator)
    write_orchestrator_control(config.judge)
    await monitors.setup(
        wandb=config.generator.orchestrator.wandb,
        prime=config.generator.orchestrator.prime_monitor,
        output_dir=config.output_dir,
        run_config=config,
    )

    generator: RoleRuntime | None = None
    judge: RoleRuntime | None = None
    try:
        generator = await setup_role("generator", config.generator)
        judge = await setup_role("judge", config.judge)
        start_step = config.resume_step or 0
        if start_step:
            logger.info(f"Resuming generator and judge inference from step {start_step}")
            await asyncio.gather(
                initialize_from_resume(generator, start_step),
                initialize_from_resume(judge, start_step),
            )
        examples = list(generator.env.get_dataset(seed=config.generator.orchestrator.buffer.seed))
        if not examples:
            raise ValueError("Generator dataset is empty")
        eval_config = config.generator.orchestrator.eval
        eval_examples: list[dict[str, Any]] = []
        if eval_config is not None:
            eval_examples = generator.env.env.get_eval_dataset(n=eval_config.num_examples).to_list()
            if not eval_examples:
                raise ValueError("Generator eval dataset is empty")
            logger.info(
                f"Adversarial eval ready: interval={eval_config.interval}, "
                f"num_examples={len(eval_examples)}, group_size={eval_config.group_size}"
            )

        judge_gate_state = JudgeUpdateGateState()
        group_size = config.generator.orchestrator.group_size
        batch_size = config.generator.orchestrator.batch_size
        assert batch_size is not None
        groups_per_step = batch_size // group_size
        logger.info(
            f"Loop ready: max_steps={config.generator.max_steps}, batch_size={batch_size}, group_size={group_size}"
        )
        if eval_config is not None and start_step == 0:
            logger.info("Running pre-training adversarial eval at step 0")
            eval_rollouts = await run_generator_eval(
                generator,
                eval_examples,
                step=0,
                group_size=eval_config.group_size,
                reward_mode=config.generator_reward_mode,
            )
            eval_step_path = get_step_path(get_rollout_dir(config.output_dir), 0)
            await asyncio.to_thread(
                save_rollouts,
                eval_rollouts,
                eval_step_path / "eval_generator_rollouts.jsonl",
                exclude_keys={"trajectory"},
            )
            eval_log_values = {
                "adversarial/eval/generator/reward": mean([float(rollout["reward"]) for rollout in eval_rollouts]),
                "adversarial/eval/generator/rollouts": len(eval_rollouts),
            }
            for metric_name in (
                "format_reward",
                "exact_reward",
                "relative_error",
                "relative_error_clipped",
                "countdown_proxy_reward",
                "countdown_gold_reward",
                "countdown_reward_hacking_gap",
                "countdown_expr_present",
                "countdown_uses_required_numbers",
                "countdown_allowed_chars",
                "countdown_target_error",
                "countdown_target_error_clipped",
                "countdown_test_unchanged",
                "countdown_solution_changed",
            ):
                value = mean_rollout_metric(eval_rollouts, metric_name)
                if value is not None:
                    eval_log_values[f"adversarial/eval/generator/{metric_name}"] = value
            await monitors.log(eval_log_values, step=0)
            logger.success(
                f"Step 0 eval | Eval reward: {eval_log_values['adversarial/eval/generator/reward']:.4f}"
            )

        for step in range(start_step, get_max_steps(config.generator)):
            step_start = time.perf_counter()
            offset = step * groups_per_step
            batch_examples = [dict(examples[(offset + idx) % len(examples)]) for idx in range(groups_per_step)]
            generator_groups = await asyncio.gather(
                *(
                    run_generator_group(generator, example, group_size, cache_salt=str(generator.ckpt_step))
                    for example in batch_examples
                )
            )
            generator_rollouts = [rollout for group in generator_groups for rollout in group]
            scored = await score_and_build_judge_batch(
                generator.env.env,
                judge,
                step=step,
                generator_groups=generator_groups,
                examples=batch_examples,
                reward_mode=config.generator_reward_mode,
            )

            generator_metrics = {
                key: mean_rollout_metric(generator_rollouts, key)
                for key in (
                    "format_reward",
                    "exact_reward",
                    "relative_error",
                    "relative_error_clipped",
                    "valid_formula",
                    "formula_l1_distance",
                    "formula_heavy_l1_distance",
                    "formula_l1_normalized",
                    "formula_heavy_l1_normalized",
                    "formula_target_atom_count",
                    "formula_target_heavy_atom_count",
                    "abstract_word_count",
                    "abstract_reference_word_count",
                    "abstract_word_count_error",
                    "abstract_word_count_relative_error",
                    "abstract_char_length",
                    "abstract_reference_char_length",
                    "abstract_char_length_error",
                    "abstract_char_length_relative_error",
                    "countdown_proxy_reward",
                    "countdown_gold_reward",
                    "countdown_reward_hacking_gap",
                    "countdown_numbers_count",
                    "countdown_expr_present",
                    "countdown_uses_required_numbers",
                    "countdown_allowed_chars",
                    "countdown_target_error",
                    "countdown_target_error_clipped",
                    "countdown_test_unchanged",
                    "countdown_solution_changed",
                    "judge_reference_probability_proxy",
                    "judge_reference_logit_proxy",
                )
            }
            judge_rewards = [float(r["reward"]) for r in scored.judge_rollouts]
            generated_judge_rollouts = [
                r for r in scored.judge_rollouts if r.get("info", {}).get("kind") == "generated"
            ]
            reference_judge_rollouts = [
                r for r in scored.judge_rollouts if r.get("info", {}).get("kind") == "reference"
            ]
            generated_judge_rewards = [float(r["reward"]) for r in generated_judge_rollouts]
            reference_judge_rewards = [float(r["reward"]) for r in reference_judge_rollouts]
            judge_accuracy = mean(judge_rewards)
            skip_judge_update = not scored.judge_rollouts or should_skip_judge_update(
                config,
                judge_gate_state,
                step - start_step,
                judge_accuracy,
            )

            send_tasks = [send_role_batch(generator, generator_rollouts, group_size, generator.train_step)]
            if not skip_judge_update:
                send_tasks.append(send_role_batch(judge, scored.judge_rollouts, 1, judge.train_step))
            await asyncio.gather(*send_tasks)
            step_path = get_step_path(get_rollout_dir(config.output_dir), step)
            await asyncio.gather(
                asyncio.to_thread(save_rollouts, generator_rollouts, step_path / "generator_rollouts.jsonl"),
                asyncio.to_thread(save_rollouts, scored.judge_rollouts, step_path / "judge_rollouts.jsonl"),
            )
            generator_next_step = generator.train_step + 1
            update_tasks = [wait_and_update(generator, generator_next_step)]
            if not skip_judge_update:
                update_tasks.append(wait_and_update(judge, judge.train_step + 1))
            await asyncio.gather(*update_tasks)

            generated_reference_probability_proxies = [
                judge.env.env.adversarial_reference_probability_proxy(r) for r in generated_judge_rollouts
            ]
            reference_reference_probability_proxies = [
                judge.env.env.adversarial_reference_probability_proxy(r) for r in reference_judge_rollouts
            ]
            judge_proxy_losses = [
                -math.log(max(1.0 - probability, 1e-7))
                for probability in generated_reference_probability_proxies
                if probability is not None
            ] + [
                -math.log(max(probability, 1e-7))
                for probability in reference_reference_probability_proxies
                if probability is not None
            ]
            judge_a_rate = mean(
                [float(judge.env.env.adversarial_choice(r.get("completion")) == "A") for r in scored.judge_rollouts]
            )
            generator_completion_lens = [float(get_model_completion_len(r)) for r in generator_rollouts]
            judge_completion_lens = [float(get_model_completion_len(r)) for r in scored.judge_rollouts]
            log_values = {
                "adversarial/generator/reward": mean(scored.generator_rewards),
                "adversarial/generator/reward_std": (
                    pstdev(scored.generator_rewards) if scored.generator_rewards else 0.0
                ),
                "adversarial/generator/format_accuracy": generator_metrics["format_reward"] or 0.0,
                "adversarial/generator/exact_reward": generator_metrics["exact_reward"] or 0.0,
                "adversarial/generator/relative_error": generator_metrics["relative_error"] or 0.0,
                "adversarial/generator/relative_error_clipped": generator_metrics["relative_error_clipped"] or 0.0,
                "adversarial/generator/judge_reference_probability_proxy": (
                    generator_metrics["judge_reference_probability_proxy"] or 0.0
                ),
                "adversarial/generator/judge_reference_logit_proxy": (
                    generator_metrics["judge_reference_logit_proxy"] or 0.0
                ),
                "adversarial/judge/accuracy": mean(judge_rewards),
                "adversarial/judge/generated_accuracy": mean(generated_judge_rewards),
                "adversarial/judge/reference_accuracy": mean(reference_judge_rewards),
                "adversarial/judge/reference_probability_proxy/generated": mean_optional_float(
                    generated_reference_probability_proxies
                ),
                "adversarial/judge/reference_probability_proxy/reference": mean_optional_float(
                    reference_reference_probability_proxies
                ),
                "adversarial/judge/proxy_loss": mean(judge_proxy_losses),
                "adversarial/judge/predicted_reference_rate": judge_a_rate,
                "adversarial/generator/completion_tokens/mean": mean(generator_completion_lens),
                "adversarial/generator/completion_tokens/min": min(generator_completion_lens, default=0.0),
                "adversarial/generator/completion_tokens/max": max(generator_completion_lens, default=0.0),
                "adversarial/judge/completion_tokens/mean": mean(judge_completion_lens),
                "adversarial/judge/completion_tokens/min": min(judge_completion_lens, default=0.0),
                "adversarial/judge/completion_tokens/max": max(judge_completion_lens, default=0.0),
                "adversarial/generator/rollouts": len(generator_rollouts),
                "adversarial/generator/trainer_step": generator.train_step,
                "adversarial/judge/scored_rollouts": len(scored.judge_rollouts),
                "adversarial/judge/train_rollouts": 0 if skip_judge_update else len(scored.judge_rollouts),
                "adversarial/judge/trainer_step": judge.train_step,
                "adversarial/judge/update_skipped": float(skip_judge_update),
                "adversarial/judge/update_skipped_no_rollouts": float(not scored.judge_rollouts),
                "adversarial/judge/update_skip_streak": judge_gate_state.skip_streak,
                "adversarial/step_time_seconds": time.perf_counter() - step_start,
            }
            if judge_gate_state.accuracy_ema is not None:
                log_values["adversarial/judge/accuracy_ema"] = judge_gate_state.accuracy_ema
            optional_generator_metrics = {
                "valid_formula": "adversarial/generator/valid_formula_rate",
                "formula_l1_distance": "adversarial/generator/formula_l1_distance",
                "formula_heavy_l1_distance": "adversarial/generator/formula_heavy_l1_distance",
                "formula_l1_normalized": "adversarial/generator/formula_l1_normalized",
                "formula_heavy_l1_normalized": "adversarial/generator/formula_heavy_l1_normalized",
                "formula_target_atom_count": "adversarial/generator/formula_target_atom_count",
                "formula_target_heavy_atom_count": "adversarial/generator/formula_target_heavy_atom_count",
                "abstract_word_count": "adversarial/generator/abstract_word_count",
                "abstract_reference_word_count": "adversarial/generator/abstract_reference_word_count",
                "abstract_word_count_error": "adversarial/generator/abstract_word_count_error",
                "abstract_word_count_relative_error": "adversarial/generator/abstract_word_count_relative_error",
                "abstract_char_length": "adversarial/generator/abstract_char_length",
                "abstract_reference_char_length": "adversarial/generator/abstract_reference_char_length",
                "abstract_char_length_error": "adversarial/generator/abstract_char_length_error",
                "abstract_char_length_relative_error": "adversarial/generator/abstract_char_length_relative_error",
                "countdown_proxy_reward": "adversarial/generator/countdown_proxy_reward",
                "countdown_gold_reward": "adversarial/generator/countdown_gold_reward",
                "countdown_reward_hacking_gap": "adversarial/generator/countdown_reward_hacking_gap",
                "countdown_numbers_count": "adversarial/generator/countdown_numbers_count",
                "countdown_expr_present": "adversarial/generator/countdown_expr_present",
                "countdown_uses_required_numbers": "adversarial/generator/countdown_uses_required_numbers",
                "countdown_allowed_chars": "adversarial/generator/countdown_allowed_chars",
                "countdown_target_error": "adversarial/generator/countdown_target_error",
                "countdown_target_error_clipped": "adversarial/generator/countdown_target_error_clipped",
                "countdown_test_unchanged": "adversarial/generator/countdown_test_unchanged",
                "countdown_solution_changed": "adversarial/generator/countdown_solution_changed",
            }
            for metric_name, log_name in optional_generator_metrics.items():
                value = generator_metrics.get(metric_name)
                if value is not None:
                    log_values[log_name] = value
            if eval_config is not None and (step + 1) % eval_config.interval == 0:
                eval_rollouts = await run_generator_eval(
                    generator,
                    eval_examples,
                    step=step + 1,
                    group_size=eval_config.group_size,
                    reward_mode=config.generator_reward_mode,
                )
                await asyncio.to_thread(
                    save_rollouts,
                    eval_rollouts,
                    step_path / "eval_generator_rollouts.jsonl",
                    exclude_keys={"trajectory"},
                )
                log_values["adversarial/eval/generator/reward"] = mean(
                    [float(rollout["reward"]) for rollout in eval_rollouts]
                )
                log_values["adversarial/eval/generator/rollouts"] = len(eval_rollouts)
                for metric_name in ("format_reward", "exact_reward", "relative_error", "relative_error_clipped"):
                    value = mean_rollout_metric(eval_rollouts, metric_name)
                    if value is not None:
                        log_values[f"adversarial/eval/generator/{metric_name}"] = value
                for metric_name, log_name in optional_generator_metrics.items():
                    value = mean_rollout_metric(eval_rollouts, metric_name)
                    if value is not None:
                        log_values[log_name.replace("adversarial/generator/", "adversarial/eval/generator/")] = value
            add_judge_alignment_diagnostics(log_values, generator_rollouts)
            await monitors.log(log_values, step=step)
            logger.success(
                f"Step {step} | Time: {time.perf_counter() - step_start:.2f}s | "
                f"Gen reward: {log_values['adversarial/generator/reward']:.4f} | "
                f"Gen exact: {log_values['adversarial/generator/exact_reward']:.4f} | "
                f"Judge acc: {log_values['adversarial/judge/accuracy']:.4f} | "
                f"Judge skip: {int(skip_judge_update)}"
            )
    finally:
        await monitors.finalize()
        if generator is not None:
            generator.env.shutdown()
            await generator.inference_pool.stop()
        if judge is not None:
            judge.env.shutdown()
            await judge.inference_pool.stop()

    logger.success("Adversarial orchestrator finished")


def main() -> None:
    set_proc_title("AdversarialOrchestrator")
    asyncio.run(adversarial_orchestrate(cli(AdversarialRLConfig)))


if __name__ == "__main__":
    main()
