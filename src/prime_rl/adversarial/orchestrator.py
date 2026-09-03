import asyncio
import time
from dataclasses import dataclass
from statistics import correlation, fmean
from typing import Any

import verifiers as vf
from renderers.base import create_renderer

import prime_rl._compat  # noqa: F401
import prime_rl.monitors as monitors
from prime_rl.adversarial.utils import (
    backfill_rollout_tokens,
    get_model_completion_len,
    latest_common_ckpt_step,
    rollout_to_sample,
    save_rollouts,
)
from prime_rl.configs.adversarial import AdversarialRLConfig, AdversarialRoleConfig
from prime_rl.orchestrator.clients import AdminClients
from prime_rl.orchestrator.packing import BatchPacker
from prime_rl.trainer.model import setup_tokenizer
from prime_rl.transports.batch import setup_batch_sender
from prime_rl.transports.batch.types import TrainingSample
from prime_rl.transports.weights import setup_weight_receiver
from prime_rl.transports.weights.base import WeightReceiver, write_ema_clock
from prime_rl.utils.config import cli
from prime_rl.utils.logger import setup_logger
from prime_rl.utils.pathing import get_broadcast_dir, get_rollout_dir, get_step_path
from prime_rl.utils.process import set_proc_title
from prime_rl.utils.utils import clean_exit

STATE_COLUMNS = ["trajectory"]


@dataclass
class RoleRuntime:
    name: str
    config: AdversarialRoleConfig
    env: vf.Environment
    env_name: str
    sampling_args: dict[str, Any]
    max_retries: int
    tokenizer: Any
    renderer: Any
    admin: AdminClients
    receiver: WeightReceiver
    model_name: str
    train_client: vf.ClientConfig
    ckpt_step: int = 0
    train_step: int = 0

    def sampling_args_with_salt(self, cache_salt: str) -> dict[str, Any]:
        return {
            **self.sampling_args,
            "extra_body": {**self.sampling_args.get("extra_body", {}), "cache_salt": cache_salt},
        }


@dataclass
class ScoredBatch:
    generator_groups: list[list[vf.RolloutOutput]]
    judge_rollouts: list[vf.RolloutOutput]
    generator_rewards: list[float]
    generation_seconds: float
    judge_tail_seconds: float
    """Judge scoring time left over after the last generation finished."""


@dataclass
class JudgeUpdateGateState:
    accuracy_ema: float | None = None
    skip_streak: int = 0


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
    if gen.batch_size is None or judge.batch_size is None:
        raise ValueError("generator and judge batch_size are required.")
    if gen.batch_size % gen.group_size != 0:
        raise ValueError("generator batch_size must be divisible by group_size.")
    pairwise = config.judge_objective == "pairwise"
    max_judge_batch_size = gen.batch_size if pairwise else 2 * gen.batch_size
    if judge.batch_size > max_judge_batch_size:
        raise ValueError(
            f"judge batch_size must be at most {max_judge_batch_size} for judge_objective={config.judge_objective}."
        )
    if bool(judge.train.env.args.get("pairwise")) != pairwise:
        raise ValueError("the judge env args must set pairwise = true exactly when judge_objective is pairwise.")
    if judge.group_size != 1:
        raise ValueError("judge group_size must be 1 for classifier training.")


def legacy_train_client(orch) -> vf.ClientConfig:
    """Rollout client for the legacy verifiers envs. The upstream ``setup_client``
    builds verifiers.v1 client configs, which legacy ``run_rollout`` rejects."""
    client = orch.client
    return vf.ClientConfig(
        client_type="renderer",
        api_base_url=client.base_url,
        api_key_var=client.api_key_var,
        max_connections=8192,
        max_keepalive_connections=8192,
        # Sticky DP-aware routing at the inference router.
        extra_headers_from_state={"X-Session-ID": "trajectory_id"},
        renderer_config=orch.renderer,
        renderer_model_name=orch.model.name,
    )


async def setup_role(name: str, config: AdversarialRoleConfig) -> RoleRuntime:
    logger = setup_logger(config.log.level or "info", json_logging=config.log.json_logging)
    orch = config.orchestrator
    env_config = orch.train.env
    env = vf.load_environment(env_config.id, **env_config.args)
    tokenizer = setup_tokenizer(orch.tokenizer)
    rollout_model_name = orch.model.name
    renderer = create_renderer(tokenizer, orch.renderer)
    admin = AdminClients(orch.client)
    logger.info(f"Waiting for {name} inference pool")
    await admin.wait_for_ready(rollout_model_name)
    logger.success(f"{name} inference pool ready")
    logger.info(f"Initializing {name} weight broadcast ({orch.weight_broadcast})")
    receiver = setup_weight_receiver(
        get_broadcast_dir(orch.output_dir),
        orch.weight_broadcast,
        admin_clients=admin.clients,
        model_name=rollout_model_name,
    )
    await receiver.initialize()
    return RoleRuntime(
        name=name,
        config=config,
        env=env,
        env_name=env_config.resolved_name,
        sampling_args=orch.train.sampling.to_sampling_args(),
        max_retries=orch.train.max_retries,
        tokenizer=tokenizer,
        renderer=renderer,
        model_name=rollout_model_name,
        train_client=legacy_train_client(orch),
        admin=admin,
        receiver=receiver,
    )


async def run_generator_group(
    runtime: RoleRuntime,
    example: dict[str, Any],
    group_size: int,
    cache_salt: str,
) -> list[vf.RolloutOutput]:
    outputs = await runtime.env.run_group(
        [vf.RolloutInput(**example) for _ in range(group_size)],
        client=runtime.train_client,
        model=runtime.model_name,
        sampling_args=runtime.sampling_args_with_salt(cache_salt),
        max_retries=runtime.max_retries,
        state_columns=STATE_COLUMNS,
    )
    for output in outputs:
        output["env_name"] = runtime.env_name
    return outputs


async def run_generator_eval(
    generator: RoleRuntime,
    examples: list[dict[str, Any]],
    *,
    step: int,
    group_size: int,
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
            result = generator.env.adversarial_generator_result(rollout.get("completion"), example)
            # Free-generation envs have no programmatic check; their eval tracks format compliance.
            rollout["reward"] = float(result.get("verifiable_reward", result["metrics"]["format_reward"]))
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
        vf.RolloutInput(
            example_id=example_id,
            prompt=[{"role": "user", "content": prompt}],
            answer=answer,
            info=rollout_info,
        ),
        client=runtime.train_client,
        model=runtime.model_name,
        sampling_args=runtime.sampling_args_with_salt(cache_salt),
        max_retries=runtime.max_retries,
        state_columns=STATE_COLUMNS,
    )
    output["env_name"] = runtime.env_name
    output.setdefault("metrics", {})["binary_classifier_label"] = label
    output["metrics"]["judge_reference_label"] = label
    output["metrics"]["judge_generated_sample"] = float(kind == "generated")
    output["metrics"]["judge_reference_sample"] = float(kind == "reference")
    return output


def single_candidate_judge_tasks(
    env: Any, question: str, reference: str, candidate: str
) -> list[tuple[str, float, str]]:
    """One judge call per answer as ``(prompt, label, kind)``. Label 1.0 means the correct choice is A."""
    return [
        (env.adversarial_judge_prompt(question, candidate), 0.0, "generated"),
        (env.adversarial_judge_prompt(question, reference), 1.0, "reference"),
    ]


def pairwise_judge_tasks(
    env: Any, question: str, reference: str, candidate: str, *, reference_first: bool
) -> list[tuple[str, float, str]]:
    """One judge call showing both answers. The correct choice is the reference's slot."""
    first, second = (reference, candidate) if reference_first else (candidate, reference)
    return [(env.adversarial_pairwise_judge_prompt(question, first, second), float(reference_first), "pairwise")]


def generated_win_probability(slot_a_probability: float | None, first_label: float) -> float | None:
    """P(the judge deems the generated answer real). In both objectives the generated answer
    occupies slot A exactly when the rollout's first judge task carries label 0."""
    if slot_a_probability is None:
        return None
    return slot_a_probability if first_label == 0.0 else 1.0 - slot_a_probability


def judge_verifiable_correlation(rollouts: list[vf.RolloutOutput]) -> float | None:
    """Pearson correlation between the judge probability and the env's verifiable
    distance (each env defines its own, e.g. L1 atom-count distance for formulas),
    over the rollouts carrying both. An aligned judge is anticorrelated with the
    distance. None when the env provides no distance or either value is constant
    across the batch."""
    pairs = [
        (metrics["judge_reference_probability_proxy"], float(metrics["verifiable_distance"]))
        for rollout in rollouts
        if (metrics := rollout["metrics"]).get("judge_reference_probability_proxy") is not None
        and metrics.get("verifiable_distance") is not None
    ]
    if len(pairs) < 2:
        return None
    probabilities, distances = zip(*pairs)
    if len(set(probabilities)) < 2 or len(set(distances)) < 2:
        return None
    return correlation(probabilities, distances)


async def score_and_build_judge_batch(
    generator_env: Any,
    judge: RoleRuntime,
    *,
    step: int,
    generator_group_tasks: list[asyncio.Task[list[vf.RolloutOutput]]],
    examples: list[dict[str, Any]],
    reward_mode: str,
    judge_objective: str,
) -> ScoredBatch:
    """Judging is pipelined into generation: each group's judge calls dispatch as
    soon as that group finishes generating, so judge prefill overlaps the still
    running generation instead of serializing after it."""
    phase_start = time.perf_counter()
    generator_groups: list[list[vf.RolloutOutput]] = []
    judge_tasks: list[asyncio.Task[vf.RolloutOutput]] = []
    scored_positions: list[tuple[vf.RolloutOutput, dict[str, Any], float, int]] = []
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
        rollout["metrics"].update(metrics)
        generator_rewards.append(reward)

    for group_idx, (example, group_task) in enumerate(zip(examples, generator_group_tasks)):
        rollouts = await group_task
        generator_groups.append(rollouts)
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

            if judge_objective == "pairwise":
                # Alternate the reference's slot deterministically to avoid position bias.
                tasks = pairwise_judge_tasks(
                    generator_env,
                    question,
                    reference,
                    result["candidate"],
                    reference_first=(step + group_idx + rollout_idx) % 2 == 0,
                )
            else:
                tasks = single_candidate_judge_tasks(generator_env, question, reference, result["candidate"])
            scored_positions.append((rollout, result, tasks[0][1], len(judge_tasks)))
            # Unique judge example ids across steps/groups/rollouts.
            base_id = step * 1_000_000 + group_idx * 10_000 + rollout_idx * 2
            for offset, (prompt, label, kind) in enumerate(tasks):
                judge_tasks.append(
                    asyncio.create_task(
                        run_judge_rollout(
                            judge,
                            example_id=base_id + offset,
                            prompt=prompt,
                            label=label,
                            kind=kind,
                            cache_salt=str(judge.ckpt_step),
                        )
                    )
                )

    generation_seconds = time.perf_counter() - phase_start
    judge_rollouts = await asyncio.gather(*judge_tasks) if judge_tasks else []
    judge_tail_seconds = time.perf_counter() - phase_start - generation_seconds
    judge_env = judge.env

    for rollout, result, first_label, task_idx in scored_positions:
        slot_a_probability = judge_env.adversarial_reference_probability_proxy(judge_rollouts[task_idx])
        p_generated = generated_win_probability(slot_a_probability, first_label)
        if not result["format_ok"]:
            reward = 0.0
        elif reward_mode == "exact_only":
            reward = result["verifiable_reward"]
        elif reward_mode == "judge_times_exact":
            reward = (p_generated or 0.0) * result["verifiable_reward"]
        else:
            reward = p_generated or 0.0
        assign_generator_reward(
            rollout,
            result,
            reward=reward,
            judge_reference_probability_proxy=p_generated,
        )

    return ScoredBatch(
        generator_groups=generator_groups,
        judge_rollouts=list(judge_rollouts),
        generator_rewards=generator_rewards,
        generation_seconds=generation_seconds,
        judge_tail_seconds=judge_tail_seconds,
    )


async def rollouts_to_samples(runtime: RoleRuntime, rollouts: list[vf.RolloutOutput]) -> list[TrainingSample]:
    needs_backfill = any(step["tokens"] is None for rollout in rollouts for step in rollout["trajectory"])
    if needs_backfill:
        await asyncio.gather(
            *(asyncio.to_thread(backfill_rollout_tokens, rollout, runtime.renderer) for rollout in rollouts)
        )
    train_examples: list[TrainingSample] = []
    temperature = float(runtime.sampling_args.get("temperature", 1.0) or 1.0)
    for rollout in rollouts:
        sample = rollout_to_sample(rollout)
        if sample is None:
            continue
        classifier_label = (rollout.get("metrics") or {}).get("binary_classifier_label")
        if classifier_label is not None:
            target = "A" if float(classifier_label) >= 0.5 else "B"
            target_ids = runtime.tokenizer.encode(target, add_special_tokens=False)
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
        # Plain GRPO: reward minus the group-mean baseline.
        for start in range(0, len(rollouts), group_size):
            group = rollouts[start : start + group_size]
            baseline = sum(float(r["reward"]) for r in group) / len(group)
            for rollout in group:
                rollout["advantage"] = float(rollout["reward"]) - baseline
    examples = await rollouts_to_samples(runtime, rollouts)
    if not examples:
        raise RuntimeError(f"{runtime.name} produced no trainable examples at step {step}")
    micro_batch_grid = await asyncio.to_thread(BatchPacker(runtime.config.orchestrator).pack, examples)
    sender = setup_batch_sender(
        runtime.config.orchestrator.output_dir,
        runtime.config.orchestrator.num_train_workers,
        step,
        runtime.config.orchestrator.rollout_transport,
    )
    await sender.send(micro_batch_grid)
    sender.close()


async def wait_and_update(runtime: RoleRuntime, next_step: int) -> None:
    await runtime.receiver.wait_published(next_step)
    await runtime.receiver.receive(next_step)
    runtime.ckpt_step = next_step
    runtime.train_step = next_step


def should_skip_judge_update(
    config: AdversarialRLConfig, state: JudgeUpdateGateState, step: int, accuracy: float
) -> bool:
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
    await monitors.setup(
        wandb=config.generator.orchestrator.wandb,
        output_dir=config.output_dir,
        run_config=config,
    )

    generator: RoleRuntime | None = None
    judge: RoleRuntime | None = None
    try:
        generator = await setup_role("generator", config.generator)
        judge = await setup_role("judge", config.judge)
        start_step = 0
        if config.resume:
            start_step = latest_common_ckpt_step(config.generator.output_dir, config.judge.output_dir)
            logger.info(f"Resuming generator and judge from checkpoint step {start_step}")
        # The trainers offer their startup weights (v{start_step}) and block
        # until each role's receiver acknowledges and joins the transfer.
        await asyncio.gather(
            wait_and_update(generator, start_step),
            wait_and_update(judge, start_step),
        )
        examples = list(generator.env.get_dataset(seed=config.generator.orchestrator.seed))
        if not examples:
            raise ValueError("Generator dataset is empty")
        eval_config = config.generator.orchestrator.eval
        eval_examples: list[dict[str, Any]] = []
        if eval_config is not None:
            eval_examples = generator.env.get_eval_dataset(n=eval_config.num_examples).to_list()
            if not eval_examples:
                raise ValueError("Generator eval dataset is empty")
            logger.info(
                f"Adversarial eval ready: interval={eval_config.interval}, "
                f"num_examples={len(eval_examples)}, group_size={eval_config.group_size}"
            )

        judge_gate_state = JudgeUpdateGateState()
        group_size = config.generator.orchestrator.group_size
        batch_size = config.generator.orchestrator.batch_size
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
            )
            eval_step_path = get_step_path(get_rollout_dir(config.output_dir), 0)
            await asyncio.to_thread(
                save_rollouts,
                eval_rollouts,
                eval_step_path / "eval_generator_rollouts.jsonl",
                exclude_keys={"trajectory"},
            )
            eval_log_values = {
                "eval/generator/reward/mean": fmean(float(rollout["reward"]) for rollout in eval_rollouts),
            }
            if "verifiable_reward" in eval_rollouts[0]["metrics"]:
                eval_log_values["eval/generator/metrics/verifiable_reward/mean"] = fmean(
                    float(r["metrics"]["verifiable_reward"]) for r in eval_rollouts
                )
            await monitors.log(eval_log_values, step=0)
            logger.success(f"Step 0 eval | Eval reward: {eval_log_values['eval/generator/reward/mean']:.4f}")

        pending_judge_update: asyncio.Task[None] | None = None
        for step in range(start_step, get_max_steps(config.generator)):
            step_start = time.perf_counter()
            offset = step * groups_per_step
            batch_examples = [dict(examples[(offset + idx) % len(examples)]) for idx in range(groups_per_step)]
            generator_group_tasks = [
                asyncio.create_task(
                    run_generator_group(generator, example, group_size, cache_salt=str(generator.ckpt_step))
                )
                for example in batch_examples
            ]
            # Generation only needs the generator weights, so the judge's slow
            # broadcast from the previous step overlaps it; scoring is what
            # requires the updated judge, and it starts after this await.
            if pending_judge_update is not None:
                await pending_judge_update
                pending_judge_update = None
            scored = await score_and_build_judge_batch(
                generator.env,
                judge,
                step=step,
                generator_group_tasks=generator_group_tasks,
                examples=batch_examples,
                reward_mode=config.generator_reward_mode,
                judge_objective=config.judge_objective,
            )
            generator_rollouts = [rollout for group in scored.generator_groups for rollout in group]

            generator_metrics: dict[str, list[float]] = {}
            for rollout in generator_rollouts:
                for key, value in rollout["metrics"].items():
                    if value is not None:
                        generator_metrics.setdefault(key, []).append(float(value))
            judge_rewards = [float(r["reward"]) for r in scored.judge_rollouts]
            generated_judge_rollouts = [
                r for r in scored.judge_rollouts if r.get("info", {}).get("kind") == "generated"
            ]
            reference_judge_rollouts = [
                r for r in scored.judge_rollouts if r.get("info", {}).get("kind") == "reference"
            ]
            generated_judge_rewards = [float(r["reward"]) for r in generated_judge_rollouts]
            reference_judge_rewards = [float(r["reward"]) for r in reference_judge_rollouts]
            judge_accuracy = fmean(judge_rewards) if judge_rewards else 0.0
            skip_judge_update = not scored.judge_rollouts or should_skip_judge_update(
                config,
                judge_gate_state,
                step,
                judge_accuracy,
            )
            # The judge's EMA target advances on generator steps, not judge
            # updates: gate skips accumulate into the next broadcast's blend.
            write_ema_clock(judge.receiver.broadcast_dir, step + 1)

            # The trainer consumes the batch for its upcoming step: fresh trainer progress
            # starts at 1 and step N trains on rollouts/step_N, then broadcasts v_N.
            send_tasks = [send_role_batch(generator, generator_rollouts, group_size, generator.train_step + 1)]
            if not skip_judge_update:
                send_tasks.append(send_role_batch(judge, scored.judge_rollouts, 1, judge.train_step + 1))
            await asyncio.gather(*send_tasks)
            step_path = get_step_path(get_rollout_dir(config.output_dir), step)
            if not skip_judge_update:
                pending_judge_update = asyncio.create_task(wait_and_update(judge, judge.train_step + 1))
            generator_next_step = generator.train_step + 1
            await asyncio.gather(
                wait_and_update(generator, generator_next_step),
                asyncio.to_thread(save_rollouts, generator_rollouts, step_path / "generator_rollouts.jsonl"),
                asyncio.to_thread(save_rollouts, scored.judge_rollouts, step_path / "judge_rollouts.jsonl"),
            )

            generator_completion_lens = [float(get_model_completion_len(r)) for r in generator_rollouts]
            judge_completion_lens = [float(get_model_completion_len(r)) for r in scored.judge_rollouts]
            log_values = {
                "train/generator/reward/mean": fmean(scored.generator_rewards),
                "train/judge/reward/mean": judge_accuracy,
                "train/generator/num_output_tokens/mean": fmean(generator_completion_lens),
                "train/generator/num_output_tokens/min": min(generator_completion_lens, default=0.0),
                "train/generator/num_output_tokens/max": max(generator_completion_lens, default=0.0),
                "train/judge/num_output_tokens/mean": fmean(judge_completion_lens) if judge_completion_lens else 0.0,
                "train/judge/num_output_tokens/min": min(judge_completion_lens, default=0.0),
                "train/judge/num_output_tokens/max": max(judge_completion_lens, default=0.0),
                "train/judge/update_skipped": float(skip_judge_update),
                "time/step": time.perf_counter() - step_start,
                "time/generation": scored.generation_seconds,
                "time/judge_tail": scored.judge_tail_seconds,
            }
            for key, values in generator_metrics.items():
                log_values[f"train/generator/metrics/{key}/mean"] = fmean(values)
            judge_alignment = judge_verifiable_correlation(generator_rollouts)
            if judge_alignment is not None:
                log_values["train/generator/metrics/judge_verifiable_correlation"] = judge_alignment
            max_completion_tokens = generator.sampling_args.get("max_completion_tokens")
            if max_completion_tokens:
                log_values["train/generator/metrics/is_truncated/mean"] = fmean(
                    float(length >= max_completion_tokens) for length in generator_completion_lens
                )
            # Per-kind accuracy only exists for single-candidate judging.
            if generated_judge_rewards:
                log_values["train/judge/metrics/generated_accuracy/mean"] = fmean(generated_judge_rewards)
            if reference_judge_rewards:
                log_values["train/judge/metrics/reference_accuracy/mean"] = fmean(reference_judge_rewards)
            if judge_gate_state.accuracy_ema is not None:
                log_values["train/judge/accuracy_ema"] = judge_gate_state.accuracy_ema
            if eval_config is not None and (step + 1) % eval_config.interval == 0:
                eval_rollouts = await run_generator_eval(
                    generator,
                    eval_examples,
                    step=step + 1,
                    group_size=eval_config.group_size,
                )
                await asyncio.to_thread(
                    save_rollouts,
                    eval_rollouts,
                    step_path / "eval_generator_rollouts.jsonl",
                    exclude_keys={"trajectory"},
                )
                log_values["eval/generator/reward/mean"] = fmean(float(rollout["reward"]) for rollout in eval_rollouts)
                if "verifiable_reward" in eval_rollouts[0]["metrics"]:
                    log_values["eval/generator/metrics/verifiable_reward/mean"] = fmean(
                        float(r["metrics"]["verifiable_reward"]) for r in eval_rollouts
                    )
            await monitors.log(log_values, step=step)
            logger.success(
                f"Step {step} | Time: {time.perf_counter() - step_start:.2f}s | "
                f"Gen reward: {log_values['train/generator/reward/mean']:.4f} | "
                + (
                    f"Gen verifiable: {fmean(generator_metrics['verifiable_reward']):.4f} | "
                    if "verifiable_reward" in generator_metrics
                    else ""
                )
                + f"Judge acc: {log_values['train/judge/reward/mean']:.4f} | "
                f"Judge skip: {int(skip_judge_update)}"
            )
        # The judge trainer must not be stranded mid-handshake at max steps.
        if pending_judge_update is not None:
            await pending_judge_update
    finally:
        await monitors.finalize()
        if generator is not None:
            await generator.admin.aclose()
        if judge is not None:
            await judge.admin.aclose()

    logger.success("Adversarial orchestrator finished")


def main() -> None:
    set_proc_title("AdversarialOrchestrator")
    asyncio.run(adversarial_orchestrate(cli(AdversarialRLConfig)))


if __name__ == "__main__":
    main()
