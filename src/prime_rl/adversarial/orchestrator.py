import asyncio
import time
from dataclasses import dataclass
from statistics import fmean
from typing import Any

import verifiers as vf

import prime_rl._compat  # noqa: F401
import prime_rl.monitors as monitors
from prime_rl.adversarial.utils import (
    backfill_rollout_tokens,
    get_model_completion_len,
    rollout_to_sample,
    save_rollouts,
)
from prime_rl.configs.adversarial import AdversarialRLConfig, AdversarialRoleConfig
from prime_rl.orchestrator.packing import BatchPacker
from prime_rl.trainer.model import setup_tokenizer
from prime_rl.transports.rollouts import setup_micro_batch_sender
from prime_rl.transports.rollouts.types import TrainingSample
from prime_rl.utils.client import InferencePool, init_nccl_broadcast
from prime_rl.utils.config import cli
from prime_rl.utils.logger import get_logger, setup_logger
from prime_rl.utils.pathing import get_broadcast_dir, get_rollout_dir, get_step_path, wait_for_path
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
    inference_pool: InferencePool
    model_name: str
    train_client: vf.ClientConfig
    ckpt_step: int = 0
    train_step: int = 0

    def sampling_args_with_salt(self, cache_salt: str) -> dict[str, Any]:
        return {**self.sampling_args, "extra_body": {**self.sampling_args.get("extra_body", {}), "cache_salt": cache_salt}}


@dataclass
class ScoredBatch:
    judge_rollouts: list[vf.RolloutOutput]
    generator_rewards: list[float]


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


async def setup_student_inference_pool(*, config, tokenizer):
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


async def setup_role(name: str, config: AdversarialRoleConfig) -> RoleRuntime:
    logger = setup_logger(config.log.level or "info", json_logging=config.log.json_logging)
    orch = config.orchestrator
    env_config = orch.train.env[0]
    env = vf.load_environment(env_config.stripped_id, **env_config.args)
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
        env_name=env_config.resolved_name,
        sampling_args=env_config.sampling.to_sampling_args(),
        max_retries=env_config.max_retries,
        tokenizer=tokenizer,
        renderer=renderer,
        inference_pool=inference_pool,
        model_name=rollout_model_name,
        train_client=inference_pool.train_client,
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
            rollout["reward"] = float(result["verifiable_reward"])
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
    judge_env = judge.env

    for (rollout, result), judge_rollout in zip(generated_task_positions, generated_judge_rollouts):
        judge_reference_probability_proxy = judge_env.adversarial_reference_probability_proxy(judge_rollout)
        if not result["format_ok"]:
            reward = 0.0
        elif reward_mode == "exact_only":
            reward = result["verifiable_reward"]
        elif reward_mode == "judge_times_exact":
            reward = (judge_reference_probability_proxy or 0.0) * result["verifiable_reward"]
        else:
            reward = judge_reference_probability_proxy or 0.0
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
                wait_and_update(generator, start_step),
                wait_and_update(judge, start_step),
            )
        examples = list(generator.env.get_dataset(seed=config.generator.orchestrator.buffer.seed))
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
                "adversarial/eval/generator/reward": fmean(float(rollout["reward"]) for rollout in eval_rollouts),
                "adversarial/eval/generator/verifiable_reward": fmean(
                    float(r["metrics"]["verifiable_reward"]) for r in eval_rollouts
                ),
            }
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

            generator_verifiable_reward = fmean(float(r["metrics"]["verifiable_reward"]) for r in generator_rollouts)
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

            generator_completion_lens = [float(get_model_completion_len(r)) for r in generator_rollouts]
            judge_completion_lens = [float(get_model_completion_len(r)) for r in scored.judge_rollouts]
            log_values = {
                "adversarial/generator/reward": fmean(scored.generator_rewards),
                "adversarial/generator/verifiable_reward": generator_verifiable_reward,
                "adversarial/judge/accuracy": judge_accuracy,
                "adversarial/judge/generated_accuracy": fmean(generated_judge_rewards) if generated_judge_rewards else 0.0,
                "adversarial/judge/reference_accuracy": fmean(reference_judge_rewards) if reference_judge_rewards else 0.0,
                "adversarial/generator/completion_tokens/mean": fmean(generator_completion_lens),
                "adversarial/generator/completion_tokens/min": min(generator_completion_lens, default=0.0),
                "adversarial/generator/completion_tokens/max": max(generator_completion_lens, default=0.0),
                "adversarial/judge/completion_tokens/mean": fmean(judge_completion_lens) if judge_completion_lens else 0.0,
                "adversarial/judge/completion_tokens/min": min(judge_completion_lens, default=0.0),
                "adversarial/judge/completion_tokens/max": max(judge_completion_lens, default=0.0),
                "adversarial/judge/update_skipped": float(skip_judge_update),
                "adversarial/step_time_seconds": time.perf_counter() - step_start,
            }
            if judge_gate_state.accuracy_ema is not None:
                log_values["adversarial/judge/accuracy_ema"] = judge_gate_state.accuracy_ema
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
                log_values["adversarial/eval/generator/reward"] = fmean(
                    float(rollout["reward"]) for rollout in eval_rollouts
                )
                log_values["adversarial/eval/generator/verifiable_reward"] = fmean(
                    float(r["metrics"]["verifiable_reward"]) for r in eval_rollouts
                )
            await monitors.log(log_values, step=step)
            logger.success(
                f"Step {step} | Time: {time.perf_counter() - step_start:.2f}s | "
                f"Gen reward: {log_values['adversarial/generator/reward']:.4f} | "
                f"Gen verifiable: {log_values['adversarial/generator/verifiable_reward']:.4f} | "
                f"Judge acc: {log_values['adversarial/judge/accuracy']:.4f} | "
                f"Judge skip: {int(skip_judge_update)}"
            )
    finally:
        await monitors.finalize()
        if generator is not None:
            await generator.inference_pool.stop()
        if judge is not None:
            await judge.inference_pool.stop()

    logger.success("Adversarial orchestrator finished")


def main() -> None:
    set_proc_title("AdversarialOrchestrator")
    asyncio.run(adversarial_orchestrate(cli(AdversarialRLConfig)))


if __name__ == "__main__":
    main()
