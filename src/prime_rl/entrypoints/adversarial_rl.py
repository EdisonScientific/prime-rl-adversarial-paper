import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path
from subprocess import Popen
from threading import Event, Thread
from typing import Any

import tomli_w

import prime_rl._compat  # noqa: F401
from prime_rl.adversarial.utils import resolve_resume_point
from prime_rl.configs.adversarial import AdversarialRLConfig
from prime_rl.configs.shared import ResumeConfig
from prime_rl.entrypoints.rl import (
    INFERENCE_CONFIG,
    INFERENCE_CONFIG_EXCLUDE,
    TRAINER_CONFIG,
    get_physical_gpu_ids,
)
from prime_rl.trainer.model import pre_download_model
from prime_rl.utils.config import cli, dump_resolved_config
from prime_rl.utils.logger import setup_logger
from prime_rl.utils.pathing import clean_future_steps, get_log_dir, validate_run_dir
from prime_rl.utils.process import cleanup_processes, cleanup_threads, monitor_process, set_proc_title
from prime_rl.utils.utils import get_free_port

ADVERSARIAL_TOML = "adversarial.toml"
ENGINE_CONFIG = "engine.json"
GENERATOR = "generator"
JUDGE = "judge"


def slurm_nodes() -> list[str]:
    nodelist = os.environ.get("SLURM_JOB_NODELIST")
    if not nodelist:
        return []
    return subprocess.check_output(["scontrol", "show", "hostnames", nodelist], text=True).splitlines()


def allocate_gpus(config: AdversarialRLConfig) -> dict[str, dict[str, Any]]:
    visible = get_physical_gpu_ids()
    nodes = slurm_nodes()

    if any(run.deployment.type == "multi_node" for run in (config.generator, config.judge)):
        if not nodes:
            raise ValueError("adversarial-rl multi_node roles require a SLURM allocation.")
        required_nodes = 0
        for run in (config.generator, config.judge):
            if run.deployment.type == "multi_node":
                required_nodes += run.deployment.num_train_nodes + run.deployment.total_infer_nodes
            else:
                required_nodes += 1
        if len(nodes) < required_nodes:
            raise ValueError(f"Adversarial run requires {required_nodes} nodes, but SLURM provided {nodes}.")
        out = {}
        cursor = 0
        for role, run in ((GENERATOR, config.generator), (JUDGE, config.judge)):
            if run.deployment.type == "multi_node":
                infer_node_count = run.deployment.total_infer_nodes if run.inference is not None else 0
                inference_nodes = nodes[cursor : cursor + infer_node_count]
                cursor += infer_node_count
                trainer_nodes = nodes[cursor : cursor + run.deployment.num_train_nodes]
                cursor += run.deployment.num_train_nodes
                inference_gpus = visible
                trainer_gpus = visible
            else:
                num_infer_gpus = run.deployment.num_infer_gpus if run.inference is not None else 0
                required_gpus = num_infer_gpus + run.deployment.num_train_gpus
                if required_gpus > len(visible):
                    raise ValueError(
                        f"{role} requires {required_gpus} GPU(s), but only {len(visible)} are visible: {visible}"
                    )
                node = nodes[cursor]
                cursor += 1
                inference_nodes = [node] if run.inference is not None else []
                trainer_nodes = [node]
                inference_gpus = visible[:num_infer_gpus]
                trainer_gpus = visible[num_infer_gpus:required_gpus]
            out[role] = {
                "inference_nodes": inference_nodes,
                "trainer_nodes": trainer_nodes,
                "inference": inference_gpus,
                "trainer": trainer_gpus,
            }
        return out

    counts: dict[str, dict[str, int]] = {}
    for role, run in ((GENERATOR, config.generator), (JUDGE, config.judge)):
        num_infer_gpus = run.deployment.num_infer_gpus if run.inference is not None else 0
        counts[role] = {"inference": num_infer_gpus, "trainer": run.deployment.num_train_gpus}
    generator_required = counts[GENERATOR]["inference"] + counts[GENERATOR]["trainer"]
    judge_required = counts[JUDGE]["inference"] + counts[JUDGE]["trainer"]
    total_required = generator_required + judge_required

    if total_required > len(visible) and len(nodes) >= 2:
        if generator_required > len(visible) or judge_required > len(visible):
            raise ValueError(
                f"Each adversarial role must fit on one node. "
                f"Visible GPUs per node: {visible}; generator needs {generator_required}, judge needs {judge_required}."
            )
        return {
            GENERATOR: {
                "inference_nodes": [nodes[0]],
                "trainer_nodes": [nodes[0]],
                "inference": visible[: counts[GENERATOR]["inference"]],
                "trainer": visible[counts[GENERATOR]["inference"] : generator_required],
            },
            JUDGE: {
                "inference_nodes": [nodes[1]],
                "trainer_nodes": [nodes[1]],
                "inference": visible[: counts[JUDGE]["inference"]],
                "trainer": visible[counts[JUDGE]["inference"] : judge_required],
            },
        }

    if total_required > len(visible):
        raise ValueError(
            f"Adversarial run requires {total_required} GPU(s) "
            f"({generator_required} generator + {judge_required} judge), "
            f"but only {len(visible)} are visible: {visible}"
        )

    cursor = 0
    out = {}
    for role in (GENERATOR, JUDGE):
        infer_count = counts[role]["inference"]
        train_count = counts[role]["trainer"]
        infer = visible[cursor : cursor + infer_count]
        cursor += infer_count
        train = visible[cursor : cursor + train_count]
        cursor += train_count
        out[role] = {"inference_nodes": [], "trainer_nodes": [], "inference": infer, "trainer": train}
    return out


def apply_role_hosts(config: AdversarialRLConfig, allocation: dict[str, dict[str, Any]]) -> None:
    for role, run in ((GENERATOR, config.generator), (JUDGE, config.judge)):
        inference_nodes = allocation[role]["inference_nodes"]
        if run.inference is not None:
            host = inference_nodes[0] if inference_nodes else "localhost"
            run.orchestrator.client.base_url = f"http://{host}:{run.inference.server.port}/v1"
            if len(inference_nodes) > 1:
                # Admin ops must reach every replica engine directly. The router
                # only forwards the data plane.
                run.orchestrator.client.admin_base_url = [
                    f"http://{node}:{run.inference.backend_port}/v1" for node in inference_nodes
                ]

        trainer_nodes = allocation[role]["trainer_nodes"]
        if run.trainer.weight_broadcast.type == "nccl":
            trainer_host = trainer_nodes[0] if trainer_nodes else "localhost"
            run.trainer.weight_broadcast.host = "0.0.0.0"
            run.orchestrator.weight_broadcast.host = trainer_host
            # The orchestrator subprocess re-parses the dumped config, and the
            # shared-block validator re-derives the orchestrator's broadcast
            # config from it, so the host must survive that round trip.
            if run.weight_broadcast is not None and run.weight_broadcast.type == "nccl":
                run.weight_broadcast.host = trainer_host


def write_configs(config: AdversarialRLConfig, config_dir: Path) -> None:
    config_dir.mkdir(parents=True, exist_ok=True)
    adversarial_config = config.model_dump(exclude_none=True, mode="json")
    with open(config_dir / ADVERSARIAL_TOML, "wb") as f:
        tomli_w.dump(adversarial_config, f)

    for role, role_config in ((GENERATOR, config.generator), (JUDGE, config.judge)):
        role_dir = config_dir / role
        role_dir.mkdir(parents=True, exist_ok=True)
        with open(role_dir / TRAINER_CONFIG, "w") as f:
            json.dump(dump_resolved_config(role_config.trainer), f, indent=2)
        if role_config.inference is not None:
            with open(role_dir / INFERENCE_CONFIG, "w") as f:
                json.dump(dump_resolved_config(role_config.inference, exclude=INFERENCE_CONFIG_EXCLUDE), f, indent=2)
            if role_config.deployment.type == "multi_node" and role_config.deployment.total_infer_nodes > 1:
                # Replica engines run bare on backend_port. A shared router on
                # server.port load-balances across them.
                engine = role_config.inference.model_copy(deep=True)
                engine.router = None
                engine.server.port = engine.backend_port
                with open(role_dir / ENGINE_CONFIG, "w") as f:
                    json.dump(dump_resolved_config(engine, exclude=INFERENCE_CONFIG_EXCLUDE), f, indent=2)


def start_process(cmd: list[str], log_path: Path, env: dict[str, str], node: str | list[str] | None = None) -> Popen:
    process_env = {**os.environ, **env}
    if node is not None:
        nodes = [node] if isinstance(node, str) else node
        remote_env = " ".join(f"{key}={shlex.quote(value)}" for key, value in env.items())
        # A literal {SLURM_PROCID} arg becomes $SLURM_PROCID, expanded per task on the
        # remote side (multi-node torchrun --node-rank). Everything else is quoted.
        remote_cmd = " ".join("$SLURM_PROCID" if arg == "{SLURM_PROCID}" else shlex.quote(arg) for arg in cmd)
        cmd = [
            "srun",
            "--overlap",
            f"--nodes={len(nodes)}",
            f"--ntasks={len(nodes)}",
            "--ntasks-per-node=1",
            f"--nodelist={','.join(nodes)}",
            "--export=ALL",
            "bash",
            "-lc",
            f"{remote_env} exec {remote_cmd}",
        ]
    with open(log_path, "w") as log_file:
        return Popen(cmd, env=process_env, stdout=log_file, stderr=log_file)


def adversarial_rl(config: AdversarialRLConfig) -> None:
    logger = setup_logger(config.log.level or os.environ.get("PRIME_LOG_LEVEL", "info"), config.log.json_logging)

    if config.clean_output_dir and config.output_dir.exists():
        logger.warning(f"Cleaning existing directory: {config.output_dir}")
        shutil.rmtree(config.output_dir)

    resume_step = -1
    judge_resume_step = -1
    if config.resume:
        snapshot = resolve_resume_point(config.output_dir, config.generator.output_dir, config.judge.output_dir)
        resume_step = snapshot["loop_step"]
        judge_resume_step = snapshot["judge_train_step"]
        logger.info(f"Resuming from loop step {resume_step} (judge train step {judge_resume_step})")
        config.generator.trainer.resume = ResumeConfig(step=resume_step)
        # A judge with no update yet restarts fresh from the pretrained weights.
        if judge_resume_step > 0:
            config.judge.trainer.resume = ResumeConfig(step=judge_resume_step)

    for role_config, role_resume_step in ((config.generator, resume_step), (config.judge, judge_resume_step)):
        validate_run_dir(
            role_config.output_dir,
            output_dir=config.output_dir,
            resuming=config.resume,
            clean=config.clean_output_dir or role_config.clean,
        )
        role_config.output_dir.mkdir(parents=True, exist_ok=True)
        clean_future_steps(role_config.output_dir, role_resume_step)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    # The orchestrator and the generator trainer share one W&B run. The judge
    # trainer is muted because the orchestrator logs the judge metrics. The persisted
    # run id makes resumed runs continue the same W&B run.
    run_id_file = config.output_dir / "wandb_run_id"
    if not run_id_file.exists():
        run_id_file.write_text(uuid.uuid4().hex)
    wandb_shared_env = {"WANDB_SHARED_MODE": "1", "WANDB_RUN_ID": run_id_file.read_text().strip()}
    config.judge.trainer.monitors.wandb = None
    pre_download_model(
        config.generator.trainer.model.name,
        skip_weights=config.generator.trainer.model.debug.random_init,
    )
    if config.judge.trainer.model.name != config.generator.trainer.model.name:
        pre_download_model(
            config.judge.trainer.model.name,
            skip_weights=config.judge.trainer.model.debug.random_init,
        )

    allocation = allocate_gpus(config)
    apply_role_hosts(config, allocation)

    config_dir = config.output_dir / "configs"
    write_configs(config, config_dir)
    logger.info(f"Wrote adversarial configs to {config_dir}")

    log_dir = get_log_dir(config.output_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    start_command = sys.argv
    processes: list[Popen] = []
    monitor_threads: list[Thread] = []
    error_queue: list[Exception] = []
    stop_events: dict[str, Event] = {}

    def sigterm_handler(signum, frame):
        logger.warning("Received SIGTERM, terminating adversarial run...")
        sys.exit(1)

    signal.signal(signal.SIGTERM, sigterm_handler)

    def monitor(name: str, process: Popen) -> None:
        event = Event()
        stop_events[name] = event
        thread = Thread(target=monitor_process, args=(process, event, error_queue, name), daemon=True)
        thread.start()
        monitor_threads.append(thread)

    try:
        for role, run in ((GENERATOR, config.generator), (JUDGE, config.judge)):
            inference_config = config_dir / role / INFERENCE_CONFIG
            if not inference_config.exists():
                continue
            inference_nodes = allocation[role]["inference_nodes"]
            inference_gpus = {"CUDA_VISIBLE_DEVICES": ",".join(map(str, allocation[role]["inference"]))}
            if len(inference_nodes) > 1:
                engine_config = config_dir / role / ENGINE_CONFIG
                for replica, node in enumerate(inference_nodes):
                    engine = start_process(
                        ["inference", "@", engine_config.as_posix()],
                        log_dir / f"{role}_engine{replica}.log",
                        inference_gpus,
                        node,
                    )
                    processes.append(engine)
                    monitor(f"{role}_engine{replica}", engine)
                assert run.inference is not None and run.inference.router is not None
                router = start_process(
                    [
                        "vllm-router",
                        "--policy",
                        run.inference.router.policy,
                        "--host",
                        "0.0.0.0",
                        "--port",
                        str(run.inference.server.port),
                        "--worker-urls",
                        *[f"http://{node}:{run.inference.backend_port}" for node in inference_nodes],
                        "--request-id-headers",
                        "x-session-id",
                        "--worker-startup-timeout-secs",
                        "4200",
                        "--prometheus-port",
                        str(run.inference.server.port + 21000),
                    ],
                    log_dir / f"{role}_router.log",
                    {},
                    inference_nodes[0],
                )
                processes.append(router)
                monitor(f"{role}_router", router)
            else:
                inference = start_process(
                    ["inference", "@", inference_config.as_posix()],
                    log_dir / f"{role}_inference.log",
                    inference_gpus,
                    inference_nodes[0] if inference_nodes else None,
                )
                processes.append(inference)
                monitor(f"{role}_inference", inference)

        for role in (GENERATOR, JUDGE):
            trainer_nodes = allocation[role]["trainer_nodes"]
            trainer_location: str | list[str] | None
            if len(trainer_nodes) > 1:
                trainer_location = trainer_nodes
                rdzv_endpoint = f"{trainer_nodes[0]}:{get_free_port()}"
                torchrun_dist_args = [f"--nnodes={len(trainer_nodes)}", "--node-rank", "{SLURM_PROCID}"]
            else:
                trainer_location = trainer_nodes[0] if trainer_nodes else None
                rdzv_endpoint = f"localhost:{get_free_port()}"
                torchrun_dist_args = []
            trainer = start_process(
                [
                    "torchrun",
                    "--role=trainer",
                    f"--rdzv-endpoint={rdzv_endpoint}",
                    f"--rdzv-id={uuid.uuid4().hex}",
                    f"--log-dir={log_dir / role / 'trainer' / 'torchrun'}",
                    "--redirect=3",
                    "--tee=3",
                    *torchrun_dist_args,
                    f"--nproc-per-node={len(allocation[role]['trainer'])}",
                    "-m",
                    "prime_rl.trainer.rl.train",
                    "@",
                    (config_dir / role / TRAINER_CONFIG).as_posix(),
                ],
                log_dir / f"{role}_trainer.log",
                {
                    **(wandb_shared_env if role == GENERATOR else {}),
                    "WANDB_SHARED_LABEL": f"{role}_trainer",
                    "CUDA_VISIBLE_DEVICES": ",".join(map(str, allocation[role]["trainer"])),
                    "PYTHONUNBUFFERED": "1",
                    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                    "LOGURU_FORCE_COLORS": "1",
                    "WANDB_PROGRAM": "uv run adversarial-rl",
                    "WANDB_ARGS": json.dumps(start_command),
                },
                trainer_location,
            )
            processes.append(trainer)
            monitor(f"{role}_trainer", trainer)

        orchestrator = start_process(
            [
                sys.executable,
                "-m",
                "prime_rl.adversarial.orchestrator",
                "@",
                (config_dir / ADVERSARIAL_TOML).as_posix(),
            ],
            log_dir / "adversarial_orchestrator.log",
            {
                **wandb_shared_env,
                "WANDB_SHARED_LABEL": "orchestrator",
                "LOGURU_FORCE_COLORS": "1",
                "WANDB_PROGRAM": "uv run adversarial-rl",
                "WANDB_ARGS": json.dumps(start_command),
            },
        )
        processes.append(orchestrator)
        monitor("adversarial_orchestrator", orchestrator)

        logger.success(f"Startup complete. Logs are in {log_dir}")
        while not stop_events["adversarial_orchestrator"].is_set():
            if error_queue:
                raise error_queue[0]
            time.sleep(1)

        if orchestrator.returncode != 0:
            raise RuntimeError(f"adversarial orchestrator failed with exit code {orchestrator.returncode}")

        while not (stop_events[GENERATOR + "_trainer"].is_set() and stop_events[JUDGE + "_trainer"].is_set()):
            if error_queue:
                raise error_queue[0]
            time.sleep(1)

        logger.success("Adversarial training finished!")
    except KeyboardInterrupt:
        logger.warning("Received interrupt signal, terminating adversarial run...")
        sys.exit(1)
    finally:
        cleanup_threads(monitor_threads)
        cleanup_processes(processes)


def main() -> None:
    set_proc_title("AdversarialLauncher")
    adversarial_rl(cli(AdversarialRLConfig))


if __name__ == "__main__":
    main()
