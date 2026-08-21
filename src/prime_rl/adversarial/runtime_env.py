import multiprocessing as mp
from multiprocessing.process import BaseProcess
from pathlib import Path

import verifiers as vf
from verifiers.serve import ZMQEnvClient, ZMQEnvServer
from verifiers.utils.serve_utils import get_free_port

from prime_rl.configs.adversarial_orchestrator import EnvConfig, TrainEnvConfig
from prime_rl.utils.logger import get_logger

REQUIRED_STATE_COLUMNS = ["trajectory"]


class Env:
    """Wraps a vf.Environment - only exposes features used in PRIME-RL."""

    def __init__(self, config: EnvConfig):
        self.config = config
        self.sampling_args: dict = {}

        get_logger().debug(f"Initializing {config.resolved_name} ({config})")
        self._env: vf.Environment = vf.load_environment(config.stripped_id, **config.args)
        self._env_client: ZMQEnvClient | None = None
        self._env_server_process: BaseProcess | None = None

    @property
    def name(self) -> str:
        return self.config.resolved_name

    @property
    def env(self) -> vf.Environment:
        return self._env

    @property
    def env_client(self) -> ZMQEnvClient:
        if not self._env_client:
            raise RuntimeError(
                f"Env {self.name} has no env client connected. Call connect() first to connect to an env server."
            )
        return self._env_client

    @property
    def requires_group_scoring(self) -> bool:
        return any(self.env.rubric._is_group_func(func) for func in self.env.rubric._get_reward_funcs())

    async def start(
        self,
        log_dir: Path,
        log_level: str | None = None,
        json_logging: bool = False,
    ) -> None:
        """Spawn an env server (if needed) and connect to it."""
        if self.config.address is None:
            address = self._spawn(log_dir=log_dir, log_level=log_level, json_logging=json_logging)
        else:
            address = self.config.address
        get_logger().debug(f"Connecting {self.name} to env server {address}")
        self._env_client = ZMQEnvClient(address=address, name=self.name)
        await self.env_client.wait_for_server_startup()

    def _spawn(
        self,
        log_dir: Path,
        log_level: str | None = None,
        json_logging: bool = False,
    ) -> str:
        assert isinstance(self.config.num_workers, int), (
            f"num_workers must be resolved before spawn, got {self.config.num_workers!r}"
        )
        num_workers = self.config.num_workers
        address = f"tcp://127.0.0.1:{get_free_port()}"
        get_logger().debug(f"Spawning env server {self.name} ({address=}, {num_workers=})")
        process = mp.get_context("spawn").Process(
            target=ZMQEnvServer.run_server,
            args=(
                self.config.stripped_id,
                self.config.args,
                self.config.extra_env_kwargs,
                log_level,
                (log_dir / self.name).as_posix(),
            ),
            kwargs=dict(
                address=address,
                json_logging=json_logging,
                console_logging=False,
                num_workers=num_workers,
            ),
            daemon=False,
        )
        process.start()
        self._env_server_process = process
        return address

    def _sampling_args_with_salt(self, cache_salt: str | None) -> dict:
        sampling_args = {**self.sampling_args}
        if cache_salt is None:
            return sampling_args
        extra_body = {**sampling_args.get("extra_body", {}), "cache_salt": cache_salt}
        sampling_args["extra_body"] = extra_body
        return sampling_args

    @property
    def state_columns(self) -> list[str]:
        """Required columns plus any extras configured on the env, deduped (required first)."""
        merged: list[str] = []
        for col in (*REQUIRED_STATE_COLUMNS, *self.config.state_columns):
            if col not in merged:
                merged.append(col)
        return merged

    async def run_rollout(
        self,
        client: vf.ClientConfig,
        example: dict,
        model_name: str,
        cache_salt: str | None,
    ) -> vf.RolloutOutput:
        """Run a single rollout for an example."""
        return await self.env.run_rollout(
            vf.RolloutInput(**example),
            client=client,
            model=model_name,
            sampling_args=self._sampling_args_with_salt(cache_salt),
            max_retries=self.config.max_retries,
            state_columns=self.state_columns,
            env_client=self.env_client,
        )

    async def run_group(
        self,
        client: vf.ClientConfig,
        example: dict,
        model_name: str,
        group_size: int,
        cache_salt: str | None,
    ) -> list[vf.RolloutOutput]:
        """Run a group of rollouts for an example. Required for group-scoring envs."""
        return await self.env.run_group(
            [vf.RolloutInput(**example) for _ in range(group_size)],
            client=client,
            model=model_name,
            sampling_args=self._sampling_args_with_salt(cache_salt),
            max_retries=self.config.max_retries,
            state_columns=self.state_columns,
            env_client=self.env_client,
        )

    def shutdown(self) -> None:
        if self._env_server_process is None:
            return
        self._env_server_process.terminate()
        self._env_server_process = None


class TrainEnv(Env):
    config: TrainEnvConfig

    def __init__(self, config: TrainEnvConfig):
        super().__init__(config)
        self.sampling_args = config.sampling.to_sampling_args()

    def get_dataset(self, seed: int | None = None):
        return self.env.get_dataset(seed=seed)
