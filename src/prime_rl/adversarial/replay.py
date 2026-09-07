# Judge replay: the judge trains against a sliding window of generator history
# alongside the current batch, so previously beaten generator strategies stay
# in its training distribution instead of being forgotten and rediscovered.

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import orjson

from prime_rl.utils.logger import get_logger
from prime_rl.utils.pathing import get_all_ckpt_steps, get_step_path


@dataclass
class ReplayItem:
    question: str
    candidate: str
    reference: str
    step: int


class ReplayBuffer:
    """Fixed-capacity FIFO ring: O(1) insert, O(k) uniform sample."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._items: list[ReplayItem] = []
        self._cursor = 0

    def __len__(self) -> int:
        return len(self._items)

    def add(self, item: ReplayItem) -> None:
        if len(self._items) < self.capacity:
            self._items.append(item)
        else:
            self._items[self._cursor] = item
            self._cursor = (self._cursor + 1) % self.capacity

    def sample(self, k: int, rng: random.Random) -> list[ReplayItem]:
        if k >= len(self._items):
            return list(self._items)
        return rng.sample(self._items, k)


def seed_replay_buffer(buffer: ReplayBuffer, rollout_dir: Path, env: Any, before_step: int, batches: int) -> int:
    """Refill the window from the generator rollouts a previous run already saved."""
    if not rollout_dir.is_dir():
        return 0
    steps = sorted(s for s in get_all_ckpt_steps(rollout_dir) if s < before_step)[-batches:]
    added = 0
    skipped = 0
    for step in steps:
        path = get_step_path(rollout_dir, step) / "generator_rollouts.jsonl"
        if not path.is_file():
            continue
        for line in path.read_bytes().splitlines():
            try:
                row = orjson.loads(line)
            except orjson.JSONDecodeError:
                # Crash-truncated rollout files are expected on resume
                skipped += 1
                continue
            info = row.get("info") or {}
            question = info.get("question") or (row.get("prompt") or [{}])[-1].get("content")
            reference = info.get("reference") or row.get("answer")
            candidate = env.adversarial_candidate_answer(row.get("completion"))
            if question and reference and candidate:
                buffer.add(ReplayItem(question=question, candidate=candidate, reference=reference, step=step))
                added += 1
    if skipped:
        get_logger().warning(f"Skipped {skipped} unparseable rollout lines while seeding the replay buffer")
    return added
