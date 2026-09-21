"""Recompute train/generator/metrics/judge_verifiable_correlation from saved rollouts.

Runs started before the graded-distance fix logged a buggy correlation, but their
rollout files carry everything needed to recompute it: the judge probability
proxy per rollout and the completion text to re-derive the verifiable distance
(stored ``verifiable_distance`` is used directly where the fixed code already
wrote it). One Pearson correlation point is produced per train step, mirroring
``judge_verifiable_correlation`` in the adversarial orchestrator.

Incremental: steps already present in the output file are skipped, so repeated
invocations extend the result. Parallel: steps are processed by a process pool.

Usage (from the repo root, on the machine holding the run outputs):
    uv run --no-sync python tools/recompute_judge_verifiable_correlation.py \
        outputs/adversarial-smiles2formula-8bgen-32bjudge \
        [--output FILE] [--workers 8] [--limit N]
"""

import argparse
import json
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from statistics import correlation

import orjson

from prime_rl.adversarial.environments.smiles_to_formula import FORMULA_RE, formula_atom_counts
from prime_rl.adversarial.utils import extract_answer, has_reasoning

STEP_RE = re.compile(r"step_(\d+)$")


def rollout_distance(row: dict) -> float | None:
    """The env's verifiable distance for one rollout: stored value when the fixed
    code wrote it, else re-derived exactly like ``adversarial_generator_result``."""
    stored = row.get("metrics", {}).get("verifiable_distance")
    if stored is not None:
        return float(stored)
    completion = row.get("completion")
    if not has_reasoning(completion):
        return None
    candidate = extract_answer(completion)
    if not candidate:
        return None
    parts = list(FORMULA_RE.finditer(candidate))
    if not parts or "".join(match.group(0) for match in parts) != candidate:
        return None
    reference = row.get("answer") or (row.get("info") or {}).get("reference")
    if not reference:
        return None
    candidate_counts = formula_atom_counts(candidate)
    target_counts: Counter[str] = formula_atom_counts(reference)
    elements = set(candidate_counts) | set(target_counts)
    return float(sum(abs(candidate_counts[el] - target_counts[el]) for el in elements))


def step_correlation(path: Path) -> tuple[float | None, int]:
    pairs = []
    for line in path.read_bytes().splitlines():
        try:
            row = orjson.loads(line)
        except orjson.JSONDecodeError:
            continue
        probability = row.get("metrics", {}).get("judge_reference_probability_proxy")
        if probability is None:
            continue
        distance = rollout_distance(row)
        if distance is None:
            continue
        pairs.append((float(probability), distance))
    if len(pairs) < 2:
        return None, len(pairs)
    probabilities, distances = zip(*pairs)
    if len(set(probabilities)) < 2 or len(set(distances)) < 2:
        return None, len(pairs)
    return correlation(probabilities, distances), len(pairs)


def process_step(args: tuple[int, str]) -> dict:
    step, path = args
    value, num_pairs = step_correlation(Path(path))
    return {"step": step, "judge_verifiable_correlation": value, "num_pairs": num_pairs}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Run output directory containing rollouts/")
    parser.add_argument("--output", type=Path, default=None, help="Result jsonl (default: <run_dir>/judge_verifiable_correlation_recomputed.jsonl)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="Process at most this many new steps this invocation")
    args = parser.parse_args()

    output = args.output or args.run_dir / "judge_verifiable_correlation_recomputed.jsonl"
    done: set[int] = set()
    if output.exists():
        for line in output.read_text().splitlines():
            try:
                done.add(json.loads(line)["step"])
            except (json.JSONDecodeError, KeyError):
                continue

    pending = []
    for step_dir in (args.run_dir / "rollouts").iterdir():
        match = STEP_RE.search(step_dir.name)
        if match is None or int(match.group(1)) in done:
            continue
        rollout_file = step_dir / "generator_rollouts.jsonl"
        if rollout_file.is_file():
            pending.append((int(match.group(1)), str(rollout_file)))
    pending.sort()
    if args.limit is not None:
        pending = pending[: args.limit]
    print(f"{len(done)} steps already computed, {len(pending)} to process")
    if not pending:
        return

    processed = 0
    with output.open("a") as sink, ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(process_step, item) for item in pending]
        for future in as_completed(futures):
            sink.write(json.dumps(future.result()) + "\n")
            sink.flush()
            processed += 1
            if processed % 200 == 0:
                print(f"{processed}/{len(pending)}")
    print(f"done: {processed} new steps -> {output}")


if __name__ == "__main__":
    main()
