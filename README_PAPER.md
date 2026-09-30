# Adversarial RL on prime-rl

This repository accompanies the paper. It extends
[prime-rl](https://github.com/PrimeIntellect-ai/prime-rl) with adversarial
training: a generator and a judge train against each other. The generator
learns to produce answers the judge accepts. The judge learns to tell
generated answers from references. One orchestrator drives both: it generates
rollouts, scores them with the served judge, and feeds each side its training
batch.

## What this adds on top of prime-rl

- `adversarial-rl` entrypoint (`src/prime_rl/entrypoints/adversarial_rl.py`):
  launches the orchestrator and a trainer and inference server per role from
  one layered config, on a single node or across many.
- Synchronous adversarial orchestrator
  (`src/prime_rl/adversarial/orchestrator.py`): the generate, judge, train,
  broadcast loop, with a judge update gate and alignment metrics.
- Task environments (`src/prime_rl/adversarial/environments/`): multiplication,
  countdown code, SMILES-to-formula, and paper abstracts, each a thin protocol
  over question, reference, candidate parsing, and judge prompt.
- Judge replay buffer (`src/prime_rl/adversarial/replay.py`): the judge batch
  extends with samples from a sliding window of generator history, so beaten
  strategies stay in its training distribution.
- EMA target network (`weight_broadcast.ema_decay`): the served judge is a
  Polyak average trailing training, a SAC-style target that stabilizes the
  judge as a reward source.
- Crash-resumable runs (`--resume`): boundary-aligned checkpoints for both
  roles, exact pairing snapshots, replay reseeding, and EMA target restore.
- SFT baselines (`environments/tasksets/`, SFT trainer accuracy metrics):
  per-task baselines trained on the same data and scored by the same
  verifiable rules.

## Getting started

The runnable example, and the guide to configuring, running, and resuming a
run, live in [`configs/adversarial/`](configs/adversarial/README.md):

```bash
uv run adversarial-rl @ configs/adversarial/base.toml @ configs/adversarial/multiplication.toml
```

## Reproducing the paper

Every recipe behind the paper's tables lives in
[`configs/paper_experiments/`](configs/paper_experiments/README.md): the
adversarial experiments under `adversarial/` with a launch table, and the SFT
baselines under `sft/`.
