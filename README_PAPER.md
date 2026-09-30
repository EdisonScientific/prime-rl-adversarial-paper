# Adversarial RL on prime-rl

This repository accompanies the paper. It extends
[prime-rl](https://github.com/PrimeIntellect-ai/prime-rl) with adversarial
training. A generator and a judge train against each other: the generator
learns to produce answers the judge accepts, the judge learns to tell
generated answers from references, and each improves against the other.

## The loop

One orchestrator drives both roles
(`src/prime_rl/adversarial/orchestrator.py`). Each step it generates a rollout
batch, scores every answer with the served judge, and builds two training
batches. The generator trains on the judge's scores with IPO loss. The judge
trains as a binary classifier on generated versus reference answers. Both
trainers broadcast fresh weights to their inference servers, and the next step
begins. The `adversarial-rl` entrypoint launches all of it from one layered
config.

## Keeping the game balanced

An adversarial judge fails in two ways: it overpowers the generator, or it
forgets what it once knew. Three mechanisms counter this.

- Judge update gate: the judge skips training while its accuracy EMA stays
  above a threshold, so it never runs far ahead of the generator.
- Judge replay buffer (`src/prime_rl/adversarial/replay.py`): the judge also
  trains on a sliding window of generator history, so beaten strategies stay
  in its distribution instead of being rediscovered.
- EMA target network (`weight_broadcast.ema_decay`): the served judge is a
  Polyak average of the learner, a SAC-style target, so the generator's reward
  moves slowly even while the judge trains every step.

## Tasks and baselines

Four task environments (`src/prime_rl/adversarial/environments/`):
multiplication, countdown code, SMILES-to-formula, and paper abstracts. Each
defines the question, the reference, the candidate parsing, and the judge
prompt. Verifiable tasks also report a graded distance, which grades the judge
against ground truth. SFT baselines (`environments/tasksets/`) train on the
same data and are scored by the same rules.

## Long runs

Runs checkpoint both roles on shared boundaries and resume exactly with
`--resume`: models, optimizers, replay buffer, and EMA target all restore.

## Where things are

- The runnable example and the configure, run, and resume guide:
  [`configs/adversarial/`](configs/adversarial/README.md).
- Every recipe behind the paper's tables:
  [`configs/paper_experiments/`](configs/paper_experiments/README.md).
