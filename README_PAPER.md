# Reinforced Adversarial Learning (RAL)

RAL post-trains a reasoning LLM from reference answers alone. A generator
produces answers, a co-trained judge compares them with references, and the
judge's score is the only reward. Built on
[prime-rl](https://github.com/PrimeIntellect-ai/prime-rl).

## Training loop

One orchestrator drives both roles
(`src/prime_rl/adversarial/orchestrator.py`). Each step it generates a rollout
batch, scores every answer with the served judge, and builds two training
batches. The generator trains on the judge's scores with IPO loss. The judge
trains as a binary classifier on generated versus reference answers. Both
trainers then broadcast fresh weights to their inference servers. The
`adversarial-rl` entrypoint launches everything from one layered config.

## Stability

- Judge update gate: judge updates are skipped while judge accuracy stays
  above a threshold.
- Judge replay buffer (`src/prime_rl/adversarial/replay.py`): the judge batch
  includes samples from a sliding window of generator history.
- Judge target network (`weight_broadcast.ema_decay`): the generator is
  scored by a Polyak average of the judge, SAC style.

## Tasks

Multiplication, countdown code, SMILES-to-formula, and paper abstracts
(`src/prime_rl/adversarial/environments/`). Each defines the question, the
reference, the candidate parsing, and the judge prompt. Verifiable tasks also
report a graded distance for scoring the judge against ground truth. SFT
baselines (`environments/tasksets/`) train on the same data and are scored by
the same rules.

## Resume

Runs checkpoint both roles on shared boundaries. `--resume` restores models,
optimizers, replay buffer, and EMA target.

## Layout

- Runnable example and usage guide:
  [`configs/adversarial/`](configs/adversarial/README.md).
- Paper recipes, adversarial and SFT:
  [`configs/paper_experiments/`](configs/paper_experiments/README.md).
