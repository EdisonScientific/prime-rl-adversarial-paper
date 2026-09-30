# Adversarial RL

`adversarial-rl` launches a full adversarial run: the orchestrator, and a
trainer and an inference server per role. The method is described in the root
[`README_PAPER.md`](../../README_PAPER.md).

## Configure

Configs are layers that deep-merge in order: later files override earlier
keys and CLI flags override everything.

- `base.toml`, the shared recipe: a 0.6B generator against a 1.7B judge on
  4 GPUs of one node.
- `multiplication.toml`, the 4-digit multiplication task. The dataset
  regenerates from its seed, so the example needs no external data.
- `countdown_code.toml`, `smiles_to_formula.toml`, `medical_abstract.toml`,
  and `grant_abstract.toml`, the task layers for the paper recipes in
  `configs/paper_experiments/adversarial/`.

Add layers with more `@ <file>` arguments, or override single values with
dotted flags (`--generator.max-steps 100`).

## Run

Inside a GPU allocation:

```bash
uv run adversarial-rl @ configs/adversarial/base.toml @ configs/adversarial/multiplication.toml
```

Or through SLURM:

```bash
sbatch configs/adversarial/smoke.sbatch
```

Everything a run produces lands in its `output_dir`: per-role checkpoints and
logs, the saved rollouts of every step, and the snapshots that pair the two
roles' checkpoints.

## Resume

Relaunch the same command with `--resume`:

```bash
uv run adversarial-rl @ configs/adversarial/base.toml @ configs/adversarial/multiplication.toml --resume
```

The run continues from the latest step at which both roles hold a complete
checkpoint. Models, optimizers, and counters restore exactly, and metrics keep
flowing into the same wandb run. When enabled, the replay buffer reseeds from
the saved rollouts and the EMA target reloads from its saved shadow.
Half-written checkpoints are skipped.
