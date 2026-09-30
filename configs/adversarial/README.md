# Adversarial RL

A generator and a judge train against each other. The generator learns to
produce answers the judge accepts. The judge learns to tell generated answers
from references. One orchestrator drives both: it generates rollouts, scores
them with the served judge, and feeds each side its training batch.

The entrypoint is `adversarial-rl`. It launches everything a run needs: the
orchestrator, and a trainer and an inference server per role.

## Configure

Config files are layers that deep-merge in order. Later files override earlier
keys. CLI flags override everything.

- `base.toml`, the shared recipe: Qwen3-0.6B generator vs Qwen3-1.7B judge,
  IPO loss, batch 128 with 8 rollouts per example, judge update gate at 0.75,
  checkpoints every 100 steps, both roles on 4 GPUs of one node.
- `multiplication.toml`, the 4-digit multiplication task. The dataset
  regenerates from its seed, so the example needs no external data.
- task env layers for the paper runs (`countdown_code.toml`,
  `smiles_to_formula.toml`, `medical_abstract.toml`, `grant_abstract.toml`):
  dataset args and run names, stacked under the experiment recipes in
  `experiments/`.

Add layers for other tasks or deployments with more `@ <file>` arguments, or
override single values with dotted flags (`--generator.max-steps 100`). The
paper's experiment recipes live in `experiments/`, and the SFT baselines they
are compared against in `configs/sft_baseline_experiments/`.

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
checkpoint. Models, optimizers, and step counters restore exactly, and metrics
keep flowing into the same wandb run. When enabled, the judge replay buffer
reseeds from the saved rollouts and the EMA target reloads from its saved
shadow. Half-written checkpoints from the crash are detected and skipped.
