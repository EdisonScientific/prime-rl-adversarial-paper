# Adversarial RL configs

Minimal runnable example of the adversarial generator/judge loop. Config files
deep-merge in order (later files override earlier keys):

- `base.toml` — the shared recipe: Qwen3-0.6B generator vs Qwen3-1.7B judge,
  IPO loss, batch 128 with 8 rollouts per example, judge update gate at 0.75,
  filesystem weight broadcast, both roles on 4 GPUs of one node.
- `multiplication.toml` — the 4-digit multiplication task. The dataset
  regenerates deterministically from its seed, so the example needs no
  external data.

Run through SLURM:

```bash
sbatch configs/adversarial/smoke.sbatch
```

or directly, inside an allocation:

```bash
uv run adversarial-rl @ configs/adversarial/base.toml @ configs/adversarial/multiplication.toml
```

Extra CLI args pass through in both cases — dotted overrides
(`--generator.max-steps 100`) or additional `@ <file>` layers for other tasks
and deployments.
