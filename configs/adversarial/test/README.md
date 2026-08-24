# Adversarial env smoke tests

One config per environment: Qwen3-0.6B generator and judge, 20 steps, small
datasets, judge reward mode, no wandb. Meant to verify every env runs end to
end on the cluster:

```bash
sbatch configs/adversarial/test/smoke.sbatch                  # multiplication
sbatch configs/adversarial/test/smoke.sbatch countdown_code   # any other env
```

or directly, inside an allocation:

```bash
uv run adversarial-rl @ configs/adversarial/test/multiplication.toml
```

Runs log to the `testing-adversarial` wandb project.

The grant_abstract config points at the prepared JSONL on the cluster; adjust
`dataset_path` if it lives elsewhere.

Pairwise judging (the judge sees generated and reference in one prompt and picks
the reference; judge batch equals the generator batch) is enabled per config with:

```toml
judge_objective = "pairwise"
```

plus `pairwise = true` in the judge env args and `batch_size = 32` under
`[judge.orchestrator]`.
