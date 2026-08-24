# Adversarial test configs

One config per environment, meant to verify every env runs end to end on the
cluster. The multiplication configs carry the known-good reference recipe
(Qwen3-0.6B generator, Qwen3-1.7B judge, 3000 steps, batch 128, 8 rollouts per
example, judge gate at 0.75); the other envs are small smoke-scale configs.
Runs log to the `testing-adversarial` wandb project.

```bash
sbatch configs/adversarial/test/smoke.sbatch                  # multiplication
sbatch configs/adversarial/test/smoke.sbatch countdown_code   # any other env
```

or directly, inside an allocation:

```bash
uv run adversarial-rl @ configs/adversarial/test/multiplication.toml
```

Two 4-GPU configs fit on one 8-GPU node if they use disjoint inference ports:

```bash
sbatch configs/adversarial/test/smoke_pair.sbatch multiplication_lora multiplication_pairwise_lora
```

The grant_abstract config points at the prepared JSONL on the cluster; adjust
`dataset_path` if it lives elsewhere.

Pairwise judging (the judge sees generated and reference in one prompt and picks
the reference; judge batch equals the generator batch) is enabled per config with
`judge_objective = "pairwise"` plus `pairwise = true` in the judge env args.

The `*_lora.toml` variants train rank-16 LoRA adapters on both roles
(`[<role>.trainer.model.lora]`) with learning rates 100x the full-finetune
recipe; everything else matches the non-LoRA configs.
