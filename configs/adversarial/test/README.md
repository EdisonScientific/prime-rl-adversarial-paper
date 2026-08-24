# Adversarial test configs

Layered TOMLs, deep-merged in order by the config loader (later files override
earlier keys; nested tables merge recursively):

- `base.toml` — the shared recipe validated on 4-digit multiplication
  (Qwen3-0.6B generator, Qwen3-1.7B judge, 3000 steps, batch 128, 8 rollouts
  per example, judge gate at 0.75, filesystem weight broadcast).
- env overlays — `multiplication.toml`, `countdown_code.toml`,
  `smiles_to_formula.toml`, `grant_abstract.toml`: dataset args, run names,
  and (where the env has an eval split) the eval block.
- feature overlays — `pairwise.toml` (pairwise judging on ports 8002/8003),
  `lora.toml` (rank-16 LoRA on both roles, 100x LR), `short.toml` (100 steps).

Launch through SLURM with `env[+overlay...]` specs:

```bash
sbatch configs/adversarial/test/smoke.sbatch multiplication
sbatch configs/adversarial/test/smoke.sbatch multiplication+pairwise+lora
sbatch configs/adversarial/test/smoke.sbatch countdown_code+short
```

or directly, inside an allocation:

```bash
uv run adversarial-rl @ configs/adversarial/test/base.toml @ configs/adversarial/test/multiplication.toml
```

Stacked sbatch specs get a derived output dir and wandb name
(`adversarial-test-<spec with dashes>`); when stacking manually, pass
`--output-dir` and `--generator.wandb.name` yourself so runs don't collide.

Two 4-GPU specs fit on one 8-GPU node if they resolve to disjoint inference
ports (the pairwise overlay moves to 8002/8003):

```bash
sbatch configs/adversarial/test/smoke_pair.sbatch multiplication+lora multiplication+pairwise+lora
```

Runs log to the `testing-adversarial` wandb project. The grant_abstract
overlay points at the prepared JSONL on the cluster; adjust `dataset_path` if
it lives elsewhere.
