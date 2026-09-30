# SFT baselines

The paper's SFT baselines, one config per task, layered over the shared
`base.toml`. Each trains the task's generator model on the same data the
adversarial runs use, with online evaluation against the same verifiable
scoring rules (the `environments/tasksets` package).

Run through SLURM:

```bash
sbatch configs/sft_baseline_experiments/sft.sbatch multiplication
```

or directly, inside a GPU allocation:

```bash
uv run sft @ configs/sft_baseline_experiments/base.toml @ configs/sft_baseline_experiments/multiplication.toml
```

Tasks: `multiplication`, `countdown`, `smiles2formula`, `grant_abstracts`.
Extra CLI args pass through in both cases.
