# Adversarial experiments

The paper's experiment configs, one directory per experiment. Files deep-merge
in order, composed as `base.toml @ <env>.toml @ <experiment>.toml
[@ <overlay>.toml ...]`.

Launch through `run.sbatch` (exclusive full nodes) or `run_shared.sbatch`
(4 GPUs on a shared node, so stacked experiments must resolve to disjoint
server and broadcast ports). Extra CLI args pass through, including additional
`@ <file>` layers.

## Experiments

| Experiment | Stack | Nodes | Launch |
|---|---|---|---|
| multiplication 0.6B gen vs 0.6B judge (family base) | `multiplication/06b06b` | half node | `sbatch configs/adversarial/experiments/run_shared.sbatch multiplication multiplication/06b06b` |
| + 1.7B or 4B judge | + `multiplication/1p7bjudge` or `multiplication/4bjudge` | half node | append the judge overlay |
| + exact reward only | + `multiplication/exact` | half node | append `@ .../multiplication/exact.toml` |
| + judge x exact reward | + `multiplication/judgexexact` | half node | append `@ .../multiplication/judgexexact.toml` |
| countdown 1.7B gen vs 1.7B judge (family base) | `countdown/1p7b1p7b` | half node | `sbatch configs/adversarial/experiments/run_shared.sbatch countdown_code countdown/1p7b1p7b` |
| + 4B or 8B judge | + `countdown/4bjudge` or `countdown/8bjudge` | half node | append the judge overlay |
| + 14B judge | + `countdown/14bjudge` | 1 (`--nodes=1`) | append the overlay on `run.sbatch` |
| + 32B judge | + `countdown/32bjudge` | 2 | append the overlay on `run.sbatch` |
| + exact reward only | + `countdown/exact` | half node | append `@ .../countdown/exact.toml` |
| + judge x exact reward | + `countdown/judgexexact` | half node | append `@ .../countdown/judgexexact.toml` |
| smiles 8B gen vs 8B judge (family base) | `smiles/8b8b` | half node | `sbatch configs/adversarial/experiments/run_shared.sbatch smiles_to_formula smiles/8b8b` |
| + 14B judge | + `smiles/14bjudge` | 1 (`--nodes=1`) | append the overlay on `run.sbatch` |
| + 32B judge | + `smiles/32bjudge` | 2 | append the overlay on `run.sbatch` |
| + exact reward only | + `smiles/exact` | 1 (`--nodes=1`) | append `@ .../smiles/exact.toml` on `run.sbatch` |
| + judge x exact reward | + `smiles/judgexexact` | half node | append `@ .../smiles/judgexexact.toml` |
| medical abstracts 8B gen vs learned 14B judge | `medical/8b14b` | 1 (`--nodes=1`) | `sbatch --nodes=1 configs/adversarial/experiments/run.sbatch medical_abstract medical/8b14b` |
| grant NIH 32B gen vs 235B judge, full finetune | `nih/32b235b` | 7 (`--nodes=7`) | `sbatch --nodes=7 configs/adversarial/experiments/run.sbatch grant_abstract nih/32b235b` |

Resume any experiment by appending `--resume` to its launch command.
