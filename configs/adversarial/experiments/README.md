# Adversarial experiments

Experiment configs layered onto the test recipes: files deep-merge in order
(later files override earlier keys), composed as
`base.toml @ <env>.toml @ <experiment>.toml [@ <stacked>.toml ...]`.

Launch through `run.sbatch` (exclusive full nodes) or `run_shared.sbatch`
(4 GPUs on a shared node; experiments must resolve to disjoint server and
broadcast ports). Extra CLI args pass through, including additional `@ <file>`
layers.

## Experiments

| Experiment | Stack on `smiles_to_formula` / `grant_abstract` | Nodes | Launch |
|---|---|---|---|
| smiles 8B gen vs 32B judge, full finetune | `smiles2formula_8b32b` | 2 | `sbatch configs/adversarial/experiments/run.sbatch smiles_to_formula smiles2formula_8b32b` |
| + LoRA both roles | `smiles2formula_8b32b` + `smiles2formula_8b32b_lora` | 2 | append `@ configs/adversarial/experiments/smiles2formula_8b32b_lora.toml` |
| + exact reward only (frozen 0.6B judge) | `smiles2formula_8b32b` + `smiles2formula_8b_exact` | 1 | append `@ .../smiles2formula_8b_exact.toml` |
| + 8B judge | `smiles2formula_8b32b` + `smiles2formula_8b8b` | half node | `run_shared.sbatch` with the stacked layers |
| + 8B judge, judge x exact reward | ... + `smiles2formula_8b8b_both` | half node | stack both files on `run_shared.sbatch` |
| grant NIH 32B gen vs 235B judge, LoRA both roles | `grant_nih_32b235b_lora` | 6 (`--nodes=6`) | `sbatch --nodes=6 configs/adversarial/experiments/run.sbatch grant_abstract grant_nih_32b235b_lora` |
| grant NIH, LoRA generator + full-finetuned judge | `grant_nih_32b235b_genlora` | 8 (`--nodes=8`) | `sbatch --nodes=8 configs/adversarial/experiments/run.sbatch grant_abstract grant_nih_32b235b_genlora` |
| deslopify 8B gen vs 8B judge, full finetune | `deslopify_8b8b` | 1 (`--nodes=1`) | `sbatch --nodes=1 configs/adversarial/experiments/run.sbatch deslopify_abstract deslopify_8b8b` |
| deslopify 14B gen vs 14B judge, full finetune | `deslopify_14b14b` | 2 | `sbatch configs/adversarial/experiments/run.sbatch deslopify_abstract deslopify_14b14b` |

Smoke any experiment before a long run: append
`--generator.max-steps 2 --judge.max-steps 2 --output-dir outputs/<smoke-dir> --generator.wandb.name <name>-smoke`
and a short `--time`. The genlora judge trainer's three-node memory fit and its NCCL
broadcast to replica engines have not been smoke-verified.

## Constraints these configs encode

- LoRA roles must broadcast weights via the filesystem (vLLM loads adapters
  from disk); full-finetuned roles use NCCL, including to replica engines
  (the upstream receiver handles the rank fan-out). Roles co-located on one
  node need distinct broadcast and server ports.
- The 235B MoE judge: LoRA adapters must target attention only (expert
  adapters do not load in vLLM), and a full-finetuned judge must serve at
  TP-4 (at TP-8 the expert intermediate size violates the CUTLASS MoE
  kernel's 128-multiple requirement; LoRA-enabled serving avoids that kernel).
- `thinking_token_budget` requires the serving engine to run with a reasoning
  parser (`reasoning_parser = "qwen3"` for hybrid Qwen3 models, which parser
  auto-detection leaves unset).
- Cold model loads read from Lustre at ~200MB/s (the 235B judge takes ~27
  minutes on a node that has not served it before; warm page cache cuts that
  to ~11). Boot cost is negligible for long runs; for rapid restart loops,
  pre-stage weights to node-local disk and point ``model.name`` at the local
  path, as the legacy runs did with ``/tmp/qwen3-32b-hf``.
- Generation dominates step time; the fast layout gives the generator a
  dedicated TP-8 inference node and a `max_model_len` just above
  prompt + completion so the KV cache holds more concurrent sequences.
