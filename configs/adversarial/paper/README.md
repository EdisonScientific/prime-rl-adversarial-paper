# Paper adversarial configs

This directory contains the adversarial-training configs used for the paper experiments. They are organized by environment so each reported run family can be inspected without searching through cluster job names.

The configs were copied from the saved experiment configs on the powerline cluster. Two schema-alignment edits were applied so they parse with this repository:

- HF trainer blocks set `attn = "flash_attention_3"` explicitly. This preserves the attention implementation used by these runs and avoids the current `auto` default selecting FA4 with `impl = "hf"` on newer GPUs.
- Legacy `[*.inference.vllm_extra]` blocks were moved under `[*.inference.vllm]`, which is the current pass-through location for vLLM arguments.

No optimization or reduction dtype settings were changed.

## Directory layout

### `multiplication/`

Four-digit multiplication experiments with a `Qwen/Qwen3-0.6B` generator. The directory includes exact-only, exact gated by judge, and judge-only runs with judge sizes from 0.6B to 8B.

### `countdown_code/`

Code-based arithmetic-expression synthesis experiments. The directory includes exact-only, exact gated by judge, and judge-only runs with judge sizes from 1.7B to 32B.

### `smiles_to_formula/`

SMILES-to-formula prediction experiments using the inverted Ether0/MolRQA setup. The directory includes exact-only, exact gated by judge, and judge-only runs with 14B and 32B judges.

### `nih_grant_abstract/`

NIH grant abstract generation experiments. These configs capture the retained paper grant-abstract run families, including 32B-generator/235B-judge and 32B-generator/32B-judge variants with the reasoning-budget parser setup.

## Notes for reproduction

The configs intentionally retain the original cluster paths for output directories, local model snapshots, and prepared datasets. Before running outside the powerline environment, update model and dataset paths to the corresponding local or Hugging Face locations.

Typical usage:

```bash
uv run adversarial-rl --config configs/adversarial/paper/<task>/<config>.toml
```
