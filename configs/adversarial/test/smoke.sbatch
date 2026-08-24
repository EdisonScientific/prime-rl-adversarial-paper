#!/bin/bash
#SBATCH --job-name=adv-smoke
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --partition=gpu
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=slurm-adv-smoke-%j.log

# Usage (from the repo root): sbatch configs/adversarial/test/smoke.sbatch [spec]
# where spec is an env overlay (multiplication (default), countdown_code,
# smiles_to_formula, grant_abstract), optionally stacked with feature overlays
# via `+` (e.g. multiplication+pairwise+lora). Files deep-merge onto base.toml
# in order; stacked runs get a derived output dir and wandb name.
set -euo pipefail
SPEC="${1:-multiplication}"

cd "${SLURM_SUBMIT_DIR}"
[ -f .env ] && source .env
export TORCH_BLAS_PREFER_CUBLASLT=1
export VLLM_ENGINE_READY_TIMEOUT_S=1200
export HF_HOME="${HF_HOME:-$HOME/data/cache/huggingface}"
export WANDB_ENTITY="${WANDB_ENTITY:-edison-scientific}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

CONFIG_ARGS=(@ "configs/adversarial/test/base.toml")
for part in ${SPEC//+/ }; do
  CONFIG_ARGS+=(@ "configs/adversarial/test/${part}.toml")
done
NAME_ARGS=()
if [[ "${SPEC}" == *+* ]]; then
  RUN_NAME="adversarial-test-${SPEC//+/-}"
  NAME_ARGS=(--output-dir "outputs/${RUN_NAME}" --generator.wandb.name "${RUN_NAME}")
fi

uv run --no-sync --extra flash-attn adversarial-rl "${CONFIG_ARGS[@]}" "${NAME_ARGS[@]}"
