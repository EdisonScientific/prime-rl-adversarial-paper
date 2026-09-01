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

# Usage (from the repo root): sbatch configs/adversarial/smoke.sbatch
# Extra CLI args pass through, including additional `@ <file>` config layers.
set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
set -a
[ -f .env ] && source .env
set +a
export TORCH_BLAS_PREFER_CUBLASLT=1
export VLLM_ENGINE_READY_TIMEOUT_S=1200
export HF_HOME="${HF_HOME:-$HOME/data/cache/huggingface}"
export WANDB_ENTITY="${WANDB_ENTITY:-edison-scientific}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

uv run --no-sync --extra flash-attn adversarial-rl \
  @ configs/adversarial/base.toml \
  @ configs/adversarial/multiplication.toml \
  "$@"
