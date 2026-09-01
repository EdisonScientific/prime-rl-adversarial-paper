#!/bin/bash
#SBATCH --job-name=adv-exp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=48
#SBATCH --mem=350G
#SBATCH --time=10-00:00:00
#SBATCH --output=slurm-adv-exp-%j.log

# Usage (from the repo root): sbatch configs/adversarial/experiments/run_shared.sbatch <env> <experiment> [...]
# Like run.sbatch, but shares a node with other jobs instead of taking it
# exclusively; experiments must resolve to disjoint server and broadcast ports.
set -euo pipefail
ENV_NAME="$1"
EXPERIMENT="$2"

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
  @ "configs/adversarial/${ENV_NAME}.toml" \
  @ "configs/adversarial/experiments/${EXPERIMENT}.toml" \
  "${@:3}"
