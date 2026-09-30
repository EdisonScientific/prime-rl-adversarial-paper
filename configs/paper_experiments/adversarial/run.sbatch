#!/bin/bash
#SBATCH --job-name=adv-exp
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --partition=gpu
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm-adv-exp-%j.log

# Usage (from the repo root): sbatch configs/paper_experiments/adversarial/run.sbatch <env> <experiment>
# where env is a test env overlay (e.g. smiles_to_formula) and experiment is a
# file under configs/paper_experiments/adversarial/. Files deep-merge onto base.toml
# in order. Each role takes one node (generator first).
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
  @ "configs/paper_experiments/adversarial/${EXPERIMENT}.toml" \
  "${@:3}"
