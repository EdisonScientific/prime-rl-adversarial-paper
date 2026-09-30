#!/bin/bash
#SBATCH --job-name=sft-baseline
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-sft-baseline-%j.log

# Usage (from the repo root): sbatch configs/sft_baseline_experiments/sft.sbatch <config>
# where config is a file name under configs/sft_baseline_experiments/ (e.g. multiplication).
set -euo pipefail
CONFIG_NAME="$1"

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

uv run --no-sync sft @ configs/sft_baseline_experiments/base.toml @ "configs/sft_baseline_experiments/${CONFIG_NAME}.toml" "${@:2}"
