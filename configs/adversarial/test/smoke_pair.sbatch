#!/bin/bash
#SBATCH --job-name=adv-smoke-pair
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --partition=gpu
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm-adv-smoke-pair-%j.log

# Usage (from the repo root): sbatch configs/adversarial/test/smoke_pair.sbatch <env1> <env2>
# Runs two 4-GPU test configs side by side on one 8-GPU node. The two configs
# must use disjoint inference ports (e.g. 8000/8001 and 8002/8003).
set -euo pipefail
ENV1="$1"
ENV2="$2"

cd "${SLURM_SUBMIT_DIR}"
[ -f .env ] && source .env
export TORCH_BLAS_PREFER_CUBLASLT=1
export VLLM_ENGINE_READY_TIMEOUT_S=1200
export HF_HOME="${HF_HOME:-$HOME/data/cache/huggingface}"
export WANDB_ENTITY="${WANDB_ENTITY:-edison-scientific}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

CUDA_VISIBLE_DEVICES=0,1,2,3 uv run --no-sync --extra flash-attn adversarial-rl @ "configs/adversarial/test/${ENV1}.toml" \
  > "slurm-adv-${ENV1}-${SLURM_JOB_ID}.log" 2>&1 &
PID1=$!
CUDA_VISIBLE_DEVICES=4,5,6,7 uv run --no-sync --extra flash-attn adversarial-rl @ "configs/adversarial/test/${ENV2}.toml" \
  > "slurm-adv-${ENV2}-${SLURM_JOB_ID}.log" 2>&1 &
PID2=$!

# Fail the job if either run fails; SLURM then tears down the survivor.
wait "$PID1"
wait "$PID2"
