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

# Usage (from the repo root): sbatch configs/adversarial/test/smoke_pair.sbatch <spec1> <spec2>
# Runs two 4-GPU test specs side by side on one 8-GPU node. Specs use the same
# `env[+overlay...]` syntax as smoke.sbatch. The two specs must resolve to
# disjoint inference ports — the pairwise overlay moves to 8002/8003, so e.g.
# `multiplication+lora multiplication+pairwise+lora` packs cleanly.
set -euo pipefail
SPEC1="$1"
SPEC2="$2"

cd "${SLURM_SUBMIT_DIR}"
[ -f .env ] && source .env
export TORCH_BLAS_PREFER_CUBLASLT=1
export VLLM_ENGINE_READY_TIMEOUT_S=1200
export HF_HOME="${HF_HOME:-$HOME/data/cache/huggingface}"
export WANDB_ENTITY="${WANDB_ENTITY:-edison-scientific}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

launch_args() {
  local spec="$1"
  local args=(@ "configs/adversarial/test/base.toml")
  for part in ${spec//+/ }; do
    args+=(@ "configs/adversarial/test/${part}.toml")
  done
  if [[ "${spec}" == *+* ]]; then
    local run_name="adversarial-test-${spec//+/-}"
    args+=(--output-dir "outputs/${run_name}" --generator.wandb.name "${run_name}")
  fi
  printf '%s\n' "${args[@]}"
}

mapfile -t ARGS1 < <(launch_args "${SPEC1}")
mapfile -t ARGS2 < <(launch_args "${SPEC2}")

CUDA_VISIBLE_DEVICES=0,1,2,3 uv run --no-sync --extra flash-attn adversarial-rl "${ARGS1[@]}" "${@:3}" \
  > "slurm-adv-${SPEC1//+/-}-${SLURM_JOB_ID}.log" 2>&1 &
PID1=$!
CUDA_VISIBLE_DEVICES=4,5,6,7 uv run --no-sync --extra flash-attn adversarial-rl "${ARGS2[@]}" "${@:3}" \
  > "slurm-adv-${SPEC2//+/-}-${SLURM_JOB_ID}.log" 2>&1 &
PID2=$!

# Fail the job if either run fails; SLURM then tears down the survivor.
wait "$PID1"
wait "$PID2"
