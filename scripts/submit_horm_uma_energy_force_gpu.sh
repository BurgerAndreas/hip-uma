#!/usr/bin/env bash
#SBATCH --job-name=horm_uma_ef_gpu
#SBATCH --account=nvr_qualg_lmbm
#SBATCH --partition=batch_singlenode
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=0
#SBATCH --output=outputs/slurm_%x_%j.log
#SBATCH --error=outputs/slurm_%x_%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"

source ./setup_cuda.sh >/dev/null

export FAIRCHEM_CACHE_DIR="$REPO_ROOT/.cache/fairchem"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

uv run python main.py \
  --config configs/uma/training_release/horm_uma_energy_force.yaml \
  steps="${HORM_UMA_STEPS:-1000}" \
  batch_size="${HORM_UMA_BATCH_SIZE:-8}" \
  eval_every_n_steps="${HORM_UMA_EVAL_EVERY:-250}" \
  checkpoint_every_n_steps="${HORM_UMA_CHECKPOINT_EVERY:-250}" \
  bf16="${HORM_UMA_BF16:-False}"
