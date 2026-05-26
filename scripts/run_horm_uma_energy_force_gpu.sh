#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(pwd)}"
cd "$REPO_ROOT"

source ./setup_cuda.sh >/dev/null

export FAIRCHEM_CACHE_DIR="$REPO_ROOT/.cache/fairchem"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

uv run python main.py \
  --config configs/uma/training_release/horm_uma_energy_force.yaml \
  steps="${HORM_UMA_STEPS:-1000}" \
  batch_size="${HORM_UMA_BATCH_SIZE:-8}" \
  eval_every_n_steps="${HORM_UMA_EVAL_EVERY:-250}" \
  checkpoint_every_n_steps="${HORM_UMA_CHECKPOINT_EVERY:-250}" \
  bf16="${HORM_UMA_BF16:-False}"
