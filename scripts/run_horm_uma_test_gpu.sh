#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(pwd)}"
cd "$REPO_ROOT"

source ./setup_cuda.sh >/dev/null

export FAIRCHEM_CACHE_DIR="$REPO_ROOT/.cache/fairchem"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

uv run python main.py \
  --config configs/uma/training_release/horm_uma_test.yaml \
  job.device_type=CUDA \
  job.run_dir=outputs/horm_uma_test_from_scratch_gpu \
  job.run_name=horm_uma_test_gpu
