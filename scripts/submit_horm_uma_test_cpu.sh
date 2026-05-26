#!/usr/bin/env bash
#SBATCH --job-name=horm_uma_test_cpu
#SBATCH --account=nvr_qualg_lmbm
#SBATCH --partition=cpu
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=outputs/slurm_%x_%j.log
#SBATCH --error=outputs/slurm_%x_%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"

source ./setup_cuda.sh >/dev/null

export FAIRCHEM_CACHE_DIR="$REPO_ROOT/.cache/fairchem"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

uv run python main.py \
  --config configs/uma/training_release/horm_uma_test.yaml
