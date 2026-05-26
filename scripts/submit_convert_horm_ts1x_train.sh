#!/usr/bin/env bash
#SBATCH --job-name=convert_horm_ts1x_train
#SBATCH --account=nvr_qualg_lmbm
#SBATCH --partition=cpu
#SBATCH --time=04:00:00
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

INPUT="data/kagglehub/datasets/yunhonghan/hessian-dataset-for-optimizing-reactive-mliphorm/versions/6/ts1x_hess_train.lmdb"
OUTPUT="data/horm_atomic/ts1x_hess_train.atomic.lmdb"
METADATA="data/horm_atomic/ts1x_hess_train.atomic.metadata.npz"

if [[ ! -f "$INPUT" ]]; then
  echo "Missing input LMDB: $INPUT" >&2
  exit 1
fi

if [[ -e "$OUTPUT" || -e "$METADATA" ]]; then
  echo "Refusing to overwrite existing output: $OUTPUT or $METADATA" >&2
  exit 1
fi

echo "Starting HORM ts1x train conversion at $(date --iso-8601=seconds)"
echo "Repo: $REPO_ROOT"
echo "Input: $INPUT"
echo "Output: $OUTPUT"

.venv/bin/python tools/convert_hip_lmdb_to_atomic_lmdb.py \
  --input "$INPUT" \
  --output "$OUTPUT" \
  --metadata "$METADATA" \
  --dataset-name ts1x_train \
  --map-size-gb "${MAP_SIZE_GB:-128}" \
  --commit-every "${COMMIT_EVERY:-10000}" \
  --progress-every "${PROGRESS_EVERY:-50000}"

echo "Finished HORM ts1x train conversion at $(date --iso-8601=seconds)"
