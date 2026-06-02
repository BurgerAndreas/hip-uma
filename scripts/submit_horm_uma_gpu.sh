#!/usr/bin/env bash
#SBATCH --account=nvr_qualg_lmbm
#SBATCH --partition=batch_singlenode
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=128G
#SBATCH --output=outputs/slurm_%x_%j.log
#SBATCH --error=outputs/slurm_%x_%j.err

set -euo pipefail

if [[ "$#" -lt 1 ]]; then
  echo "Usage: sbatch --job-name=<name> $0 <config.yaml> [hydra overrides...]" >&2
  exit 2
fi

CONFIG="$1"
shift
EXTRA_ARGS=("$@")

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"

source ./setup_cuda.sh >/dev/null

export FAIRCHEM_CACHE_DIR="$REPO_ROOT/.cache/fairchem"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

mkdir -p outputs/slurm_run_ids
JOB_KEY="${SLURM_JOB_NAME:-$(basename "$CONFIG" .yaml)}"
RUN_ID_FILE="outputs/slurm_run_ids/${JOB_KEY}.run_id"
HAS_TIMESTAMP_OVERRIDE=0
for arg in "${EXTRA_ARGS[@]}"; do
  if [[ "$arg" == job.timestamp_id=* || "$arg" == +job.timestamp_id=* ]]; then
    HAS_TIMESTAMP_OVERRIDE=1
    break
  fi
done

TIMESTAMP_ARG=()
if [[ "$HAS_TIMESTAMP_OVERRIDE" -eq 0 ]]; then
  if [[ -f "$RUN_ID_FILE" ]]; then
    RUN_ID="$(<"$RUN_ID_FILE")"
  else
    RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-${JOB_KEY}"
    printf '%s\n' "$RUN_ID" > "$RUN_ID_FILE"
  fi
  TIMESTAMP_ARG=("+job.timestamp_id=$RUN_ID")
fi

printf '%s %s %s %s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  "${SLURM_JOB_ID:-manual}" \
  "$CONFIG" \
  "${EXTRA_ARGS[*]}" \
  >> "outputs/slurm_run_ids/${JOB_KEY}.jobs"

TRAIN_SECONDS="${TRAIN_SECONDS:-13800}"  # 3h50m inside a 4h allocation.
set +e
timeout --signal=TERM --kill-after=300s "$TRAIN_SECONDS" \
  uv run python main.py \
    --config "$CONFIG" \
    "${TIMESTAMP_ARG[@]}" \
    "${EXTRA_ARGS[@]}"
status=$?
set -e

if [[ "$status" -eq 0 ]]; then
  echo "Training command completed for $JOB_KEY"
  exit 0
fi

if [[ "$status" -eq 124 || "$status" -eq 137 || "$status" -eq 143 ]]; then
  echo "Training slice ended with status=$status; submitting continuation for $JOB_KEY"
  next_job="$(sbatch --job-name="$JOB_KEY" "$0" "$CONFIG" "${EXTRA_ARGS[@]}")"
  echo "$next_job"
  printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$next_job" \
    >> "outputs/slurm_run_ids/${JOB_KEY}.resubmissions"
  exit 0
fi

echo "Training failed with status=$status; not resubmitting" >&2
exit "$status"
