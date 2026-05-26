#!/usr/bin/env bash
#SBATCH --job-name=horm_uma_hess_full
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

RUN_DIR="${HORM_UMA_HESSIAN_RUN_DIR:-$REPO_ROOT/outputs/horm_uma_hessian_full}"
mkdir -p "$RUN_DIR"

RUN_ID_FILE="$RUN_DIR/run_id"
if [[ -n "${HORM_UMA_HESSIAN_RUN_ID:-}" ]]; then
  RUN_ID="$HORM_UMA_HESSIAN_RUN_ID"
elif [[ -f "$RUN_ID_FILE" ]]; then
  RUN_ID="$(<"$RUN_ID_FILE")"
else
  RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-${SLURM_JOB_ID:-manual}"
fi
printf '%s\n' "$RUN_ID" > "$RUN_ID_FILE"

printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${SLURM_JOB_ID:-manual}" \
  >> "$RUN_DIR/slurm_job_ids.txt"

WANDB_ENABLED="${HORM_UMA_WANDB_ENABLED:-1}"
WANDB_ENTITY="${HORM_UMA_WANDB_ENTITY:-fairchem}"
WANDB_PROJECT="${HORM_UMA_WANDB_PROJECT:-uma}"
WANDB_ARGS=()
if [[ "$WANDB_ENABLED" != "0" ]]; then
  export WANDB_RESUME=allow
  WANDB_ARGS=(
    "job.debug=False"
    "+job.logger._target_=fairchem.core.common.logger.WandBSingletonLogger.init_wandb"
    "+job.logger._partial_=true"
    "+job.logger.entity=${WANDB_ENTITY}"
    "+job.logger.project=${WANDB_PROJECT}"
    "+job.logger.group=${HORM_UMA_WANDB_GROUP:-horm_uma_hessian_full}"
    "+job.logger.job_type=${HORM_UMA_WANDB_JOB_TYPE:-training}"
  )
fi

TRAIN_SECONDS="${HORM_UMA_HESSIAN_TRAIN_SECONDS:-13800}"  # 3h50m inside a 4h allocation.
set +e
timeout --signal=TERM --kill-after=300s "$TRAIN_SECONDS" \
  uv run python main.py \
    --config configs/uma/training_release/horm_uma_hessian_full.yaml \
    "job.run_dir=$RUN_DIR" \
    "job.run_name=horm_uma_hessian_full" \
    "+job.timestamp_id=$RUN_ID" \
    "batch_size=${HORM_UMA_HESSIAN_BATCH_SIZE:-128}" \
    "steps=${HORM_UMA_HESSIAN_STEPS:-200000}" \
    "eval_max_batches=${HORM_UMA_HESSIAN_MAX_EVAL_BATCHES:-4}" \
    "runner.evaluate_every_n_steps=${HORM_UMA_HESSIAN_EVAL_EVERY:-1000}" \
    "runner.callbacks.0.checkpoint_every_n_steps=${HORM_UMA_HESSIAN_CHECKPOINT_EVERY:-500}" \
    "${WANDB_ARGS[@]}"
status=$?
set -e

if [[ "$status" -eq 0 ]]; then
  touch "$RUN_DIR/TRAINING_COMPLETE"
  echo "Training completed for run_id=$RUN_ID"
  exit 0
fi

if [[ "$status" -eq 124 || "$status" -eq 137 || "$status" -eq 143 ]]; then
  echo "Training slice ended with status=$status; submitting continuation for run_id=$RUN_ID"
  next_job="$(
    sbatch \
      --export=ALL,HORM_UMA_HESSIAN_RUN_ID="$RUN_ID",HORM_UMA_HESSIAN_RUN_DIR="$RUN_DIR" \
      scripts/submit_horm_uma_hessian_full_gpu.sh
  )"
  echo "$next_job"
  printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$next_job" \
    >> "$RUN_DIR/slurm_resubmissions.txt"
  exit 0
fi

echo "Training failed with status=$status; not resubmitting" >&2
exit "$status"
