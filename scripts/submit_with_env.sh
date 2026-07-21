#!/usr/bin/env bash
# Submit a repo job using cluster settings from the local .env file.

set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 {gpu|eval|cpu} JOB_NAME SCRIPT [SCRIPT_ARGS...]" >&2
  exit 2
fi

PROFILE=$1
JOB_NAME=$2
SCRIPT=$3
shift 3

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
source scripts/load_local_env.sh

case "$PROFILE" in
  gpu) PARTITION="${SLURM_GPU_PARTITION:?Set SLURM_GPU_PARTITION in .env}" ;;
  eval) PARTITION="${SLURM_EVAL_PARTITION:-${SLURM_GPU_PARTITION:?Set SLURM_EVAL_PARTITION or SLURM_GPU_PARTITION in .env}}" ;;
  cpu) PARTITION="${SLURM_CPU_PARTITION:?Set SLURM_CPU_PARTITION in .env}" ;;
  *) echo "Unknown profile '$PROFILE'; expected gpu, eval, or cpu" >&2; exit 2 ;;
esac

SBATCH_ARGS=(--partition="$PARTITION" --job-name="$JOB_NAME")
if [[ -n "${SLURM_ACCOUNT:-}" ]]; then
  SBATCH_ARGS+=(--account="$SLURM_ACCOUNT")
fi

if [[ "${HIP_UMA_SBATCH_DRY_RUN:-0}" == "1" ]]; then
  printf 'sbatch'
  printf ' %q' "${SBATCH_ARGS[@]}" "$SCRIPT" "$@"
  printf '\n'
  exit 0
fi

exec sbatch "${SBATCH_ARGS[@]}" "$SCRIPT" "$@"
