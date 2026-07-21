#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=outputs/slurm_%x_%j.log
#SBATCH --error=outputs/slurm_%x_%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"
source scripts/load_local_env.sh

if [[ $# -lt 3 ]]; then
    echo "Usage: scripts/submit_with_env.sh eval JOB_NAME $0 CHECKPOINT {predict|autograd} OUTPUT_CSV [MAX_SAMPLES] [DEVICE]" >&2
    exit 2
fi

CHECKPOINT=$1
HESSIAN_METHOD=$2
OUTPUT_CSV=$3
MAX_SAMPLES=${4:-10000}
DEVICE=${5:-cuda}

source setup_cuda.sh

uv run python tools/eval_horm_uma.py \
    --checkpoint "$CHECKPOINT" \
    --hessian-method "$HESSIAN_METHOD" \
    --max-samples "$MAX_SAMPLES" \
    --device "$DEVICE" \
    --output "$OUTPUT_CSV"
