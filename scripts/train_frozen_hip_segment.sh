#!/usr/bin/env bash
# Resume a frozen HIP experiment through bounded scheduler allocations.
# Supply scheduler settings and private paths through the environment.
set -euo pipefail

: "${PYTHON_BIN:?Set PYTHON_BIN}"
: "${CODE_ROOT:?Set CODE_ROOT}"
: "${DATA_ROOT:?Set DATA_ROOT}"
: "${PACKED_ROOT:?Set PACKED_ROOT}"
: "${RUN_ROOT:?Set RUN_ROOT}"
: "${LOSS_NAME:?Set LOSS_NAME to mae, block-rmse, or mass-block-rmse}"
: "${TARGET_STEPS:=2000000}"
: "${SEGMENT_SECONDS:=13200}"
: "${SEGMENT_INDEX:=0}"
: "${MAX_SEGMENTS:=128}"

case "$LOSS_NAME" in mae|block-rmse|mass-block-rmse) ;; *) exit 2 ;; esac
if (( SEGMENT_INDEX >= MAX_SEGMENTS )); then
    echo "Maximum allocation count reached; checkpoint remains resumable." >&2
    exit 2
fi
mkdir -p "$RUN_ROOT"
exec 9>"$RUN_ROOT/allocation.lock"
if ! flock -n 9; then
    echo "Another allocation owns this run." >&2
    exit 2
fi
if [[ -e "$RUN_ROOT/STOP" ]]; then
    echo "Stop requested; checkpoint remains resumable."
    exit 0
fi
if [[ -s "$RUN_ROOT/final_evaluation.json" ]]; then
    finished_step=$("$PYTHON_BIN" - "$RUN_ROOT/final_evaluation.json" <<'PYTHON'
import json, sys
from pathlib import Path
row = json.loads(Path(sys.argv[1]).read_text())
assert row["from_last_checkpoint"]
print(int(row["step"]))
PYTHON
)
    if (( finished_step >= TARGET_STEPS )); then
        echo "Training and final evaluation already complete at step $finished_step."
        exit 0
    fi
fi

# Queue one successor while the scheduler can still serve this allocation.
# A later controller outage cannot discard an already accepted dependency.
# Scheduler timeouts can still resume the last checkpoint. Handled trainer
# failures cancel their child explicitly instead of repeating a numerical error.
continuation_job=""
queue_successor() {
    if [[ -z "${CONTINUATION_SCRIPT:-}" ]]; then return 1; fi
    : "${SLURM_JOB_ID:?Automatic continuation requires a scheduler allocation}"
    if (( SEGMENT_INDEX + 1 >= MAX_SEGMENTS )); then return 1; fi
    local parent="" previous="" previous_restart="" restart="${SLURM_RESTART_COUNT:-0}"
    if [[ -s "$RUN_ROOT/continuation.record" ]]; then
        if read -r parent previous previous_restart <"$RUN_ROOT/continuation.record" &&
            [[ "$parent" == "$SLURM_JOB_ID" && "$previous" =~ ^[0-9]+$ ]]; then
            if [[ "${previous_restart:-0}" == "$restart" ]]; then
                continuation_job="$previous"
                echo "Reusing continuation $continuation_job for this allocation."
                return 0
            fi
            # A predecessor's earlier failed state can invalidate its dependency.
            # Replace that dependency after a scheduler requeue, without polling.
            timeout -k 5s 60s scancel "$previous" || true
        fi
    fi
    local response
    if response=$(SEGMENT_INDEX=$((SEGMENT_INDEX + 1)) timeout -k 5s 150s sbatch \
        --parsable --dependency="afterany:$SLURM_JOB_ID" --kill-on-invalid-dep=yes \
        --export=ALL "$CONTINUATION_SCRIPT"); then
        continuation_job="${response%%;*}"
        if [[ ! "$continuation_job" =~ ^[0-9]+$ ]]; then
            echo "The scheduler returned an invalid continuation id." >&2
            continuation_job=""
            return 1
        fi
        printf '%s %s %s\n' "$SLURM_JOB_ID" "$continuation_job" "$restart" >"$RUN_ROOT/continuation.tmp"
        mv "$RUN_ROOT/continuation.tmp" "$RUN_ROOT/continuation.record"
        printf '%s\n' "$continuation_job" >"$RUN_ROOT/next_job_id"
        echo "Queued continuation $continuation_job."
        return 0
    fi
    echo "Continuation submission failed; training can still save progress." >&2
    return 1
}
cancel_successor() {
    if [[ -n "$continuation_job" ]]; then
        timeout -k 5s 60s scancel "$continuation_job" || true
    fi
}
queue_successor || true

cd "$CODE_ROOT"
export PYTHONPATH="$CODE_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
args=(
    --data "$DATA_ROOT" --packed-data "$PACKED_ROOT" --output "$RUN_ROOT"
    --loss "$LOSS_NAME" --steps "$TARGET_STEPS" --duration "$SEGMENT_SECONDS"
    --weight-decay "${WEIGHT_DECAY:-0}"
    --lr "${LEARNING_RATE:-0.001}" --layers "${HEAD_LAYERS:-1}"
    --head-width-multiplier "${HEAD_WIDTH_MULTIPLIER:-1}"
    --max-batch "${GLOBAL_BATCH_CAP:-64}" --max-atoms "${GLOBAL_ATOM_TARGET:-2560}"
    --microbatch-pairs "${MICROBATCH_PAIRS:-524288}"
    --workers "${LOADER_WORKERS:-2}" --prefetch-factor "${PREFETCH_FACTOR:-8}" --window-shards 8
    --tf32 --backbone-precision fp16 --head-precision "${HEAD_PRECISION:-fp32}"
    --backbone-graph-version 4 --grouped-mole --fused-backbone
    --fused-rotations --block-so2 --projected-readout --compile-head
    --checkpoint-every "${CHECKPOINT_EVERY:-5000}" --validate-every "${VALIDATE_EVERY:-400000}" --log-every 100
    --refresh-data-every 0
    --stop-file "$RUN_ROOT/STOP"
)
if [[ -n "${VALIDATION_MANIFEST:-}" ]]; then args+=(--validation-manifest "$VALIDATION_MANIFEST"); fi
if [[ -n "${FINAL_VALIDATION_MANIFEST:-}" ]]; then args+=(--final-validation-manifest "$FINAL_VALIDATION_MANIFEST"); fi
if [[ -n "${DATASET_REVISION:-}" ]]; then args+=(--dataset-revision "$DATASET_REVISION"); fi
if [[ "${CUDA_GRAPHS:-0}" == 1 ]]; then args+=(--cuda-graphs --cuda-graph-cache "${CUDA_GRAPH_CACHE:-2}"); fi
if [[ -n "${FEATURE_CACHE_ROOT:-}" ]]; then
    args+=(--feature-cache "$FEATURE_CACHE_ROOT" --feature-cache-read-only)
fi
if [[ "${SCALED_LINEAR:-0}" == 1 ]]; then args+=(--scaled-linear); fi
if [[ "${FUSED_SCATTER:-0}" == 1 ]]; then args+=(--fused-scatter); fi
if [[ "${SEGMENTED_SCATTER:-0}" == 1 ]]; then args+=(--segmented-scatter); fi
if [[ "${FUSED_WIGNER:-0}" == 1 ]]; then args+=(--fused-wigner); fi
if [[ "${ASYNC_GRAD_CHECK:-0}" == 1 ]]; then args+=(--async-grad-check); fi
if [[ "${DEVICE_PREFETCH:-0}" == 1 ]]; then args+=(--device-prefetch); fi
if [[ "${THREADED_PREFETCH:-0}" == 1 ]]; then args+=(--threaded-prefetch); fi
if [[ "${FAST_PACKED_LOADER:-0}" == 1 ]]; then args+=(--fast-packed-loader); fi
if [[ "${TRANSLATION_PROJECTION:-0}" == 1 ]]; then args+=(--translation-projection); fi
if [[ "${HEAD_CHARGE_SPIN:-0}" == 1 ]]; then args+=(--head-charge-spin); fi
if [[ -n "${HEAD_EXPERTS:-}" ]]; then args+=(--head-experts "$HEAD_EXPERTS"); fi
if [[ -n "${LOSS_SCALE:-}" ]]; then args+=(--loss-scale "$LOSS_SCALE"); fi
if [[ "${DYNAMIC_LOSS_SCALE:-0}" == 1 ]]; then args+=(--dynamic-loss-scale); fi
if [[ -n "${LOSS_SCALE_GROWTH_INTERVAL:-}" ]]; then
    args+=(--loss-scale-growth-interval "$LOSS_SCALE_GROWTH_INTERVAL")
fi
if [[ "${PACKED_WINDOW_PREFETCH:-0}" == 1 ]]; then args+=(--packed-window-prefetch); fi
if [[ -n "${PACKED_WINDOW_MEMORY_GB:-}" ]]; then
    args+=(--packed-window-memory-gb "$PACKED_WINDOW_MEMORY_GB")
fi
if [[ "${FUSED_RADIAL:-0}" == 1 ]]; then args+=(--fused-radial); fi
if [[ -n "${RADIAL_TILE:-}" ]]; then args+=(--radial-tile "$RADIAL_TILE"); fi
if [[ "${SKINNY_READOUT:-0}" == 1 ]]; then args+=(--skinny-readout); fi
if [[ "${TUNED_EDGE_KERNELS:-0}" == 1 ]]; then args+=(--tuned-edge-kernels); fi
if [[ "${ALLOW_FP16_CACHE:-0}" == 1 ]]; then args+=(--allow-fp16-cache); fi
if [[ "${PACK_FEATURES_FP16:-0}" == 1 ]]; then args+=(--pack-features-fp16); fi
if [[ -n "${COMPILE_MODE:-}" ]]; then args+=(--compile-mode "$COMPILE_MODE"); fi
if [[ -n "${CODE_REVISION:-}" ]]; then args+=(--code-revision "$CODE_REVISION"); fi

stop_continuation=0
forward_signal() {
    kill -USR1 "$trainer_pid" 2>/dev/null || true
    wait "$trainer_pid"
}
"$PYTHON_BIN" -m fairchem.core.frozen_hip.train "${args[@]}" &
trainer_pid=$!
trap forward_signal USR1
trap 'stop_continuation=1; forward_signal' TERM
status=0
wait "$trainer_pid" || status=$?
if (( status >= 128 )); then
    status=0
    wait "$trainer_pid" || status=$?
fi
if (( status != 0 )); then
    cancel_successor
    rm -f "$RUN_ROOT/continuation.record"
    exit "$status"
fi
trap - USR1 TERM
if [[ -e "$RUN_ROOT/STOP" ]]; then
    cancel_successor
    echo "Stop requested; checkpoint remains resumable."
    exit 0
fi
if (( stop_continuation )); then
    echo "Allocation terminated; checkpoint and accepted continuation remain resumable."
    exit 0
fi

# The trainer writes this record only after it saves its last checkpoint.
step=$("$PYTHON_BIN" - "$RUN_ROOT/session.json" <<'PY'
import json, sys
from pathlib import Path
session = json.loads(Path(sys.argv[1]).read_text())
assert session["frozen_backbone_verified"]
print(int(session["step"]))
PY
)
if (( step >= TARGET_STEPS )); then
    if [[ -s "$RUN_ROOT/final_evaluation_pending.json" ]]; then
        cancel_successor
        echo "Training complete; final evaluation awaits the full label set."
        exit 0
    fi
    test -s "$RUN_ROOT/final_evaluation.json"
    cancel_successor
    echo "Training and final evaluation complete at step $step."
    exit 0
fi
if [[ -z "${CONTINUATION_SCRIPT:-}" ]]; then
    echo "Saved step $step. Set CONTINUATION_SCRIPT to continue automatically."
    exit 0
fi
if [[ -z "$continuation_job" ]]; then
    # This second attempt follows completed computation. It does not poll status.
    queue_successor
fi
echo "Saved step $step; continuation $continuation_job will resume it."
