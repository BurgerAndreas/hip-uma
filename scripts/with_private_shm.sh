#!/usr/bin/env bash
# Keep a trainer's POSIX semaphores and tensor buffers in its own mount.
# No files in the host shared-memory mount are changed or removed.
set -euo pipefail
if (( $# == 0 )); then
    echo "Supply a command to run with private shared memory." >&2
    exit 2
fi
export HIP_SHM_SIZE="${HIP_SHM_SIZE:-16G}"
if [[ ! "$HIP_SHM_SIZE" =~ ^[1-9][0-9]*[KMGT]?$ ]]; then
    echo "HIP_SHM_SIZE must be a positive size such as 16G." >&2
    exit 2
fi
# Keep the trainer PID unchanged so checkpoint signals reach it directly.
# The kernel releases the mount when the trainer and its workers exit.
exec unshare --user --map-root-user --mount --ipc --propagation private \
    bash -c '
        set -euo pipefail
        mount -t tmpfs -o "size=$HIP_SHM_SIZE,mode=1777,nosuid,nodev" tmpfs /dev/shm
        exec "$@"
    ' bash "$@"
