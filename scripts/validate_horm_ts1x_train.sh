#!/usr/bin/env bash
#SBATCH --job-name=validate_horm_ts1x_train
#SBATCH --account=nvr_qualg_lmbm
#SBATCH --partition=cpu
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --output=outputs/slurm_%x_%j.log
#SBATCH --error=outputs/slurm_%x_%j.err

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"

source ./setup_cuda.sh >/dev/null

export FAIRCHEM_CACHE_DIR="$REPO_ROOT/.cache/fairchem"

.venv/bin/python -c '
import lmdb
import numpy as np
import pickle

path = "data/horm_atomic/ts1x_hess_train.atomic.lmdb"
meta_path = "data/horm_atomic/ts1x_hess_train.atomic.metadata.npz"

env = lmdb.open(
    path,
    subdir=False,
    readonly=True,
    lock=False,
    readahead=False,
    meminit=False,
    max_readers=1,
)
with env.begin() as txn:
    length = pickle.loads(txn.get(b"length"))
    first = pickle.loads(txn.get(b"0"))
    last_exists = txn.get(str(length - 1).encode("ascii")) is not None
env.close()

meta = np.load(meta_path)
print("length", length)
print("metadata_n", meta["natoms"].shape[0])
print("has_hessian_count", int(meta["has_hessian"].sum()))
print("keys", sorted(first.keys()))
print("pos", tuple(first["pos"].shape), first["pos"].dtype)
print("forces", tuple(first["forces"].shape), first["forces"].dtype)
print("energy", tuple(first["energy"].shape), first["energy"].dtype)
print("hessian", tuple(first["hessian"].shape), first["hessian"].dtype)
print("last_key_exists", last_exists)
'
