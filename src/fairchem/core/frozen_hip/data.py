"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Stream verified dense Hessian shards in bounded, size-balanced batches.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.frozen_hip.cache import cached_features


def validation_mask(indices):
    """Reserve a deterministic one percent of parent indices for validation."""
    values = np.asarray(indices, dtype=np.uint64)
    return ((values * np.uint64(6364136223846793005)) % np.uint64(1000)) < 10


def make_batch(values, rows):
    """Pack an equal-size batch directly, without per-molecule ASE objects."""
    n = int(values["natoms"][rows[0]])
    assert np.all(values["natoms"][rows] == n)
    b = len(rows)
    if np.all(values["natoms"] == n):
        numbers = values["atomic_numbers"].reshape(-1, n)[rows].reshape(-1)
        coords = values["coords"].reshape(-1, n, 3)[rows].reshape(-1, 3)
        labels = values["hessian_flat"].reshape(-1, 9 * n * n)[rows].reshape(-1)
    else:
        ptr, hp = values["atom_ptr"], values["hessian_ptr"]
        numbers = np.concatenate(
            [values["atomic_numbers"][ptr[r] : ptr[r + 1]] for r in rows]
        )
        coords = np.concatenate([values["coords"][ptr[r] : ptr[r + 1]] for r in rows])
        labels = np.concatenate(
            [values["hessian_flat"][hp[r] : hp[r + 1]] for r in rows]
        )
    # No periodic images are used for the molecular teacher or student.
    return AtomicData(
        pos=torch.from_numpy(coords.copy()),
        atomic_numbers=torch.from_numpy(numbers.astype(np.int64)),
        cell=torch.eye(3).repeat(b, 1, 1) * 120,
        pbc=torch.zeros(b, 3, dtype=torch.bool),
        natoms=torch.full((b,), n, dtype=torch.long),
        edge_index=torch.empty(2, 0, dtype=torch.long),
        cell_offsets=torch.empty(0, 3),
        nedges=torch.zeros(b, dtype=torch.long),
        charge=torch.from_numpy(values["charge"][rows].astype(np.int64)),
        spin=torch.from_numpy(values["spin"][rows].astype(np.int64)),
        fixed=torch.zeros(b * n, dtype=torch.long),
        tags=torch.ones(b * n, dtype=torch.long),
        hessian=torch.from_numpy(labels.copy()),
        batch=torch.arange(b).repeat_interleave(n),
        dataset=["omol"] * b,
        sid=[str(x) for x in values["omol_index"][rows]],
    )


class HessianStream(IterableDataset):
    """Read each compressed shard once per pass and partition files by rank."""

    def __init__(
        self,
        root,
        *,
        rank=0,
        world=1,
        seed=7,
        pair_budget=8192,
        max_batch=32,
        split="train",
        max_passes=None,
        feature_cache=None,
    ):
        super().__init__()
        self.root = Path(root)
        self.rank, self.world, self.seed = rank, world, seed
        self.pair_budget, self.max_batch = pair_budget, max_batch
        self.split, self.max_passes = split, max_passes
        self.feature_cache = feature_cache

    def __iter__(self):
        worker = get_worker_info()
        wid, workers = (worker.id, worker.num_workers) if worker else (0, 1)
        pass_id = 0
        while self.max_passes is None or pass_id < self.max_passes:
            # Immutable verified markers allow downloads to add complete shards.
            markers = sorted(self.root.glob("shards/*/*.verified.json"))
            if len(markers) < self.world * workers:
                raise RuntimeError("Too few verified shards for the loader ranks")
            rng = random.Random(self.seed + pass_id)
            rng.shuffle(markers)
            mine = markers[self.rank :: self.world][wid::workers]
            for marker in mine:
                row = json.loads(marker.read_text())
                path = self.root / row["path"]
                with h5py.File(path, "r") as f:
                    if (
                        not f.attrs.get("complete")
                        or f.attrs.get("model") != "uma-s-1p2"
                    ):
                        raise ValueError("Require complete UMA-s-1.2 teacher shards")
                    if f.attrs.get("hessian_unit") != "eV/Angstrom^2":
                        raise ValueError("Unexpected Hessian units")
                    fields = (
                        "natoms",
                        "atomic_numbers",
                        "coords",
                        "hessian_flat",
                        "charge",
                        "spin",
                        "omol_index",
                        "atom_ptr",
                        "hessian_ptr",
                    )
                    values = {k: f[k][:] for k in fields}
                sizes = values["natoms"]
                if not np.array_equal(values["atom_ptr"], np.r_[0, np.cumsum(sizes)]):
                    raise ValueError("Atom pointers differ from sample sizes")
                if not np.array_equal(
                    values["hessian_ptr"],
                    np.r_[0, np.cumsum(9 * sizes.astype(np.int64) ** 2)],
                ):
                    raise ValueError("Hessian pointers differ from dense sample sizes")
                mask = validation_mask(values["omol_index"])
                rows = np.flatnonzero(mask if self.split == "val" else ~mask)
                key = str(Path(row["path"]).with_suffix(""))
                features = (
                    cached_features(self.feature_cache, key, row["sha256"])
                    if self.feature_cache and self.split == "train"
                    else None
                )
                groups = np.unique(sizes[rows]).tolist()
                rng.shuffle(groups)
                for n in groups:
                    selected = rows[sizes[rows] == n]
                    rng.shuffle(selected)
                    batch_size = min(
                        self.max_batch, max(1, self.pair_budget // (n * n))
                    )
                    for start in range(0, len(selected), batch_size):
                        chosen = selected[start : start + batch_size]
                        data = make_batch(values, chosen)
                        ptr = values["atom_ptr"]
                        spans = [(int(ptr[r]), int(ptr[r + 1])) for r in chosen]
                        if features is not None:
                            data.hip_features = torch.from_numpy(
                                np.concatenate([features[a:b] for a, b in spans])
                            )
                        data.hip_cache_key = key
                        data.hip_total_atoms = int(ptr[-1])
                        data.hip_row_count = len(sizes)
                        data.hip_rows = chosen.tolist()
                        data.hip_atom_spans = spans
                        data.hip_training_rows = np.flatnonzero(~mask).tolist()
                        data.hip_data_sha256 = row["sha256"]
                        yield data
            pass_id += 1
