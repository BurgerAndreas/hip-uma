"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Stream verified dense Hessian shards in bounded, size-balanced batches.
"""

from __future__ import annotations

import json
import random
from bisect import bisect_left, insort
from pathlib import Path
from types import SimpleNamespace

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


def make_batch(values, rows, *, include_hessian=True):
    """Pack ragged molecules directly, without per-molecule ASE objects."""
    sizes = values["natoms"][rows].astype(np.int64)
    b = len(rows)
    ptr = values.get("atom_ptr")
    if ptr is None:
        ptr = np.r_[0, np.cumsum(values["natoms"])]
    numbers = np.concatenate(
        [values["atomic_numbers"][ptr[r] : ptr[r + 1]] for r in rows]
    )
    positions, cells = [], np.zeros((b, 3, 3), dtype=np.float32)
    for i, r in enumerate(rows):
        # Match ASE teacher centering in float64 before conversion to FP32.
        xyz = values["coords"][ptr[r] : ptr[r + 1]].astype(np.float64)
        low = xyz.min(axis=0)
        cells[i, np.arange(3), np.arange(3)] = xyz.max(axis=0) - low + 240.0
        positions.append((xyz - low + 120.0).astype(np.float32))
    labels = None
    if include_hessian:
        hp = values.get("hessian_ptr")
        if hp is None:
            hp = np.r_[0, np.cumsum(9 * values["natoms"].astype(np.int64) ** 2)]
        labels = torch.from_numpy(
            np.concatenate([values["hessian_flat"][hp[r] : hp[r + 1]] for r in rows])
        )
    return AtomicData(
        pos=torch.from_numpy(np.concatenate(positions)),
        atomic_numbers=torch.from_numpy(numbers.astype(np.int64)),
        cell=torch.from_numpy(cells),
        pbc=torch.zeros(b, 3, dtype=torch.bool),
        natoms=torch.from_numpy(sizes),
        edge_index=torch.empty(2, 0, dtype=torch.long),
        cell_offsets=torch.empty(0, 3),
        nedges=torch.zeros(b, dtype=torch.long),
        charge=torch.from_numpy(values["charge"][rows].astype(np.int64)),
        spin=torch.from_numpy(values["spin"][rows].astype(np.int64)),
        fixed=torch.zeros(int(sizes.sum()), dtype=torch.long),
        tags=torch.ones(int(sizes.sum()), dtype=torch.long),
        hessian=labels,
        batch=torch.arange(b).repeat_interleave(torch.from_numpy(sizes)),
        dataset=["omol"] * b,
        sid=[str(x) for x in values["omol_index"][rows]],
    )


class BalancedBatchSampler:
    """Best-fit batches with atom, molecule, and quadratic memory limits.

    Oversized molecules form singleton batches. No sample is dropped or repeated.
    Largest-first packing reduces unused atom capacity within a shard window.
    """

    def __init__(self, sizes, *, max_atoms, batch_size=64, pair_budget=131072):
        if min(max_atoms, batch_size, pair_budget) < 1:
            raise ValueError("Batch limits must be positive")
        self.sizes = np.asarray(sizes, dtype=np.int64)
        if np.any(self.sizes < 1):
            raise ValueError("Each molecule must contain at least one atom")
        self.max_atoms = max_atoms
        self.batch_size = batch_size
        self.pair_budget = pair_budget

    def __iter__(self):
        batches, atoms, pairs = [], [], []
        available = []
        for row in np.argsort(-self.sizes, kind="stable"):
            n = int(self.sizes[row])
            # Increasing free atom capacity gives the same best-fit order.
            # The batch index resolves ties exactly as the original scan did.
            position = bisect_left(available, (n, -1))
            while position < len(available):
                _, i = available[position]
                if pairs[i] + n * n <= self.pair_budget:
                    break
                position += 1
            if position < len(available):
                _, i = available.pop(position)
                batches[i].append(int(row))
                atoms[i] += n
                pairs[i] += n * n
            else:
                i = len(batches)
                batches.append([int(row)])
                atoms.append(n)
                pairs.append(n * n)
            remaining = self.max_atoms - atoms[i]
            if (
                len(batches[i]) < self.batch_size
                and remaining > 0
                and pairs[i] < self.pair_budget
            ):
                insort(available, (remaining, i))
        yield from batches


def cache_balanced_batches(sizes, cached, *, max_atoms, batch_size, pair_budget):
    """
    Keep cache hits usable when a shard window also contains cache misses.
    """
    sizes = np.asarray(sizes)
    cached = np.asarray(cached, dtype=bool)
    for ready in (False, True):
        rows = np.flatnonzero(cached == ready)
        for batch in BalancedBatchSampler(
            sizes[rows],
            max_atoms=max_atoms,
            batch_size=batch_size,
            pair_budget=pair_budget,
        ):
            yield rows[batch].tolist()


def microbatch_indices(batch, sizes, pair_budget):
    """
    Split one optimizer batch into bounded quadratic work units without losses.
    """
    result, current, pairs = [], [], 0
    for index in batch:
        cost = int(sizes[index]) ** 2
        if current and pairs + cost > pair_budget:
            result.append(current)
            current, pairs = [], 0
        current.append(index)
        pairs += cost
    if current:
        result.append(current)
    return result


def pack_window_batch(window, selected):
    """Pack samples across source shards and retain cache scatter metadata."""
    chunks = []
    groups = []
    cached = []
    offset = 0
    for source, rows in selected:
        values, meta, features = window[source]
        part = make_batch(values, np.asarray(rows))
        chunks.append(part)
        ptr = values["atom_ptr"]
        spans = [(int(ptr[r]), int(ptr[r + 1])) for r in rows]
        length = sum(b - a for a, b in spans)
        groups.append(
            SimpleNamespace(
                hip_cache_key=str(Path(meta["path"]).with_suffix("")),
                hip_total_atoms=int(ptr[-1]),
                hip_row_count=len(values["natoms"]),
                hip_rows=rows,
                hip_atom_spans=spans,
                hip_training_rows=np.flatnonzero(
                    ~validation_mask(values["omol_index"])
                ).tolist(),
                hip_data_sha256=meta["sha256"],
                hip_feature_start=offset,
                hip_feature_end=offset + length,
            )
        )
        offset += length
        if features is not None:
            cached.append(np.concatenate([features[a:b] for a, b in spans]))
    tensor_fields = (
        "pos",
        "atomic_numbers",
        "cell",
        "pbc",
        "natoms",
        "nedges",
        "charge",
        "spin",
        "fixed",
        "tags",
        "hessian",
    )
    combined = {key: torch.cat([part[key] for part in chunks]) for key in tensor_fields}
    count = len(combined["natoms"])
    data = AtomicData(
        **combined,
        edge_index=torch.empty(2, 0, dtype=torch.long),
        cell_offsets=torch.empty(0, 3),
        batch=torch.arange(count).repeat_interleave(combined["natoms"]),
        dataset=["omol"] * count,
        sid=[sid for part in chunks for sid in part.sid],
    )
    data.hip_cache_groups = groups
    if len(cached) == len(chunks):
        data.hip_features = torch.from_numpy(np.concatenate(cached))
    return data


class HessianStream(IterableDataset):
    """Read each compressed shard once per pass and partition files by rank."""

    def __init__(
        self,
        root,
        *,
        rank=0,
        world=1,
        seed=7,
        pair_budget=131072,
        max_batch=64,
        max_atoms=2560,
        window_shards=8,
        microbatch_pairs=None,
        split="train",
        max_passes=None,
        feature_cache=None,
        markers=None,
    ):
        super().__init__()
        self.root = Path(root)
        self.rank, self.world, self.seed = rank, world, seed
        self.pair_budget, self.max_batch = pair_budget, max_batch
        self.max_atoms, self.window_shards = max_atoms, window_shards
        self.microbatch_pairs = microbatch_pairs
        self.split, self.max_passes = split, max_passes
        self.feature_cache = feature_cache
        self.markers = markers

    def __iter__(self):
        worker = get_worker_info()
        wid, workers = (worker.id, worker.num_workers) if worker else (0, 1)
        pass_id = 0
        while self.max_passes is None or pass_id < self.max_passes:
            # Immutable verified markers allow downloads to add complete shards.
            markers = (
                [self.root / p for p in self.markers]
                if self.markers is not None
                else sorted(self.root.glob("shards/*/*.verified.json"))
            )
            if len(markers) < self.world * workers:
                raise RuntimeError("Too few verified shards for the loader ranks")
            rng = random.Random(self.seed + pass_id)
            rng.shuffle(markers)
            mine = markers[self.rank :: self.world][wid::workers]
            window = []
            for marker_index, marker in enumerate(mine):
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
                key = str(Path(row["path"]).with_suffix(""))
                features = (
                    cached_features(self.feature_cache, key, row["sha256"])
                    if self.feature_cache and self.split == "train"
                    else None
                )
                window.append((values, row, features))
                if len(window) < self.window_shards and marker_index + 1 < len(mine):
                    continue
                references = []
                sizes_window = []
                for source, (v, _, _) in enumerate(window):
                    held_out = validation_mask(v["omol_index"])
                    eligible = np.flatnonzero(
                        held_out if self.split == "val" else ~held_out
                    ).tolist()
                    rng.shuffle(eligible)
                    references.extend((source, r) for r in eligible)
                    sizes_window.extend(int(v["natoms"][r]) for r in eligible)
                batches = list(
                    cache_balanced_batches(
                        sizes_window,
                        [window[source][2] is not None for source, _ in references],
                        max_atoms=self.max_atoms,
                        batch_size=self.max_batch,
                        pair_budget=self.pair_budget,
                    )
                )
                rng.shuffle(batches)
                for batch in batches:
                    fragments = (
                        microbatch_indices(batch, sizes_window, self.microbatch_pairs)
                        if self.microbatch_pairs
                        else [batch]
                    )
                    packed = []
                    for fragment in fragments:
                        by_source = {}
                        for index in fragment:
                            source, r = references[index]
                            by_source.setdefault(source, []).append(r)
                        packed.append(
                            pack_window_batch(window, list(by_source.items()))
                        )
                    yield packed if self.microbatch_pairs else packed[0]
                window = []
            pass_id += 1
