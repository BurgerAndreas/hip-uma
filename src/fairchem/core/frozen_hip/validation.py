"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Select and stream a fixed evaluation cohort independently of training settings.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from fairchem.core.frozen_hip.data import (
    BalancedBatchSampler,
    make_batch,
    pack_window_batch,
    validation_mask,
)

DATA_REVISION = "7ce83e7425649619c9f85dad485bb2cfd7ee12bd"
SELECTION = "parent-holdout-splitmix64-v1"


def monitoring_complete(path, step, cohort):
    """Recognize a complete fixed-cohort record after a checkpoint restart."""
    from fairchem.core.frozen_hip.metrics import PROJECTED_METRIC_NAMES

    path = Path(path)
    if not path.exists():
        return False
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
            if (
                row.get("event") == "validation"
                and row.get("step") == step
                and row.get("samples") == cohort["samples"]
                and row.get("sample_ids_sha256") == cohort["sample_ids_sha256"]
                and all(
                    math.isfinite(row.get(key, math.nan))
                    for key in PROJECTED_METRIC_NAMES
                )
                and row.get("mass_weighted_eckart_samples", 0)
                + row.get("mass_weighted_eckart_samples_without_modes", 0)
                == cohort["samples"]
            ):
                return True
        except (json.JSONDecodeError, TypeError, AttributeError):
            # A terminated writer can leave an incomplete trailing record.
            continue
    return False


def evaluation_mask(indices):
    """
    Select a fixed tenth of the original one-percent parent-index holdout.

    SplitMix64 avoids correlation with the original linear holdout hash.
    The selection has no dependency on training seeds, ordering, or GPU counts.
    """
    x = np.asarray(indices, dtype=np.uint64)
    with np.errstate(over="ignore"):
        z = x + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z ^= z >> np.uint64(31)
    return validation_mask(x) & (z % np.uint64(10) == 0)


def sample_fingerprint(indices):
    """
    Identify cohort membership independently of its partition and presentation.
    """
    values = np.sort(np.asarray(indices, dtype="<u8"))
    return hashlib.sha256(values.tobytes()).hexdigest()


def _read_manifest_marker(request):
    root, marker = request
    record = json.loads(marker.read_text())
    entry, chosen = None, []
    with h5py.File(root / record["path"], "r") as f:
        indices = f["omol_index"][:]
        rows = np.flatnonzero(evaluation_mask(indices))
        if len(rows):
            chosen = indices[rows].astype(np.int64).tolist()
            entry = dict(
                path=record["path"],
                source_sha256=record["sha256"],
                rows=rows.tolist(),
                omol_indices=chosen,
                natoms=f["natoms"][:][rows].astype(np.int64).tolist(),
            )
    return len(indices), entry, chosen


def build_manifest(root, completion):
    """
    Scan only sample metadata from the complete, verified source snapshot.
    """
    root = Path(root)
    markers = sorted(root.glob("shards/*/*.verified.json"))
    if completion["revision"] != DATA_REVISION:
        raise ValueError("Evaluation requires the pinned scientific snapshot")
    if len(markers) != completion["shards"]:
        raise ValueError("Complete all verified shards before fixing evaluation")
    entries, ids = [], []
    scanned = 0

    with ProcessPoolExecutor(max_workers=8) as pool:
        for index, (count, entry, chosen) in enumerate(
            pool.map(
                _read_manifest_marker,
                ((root, marker) for marker in markers),
                chunksize=16,
            )
        ):
            scanned += count
            if entry:
                entries.append(entry)
                ids.extend(chosen)
            if (index + 1) % 1000 == 0:
                print(f"Scanned {index + 1}/{len(markers)} shards", flush=True)
    if scanned != completion["samples"]:
        raise ValueError("Source sample count differs from the completed snapshot")
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Evaluation IDs must be nonempty and unique")
    return dict(
        version=1,
        selection=SELECTION,
        dataset_revision=DATA_REVISION,
        samples=len(ids),
        sample_ids_sha256=sample_fingerprint(ids),
        entries=entries,
    )


def write_manifest(path, manifest):
    """
    Atomically publish portable sample identities and relative source locations.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(path)


def load_manifest(path):
    """
    Reject changed membership, duplicate samples, or a different source snapshot.
    """
    manifest = json.loads(Path(path).read_text())
    if manifest.get("selection") == "release-holdout-v2":
        ids = manifest["sample_ids"]
        if (
            len(ids) != manifest["samples"]
            or len(set(ids)) != len(ids)
            or sample_fingerprint(ids) != manifest["sample_ids_sha256"]
            or any(i < (1 << 60) for i in ids)
        ):
            raise ValueError("Release validation identities differ")
        if len(manifest["dataset_revision"]) != 40:
            raise ValueError("Pin the release to a full dataset revision")
        return manifest
    if (
        manifest["selection"] != SELECTION
        or manifest["dataset_revision"] != DATA_REVISION
    ):
        raise ValueError("Unexpected evaluation cohort version or dataset revision")
    ids = [i for entry in manifest["entries"] for i in entry["omol_indices"]]
    if len(ids) != manifest["samples"] or len(set(ids)) != len(ids) or not ids:
        raise ValueError("Evaluation cohort count or uniqueness differs")
    if not evaluation_mask(ids).all():
        raise ValueError("Evaluation contains samples outside the fixed holdout")
    if sample_fingerprint(ids) != manifest["sample_ids_sha256"]:
        raise ValueError("Evaluation cohort fingerprint differs")
    return manifest


def read_selected(root, entry):
    """
    Read selected atom and Hessian spans, without decompressing whole shards.
    """
    relative = Path(entry["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Evaluation source paths must be relative")
    path = Path(root) / relative
    marker = json.loads(path.with_suffix(".verified.json").read_text())
    if marker["sha256"] != entry["source_sha256"]:
        raise ValueError("Evaluation source checksum marker differs")
    rows = entry["rows"]
    with h5py.File(path, "r") as f:
        if not f.attrs.get("complete") or f.attrs.get("model") != "uma-s-1p2":
            raise ValueError("Require complete UMA-s-1.2 evaluation shards")
        if f.attrs.get("hessian_unit") != "eV/Angstrom^2":
            raise ValueError("Unexpected evaluation Hessian units")
        indices = f["omol_index"][:][rows]
        sizes = f["natoms"][:][rows].astype(np.int64)
        if (
            indices.tolist() != entry["omol_indices"]
            or sizes.tolist() != entry["natoms"]
        ):
            raise ValueError("Evaluation row identities or atom counts differ")
        ap, hp = f["atom_ptr"][:], f["hessian_ptr"][:]
        values = {
            key: np.concatenate([f[key][ap[r] : ap[r + 1]] for r in rows])
            for key in ("atomic_numbers", "coords")
        }
        values.update(
            natoms=sizes,
            omol_index=indices,
            atom_ptr=np.r_[0, np.cumsum(sizes)],
            hessian_ptr=np.r_[0, np.cumsum(9 * sizes**2)],
            hessian_flat=np.concatenate(
                [f["hessian_flat"][hp[r] : hp[r + 1]] for r in rows]
            ),
            charge=f["charge"][:][rows],
            spin=f["spin"][:][rows],
        )
    if len(values["atomic_numbers"]) != values["atom_ptr"][-1]:
        raise ValueError("Evaluation atom spans differ from molecule sizes")
    if len(values["hessian_flat"]) != values["hessian_ptr"][-1]:
        raise ValueError("Evaluation Hessian spans differ from molecule sizes")
    return values, {"path": entry["path"], "sha256": entry["source_sha256"]}, None


def manifest_fingerprint(manifest):
    """
    Bind cached arrays to source checksums, selected rows, and sample identities.
    """
    return hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _cache_entry(request):
    root, entry = request
    return read_selected(root, entry)[0]


def build_validation_cache(root, manifest, output=None, workers=8):
    """
    Retain original dense labels and coordinates in one portable memory map.
    """
    from fairchem.core.frozen_hip.packed import write_packed

    root = Path(root)
    output = Path(output) if output else root / "validation_10000.hip"
    parts = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for index, values in enumerate(
            pool.map(
                _cache_entry,
                ((root, entry) for entry in manifest["entries"]),
                chunksize=8,
            )
        ):
            parts.append(values)
            if (index + 1) % 500 == 0:
                print(
                    f"Cached {index + 1}/{len(manifest['entries'])} sources", flush=True
                )
    keys = (
        "natoms",
        "omol_index",
        "atomic_numbers",
        "coords",
        "hessian_flat",
        "charge",
        "spin",
    )
    arrays = {key: np.concatenate([part[key] for part in parts]) for key in keys}
    sizes = arrays["natoms"]
    arrays["atom_ptr"] = np.r_[0, np.cumsum(sizes)]
    arrays["hessian_ptr"] = np.r_[0, np.cumsum(9 * sizes**2)]
    arrays["entry_ptr"] = np.r_[0, np.cumsum([len(part["natoms"]) for part in parts])]
    if (
        len(sizes) != manifest["samples"]
        or sample_fingerprint(arrays["omol_index"]) != manifest["sample_ids_sha256"]
    ):
        raise ValueError("Cached evaluation identities differ from the manifest")
    write_packed(
        output,
        arrays,
        dict(
            kind="dense_validation_v1", manifest_sha256=manifest_fingerprint(manifest)
        ),
    )
    return output


class EvaluationStream(IterableDataset):
    """
    Evaluate every fixed cohort member exactly once across any GPU count.
    """

    def __init__(
        self,
        root,
        manifest,
        *,
        rank=0,
        world=1,
        max_batch=32,
        max_atoms=1280,
        pair_budget=196608,
        window_files=64,
    ):
        super().__init__()
        self.root, self.manifest = root, manifest
        self.rank, self.world = rank, world
        self.max_batch, self.max_atoms = max_batch, max_atoms
        self.pair_budget, self.window_files = pair_budget, window_files

    def __iter__(self):
        if self.manifest.get("selection") == "release-holdout-v2":
            if not self.manifest.get("complete"):
                raise ValueError(
                    "Final validation labels are incomplete; retain the last checkpoint"
                )
            from fairchem.core.frozen_hip.packed import open_packed

            entries = self.manifest["entries"][self.rank :: self.world]
            worker = get_worker_info()
            if worker:
                entries = entries[worker.id :: worker.num_workers]
            for entry in entries:
                path = Path(entry["path"])
                if path.is_absolute() or ".." in path.parts:
                    raise ValueError("Evaluation paths must be relative")
                values, header = open_packed(Path(self.root) / path)
                if header.get("dataset_revision") != self.manifest["dataset_revision"]:
                    raise ValueError("Evaluation source revision differs")
                if header.get("backbone_sha256") != self.manifest.get(
                    "backbone_sha256"
                ):
                    raise ValueError(
                        "Evaluation features use a different frozen backbone"
                    )
                rows = np.asarray(entry["rows"], dtype=np.int64)
                if values["omol_index"][rows].tolist() != entry["omol_indices"]:
                    raise ValueError("Evaluation source identities differ")
                sampler = BalancedBatchSampler(
                    values["natoms"][rows],
                    max_atoms=self.max_atoms,
                    batch_size=self.max_batch,
                    pair_budget=self.pair_budget,
                )
                for indices in sampler:
                    selected = rows[indices]
                    # Evaluation retains dense labels for the projected eigensystem.
                    data = make_batch(values, selected)
                    ap = values["atom_ptr"]
                    data.hip_features = torch.from_numpy(
                        np.concatenate(
                            [values["features"][ap[i] : ap[i + 1]] for i in selected]
                        )
                    )
                    data.hip_feature_scale = header.get("feature_scale", 1.0)
                    yield data
            return
        cache_path = Path(self.root) / "validation_10000.hip"
        if cache_path.exists():
            from fairchem.core.frozen_hip.packed import open_packed

            values, header = open_packed(cache_path)
            if header.get("kind") != "dense_validation_v1" or header.get(
                "manifest_sha256"
            ) != manifest_fingerprint(self.manifest):
                raise ValueError("Validation cache differs from the fixed manifest")
            entries = np.arange(len(self.manifest["entries"]))[self.rank :: self.world]
            worker = get_worker_info()
            if worker is not None:
                entries = entries[worker.id :: worker.num_workers]
            ep = values["entry_ptr"]
            for start in range(0, len(entries), self.window_files):
                rows = np.concatenate(
                    [
                        np.arange(ep[index], ep[index + 1])
                        for index in entries[start : start + self.window_files]
                    ]
                )
                sampler = BalancedBatchSampler(
                    values["natoms"][rows],
                    max_atoms=self.max_atoms,
                    batch_size=self.max_batch,
                    pair_budget=self.pair_budget,
                )
                for batch in sampler:
                    yield make_batch(values, rows[np.asarray(batch)])
            return
        entries = self.manifest["entries"][self.rank :: self.world]
        worker = get_worker_info()
        if worker is not None:
            entries = entries[worker.id :: worker.num_workers]
        for start in range(0, len(entries), self.window_files):
            window = [
                read_selected(self.root, row)
                for row in entries[start : start + self.window_files]
            ]
            refs = [
                (source, row)
                for source, (values, _, _) in enumerate(window)
                for row in range(len(values["natoms"]))
            ]
            sizes = [int(window[source][0]["natoms"][row]) for source, row in refs]
            batches = BalancedBatchSampler(
                sizes,
                max_atoms=self.max_atoms,
                batch_size=self.max_batch,
                pair_budget=self.pair_budget,
            )
            for batch in batches:
                groups = {}
                for index in batch:
                    source, row = refs[index]
                    groups.setdefault(source, []).append(row)
                yield pack_window_batch(window, list(groups.items()))


def load_last_head(head, checkpoint_path, backbone_sha256, device):
    """
    Reload the named last checkpoint rather than selecting weights by metrics.
    """
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if state["backbone_sha256"] != backbone_sha256:
        raise ValueError("Last evaluation checkpoint has a different backbone")
    head.load_state_dict(state["head"])
    return state
