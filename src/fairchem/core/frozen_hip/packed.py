"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Memory-map packed features and symmetric Hessian blocks without label padding.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.frozen_hip.cache import cached_features
from fairchem.core.frozen_hip.data import (
    cache_balanced_batches,
    make_batch,
    microbatch_indices,
    validation_mask,
)

MAGIC = b"HIPPACK1"


def pack_hessian(hessian):
    """
    Pack symmetric targets and retain the antisymmetric MSE constant exactly.
    """
    h = np.asarray(hessian, dtype=np.float64)
    if h.ndim != 2 or h.shape[0] != h.shape[1] or h.shape[0] % 3:
        raise ValueError(
            "A molecular Hessian must be square with size divisible by three"
        )
    if not np.isfinite(h).all():
        raise ValueError("Hessian targets must be finite")
    n = h.shape[0] // 3
    symmetric = ((h + h.T) * 0.5).astype(np.float32)
    blocks = symmetric.reshape(n, 3, n, 3).transpose(0, 2, 1, 3)
    j, i = np.tril_indices(n, k=-1)
    pairs = blocks[i, j].reshape(-1)
    upper = np.triu_indices(3)
    diagonal = blocks[np.arange(n), np.arange(n)][:, upper[0], upper[1]].reshape(-1)
    constant = float(np.square((h - h.T) * 0.5).mean())
    return pairs, diagonal, constant


def write_packed(path, arrays, metadata):
    """
    Write aligned arrays and a JSON header into one atomically published file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {key: np.ascontiguousarray(value) for key, value in arrays.items()}
    layout, offset = {}, 0
    for name, value in arrays.items():
        offset = (offset + 255) // 256 * 256
        layout[name] = dict(
            dtype=value.dtype.str, shape=list(value.shape), offset=offset
        )
        offset += value.nbytes
    header = json.dumps(
        dict(version=1, arrays=layout, **metadata), sort_keys=True
    ).encode()
    begin = (16 + len(header) + 255) // 256 * 256
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    with temporary.open("wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<Q", begin - 16))
        f.write(header)
        f.write(b" " * (begin - f.tell()))
        for name, value in arrays.items():
            f.seek(begin + layout[name]["offset"])
            value.tofile(f)
        f.flush()
        os.fsync(f.fileno())
    temporary.replace(path)


def open_packed(path):
    """
    Open arrays as read-only memory maps with no decompression or whole-file copy.
    """
    path = Path(path)
    with path.open("rb") as f:
        if f.read(8) != MAGIC:
            raise ValueError("Unexpected HIP packed file version")
        size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(size))
        mapping = np.memmap(f, mode="r", dtype=np.uint8)
    begin = 16 + size
    arrays = {}
    for name, row in header["arrays"].items():
        shape = tuple(row["shape"])
        dtype = np.dtype(row["dtype"])
        count = int(np.prod(shape))
        offset = begin + row["offset"]
        arrays[name] = (
            mapping[offset : offset + count * dtype.itemsize].view(dtype).reshape(shape)
            if count
            else np.empty(shape, dtype=dtype)
        )
    return arrays, header


def make_packed_batch(
    values, rows, feature_scale=1.0, cache_meta=None, *, include_targets=True
):
    """
    Gather variable-size rows while retaining compact upper-triangle targets.
    """
    data = make_batch(values, np.asarray(rows), include_hessian=False)
    sizes = values["natoms"][rows].astype(np.int64)
    data.hip_atom_ptr = torch.from_numpy(np.r_[0, np.cumsum(sizes)])
    data.hip_edge_ptr = torch.from_numpy(np.r_[0, np.cumsum(sizes * (sizes - 1))])
    data.hip_pair_ptr = torch.from_numpy(
        np.r_[0, np.cumsum(9 * sizes * (sizes - 1) // 2)]
    )
    data.hip_fused_graph = True
    ap = values["atom_ptr"]
    if include_targets:
        pp = values["pair_ptr"]
        data.hip_pair_targets = torch.from_numpy(
            np.concatenate([values["pair_targets"][pp[r] : pp[r + 1]] for r in rows])
        )
        data.hip_diagonal_targets = torch.from_numpy(
            np.concatenate(
                [values["diagonal_targets"][ap[r] * 6 : ap[r + 1] * 6] for r in rows]
            )
        )
        data.hip_asymmetry_mse = torch.from_numpy(
            np.asarray(values["asymmetry_mse"])[rows].copy()
        )
    if "features" in values:
        data.hip_features = torch.from_numpy(
            np.concatenate([values["features"][ap[r] : ap[r + 1]] for r in rows])
        )
        data.hip_feature_scale = feature_scale
    if cache_meta is not None:
        length = int(sizes.sum())
        data.hip_cache_groups = [
            SimpleNamespace(
                **cache_meta,
                hip_rows=list(rows),
                hip_atom_spans=[(int(ap[r]), int(ap[r + 1])) for r in rows],
                hip_feature_start=0,
                hip_feature_end=length,
            )
        ]
    return data


def pack_cached_window(window, references, *, include_targets=True):
    """Gather cached rows once, retaining the original bundle grouping order."""
    grouped = {}
    for source, row in references:
        grouped.setdefault(source, []).append(row)
    if any(
        "features" not in window[source][0] or "_cache_meta" in window[source][1]
        for source in grouped
    ):
        return pack_packed_window(window, references, include_targets=include_targets)
    sizes, positions, cells, numbers, features = [], [], [], [], []
    pairs, diagonals, asymmetry, charge, spin, sid, scales = [], [], [], [], [], [], []
    for source, rows in grouped.items():
        values, header = window[source]
        for row in rows:
            lo, hi = values["atom_ptr"][row : row + 2]
            n = int(hi - lo)
            sizes.append(n)
            xyz = values["coords"][lo:hi].astype(np.float64)
            low = xyz.min(axis=0)
            positions.append((xyz - low + 120.0).astype(np.float32))
            cells.append(np.diag(xyz.max(axis=0) - low + 240.0).astype(np.float32))
            numbers.append(values["atomic_numbers"][lo:hi])
            features.append(values["features"][lo:hi])
            charge.append(values["charge"][row])
            spin.append(values["spin"][row])
            sid.append(str(values["omol_index"][row]))
            scales.append(header.get("feature_scale", 1.0))
            if include_targets:
                pl, ph = values["pair_ptr"][row : row + 2]
                pairs.append(values["pair_targets"][pl:ph])
                diagonals.append(values["diagonal_targets"][lo * 6 : hi * 6])
                asymmetry.append(values["asymmetry_mse"][row])
    sizes = np.asarray(sizes, dtype=np.int64)
    count, atoms = len(sizes), int(sizes.sum())
    data = AtomicData(
        pos=torch.from_numpy(np.concatenate(positions)),
        atomic_numbers=torch.from_numpy(np.concatenate(numbers).astype(np.int64)),
        cell=torch.from_numpy(np.stack(cells)),
        pbc=torch.zeros(count, 3, dtype=torch.bool),
        natoms=torch.from_numpy(sizes),
        edge_index=torch.empty(2, 0, dtype=torch.long),
        cell_offsets=torch.empty(0, 3),
        nedges=torch.zeros(count, dtype=torch.long),
        charge=torch.from_numpy(np.asarray(charge, dtype=np.int64)),
        spin=torch.from_numpy(np.asarray(spin, dtype=np.int64)),
        fixed=torch.zeros(atoms, dtype=torch.long),
        tags=torch.ones(atoms, dtype=torch.long),
        batch=torch.from_numpy(np.repeat(np.arange(count, dtype=np.int64), sizes)),
        dataset=["omol"] * count,
        sid=sid,
    )
    data.hip_atom_ptr = torch.from_numpy(np.r_[0, np.cumsum(sizes)])
    data.hip_edge_ptr = torch.from_numpy(np.r_[0, np.cumsum(sizes * (sizes - 1))])
    data.hip_pair_ptr = data.hip_edge_ptr * 9 // 2
    data.hip_fused_graph = True
    data.hip_features = torch.from_numpy(np.concatenate(features))
    data.hip_feature_scale = (
        scales[0]
        if len(set(scales)) == 1
        else torch.from_numpy(
            np.repeat(np.asarray(scales, dtype=np.float32), sizes).reshape(-1, 1, 1)
        )
    )
    if include_targets:
        data.hip_pair_targets = torch.from_numpy(np.concatenate(pairs))
        data.hip_diagonal_targets = torch.from_numpy(np.concatenate(diagonals))
        data.hip_asymmetry_mse = torch.from_numpy(
            np.asarray(asymmetry, dtype=np.float32)
        )
    return data


def pack_packed_window(window, references, *, include_targets=True):
    """
    Mix bundle sizes before packing, while preserving every row's labels.
    """
    grouped = {}
    for source, row in references:
        grouped.setdefault(source, []).append(row)
    parts = [
        make_packed_batch(
            window[source][0],
            np.asarray(rows),
            window[source][1].get("feature_scale", 1.0),
            window[source][1].get("_cache_meta"),
            include_targets=include_targets,
        )
        for source, rows in grouped.items()
    ]
    if len(parts) == 1:
        return parts[0]
    fields = (
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
    )
    values = {key: torch.cat([part[key] for part in parts]) for key in fields}
    count = len(values["natoms"])
    data = AtomicData(
        **values,
        edge_index=torch.empty(2, 0, dtype=torch.long),
        cell_offsets=torch.empty(0, 3),
        batch=torch.arange(count).repeat_interleave(values["natoms"]),
        dataset=["omol"] * count,
        sid=[sid for part in parts for sid in part.sid],
    )
    if include_targets:
        for key in ("hip_pair_targets", "hip_diagonal_targets", "hip_asymmetry_mse"):
            setattr(data, key, torch.cat([getattr(part, key) for part in parts]))
    sizes = data.natoms
    data.hip_atom_ptr = torch.cat((sizes.new_zeros(1), sizes.cumsum(0)))
    data.hip_edge_ptr = torch.cat((sizes.new_zeros(1), (sizes * (sizes - 1)).cumsum(0)))
    data.hip_pair_ptr = data.hip_edge_ptr * 9 // 2
    data.hip_fused_graph = True
    if all(hasattr(part, "hip_features") for part in parts):
        data.hip_features = torch.cat([part.hip_features for part in parts])
        scales = [part.hip_feature_scale for part in parts]
        if len(set(scales)) == 1:
            data.hip_feature_scale = scales[0]
        else:
            data.hip_feature_scale = torch.cat(
                [
                    torch.full((len(part.pos), 1, 1), scale, dtype=torch.float32)
                    for part, scale in zip(parts, scales, strict=True)
                ]
            )
    if all(hasattr(part, "hip_cache_groups") for part in parts):
        data.hip_cache_groups = []
        start = 0
        for part in parts:
            for group in part.hip_cache_groups:
                group.hip_feature_start += start
                group.hip_feature_end += start
                data.hip_cache_groups.append(group)
            start += len(part.pos)
    return data


def block_mse(edge_blocks, node_blocks, data):
    """
    Match dense per-molecule MSE using symmetric blocks and diagonal weights.
    """
    if edge_blocks.is_cuda and getattr(data, "hip_fused_loss", False):
        from fairchem.core.frozen_hip.kernels import fused_block_mse

        return fused_block_mse(edge_blocks, node_blocks, data)
    sizes = data.hip_sizes
    pair_sums, offset, atom_offset, target_offset = [], 0, 0, 0
    diagonal_offset = 0
    weights = node_blocks.new_tensor([1, 2, 2, 1, 2, 1], dtype=torch.float32)
    upper = torch.triu_indices(3, 3, device=node_blocks.device)
    for index, n in enumerate(sizes):
        j, i = torch.tril_indices(n, n, offset=-1, device=edge_blocks.device)
        first = offset + j * (n - 1) + i
        reverse = offset + i * (n - 1) + j - 1
        predicted = edge_blocks[first].float() + edge_blocks[reverse].float().transpose(
            -1, -2
        )
        width = 9 * n * (n - 1) // 2
        target = data.hip_pair_targets[target_offset : target_offset + width].reshape(
            -1, 3, 3
        )
        pair_error = (predicted - target).square().sum() * 2
        nodes = node_blocks[atom_offset : atom_offset + n].float()
        diagonal = nodes + nodes.transpose(-1, -2)
        actual_diagonal = diagonal[:, upper[0], upper[1]]
        target_diagonal = data.hip_diagonal_targets[
            diagonal_offset : diagonal_offset + 6 * n
        ].reshape(n, 6)
        error = (
            pair_error + ((actual_diagonal - target_diagonal).square() * weights).sum()
        )
        pair_sums.append(error / (9 * n * n) + data.hip_asymmetry_mse[index])
        offset += n * (n - 1)
        atom_offset += n
        target_offset += width
        diagonal_offset += 6 * n
    return torch.stack(pair_sums).mean()


def block_rmse(edge_blocks, node_blocks, data, *, absolute=False, mass_weighted=False):
    """
    Average symmetrized-target block norms with equal geometry weights.
    """
    if edge_blocks.is_cuda and getattr(data, "hip_fused_loss", False):
        from fairchem.core.frozen_hip.kernels import fused_block_mae, fused_block_rmse

        operation = fused_block_mae if absolute else fused_block_rmse
        return operation(edge_blocks, node_blocks, data, mass_weighted=mass_weighted)
    from fairchem.core.frozen_hip.losses import block_rmse_values, inverse_sqrt_masses

    def block_values(error):
        return (
            error.float().abs().mean(dim=(-2, -1))
            if absolute
            else block_rmse_values(error)
        )

    inverse = inverse_sqrt_masses(data.atomic_numbers) if mass_weighted else None
    losses, offset, atom_offset, target_offset = [], 0, 0, 0
    lookup = torch.tensor([[0, 1, 2], [1, 3, 4], [2, 4, 5]], device=node_blocks.device)
    for n in data.hip_sizes:
        j, i = torch.tril_indices(n, n, offset=-1, device=edge_blocks.device)
        first = offset + j * (n - 1) + i
        reverse = offset + i * (n - 1) + j - 1
        predicted = edge_blocks[first].float() + edge_blocks[reverse].float().transpose(
            -1, -2
        )
        width = 9 * n * (n - 1) // 2
        target = data.hip_pair_targets[target_offset : target_offset + width].reshape(
            -1, 3, 3
        )
        nodes = node_blocks[atom_offset : atom_offset + n].float()
        target_diag = data.hip_diagonal_targets[
            atom_offset * 6 : (atom_offset + n) * 6
        ].reshape(n, 6)[:, lookup]
        diagonal = nodes + nodes.transpose(-1, -2)
        pair_values = block_values(predicted - target)
        diagonal_values = block_values(diagonal - target_diag)
        if inverse is not None:
            mass = inverse[atom_offset : atom_offset + n]
            pair_values = pair_values * mass[i] * mass[j]
            diagonal_values = diagonal_values * mass.square()
        losses.append((2 * pair_values.sum() + diagonal_values.sum()) / (n * n))
        offset += n * (n - 1)
        atom_offset += n
        target_offset += width
    return torch.stack(losses).mean()


def block_mae(edge_blocks, node_blocks, data):
    """Average Cartesian absolute errors with equal geometry weights."""
    return block_rmse(edge_blocks, node_blocks, data, absolute=True)


def stage_packed_arrays(values, budget, *, include_targets=True):
    """Copy contiguous payloads within a strict byte budget before random gathers."""
    staged, used = dict(values), 0
    keys = ["features"]
    if include_targets:
        keys.extend(("pair_targets", "diagonal_targets"))
    keys.extend(("coords", "atomic_numbers"))
    for key in keys:
        array = values.get(key)
        if array is not None and array.nbytes <= budget - used:
            staged[key] = np.array(array, copy=True, order="C")
            used += array.nbytes
    return staged, used


class PackedHessianStream(IterableDataset):
    """
    Stream memory-mapped training bundles with bounded quadratic GPU batches.
    """

    def __init__(
        self,
        root,
        *,
        markers,
        backbone_sha256,
        backbone_precision="fp32",
        backbone_graph_version=2,
        fused_backbone=False,
        tf32=True,
        grouped_mole=False,
        feature_cache=None,
        window_files=8,
        max_passes=None,
        rank=0,
        world=1,
        seed=7,
        max_atoms=2560,
        max_batch=64,
        pair_budget=196608,
        logical_pair_budget=1000000000,
        include_targets=True,
        include_cache_metadata=True,
        fast_packing=False,
        window_memory_bytes=0,
        prefetch_window=False,
    ):
        super().__init__()
        self.root, self.markers = Path(root), markers
        self.backbone_sha256 = backbone_sha256
        self.backbone_precision, self.tf32 = backbone_precision, tf32
        self.grouped_mole = grouped_mole
        self.backbone_graph_version = backbone_graph_version
        self.fused_backbone = fused_backbone
        self.feature_cache = feature_cache
        self.logical_pair_budget = logical_pair_budget
        self.include_targets = include_targets
        self.include_cache_metadata = include_cache_metadata
        self.fast_packing = fast_packing
        if window_memory_bytes < 0:
            raise ValueError("Window memory budget must be nonnegative")
        self.prefetch_window = prefetch_window
        self.window_memory_bytes = window_memory_bytes
        if window_files < 1:
            raise ValueError("Packed window size must be positive")
        self.window_files, self.max_passes = window_files, max_passes
        self.rank, self.world, self.seed = rank, world, seed
        self.max_atoms, self.max_batch, self.pair_budget = (
            max_atoms,
            max_batch,
            pair_budget,
        )

    def _load_window(self, sources):
        window, staged_bytes = [], 0
        for source in sources:
            path = self.root / self.markers[source]
            values, header = open_packed(path)
            if "features" in values:
                if header["backbone_sha256"] != self.backbone_sha256:
                    raise ValueError("Packed features belong to a different backbone")
                if (
                    header.get("backbone_precision", "fp32") != self.backbone_precision
                    or header.get("tf32", True) != self.tf32
                ):
                    raise ValueError(
                        "Packed features use different inference precision"
                    )
                if (
                    header.get("backbone_graph_version", 2)
                    != self.backbone_graph_version
                ):
                    raise ValueError("Packed features use a different neighbor graph")
                if header.get("fused_backbone", False) != self.fused_backbone:
                    raise ValueError("Packed features use different backbone rotations")
                if header.get("grouped_mole", False) != self.grouped_mole:
                    raise ValueError("Packed features use different expert kernels")
                if (
                    self.grouped_mole
                    and self.backbone_precision == "fp16"
                    and header.get("expert_fp32_output", False) is not True
                ):
                    raise ValueError(
                        "Packed features use unsafe expert output precision"
                    )
            if self.feature_cache:
                digest = hashlib.sha256(str(header.get("fingerprint", "")).encode())
                digest.update(np.asarray(values["omol_index"]).tobytes())
                digest.update(np.asarray(values["natoms"]).tobytes())
                source_sha = digest.hexdigest()
                key = "bundles/" + path.stem
                if "features" not in values:
                    features = cached_features(self.feature_cache, key, source_sha)
                    if features is not None:
                        values = {**values, "features": features}
                        header["feature_scale"] = 1.0
                if self.include_cache_metadata:
                    header["_cache_meta"] = dict(
                        hip_cache_key=key,
                        hip_total_atoms=int(values["atom_ptr"][-1]),
                        hip_row_count=len(values["natoms"]),
                        hip_training_rows=np.flatnonzero(
                            ~validation_mask(values["omol_index"])
                        ).tolist(),
                        hip_data_sha256=source_sha,
                    )
            if self.window_memory_bytes:
                values, used = stage_packed_arrays(
                    values,
                    self.window_memory_bytes - staged_bytes,
                    include_targets=self.include_targets,
                )
                staged_bytes += used
            window.append((values, header))
        return window

    def __iter__(self):
        worker = get_worker_info()
        wid, workers = (worker.id, worker.num_workers) if worker else (0, 1)
        epoch = 0
        while self.max_passes is None or epoch < self.max_passes:
            rng = np.random.default_rng(self.seed + epoch)
            order = rng.permutation(len(self.markers)).tolist()
            mine = order[self.rank :: self.world][wid::workers]
            if not mine:
                raise ValueError("Too few packed bundles for all loader workers")
            sources = (
                mine[start : start + self.window_files]
                for start in range(0, len(mine), self.window_files)
            )
            if self.prefetch_window:
                from fairchem.core.frozen_hip.window_prefetch import prefetch_windows

                windows = prefetch_windows(self._load_window, sources)
            else:
                windows = map(self._load_window, sources)
            for window in windows:
                references, sizes, cached = [], [], []
                for source, (values, _) in enumerate(window):
                    rows = (
                        np.arange(len(values["natoms"]))
                        if window[source][1].get("training_split") == "release-v2"
                        else np.flatnonzero(~validation_mask(values["omol_index"]))
                    )
                    rng.shuffle(rows)
                    references.extend((source, int(row)) for row in rows)
                    sizes.extend(int(values["natoms"][row]) for row in rows)
                    cached.extend(["features" in values] * len(rows))
                batches = list(
                    cache_balanced_batches(
                        sizes,
                        cached,
                        max_atoms=self.max_atoms,
                        batch_size=self.max_batch,
                        pair_budget=self.logical_pair_budget,
                    )
                )
                rng.shuffle(batches)
                for batch in batches:
                    yield [
                        (
                            pack_cached_window
                            if self.fast_packing
                            else pack_packed_window
                        )(
                            window,
                            [references[i] for i in indices],
                            include_targets=self.include_targets,
                        )
                        for indices in microbatch_indices(
                            batch, sizes, self.pair_budget
                        )
                    ]
                del window
            epoch += 1


def arrays_from_batches(batches, embeddings, *, feature_dtype="float32"):
    """
    Convert packed molecular batches to reusable features and triangular targets.
    """
    fields = {
        name: []
        for name in (
            "natoms",
            "atomic_numbers",
            "coords",
            "charge",
            "spin",
            "omol_index",
            "pair_targets",
            "diagonal_targets",
            "asymmetry_mse",
            "features",
        )
    }
    batch_ptr = [0]
    for data, emb in zip(batches, embeddings, strict=True):
        sizes = data.natoms.cpu().numpy().astype(np.int64)
        for name, value in (
            ("natoms", sizes),
            ("atomic_numbers", data.atomic_numbers.cpu().numpy()),
            ("coords", data.pos.cpu().numpy()),
            ("charge", data.charge.cpu().numpy()),
            ("spin", data.spin.cpu().numpy()),
            ("omol_index", np.asarray(data.sid, dtype=np.int64)),
        ):
            fields[name].append(value)
        if emb is not None:
            fields["features"].append(
                emb["node_embedding"].detach().float().cpu().numpy()
            )
        labels = data.hessian.detach().cpu().numpy()
        offset = 0
        for n in sizes:
            width = 9 * int(n) ** 2
            pairs, diagonal, constant = pack_hessian(
                labels[offset : offset + width].reshape(3 * n, 3 * n)
            )
            fields["pair_targets"].append(pairs)
            fields["diagonal_targets"].append(diagonal)
            fields["asymmetry_mse"].append(np.array([constant], dtype=np.float32))
            offset += width
        batch_ptr.append(batch_ptr[-1] + len(sizes))
    values = {key: np.concatenate(parts) for key, parts in fields.items() if parts}
    sizes = values["natoms"]
    values["atom_ptr"] = np.r_[0, np.cumsum(sizes)]
    values["pair_ptr"] = np.r_[0, np.cumsum(9 * sizes * (sizes - 1) // 2)]
    values["batch_ptr"] = np.asarray(batch_ptr, dtype=np.int64)
    scale = 1.0
    if "features" not in values:
        return values, scale
    if feature_dtype == "float16":
        maximum = float(np.abs(values["features"]).max())
        if not np.isfinite(maximum):
            raise ValueError("Frozen features must be finite before quantization")
        # Power-of-two scaling is exact in FP32 and protects FP16 range.
        scale = float(2.0 ** np.ceil(np.log2(max(maximum, 1e-30) / 32752.0)))
        values["features"] = (values["features"] / scale).astype(np.float16)
    elif feature_dtype != "float32":
        raise ValueError("Use float32 or float16 feature storage")
    return values, scale
