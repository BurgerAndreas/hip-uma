#!/usr/bin/env python3
"""
Convert HIP PyG-pickled LMDB samples to FAIRChem AtomicData dictionaries.

The input HIP LMDB stores torch_geometric.data.Data objects. This converter
requires torch-geometric only while converting. The output LMDB stores plain
pickled dictionaries that can be read back as fairchem.core.datasets.AtomicData.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import lmdb
import numpy as np
import torch

from fairchem.core.datasets.atomic_data import AtomicData


ONE_HOT_Z = torch.tensor([1, 6, 7, 8, 9], dtype=torch.long)


def _get_attr(data: Any, key: str, default: Any = None) -> Any:
    if hasattr(data, key):
        return getattr(data, key)
    try:
        return data[key]
    except Exception:
        return default


def _as_tensor(value: Any, dtype: torch.dtype | None = None) -> torch.Tensor:
    tensor = value if torch.is_tensor(value) else torch.as_tensor(value)
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor.detach().cpu()


def _atomic_numbers(data: Any) -> torch.Tensor:
    z = _get_attr(data, "z")
    if z is not None:
        return _as_tensor(z, torch.long).view(-1)

    atomic_numbers = _get_attr(data, "atomic_numbers")
    if atomic_numbers is not None:
        return _as_tensor(atomic_numbers, torch.long).view(-1)

    one_hot = _get_attr(data, "one_hot")
    if one_hot is None:
        raise KeyError("Sample has neither z, atomic_numbers, nor one_hot")
    one_hot = _as_tensor(one_hot)
    return ONE_HOT_Z[one_hot.long().argmax(dim=1)]


def _energy(data: Any, dtype: torch.dtype) -> torch.Tensor:
    for key in ("energy", "ae", "y"):
        value = _get_attr(data, key)
        if value is not None:
            return _as_tensor(value, dtype).view(1)
    raise KeyError("Sample has no energy, ae, or y field")


def _forces(data: Any, dtype: torch.dtype) -> torch.Tensor:
    value = _get_attr(data, "forces")
    if value is None:
        value = _get_attr(data, "force")
    if value is None:
        raise KeyError("Sample has no forces or force field")
    return _as_tensor(value, dtype).view(-1, 3)


def _optional_tensor(data: Any, key: str, dtype: torch.dtype | None = None):
    value = _get_attr(data, key)
    if value is None:
        return None
    return _as_tensor(value, dtype)


def sample_to_atomic_data_dict(
    data: Any,
    idx: int,
    *,
    dataset_name: str,
    dtype: torch.dtype,
    include_hessian: bool,
) -> dict[str, Any]:
    pos = _as_tensor(_get_attr(data, "pos"), dtype).view(-1, 3)
    atomic_numbers = _atomic_numbers(data)
    natoms = torch.tensor([pos.shape[0]], dtype=torch.long)

    cell_value = _get_attr(data, "cell")
    if cell_value is None:
        cell = torch.zeros((1, 3, 3), dtype=dtype)
    else:
        cell = _as_tensor(cell_value, dtype)
        if cell.numel() == 9:
            cell = cell.view(1, 3, 3)
        elif cell.shape == (3, 3):
            cell = cell.view(1, 3, 3)
        elif cell.shape != (1, 3, 3):
            raise ValueError(f"Unsupported cell shape for sample {idx}: {cell.shape}")

    pbc_value = _get_attr(data, "pbc")
    if pbc_value is None:
        pbc = torch.zeros((1, 3), dtype=torch.bool)
    else:
        pbc = _as_tensor(pbc_value, torch.bool).view(1, 3)

    atomic_data = AtomicData(
        pos=pos,
        atomic_numbers=atomic_numbers,
        cell=cell,
        pbc=pbc,
        natoms=natoms,
        edge_index=torch.empty((2, 0), dtype=torch.long),
        cell_offsets=torch.empty((0, 3), dtype=dtype),
        nedges=torch.tensor([0], dtype=torch.long),
        charge=torch.tensor([int(_get_attr(data, "charge", 0))], dtype=torch.long),
        spin=torch.tensor([int(_get_attr(data, "spin", 0))], dtype=torch.long),
        fixed=torch.zeros(pos.shape[0], dtype=torch.long),
        tags=torch.ones(pos.shape[0], dtype=torch.long),
        energy=_energy(data, dtype),
        forces=_forces(data, dtype),
        sid=[str(_get_attr(data, "id", idx))],
        dataset=[dataset_name],
    )
    out = atomic_data.to_dict()

    if include_hessian:
        hessian = _optional_tensor(data, "hessian", dtype)
        if hessian is not None:
            out["hessian"] = hessian.reshape(-1)

    dataset_idx = _optional_tensor(data, "dataset_idx", torch.long)
    out["source_idx"] = (
        dataset_idx.view(1) if dataset_idx is not None else torch.tensor([idx])
    )
    return out


def convert(
    input_path: Path,
    output_path: Path,
    metadata_path: Path,
    *,
    dataset_name: str,
    dtype: torch.dtype,
    include_hessian: bool,
    map_size: int,
    commit_every: int,
    progress_every: int,
) -> None:
    if output_path.exists():
        output_path.unlink()
    lock_path = output_path.with_name(output_path.name + "-lock")
    if lock_path.exists():
        lock_path.unlink()
    if metadata_path.exists():
        metadata_path.unlink()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    src_env = lmdb.open(
        str(input_path),
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=1,
    )
    dst_env = lmdb.open(
        str(output_path),
        subdir=False,
        readonly=False,
        lock=True,
        readahead=False,
        meminit=False,
        map_size=map_size,
    )

    natoms = np.empty(0, dtype=np.int64)
    has_hessian = np.empty(0, dtype=np.bool_)
    with src_env.begin() as src_txn:
        length_value = src_txn.get(b"length")
        if length_value is None:
            length = src_env.stat()["entries"]
        else:
            length = pickle.loads(length_value)

        natoms = np.empty(length, dtype=np.int64)
        has_hessian = np.empty(length, dtype=np.bool_)

        dst_txn = dst_env.begin(write=True)
        try:
            for idx in range(length):
                raw = src_txn.get(str(idx).encode("ascii"))
                if raw is None:
                    raise KeyError(f"Missing source LMDB sample {idx}")
                source_data = pickle.loads(raw)
                atomic_dict = sample_to_atomic_data_dict(
                    source_data,
                    idx,
                    dataset_name=dataset_name,
                    dtype=dtype,
                    include_hessian=include_hessian,
                )
                dst_txn.put(
                    str(idx).encode("ascii"),
                    pickle.dumps(atomic_dict, protocol=pickle.HIGHEST_PROTOCOL),
                )
                natoms[idx] = int(atomic_dict["natoms"].item())
                has_hessian[idx] = "hessian" in atomic_dict

                if (idx + 1) % commit_every == 0:
                    dst_txn.commit()
                    dst_env.sync()
                    dst_txn = dst_env.begin(write=True)

                if progress_every > 0 and (idx + 1) % progress_every == 0:
                    print(f"Converted {idx + 1}/{length}", flush=True)
            dst_txn.put(b"length", pickle.dumps(length, protocol=pickle.HIGHEST_PROTOCOL))
            dst_txn.commit()
        except Exception:
            dst_txn.abort()
            raise

    np.savez_compressed(
        metadata_path,
        natoms=natoms,
        has_hessian=has_hessian,
    )
    src_env.close()
    dst_env.sync()
    dst_env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/hip_sample_100/sample_100.lmdb")
    parser.add_argument(
        "--output", default="data/hip_sample_100/sample_100.atomic.lmdb"
    )
    parser.add_argument(
        "--metadata", default="data/hip_sample_100/sample_100.atomic.metadata.npz"
    )
    parser.add_argument("--dataset-name", default="hip_sample_100")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--no-hessian", action="store_true")
    parser.add_argument("--map-size-gb", type=float, default=128.0)
    parser.add_argument("--commit-every", type=int, default=10_000)
    parser.add_argument("--progress-every", type=int, default=50_000)
    args = parser.parse_args()

    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    convert(
        Path(args.input),
        Path(args.output),
        Path(args.metadata),
        dataset_name=args.dataset_name,
        dtype=dtype,
        include_hessian=not args.no_hessian,
        map_size=int(args.map_size_gb * 1024**3),
        commit_every=args.commit_every,
        progress_every=args.progress_every,
    )
    print(f"Wrote {args.output}")
    print(f"Wrote {args.metadata}")


if __name__ == "__main__":
    main()
