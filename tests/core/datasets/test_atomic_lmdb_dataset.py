from __future__ import annotations

import pickle

import lmdb
import numpy as np
import torch

from fairchem.core.datasets import create_dataset
from fairchem.core.datasets.atomic_data import AtomicData


def _atomic_dict(idx: int, natoms: int = 3):
    data = AtomicData(
        pos=torch.arange(natoms * 3, dtype=torch.float32).view(natoms, 3),
        atomic_numbers=torch.tensor([1, 6, 8], dtype=torch.long)[:natoms],
        cell=torch.zeros((1, 3, 3), dtype=torch.float32),
        pbc=torch.zeros((1, 3), dtype=torch.bool),
        natoms=torch.tensor([natoms], dtype=torch.long),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        cell_offsets=torch.empty((0, 3), dtype=torch.float32),
        nedges=torch.tensor([0], dtype=torch.long),
        charge=torch.tensor([0], dtype=torch.long),
        spin=torch.tensor([0], dtype=torch.long),
        fixed=torch.zeros(natoms, dtype=torch.long),
        tags=torch.ones(natoms, dtype=torch.long),
        energy=torch.tensor([float(idx)], dtype=torch.float32),
        forces=torch.ones((natoms, 3), dtype=torch.float32),
        sid=[str(idx)],
        dataset=["horm"],
    )
    out = data.to_dict()
    out["hessian"] = torch.zeros(natoms * 3 * natoms * 3, dtype=torch.float32)
    return out


def test_atomic_lmdb_dataset(tmp_path):
    lmdb_path = tmp_path / "tiny.atomic.lmdb"
    metadata_path = tmp_path / "tiny.atomic.metadata.npz"
    length = 2

    env = lmdb.open(str(lmdb_path), subdir=False, map_size=1 << 24)
    with env.begin(write=True) as txn:
        for idx in range(length):
            txn.put(
                str(idx).encode("ascii"),
                pickle.dumps(_atomic_dict(idx), protocol=pickle.HIGHEST_PROTOCOL),
            )
        txn.put(b"length", pickle.dumps(length, protocol=pickle.HIGHEST_PROTOCOL))
    env.close()
    np.savez(metadata_path, natoms=np.array([3, 3]), has_hessian=np.array([True, True]))

    dataset = create_dataset(
        {
            "format": "atomic_lmdb",
            "src": str(lmdb_path),
            "metadata_path": str(metadata_path),
            "no_shuffle": True,
        },
        split="train",
    )

    assert len(dataset) == length
    assert dataset.metadata_hasattr("natoms")
    assert dataset.get_metadata("natoms", 0) == 3
    sample = dataset[0]
    assert torch.equal(sample.atomic_numbers, torch.tensor([1, 6, 8]))
    assert sample.energy.shape == (1,)
    assert sample.forces.shape == (3, 3)
    assert sample.hessian.shape == (81,)
