"""
Dataset reader for LMDBs containing pickled AtomicData dictionaries.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import lmdb
import numpy as np

from fairchem.core.common.registry import registry
from fairchem.core.datasets._utils import rename_data_object_keys
from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.datasets.base_dataset import BaseDataset
from fairchem.core.modules.transforms import DataTransforms


@registry.register_dataset("atomic_lmdb")
class AtomicLMDBDataset(BaseDataset):
    """Read AtomicData dictionaries written one sample per LMDB key.

    Expected keys are ASCII integer sample ids, plus a pickled ``length`` entry.
    Values are pickled dictionaries compatible with ``AtomicData.from_dict``.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        if len(self.paths) != 1:
            raise ValueError("atomic_lmdb expects exactly one src path")

        self.path = Path(self.paths[0])
        self.env = lmdb.open(
            str(self.path),
            subdir=False,
            readonly=True,
            lock=False,
            readahead=config.get("readahead", False),
            meminit=False,
            max_readers=1,
        )
        with self.env.begin() as txn:
            length_value = txn.get(b"length")
            if length_value is None:
                raise ValueError(f"Missing length key in atomic_lmdb: {self.path}")
            self.num_samples = int(pickle.loads(length_value))

        self.transforms = DataTransforms(config.get("transforms", {}))
        self.key_mapping = config.get("key_mapping", None)

    @property
    def indices(self):
        return np.arange(self.num_samples, dtype=int)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self[i] for i in range(*idx.indices(len(self)))]

        with self.env.begin() as txn:
            raw = txn.get(str(idx).encode("ascii"))
        if raw is None:
            raise KeyError(f"Missing sample {idx} in atomic_lmdb: {self.path}")

        data_object = AtomicData.from_dict(pickle.loads(raw))
        data_object = self.transforms(data_object)
        if self.key_mapping is not None:
            data_object = rename_data_object_keys(data_object, self.key_mapping)
        return data_object

    def __del__(self):
        env = getattr(self, "env", None)
        if env is not None:
            env.close()
