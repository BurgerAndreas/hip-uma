"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Reuse exact frozen features after the first training pass over each shard.
"""

from __future__ import annotations

import json
import os
from collections import OrderedDict
from pathlib import Path

import numpy as np


def cached_features(root, key, source_sha256):
    """Open only complete feature arrays tied to the source content hash."""
    path = Path(root) / f"{key}.npy"
    marker = path.with_suffix(".json")
    if not path.exists() or not marker.exists():
        return None
    meta = json.loads(marker.read_text())
    if meta["source_sha256"] != source_sha256:
        raise ValueError("Cached features refer to different scientific data")
    values = np.load(path, mmap_mode="r")
    if list(values.shape) != meta["shape"] or values.dtype != np.float32:
        raise ValueError("Frozen feature cache shape or dtype differs")
    return values


class FeatureCacheWriter:
    """Publish a shard only after all of its training rows have exact features.

    Incomplete arrays remain in bounded host memory. Atomic file replacement
    prevents interrupted jobs from exposing partial features to loader workers.
    Validation features are always recomputed and never read from this cache.
    """

    def __init__(self, root, max_bytes=2048 << 20):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes
        self.active = OrderedDict()

    def store(self, data, features):
        host = features.detach().float().cpu().numpy()
        if hasattr(data, "hip_cache_groups"):
            for group in data.hip_cache_groups:
                self._store(
                    group, host[group.hip_feature_start : group.hip_feature_end]
                )
        else:
            self._store(data, host)

    def _store(self, data, host):
        key = data.hip_cache_key
        if (self.root / f"{key}.json").exists():
            return
        if key not in self.active:
            array = np.zeros((data.hip_total_atoms, *host.shape[1:]), dtype=np.float32)
            seen = np.zeros(data.hip_row_count, dtype=bool)
            self.active[key] = (array, seen)
        self.active.move_to_end(key)
        array, seen = self.active[key]
        start = 0
        for row, (first, last) in zip(data.hip_rows, data.hip_atom_spans, strict=True):
            array[first:last] = host[start : start + last - first]
            seen[row] = True
            start += last - first
        if seen[data.hip_training_rows].all():
            path = self.root / f"{key}.npy"
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(f".{os.getpid()}.tmp")
            with tmp.open("wb") as f:
                np.save(f, array)
                f.flush()
                os.fsync(f.fileno())
            tmp.replace(path)
            marker = path.with_suffix(".json")
            temporary = marker.with_suffix(f".{os.getpid()}.tmp")
            temporary.write_text(
                json.dumps(
                    {
                        "source_sha256": data.hip_data_sha256,
                        "shape": list(array.shape),
                        "training_only": True,
                    }
                )
                + "\n"
            )
            temporary.replace(marker)
            del self.active[key]
        while sum(a.nbytes for a, _ in self.active.values()) > self.max_bytes:
            self.active.popitem(last=False)


class AsyncFeatureCacheWriter:
    """
    Copy frozen features asynchronously and publish files outside GPU dispatch.

    A bounded queue owns pinned host tensors and CPU-only scatter metadata.
    CUDA events protect reads; stream recording protects GPU storage lifetimes.
    """

    def __init__(self, root, max_bytes=16 << 30, max_pending=32):
        import queue
        import threading

        if max_bytes < 1 or max_pending < 1:
            raise ValueError("Feature cache memory and queue limits must be positive")
        self.writer = FeatureCacheWriter(root, max_bytes=max_bytes)
        self.pending = queue.Queue(maxsize=max_pending)
        self.streams = {}
        self.error = None
        self.closed = False
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while True:
            item = self.pending.get()
            try:
                if item is None:
                    return
                metadata, host, event = item
                if event is not None:
                    event.synchronize()
                if self.error is None:
                    self.writer.store(metadata, host)
            except BaseException as error:
                self.error = error
            finally:
                self.pending.task_done()

    def _check_error(self):
        if self.error is not None:
            raise RuntimeError("Frozen feature cache writer failed") from self.error

    def store(self, data, features):
        import copy
        from types import SimpleNamespace

        import torch

        if self.closed:
            raise RuntimeError("Frozen feature cache writer is closed")
        self._check_error()
        if hasattr(data, "hip_cache_groups"):
            metadata = SimpleNamespace(
                hip_cache_groups=copy.deepcopy(data.hip_cache_groups)
            )
        else:
            keys = (
                "hip_cache_key",
                "hip_total_atoms",
                "hip_row_count",
                "hip_rows",
                "hip_atom_spans",
                "hip_training_rows",
                "hip_data_sha256",
            )
            metadata = SimpleNamespace(
                **{key: copy.deepcopy(getattr(data, key)) for key in keys}
            )
        values = features.detach().float()
        event = None
        if values.is_cuda:
            device = values.device
            if device not in self.streams:
                self.streams[device] = torch.cuda.Stream(device=device)
            stream = self.streams[device]
            host = torch.empty(values.shape, dtype=torch.float32, pin_memory=True)
            stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(stream):
                host.copy_(values, non_blocking=True)
                values.record_stream(stream)
                event = torch.cuda.Event()
                event.record(stream)
        else:
            host = values.clone()
        self.pending.put((metadata, host, event))
        self._check_error()

    def close(self):
        """
        Drain writes before successful exit and propagate background failures.
        """
        if not self.closed:
            self.closed = True
            self.pending.put(None)
            self.thread.join()
        self._check_error()
