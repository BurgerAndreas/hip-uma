"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Overlap one logical batch transfer with the preceding GPU update.
"""

from __future__ import annotations

import torch


class DeviceBatchPrefetcher:
    """Retain one prepared batch and track its tensors across CUDA streams."""

    def __init__(self, source, move, device):
        self.source = source
        self.move = move
        self.device = device
        self.stream = torch.cuda.Stream(device=device)
        self.pending = None
        self.exhausted = False

    def __iter__(self):
        return self

    def _preload(self):
        try:
            host = next(self.source)
        except StopIteration:
            self.exhausted = True
            self.pending = None
            return
        with torch.cuda.stream(self.stream):
            self.pending = [self.move(data) for data in host]

    def __next__(self):
        if self.pending is None and not self.exhausted:
            self._preload()
        if self.pending is None:
            raise StopIteration
        current = torch.cuda.current_stream(self.device)
        current.wait_stream(self.stream)
        batch = self.pending

        def record(value):
            if value.is_cuda:
                value.record_stream(current)
            return value

        for data in batch:
            data.apply(record)
        self._preload()
        return batch

    def _shutdown_workers(self):
        self.stream.synchronize()
        self.pending = None
        if hasattr(self.source, "_shutdown_workers"):
            self.source._shutdown_workers()
