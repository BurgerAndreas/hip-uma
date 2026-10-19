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


class ThreadedDeviceBatchPrefetcher:
    """Prepare the next device batch without blocking delivery of the ready one."""

    def __init__(self, source, move, device):
        import queue
        import threading

        self.source, self.move, self.device = source, move, device
        self.stream = torch.cuda.Stream(device=device)
        self.ready = queue.Queue(maxsize=1)
        self.stopped = threading.Event()
        self.exhausted = False
        self.worker = threading.Thread(target=self._produce, daemon=True)
        self.worker.start()

    def __iter__(self):
        return self

    def _publish(self, item):
        import queue

        while not self.stopped.is_set():
            try:
                self.ready.put(item, timeout=0.1)
                return
            except queue.Full:
                continue

    def _produce(self):
        try:
            with torch.cuda.device(self.device), torch.cuda.stream(self.stream):
                for host in self.source:
                    if self.stopped.is_set():
                        return
                    batch = [self.move(data) for data in host]
                    event = torch.cuda.Event()
                    event.record(self.stream)
                    self._publish((batch, event, None))
                self._publish((None, None, None))
        except BaseException as error:
            self._publish((None, None, error))

    def __next__(self):
        if self.exhausted:
            raise StopIteration
        batch, event, error = self.ready.get()
        if error is not None:
            self.exhausted = True
            raise error
        if batch is None:
            self.exhausted = True
            raise StopIteration
        current = torch.cuda.current_stream(self.device)
        current.wait_event(event)

        def record(value):
            if value.is_cuda:
                value.record_stream(current)
            return value

        for data in batch:
            data.apply(record)
        return batch

    def _shutdown_workers(self):
        self.exhausted = True
        self.stopped.set()
        self.worker.join(timeout=2)
        if hasattr(self.source, "_shutdown_workers"):
            self.source._shutdown_workers()
        self.worker.join(timeout=2)
        self.stream.synchronize()
        while not self.ready.empty():
            self.ready.get_nowait()
