"""Test background delivery without requiring a GPU or the model runtime."""

from __future__ import annotations

import importlib.util
import sys
import threading
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def prefetcher(monkeypatch):
    class Stream:
        def wait_event(self, event):
            assert event.recorded

        def synchronize(self):
            pass

    class Event:
        def __init__(self):
            self.recorded = False

        def record(self, stream):
            self.recorded = True

    stream = Stream()
    cuda = SimpleNamespace(
        Stream=lambda **kwargs: Stream(),
        Event=Event,
        device=lambda device: nullcontext(),
        stream=lambda stream: nullcontext(),
        current_stream=lambda device: stream,
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    source = (
        Path(__file__).resolve().parents[4] / "src/fairchem/core/frozen_hip/prefetch.py"
    )
    spec = importlib.util.spec_from_file_location("isolated_prefetch", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ThreadedDeviceBatchPrefetcher


class Batch:
    def __init__(self, value):
        self.value = value
        self.is_cuda = True
        self.recorded = False

    def apply(self, operation):
        operation(self)

    def record_stream(self, stream):
        self.recorded = True


def test_ready_batch_does_not_wait_for_the_next_host_batch(prefetcher):
    blocked, release = threading.Event(), threading.Event()

    def source():
        yield [Batch(1)]
        blocked.set()
        assert release.wait(timeout=5)
        yield [Batch(2)]

    iterator = prefetcher(source(), lambda batch: batch, "cuda")
    try:
        first = next(iterator)
        assert first[0].value == 1
        assert first[0].recorded
        assert blocked.wait(timeout=1)
        assert not release.is_set()
        release.set()
        assert next(iterator)[0].value == 2
        with pytest.raises(StopIteration):
            next(iterator)
    finally:
        release.set()
        iterator._shutdown_workers()
    assert not iterator.worker.is_alive()


def test_producer_error_reaches_consumer_after_previous_batches(prefetcher):
    def source():
        yield [Batch(7)]
        raise ValueError("source failed")

    iterator = prefetcher(source(), lambda batch: batch, "cuda")
    try:
        assert next(iterator)[0].value == 7
        with pytest.raises(ValueError, match="source failed"):
            next(iterator)
        with pytest.raises(StopIteration):
            next(iterator)
    finally:
        iterator._shutdown_workers()


def test_shutdown_unblocks_full_queue_and_closes_loader(prefetcher):
    class Source:
        closed = False

        def __iter__(self):
            while True:
                yield [Batch(1)]

        def _shutdown_workers(self):
            self.closed = True

    source = Source()
    iterator = prefetcher(source, lambda batch: batch, "cuda")
    assert next(iterator)[0].value == 1
    iterator._shutdown_workers()
    assert source.closed
    assert not iterator.worker.is_alive()
    assert iterator.ready.empty()
