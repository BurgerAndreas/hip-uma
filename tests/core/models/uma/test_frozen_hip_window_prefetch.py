"""Verify overlap, bounded read-ahead, order, and loader error propagation."""

from __future__ import annotations

from threading import Event

import pytest

from fairchem.core.frozen_hip.window_prefetch import prefetch_windows


def test_window_prefetch_overlaps_only_one_future_window():
    started, release, third = Event(), Event(), Event()

    def load(index):
        if index == 1:
            started.set()
            assert release.wait(5)
        if index == 2:
            third.set()
        return index

    windows = prefetch_windows(load, range(3))
    try:
        assert next(windows) == 0
        assert started.wait(5)
        assert not third.is_set()
        release.set()
        assert next(windows) == 1
        assert next(windows) == 2
        with pytest.raises(StopIteration):
            next(windows)
    finally:
        release.set()
        windows.close()


def test_window_prefetch_propagates_failure_after_ready_window():
    def load(index):
        if index:
            raise ValueError("invalid window")
        return index

    windows = prefetch_windows(load, range(2))
    assert next(windows) == 0
    with pytest.raises(ValueError, match="invalid window"):
        next(windows)


def test_window_prefetch_accepts_empty_sources():
    assert list(prefetch_windows(lambda _: pytest.fail("unexpected read"), [])) == []
