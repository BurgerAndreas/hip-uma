"""Load one future shard window while the caller consumes the current window."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor


def prefetch_windows(load, sources):
    """Preserve order and propagate load failures with one outstanding future."""
    sources = iter(sources)
    try:
        first = next(sources)
    except StopIteration:
        return
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hip-window")
    pending = executor.submit(load, first)
    try:
        while pending is not None:
            result = pending.result()
            try:
                source = next(sources)
            except StopIteration:
                pending = None
            else:
                pending = executor.submit(load, source)
            yield result
            del result
    finally:
        if pending is not None:
            pending.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
