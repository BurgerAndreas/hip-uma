"""Convert frozen feature caches to explicit, bounded-error FP16 storage.

Run this CPU tool on a compute node. Source arrays remain unchanged. Training
must opt in to FP16 cache storage; parameters and Cartesian outputs stay FP32.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from fairchem.core.frozen_hip.cache_quantization import (
    quantization_statistics,
    relative_error_bound,
)


def convert_file(task):
    marker, source_root, output_root, chunk_elements, max_relative_error = task
    relative = marker.relative_to(source_root)
    destination = output_root / relative.with_suffix(".npy")
    output_marker = destination.with_suffix(".json")
    metadata = json.loads(marker.read_text())
    if metadata.get("storage_dtype", "float32") != "float32":
        raise ValueError("Conversion requires FP32 source storage")
    source = np.load(marker.with_suffix(".npy"), mmap_mode="r", allow_pickle=False)
    if source.dtype != np.float32 or list(source.shape) != metadata["shape"]:
        raise ValueError("Source cache shape or dtype differs")
    if output_marker.exists() and destination.exists():
        existing = json.loads(output_marker.read_text())
        cached = np.load(destination, mmap_mode="r", allow_pickle=False)
        if (
            existing["source_sha256"] == metadata["source_sha256"]
            and existing["shape"] == metadata["shape"]
            and existing.get("storage_dtype") == "float16"
            and cached.dtype == np.float16
            and list(cached.shape) == metadata["shape"]
            and existing.get("finite") is True
            and existing.get(
                "quantization_relative_l2_upper_bound",
                existing["quantization_relative_l2"],
            )
            <= max_relative_error
        ):
            return dict(
                file=str(relative), reused=True, bytes=destination.stat().st_size
            )
        raise ValueError("Existing converted cache has a different identity")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + f".{os.getpid()}.partial")
    temporary_marker = output_marker.with_name(
        output_marker.name + f".{os.getpid()}.partial"
    )
    started = time.monotonic()
    try:
        output = np.lib.format.open_memmap(
            temporary,
            mode="w+",
            dtype=np.float16,
            shape=source.shape,
        )
        flat, target = source.reshape(-1), output.reshape(-1)
        squared_error = squared_reference = maximum_error = 0.0
        digest = hashlib.sha256()
        for start in range(0, flat.size, chunk_elements):
            block = np.asarray(flat[start : start + chunk_elements])
            if not np.isfinite(block).all():
                raise ValueError("Source cache contains nonfinite values")
            if np.max(np.abs(block), initial=0) > np.finfo(np.float16).max:
                raise ValueError("Source features exceed the FP16 storage range")
            stored = block.astype(np.float16)
            error, reference, maximum = quantization_statistics(block, stored)
            squared_error += error
            squared_reference += reference
            maximum_error = max(maximum_error, maximum)
            target[start : start + len(block)] = stored
            digest.update(stored.tobytes())
        relative_error, upper_bound = relative_error_bound(
            squared_error, squared_reference
        )
        if upper_bound > max_relative_error:
            raise ValueError("Feature quantization exceeds the requested error bound")
        output.flush()
        del target, output
        converted = {
            **metadata,
            "storage_dtype": "float16",
            "finite": True,
            "quantization_relative_l2": relative_error,
            "quantization_relative_l2_upper_bound": upper_bound,
            "quantization_max_abs": maximum_error,
            "storage_payload_sha256": digest.hexdigest(),
        }
        temporary_marker.write_text(json.dumps(converted, indent=2) + "\n")
        temporary.replace(destination)
        temporary_marker.replace(output_marker)
        return dict(
            file=str(relative),
            reused=False,
            bytes=destination.stat().st_size,
            relative_error=relative_error,
            seconds=time.monotonic() - started,
        )
    finally:
        temporary.unlink(missing_ok=True)
        temporary_marker.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk-elements", type=int, default=1 << 20)
    parser.add_argument("--max-relative-error", type=float, default=0.0003)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.source.resolve() == args.output.resolve():
        parser.error("Source and output must differ")
    if (
        args.workers < 1
        or args.chunk_elements < 1
        or not 0 < args.max_relative_error < 1
    ):
        parser.error("Workers, chunks, and the finite error bound must be positive")
    markers = sorted(
        p for p in args.source.rglob("*.json") if p.with_suffix(".npy").exists()
    )
    if args.limit is not None:
        if args.limit < 1:
            parser.error("Limit must be positive")
        markers = markers[: args.limit]
    if not markers:
        raise ValueError("No complete source feature arrays found")
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    completed = total_bytes = 0
    maximum_error = 0.0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(
                convert_file,
                (
                    p,
                    args.source,
                    args.output,
                    args.chunk_elements,
                    args.max_relative_error,
                ),
            )
            for p in markers
        ]
        for future in as_completed(futures):
            report = future.result()
            completed += 1
            total_bytes += report["bytes"]
            maximum_error = max(maximum_error, report.get("relative_error", 0.0))
            if completed % 25 == 0 or completed == len(markers):
                print(
                    json.dumps(
                        dict(
                            event="cache_converted",
                            completed=completed,
                            files=len(markers),
                            bytes=total_bytes,
                            seconds=time.monotonic() - started,
                        )
                    ),
                    flush=True,
                )
    report = dict(
        complete=True,
        files=completed,
        bytes=total_bytes,
        maximum_relative_error=maximum_error,
        seconds=time.monotonic() - started,
        storage_dtype="float16",
        source_unchanged=True,
    )
    temporary = args.output / f"READY.{os.getpid()}.tmp"
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(args.output / "READY.json")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
