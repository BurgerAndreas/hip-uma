"""Read release labels on CPU workers before allocating feature-extraction GPUs.

Each process owns its HDF5 handles and HTTP client. Completed groups survive
restarts. Numerical work belongs on compute nodes, including this CPU stage.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import multiprocessing
import os
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from tools.prepare_frozen_hip_release import (
    concatenate,
    download_file,
    membership,
    open_packed,
    sample_fingerprint,
    selected_arrays,
    write_manifest,
)

_context = None


def release_groups(inventory, group_files):
    records = sorted(
        inventory["files"],
        key=lambda x: (x["parent_partition"], x["split"], x["path"]),
    )
    groups = []
    for partition, split in [(0, "train"), (1, "train"), (1, "validation")]:
        subset = [
            r for r in records
            if r["parent_partition"] == partition and r["split"] == split
        ]
        groups.extend(
            subset[i:i + group_files] for i in range(0, len(subset), group_files)
        )
    return groups


def initialize(args, inventory, legacy_ids, allowed):
    global _context
    # Construct clients after process creation. HTTP background threads must not
    # cross a fork boundary. One process performs its own sequential HDF5 reads.
    if args.base_read_mode == "sparse":
        from huggingface_hub import HfFileSystem

        args.hub_filesystem = HfFileSystem(token=args.token.read_text().strip())
    _context = (args, inventory, legacy_ids, allowed)


def source_file(record, args, inventory):
    if record["parent_partition"] != 0 or args.base_read_mode == "sparse":
        return download_file((record, args, inventory))
    # One sequential request can outperform many high-latency range requests.
    # Original source shards are bounded; reject unexpectedly large buffers.
    if record["bytes"] > 128 << 20:
        raise ValueError("Buffered original shard exceeds the memory bound")
    url = (
        "https://huggingface.co/datasets/" + inventory["repository"]
        + "/resolve/" + inventory["revision"] + "/" + record["path"]
    )
    request = urllib.request.Request(
        url, headers={"Authorization": "Bearer " + args.token.read_text().strip()}
    )
    buffer = io.BytesIO()
    digest = hashlib.sha256()
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            while chunk := response.read(8 << 20):
                buffer.write(chunk)
                digest.update(chunk)
                if buffer.tell() > record["bytes"]:
                    raise ValueError("Buffered source exceeds its published size")
        if buffer.tell() != record["bytes"] or digest.hexdigest() != record["sha256"]:
            raise ValueError("Buffered source checksum or size differs")
        buffer.seek(0)
        return buffer
    except BaseException:
        buffer.close()
        raise


def stage_group(task):
    index, group = task
    args, inventory, legacy_ids, allowed = _context
    split, partition = group[0]["split"], group[0]["parent_partition"]
    destination = args.output / split / f"release_{index:06d}.hip"
    receipt = destination.with_suffix(".json")
    marker = args.staged_output / f"{index:06d}.json"
    staged = marker.with_suffix(".npz")
    if marker.exists():
        previous = json.loads(marker.read_text())
        if previous["dataset_revision"] != inventory["revision"]:
            raise ValueError("Staging directory contains another release")
        if (previous.get("already_packed") and destination.exists() and receipt.exists()) or (
            not previous.get("already_packed") and (previous["samples"] == 0 or staged.exists())
        ):
            return dict(group=index, reused=True, samples=previous["samples"])
    start = time.monotonic()
    status = dict(group=index, dataset_revision=inventory["revision"], split=split)
    if destination.exists() and receipt.exists():
        entry = json.loads(receipt.read_text())
        status.update(already_packed=True, samples=len(entry["ids"]))
    else:
        parts = []
        for record in group:
            # Sparse range reads avoid downloading the full original dataset.
            # Independent processes overlap the latency without h5py's lock.
            for attempt in range(3):
                source = None
                try:
                    source = source_file(record, args, inventory)
                    values = selected_arrays(
                        source, record, legacy_ids, allowed[(partition, split)]
                    )
                    break
                except (OSError, TimeoutError):
                    if attempt == 2:
                        raise
                    time.sleep(2 ** attempt)
                finally:
                    if hasattr(source, "close"):
                        source.close()
            if values is not None:
                parts.append(values)
        values = concatenate(parts) if parts else None
        status.update(already_packed=False, samples=0 if values is None else len(values["natoms"]))
        if values is not None:
            temporary = staged.with_suffix(f".{os.getpid()}.tmp")
            with temporary.open("wb") as handle:
                # Uncompressed arrays keep CPU staging and GPU loading cheap.
                np.savez(handle, **values)
            temporary.replace(staged)
            status["sample_ids_sha256"] = sample_fingerprint(values["omol_index"])
    write_manifest(marker, status)
    return dict(group=index, seconds=time.monotonic() - start, **{
        k: status[k] for k in ["samples", "already_packed", "split"]
    })


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inventory", type=Path, required=True)
    p.add_argument("--manifests", type=Path, required=True)
    p.add_argument("--legacy-packed", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--staged-output", type=Path, required=True)
    p.add_argument("--token", type=Path, required=True)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--base-read-mode", choices=["sparse", "buffered"], default="sparse")
    p.add_argument("--group-files", type=int, default=4)
    p.add_argument("--limit-groups", type=int)
    p.add_argument("--group-indices", help="Comma-separated groups for a finite smoke test")
    args = p.parse_args()
    args.sparse_base = True
    inventory = json.loads(args.inventory.read_text())
    args.staged_output.mkdir(parents=True, exist_ok=True)
    legacy = []
    for path in sorted(args.legacy_packed.glob("*.hip")):
        arrays, _ = open_packed(path)
        legacy.append(np.asarray(arrays["omol_index"]).copy())
        target = args.output / "train" / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            target.symlink_to(path.resolve())
        except FileExistsError:
            pass
    legacy_ids = np.sort(np.concatenate(legacy))
    del legacy
    allowed = {
        (partition, split): np.sort(np.load(args.manifests / filename)["indices"])
        for partition, split, filename in [
            (0, "train", "sample_10m.npz"),
            (1, "train", "training_additions_800k.npz"),
            (1, "validation", "validation_200k.npz"),
        ]
    }
    if not membership(legacy_ids, allowed[(0, "train")]).all():
        raise ValueError("Legacy labels leave the training split")
    groups = release_groups(inventory, args.group_files)
    tasks = list(enumerate(groups))
    if args.group_indices:
        selected = {int(x) for x in args.group_indices.split(",")}
        if not selected or min(selected) < 0 or max(selected) >= len(groups):
            raise ValueError("Invalid group selection")
        tasks = [(i, g) for i, g in tasks if i in selected]
    if args.limit_groups is not None:
        tasks = tasks[:args.limit_groups]
    start = time.monotonic()
    errors = []
    samples = completed = 0
    # Fork shares the read-only legacy ID arrays. No CUDA or HTTP client exists
    # in the parent. Worker initialization creates each client's event loop.
    with ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=multiprocessing.get_context("fork"),
        initializer=initialize,
        initargs=(args, inventory, legacy_ids, allowed),
    ) as pool:
        futures = {pool.submit(stage_group, task): task[0] for task in tasks}
        for future in as_completed(futures):
            try:
                report = future.result()
                samples += report["samples"]
                completed += 1
                if not report.get("reused") or completed % 100 == 0:
                    print(json.dumps(dict(event="staged", **report)), flush=True)
            except Exception as error:
                report = dict(group=futures[future], error=repr(error))
                errors.append(report)
                print(json.dumps(dict(event="stage_error", **report)), flush=True)
    result = dict(
        dataset_revision=inventory["revision"], groups=len(tasks),
        completed=completed, samples=samples, errors=errors,
        seconds=time.monotonic() - start, workers=args.workers,
        complete=not errors and len(tasks) == len(groups),
    )
    name = "STAGED_READY.json" if result["complete"] else "STAGED_PARTIAL.json"
    write_manifest(args.staged_output / name, result)
    print(json.dumps(result), flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
