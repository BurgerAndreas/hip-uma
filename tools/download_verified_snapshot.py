"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Download an immutable dataset snapshot with streaming SHA-256 verification.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import threading
import time
from pathlib import Path

import requests
from huggingface_hub import get_token


def verified(root, row):
    path = root / row["path"]
    marker = path.with_suffix(".verified.json")
    if not marker.exists() or not path.exists() or path.stat().st_size != row["size"]:
        return False
    try:
        return json.loads(marker.read_text())["sha256"] == row["sha256"]
    except (ValueError, KeyError):
        return False


def download_one(session, root, repo, revision, row, token):
    """Expose a file and its marker only after content verification succeeds."""
    if verified(root, row):
        return 0
    path = root / row["path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = f".{os.getpid()}.{threading.get_ident()}.tmp"
    temporary = path.with_suffix(suffix)
    url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{row['path']}"
    headers = {"Accept-Encoding": "identity"}
    if token:
        headers["Authorization"] = "Bearer " + token
    try:
        with session.get(
            url, headers=headers, stream=True, timeout=(20, 120)
        ) as response:
            if response.status_code != 200:
                # Never include signed redirect URLs or authentication in logs.
                error = RuntimeError(f"Download returned HTTP {response.status_code}")
                error.retry_after = response.headers.get("Retry-After")
                raise error
            digest = hashlib.sha256()
            with temporary.open("wb") as handle:
                for block in response.iter_content(4 << 20):
                    if block:
                        handle.write(block)
                        digest.update(block)
            if (
                temporary.stat().st_size != row["size"]
                or digest.hexdigest() != row["sha256"]
            ):
                raise ValueError(
                    "Downloaded content differs from the snapshot checksum"
                )
        temporary.replace(path)
        marker = path.with_suffix(".verified.json")
        temporary_marker = marker.with_suffix(suffix)
        temporary_marker.write_text(json.dumps(row) + "\n")
        temporary_marker.replace(marker)
        return row["size"]
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", args.repo):
        raise ValueError("Invalid dataset repository identifier")
    manifest = json.loads(args.manifest.read_text())
    revision = manifest["data_revision"]
    if not re.fullmatch(r"[a-f0-9]{40}", revision):
        raise ValueError("Require an immutable dataset revision")
    rows = manifest["files"]
    for row in rows:
        if not re.fullmatch(r"shards/\d{2}/shard_\d{5}\.h5", row["path"]):
            raise ValueError("Invalid canonical shard path")
        if not re.fullmatch(r"[a-f0-9]{64}", row["sha256"]):
            raise ValueError("Invalid snapshot checksum")
    token = get_token()
    local = threading.local()
    started = time.monotonic()

    def transfer(row):
        if not hasattr(local, "session"):
            local.session = requests.Session()
        for attempt in range(12):
            try:
                return download_one(
                    local.session, args.output, args.repo, revision, row, token
                )
            except (requests.RequestException, RuntimeError, ValueError) as error:
                if attempt == 11:
                    raise RuntimeError(
                        f"Cannot verify {row['path']}: {type(error).__name__}"
                    ) from None
                retry = getattr(error, "retry_after", None)
                delay = (
                    float(retry) if retry and retry.isdigit() else min(120, 2**attempt)
                )
                print(
                    json.dumps(
                        {
                            "event": "download_retry",
                            "path": row["path"],
                            "attempt": attempt + 1,
                            "delay_seconds": delay,
                            "error_type": type(error).__name__,
                        }
                    ),
                    flush=True,
                )
                time.sleep(delay)
        raise AssertionError("Unreachable retry state")

    done = 0
    transferred = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(transfer, row) for row in rows]
        for future in concurrent.futures.as_completed(futures):
            transferred += future.result()
            done += 1
            if done % 256 == 0:
                print(
                    json.dumps(
                        {
                            "verified_shards": done,
                            "total": len(rows),
                            "new_bytes": transferred,
                            "MB_s": transferred / 1e6 / (time.monotonic() - started),
                        }
                    ),
                    flush=True,
                )
    assert all(verified(args.output, row) for row in rows)
    summary = {
        "shards": len(rows),
        "samples": sum(row["samples"] for row in rows),
        "bytes": sum(row["size"] for row in rows),
        "revision": revision,
        "new_bytes": transferred,
        "seconds": time.monotonic() - started,
    }
    output = args.output.parent / "DOWNLOAD_COMPLETE.json"
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(summary, indent=2) + "\n")
    temporary.replace(output)
    print(json.dumps({"event": "download_complete", **summary}), flush=True)


if __name__ == "__main__":
    main()
