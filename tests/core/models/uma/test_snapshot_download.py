"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Verify that failed transfers never publish scientific files or ready markers.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


def download_module():
    path = Path(__file__).resolve().parents[4] / "tools/download_verified_snapshot.py"
    spec = importlib.util.spec_from_file_location("snapshot_download", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Response:
    status_code = 200

    def __init__(self, payload):
        self.payload = payload
        self.headers = {}

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def iter_content(self, _):
        yield self.payload


class Session:
    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    def get(self, *_, **__):
        self.calls += 1
        return Response(self.payload)


def test_verified_download_publishes_once_and_resumes(tmp_path):
    module = download_module()
    payload = b"dense scientific content"
    row = {
        "path": "shards/00/shard_00000.h5",
        "size": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "samples": 3,
    }
    session = Session(payload)
    assert module.download_one(
        session, tmp_path, "example/data", "a" * 40, row, None
    ) == len(payload)
    assert module.verified(tmp_path, row)
    assert (
        module.download_one(session, tmp_path, "example/data", "a" * 40, row, None) == 0
    )
    assert session.calls == 1
    assert (
        json.loads((tmp_path / row["path"]).with_suffix(".verified.json").read_text())
        == row
    )


def test_checksum_failure_does_not_publish(tmp_path):
    module = download_module()
    row = {
        "path": "shards/00/shard_00000.h5",
        "size": 3,
        "sha256": hashlib.sha256(b"abc").hexdigest(),
        "samples": 1,
    }
    with pytest.raises(ValueError, match="checksum"):
        module.download_one(
            Session(b"bad"), tmp_path, "example/data", "a" * 40, row, None
        )
    assert not (tmp_path / row["path"]).exists()
    assert not list(tmp_path.rglob("*.verified.json"))
    assert not list(tmp_path.rglob("*.tmp"))
