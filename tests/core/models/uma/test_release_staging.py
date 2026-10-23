"""CPU staging preserves Hessians and split identities across restarts."""

from __future__ import annotations

import hashlib
import io
import json
import pickle
import urllib.error
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from tools import stage_frozen_hip_release as staging


def test_staged_labels_roundtrip_and_restart(tmp_path, monkeypatch):
    path = tmp_path / "labels.h5"
    hessian = np.arange(36, dtype=np.float32).reshape(6, 6)
    hessian = (hessian + hessian.T) * 0.5
    with h5py.File(path, "w") as f:
        f.attrs.update(
            model="uma-s-1p2",
            hessian_unit="eV/Angstrom^2",
            parent_partition=1,
            split="validation",
            complete=True,
        )
        for key, value in dict(
            omol_index=[4],
            natoms=[2],
            atom_ptr=[0, 2],
            hessian_ptr=[0, 21],
            charge=[0],
            spin=[1],
            atomic_numbers=[1, 1],
            coords=np.zeros((2, 3), np.float32),
            hessian_upper=hessian[np.triu_indices(6)],
        ).items():
            f[key] = value
    args = SimpleNamespace(
        output=tmp_path / "packed",
        staged_output=tmp_path / "staged",
        base_read_mode="sparse",
    )
    args.staged_output.mkdir()
    inventory = dict(revision="a" * 40)
    record = dict(parent_partition=1, split="validation", samples=1)
    monkeypatch.setattr(
        staging,
        "_context",
        (
            args,
            inventory,
            np.array([4]),
            {(1, "validation"): np.array([4])},
        ),
    )
    monkeypatch.setattr(staging, "download_file", lambda task: path)
    report = staging.stage_group((7, [record]))
    assert report["samples"] == 1
    with np.load(args.staged_output / "000007.npz", allow_pickle=False) as arrays:
        np.testing.assert_array_equal(arrays["hessian_flat"].reshape(6, 6), hessian)
        assert arrays["omol_index"].tolist() == [(1 << 60) + 4]
    monkeypatch.setattr(
        staging, "download_file", lambda task: pytest.fail("Repeated read")
    )
    assert staging.stage_group((7, [record]))["reused"]
    marker = args.staged_output / "000007.json"
    obj = json.loads(marker.read_text())
    obj["dataset_revision"] = "b" * 40
    marker.write_text(json.dumps(obj))
    with pytest.raises(ValueError, match="another release"):
        staging.stage_group((7, [record]))


def test_staging_group_order_matches_production():
    rows = [
        dict(parent_partition=1, split="validation", path="v"),
        dict(parent_partition=0, split="train", path="b"),
        dict(parent_partition=1, split="train", path="n"),
        dict(parent_partition=0, split="train", path="a"),
    ]
    groups = staging.release_groups(dict(files=rows), 2)
    assert [[r["path"] for r in group] for group in groups] == [
        ["a", "b"],
        ["n"],
        ["v"],
    ]


def test_buffered_source_checks_full_content(tmp_path, monkeypatch):
    contents = b"immutable source bytes"
    token = tmp_path / "credential"
    token.write_text("test-token")
    args = SimpleNamespace(token=token, base_read_mode="buffered")
    inventory = dict(repository="example/labels", revision="a" * 40)
    record = dict(
        parent_partition=0,
        path="source.h5",
        bytes=len(contents),
        sha256=hashlib.sha256(contents).hexdigest(),
    )
    monkeypatch.setattr(
        staging.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(contents)
    )
    with staging.source_file(record, args, inventory) as f:
        assert f.read() == contents
    record["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="checksum"):
        staging.source_file(record, args, inventory)
    record["bytes"] = (128 << 20) + 1
    with pytest.raises(ValueError, match="memory bound"):
        staging.source_file(record, args, inventory)


def test_http_worker_error_is_serializable(tmp_path, monkeypatch):
    path = tmp_path / "error.txt"
    path.write_text("rate limit")
    response = path.open("rb")
    error = urllib.error.HTTPError(
        "https://example.test/source", 429, "Too Many Requests", {}, response
    )

    def reject(_task):
        raise error

    monkeypatch.setattr(staging, "stage_group", reject)
    report = staging.stage_group_safely((17, []))
    assert report["failed"] and report["http_status"] == 429
    assert pickle.loads(pickle.dumps(report)) == report
    assert response.closed


def test_download_retry_honors_server_window():
    error = urllib.error.HTTPError(
        "https://example.test/source",
        429,
        "Too Many Requests",
        {"Retry-After": "71"},
        None,
    )
    assert staging.retry_delay(error, 0) >= 71
    error.headers = {"RateLimit": '"resolvers";r=0;t=219'}
    assert staging.retry_delay(error, 0) >= 219


def test_resolver_budget_is_shared(monkeypatch):
    delays = []
    monkeypatch.setattr(staging.time, "monotonic", lambda: 100.0)
    monkeypatch.setattr(staging.time, "sleep", delays.append)
    limiter = staging.ResolverLimiter(4)
    for _ in range(4):
        limiter.acquire()
    assert delays == [0.25, 0.5, 0.75]


def test_legacy_index_verifies_identity(tmp_path):
    from tools.prepare_frozen_hip_release import load_legacy_index, sample_fingerprint

    ids = np.array([4, 7, 19], dtype=np.int64)
    path = tmp_path / "legacy_ids.npy"
    np.save(path, ids)
    metadata = dict(
        dataset_revision="a" * 40,
        samples=3,
        sample_ids_sha256=sample_fingerprint(ids),
        files=["bundle_000001.hip"],
    )
    path.with_suffix(".json").write_text(json.dumps(metadata))
    actual, files = load_legacy_index(path, "a" * 40)
    np.testing.assert_array_equal(actual, ids)
    assert files == metadata["files"]
    with pytest.raises(ValueError, match="another release"):
        load_legacy_index(path, "b" * 40)
    np.save(path, ids + 1)
    with pytest.raises(ValueError, match="identities"):
        load_legacy_index(path, "a" * 40)


def test_consuming_staged_cache_is_explicit_and_retains_identity(tmp_path):
    from tools.prepare_frozen_hip_release import consume_staged_input

    payload = tmp_path / "000017.npz"
    payload.write_bytes(b"staged")
    marker = tmp_path / "000017.json"
    marker.write_text('{"group": 17}')
    args = SimpleNamespace(staged_input=tmp_path, consume_staged_input=False)
    consume_staged_input(args, 17)
    assert payload.exists()
    args.consume_staged_input = True
    consume_staged_input(args, 17)
    assert not payload.exists() and marker.exists()
    consume_staged_input(args, 17)
