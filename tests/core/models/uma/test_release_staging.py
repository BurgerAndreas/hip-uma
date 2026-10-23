"""CPU staging preserves Hessians and split identities across restarts."""

import hashlib
import io
import json
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
            model="uma-s-1p2", hessian_unit="eV/Angstrom^2",
            parent_partition=1, split="validation", complete=True,
        )
        for key, value in dict(
            omol_index=[4], natoms=[2], atom_ptr=[0, 2], hessian_ptr=[0, 21],
            charge=[0], spin=[1], atomic_numbers=[1, 1],
            coords=np.zeros((2, 3), np.float32),
            hessian_upper=hessian[np.triu_indices(6)],
        ).items():
            f[key] = value
    args = SimpleNamespace(
        output=tmp_path / "packed", staged_output=tmp_path / "staged",
        base_read_mode="sparse",
    )
    args.staged_output.mkdir()
    inventory = dict(revision="a" * 40)
    record = dict(parent_partition=1, split="validation", samples=1)
    monkeypatch.setattr(staging, "_context", (
        args, inventory, np.array([4]), {(1, "validation"): np.array([4])},
    ))
    monkeypatch.setattr(staging, "download_file", lambda task: path)
    report = staging.stage_group((7, [record]))
    assert report["samples"] == 1
    with np.load(args.staged_output / "000007.npz", allow_pickle=False) as arrays:
        np.testing.assert_array_equal(arrays["hessian_flat"].reshape(6, 6), hessian)
        assert arrays["omol_index"].tolist() == [(1 << 60) + 4]
    monkeypatch.setattr(staging, "download_file", lambda task: pytest.fail("Repeated read"))
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
    assert [[r["path"] for r in group] for group in groups] == [["a", "b"], ["n"], ["v"]]


def test_buffered_source_checks_full_content(tmp_path, monkeypatch):
    contents = b"immutable source bytes"
    token = tmp_path / "credential"
    token.write_text("test-token")
    args = SimpleNamespace(token=token, base_read_mode="buffered")
    inventory = dict(repository="example/labels", revision="a" * 40)
    record = dict(
        parent_partition=0, path="source.h5", bytes=len(contents),
        sha256=hashlib.sha256(contents).hexdigest(),
    )
    monkeypatch.setattr(staging.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(contents))
    with staging.source_file(record, args, inventory) as f:
        assert f.read() == contents
    record["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="checksum"):
        staging.source_file(record, args, inventory)
    record["bytes"] = (128 << 20) + 1
    with pytest.raises(ValueError, match="memory bound"):
        staging.source_file(record, args, inventory)
