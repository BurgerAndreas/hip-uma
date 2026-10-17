"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import json

import h5py
import numpy as np
import pytest
import torch

from fairchem.core.frozen_hip.data import validation_mask
from fairchem.core.frozen_hip.validation import (
    DATA_REVISION,
    EvaluationStream,
    build_manifest,
    build_validation_cache,
    evaluation_mask,
    load_last_head,
    load_manifest,
    read_selected,
    sample_fingerprint,
    write_manifest,
)


def test_fixed_evaluation_selection_is_held_out_and_order_independent():
    ids = np.arange(1000000, dtype=np.int64)
    chosen = evaluation_mask(ids)
    assert 800 < chosen.sum() < 1200
    assert validation_mask(ids[chosen]).all()
    np.testing.assert_array_equal(evaluation_mask(ids[::-1]), chosen[::-1])
    assert sample_fingerprint(ids[chosen]) == sample_fingerprint(ids[chosen][::-1])


def cohort_fixture(root):
    candidates = np.arange(100000, dtype=np.int64)
    ids = candidates[evaluation_mask(candidates)][:24]
    ordinary = candidates[~validation_mask(candidates)][:24]
    all_ids = np.r_[ids, ordinary]
    for source, indices in enumerate(np.array_split(all_ids, 4)):
        sizes = np.arange(len(indices)) % 4 + 2
        ap = np.r_[0, np.cumsum(sizes)]
        hp = np.r_[0, np.cumsum(9 * sizes**2)]
        name = f"shards/00/shard_{source:05d}.h5"
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as f:
            f.attrs.update(
                complete=True, model="uma-s-1p2", hessian_unit="eV/Angstrom^2"
            )
            values = dict(
                natoms=sizes,
                omol_index=indices,
                atom_ptr=ap,
                hessian_ptr=hp,
                atomic_numbers=np.ones(ap[-1], dtype=np.int64),
                coords=np.arange(ap[-1] * 3, dtype=np.float32).reshape(-1, 3),
                hessian_flat=np.concatenate(
                    [
                        np.full(9 * n * n, index, dtype=np.float32)
                        for n, index in zip(sizes, indices, strict=True)
                    ]
                ),
                charge=np.zeros(len(indices)),
                spin=np.ones(len(indices)),
            )
            for key, value in values.items():
                f[key] = value
        path.with_suffix(".verified.json").write_text(
            json.dumps(dict(path=name, sha256="a" * 64))
        )
    return build_manifest(
        root, dict(revision=DATA_REVISION, shards=4, samples=len(all_ids))
    )


def test_manifest_and_ragged_evaluation_cover_same_ids_across_gpu_counts(tmp_path):
    manifest = cohort_fixture(tmp_path)
    path = tmp_path / "validation_10000.json"
    write_manifest(path, manifest)
    manifest = load_manifest(path)
    expected = {sid for entry in manifest["entries"] for sid in entry["omol_indices"]}
    for world in (1, 2, 3, 5):
        actual = []
        for rank in range(world):
            stream = EvaluationStream(
                tmp_path,
                manifest,
                rank=rank,
                world=world,
                max_batch=4,
                max_atoms=12,
                pair_budget=40,
            )
            for batch in stream:
                actual.extend(map(int, batch.sid))
                chunks = batch.hessian.split([9 * n * n for n in batch.natoms.tolist()])
                for sid, labels in zip(batch.sid, chunks, strict=True):
                    torch.testing.assert_close(
                        labels, torch.full_like(labels, int(sid))
                    )
        assert len(actual) == len(set(actual)) == len(expected)
        assert set(actual) == expected
        assert sample_fingerprint(actual) == manifest["sample_ids_sha256"]


def test_manifest_rejects_changed_sample_identity(tmp_path):
    manifest = cohort_fixture(tmp_path)
    manifest["entries"][0]["omol_indices"][0] += 1
    path = tmp_path / "changed.json"
    write_manifest(path, manifest)
    with pytest.raises(ValueError):
        load_manifest(path)
    with pytest.raises(ValueError, match="identities"):
        read_selected(tmp_path, manifest["entries"][0])


def test_final_evaluation_reloads_last_weights_not_live_or_best_weights(tmp_path):
    head = torch.nn.Linear(2, 1)
    with torch.no_grad():
        head.weight.fill_(2)
        head.bias.fill_(3)
    checkpoint = dict(step=123, head=head.state_dict(), backbone_sha256="frozen")
    path = tmp_path / "latest.pt"
    torch.save(checkpoint, path)
    with torch.no_grad():
        head.weight.fill_(99)
        head.bias.fill_(99)
    torch.save(dict(step=100, head=head.state_dict()), tmp_path / "best.pt")
    state = load_last_head(head, path, "frozen", "cpu")
    assert state["step"] == 123
    torch.testing.assert_close(head(torch.ones(1, 2)), torch.tensor([[7.0]]))
    with pytest.raises(ValueError, match="backbone"):
        load_last_head(head, path, "wrong", "cpu")


def test_dense_validation_cache_preserves_samples_labels_and_geometry(tmp_path):
    manifest = cohort_fixture(tmp_path)

    def collect(world):
        result = {}
        for rank in range(world):
            for data in EvaluationStream(tmp_path, manifest, world=world, rank=rank):
                atoms = data.pos.split(data.natoms.tolist())
                labels = data.hessian.split([9 * n * n for n in data.natoms.tolist()])
                for sid, pos, target in zip(data.sid, atoms, labels, strict=True):
                    assert int(sid) not in result
                    result[int(sid)] = (pos.clone(), target.clone())
        return result

    expected = collect(1)
    build_validation_cache(tmp_path, manifest, workers=2)
    for world in (1, 2, 5):
        actual = collect(world)
        assert actual.keys() == expected.keys()
        for sid in expected:
            for got, wanted in zip(actual[sid], expected[sid], strict=True):
                torch.testing.assert_close(got, wanted, rtol=0, atol=0)
    manifest["entries"][0]["source_sha256"] = "changed"
    with pytest.raises(ValueError, match="cache differs"):
        next(iter(EvaluationStream(tmp_path, manifest)))
