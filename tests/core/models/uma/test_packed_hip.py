"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fairchem.core.frozen_hip.model import structure_mse
from fairchem.core.frozen_hip.packed import (
    block_mse,
    make_packed_batch,
    open_packed,
    pack_hessian,
    write_packed,
)
from fairchem.core.models.uma.hessian_pred_utils import (
    add_hessian_graph_batch,
    blocks3x3_to_hessian,
)


def test_triangular_block_loss_matches_dense_loss_and_gradients_for_asymmetric_labels():
    sizes = [1, 2, 5, 3]
    data = SimpleNamespace(
        pos=torch.randn(sum(sizes), 3),
        natoms=torch.tensor(sizes),
        batch=torch.arange(len(sizes)).repeat_interleave(torch.tensor(sizes)),
        hip_sizes=sizes,
    )
    add_hessian_graph_batch(data)
    edge = torch.randn(data.edge_index_hessian.shape[1], 3, 3, requires_grad=True)
    node = torch.randn(sum(sizes), 3, 3, requires_grad=True)
    labels = [torch.randn(n * 3, n * 3) for n in sizes]
    packed = [pack_hessian(h.numpy()) for h in labels]
    data.hip_pair_targets = torch.from_numpy(np.concatenate([p[0] for p in packed]))
    data.hip_diagonal_targets = torch.from_numpy(np.concatenate([p[1] for p in packed]))
    data.hip_asymmetry_mse = torch.tensor([p[2] for p in packed])
    dense = blocks3x3_to_hessian(data.edge_index_hessian, data, edge, node)
    reference = structure_mse(dense, torch.cat([h.flatten() for h in labels]), sizes)
    ref_grad = torch.autograd.grad(reference, (edge, node), retain_graph=True)
    actual = block_mse(edge, node, data)
    actual_grad = torch.autograd.grad(actual, (edge, node))
    torch.testing.assert_close(actual, reference)
    for a, b in zip(actual_grad, ref_grad, strict=True):
        torch.testing.assert_close(a, b)


def test_packed_file_roundtrip_is_memory_mapped_and_handles_empty_arrays(tmp_path):
    path = tmp_path / "bundle.hip"
    arrays = dict(
        values=np.arange(30, dtype=np.float16).reshape(10, 3),
        pair_targets=np.empty(0, dtype=np.float32),
        counts=np.arange(5, dtype=np.int64),
    )
    write_packed(path, arrays, dict(backbone_sha256="frozen", feature_scale=0.5))
    opened, meta = open_packed(path)
    assert isinstance(opened["values"], np.memmap)
    assert opened["values"]._mmap is opened["counts"]._mmap
    for key in arrays:
        np.testing.assert_array_equal(opened[key], arrays[key])
    assert meta["feature_scale"] == 0.5


def test_packed_batch_has_no_dense_hessian_and_preserves_ragged_rows():
    sizes = np.array([2, 1, 3])
    values = dict(
        natoms=sizes,
        atom_ptr=np.r_[0, np.cumsum(sizes)],
        atomic_numbers=np.ones(6, dtype=np.int64),
        coords=np.arange(18, dtype=np.float32).reshape(6, 3),
        charge=np.zeros(3),
        spin=np.ones(3),
        omol_index=np.arange(3),
        pair_ptr=np.array([0, 9, 9, 36]),
        pair_targets=np.arange(36, dtype=np.float32),
        diagonal_targets=np.arange(36, dtype=np.float32),
        asymmetry_mse=np.zeros(3, dtype=np.float32),
        features=np.ones((6, 9, 4), dtype=np.float16),
    )
    data = make_packed_batch(values, np.array([2, 0]), 0.5)
    assert not hasattr(data, "hessian")
    assert data.natoms.tolist() == [3, 2]
    assert data.hip_pair_targets.numel() == 36
    assert data.hip_diagonal_targets.numel() == 30
    assert data.hip_features.shape == (5, 9, 4)
    assert data.hip_feature_scale == 0.5


def test_feature_preparation_reads_geometry_without_target_arrays():
    from fairchem.core.frozen_hip.packed import pack_packed_window

    values = dict(
        natoms=np.array([2, 1]),
        atom_ptr=np.array([0, 2, 3]),
        atomic_numbers=np.ones(3, dtype=np.int64),
        coords=np.arange(9, dtype=np.float32).reshape(3, 3),
        charge=np.zeros(2),
        spin=np.ones(2),
        omol_index=np.array([7, 19]),
    )
    # Target keys are absent, so any target access fails this preparation path.
    single = make_packed_batch(values, np.array([1, 0]), include_targets=False)
    mixed = pack_packed_window([(values, {})], [(0, 1), (0, 0)], include_targets=False)
    for data in (single, mixed):
        assert data.natoms.tolist() == [1, 2]
        assert data.sid == ["19", "7"]
        assert data.hip_atom_ptr.tolist() == [0, 1, 3]
        assert not hasattr(data, "hip_pair_targets")
        assert not hasattr(data, "hip_diagonal_targets")
        assert not hasattr(data, "hessian")
    torch.testing.assert_close(single.pos, mixed.pos)


def test_read_only_features_skip_writer_metadata_without_losing_cache_hits(tmp_path):
    from fairchem.core.frozen_hip.cache import FeatureCacheWriter
    from fairchem.core.frozen_hip.data import validation_mask
    from fairchem.core.frozen_hip.packed import PackedHessianStream

    candidates = np.arange(1000)
    ids = candidates[~validation_mask(candidates)][:2]
    values = dict(
        natoms=np.array([2, 1]),
        atom_ptr=np.array([0, 2, 3]),
        atomic_numbers=np.ones(3, dtype=np.int64),
        coords=np.arange(9, dtype=np.float32).reshape(3, 3),
        charge=np.zeros(2),
        spin=np.ones(2),
        omol_index=ids,
    )
    write_packed(tmp_path / "bundle.hip", values, dict(backbone_sha256="frozen"))
    options = dict(
        root=tmp_path,
        markers=["bundle.hip"],
        backbone_sha256="frozen",
        feature_cache=tmp_path / "features",
        max_passes=1,
        include_targets=False,
    )
    writer = FeatureCacheWriter(tmp_path / "features")
    for logical in PackedHessianStream(**options):
        for data in logical:
            writer.store(data, torch.ones(len(data.pos), 9, 4))
    seen = []
    for logical in PackedHessianStream(**options, include_cache_metadata=False):
        for data in logical:
            seen.extend(data.sid)
            assert not hasattr(data, "hip_cache_groups")
            torch.testing.assert_close(
                data.hip_features, torch.ones(len(data.pos), 9, 4)
            )
    assert sorted(int(sid) for sid in seen) == ids.tolist()


@pytest.mark.parametrize("prefetch_window", [False, True])
def test_packed_windows_mix_sizes_and_cover_rows_once_across_ranks_and_workers(
    tmp_path, monkeypatch, prefetch_window
):
    from fairchem.core.frozen_hip import packed as module
    from fairchem.core.frozen_hip.data import validation_mask

    candidates = np.arange(1000)
    ids = candidates[~validation_mask(candidates)][:64]
    markers = []
    for source in range(8):
        sizes = np.full(8, source + 1, dtype=np.int64)
        atoms = int(sizes.sum())
        ap = np.r_[0, np.cumsum(sizes)]
        pp = np.r_[0, np.cumsum(9 * sizes * (sizes - 1) // 2)]
        values = dict(
            natoms=sizes,
            atom_ptr=ap,
            pair_ptr=pp,
            atomic_numbers=np.ones(atoms, dtype=np.int64),
            coords=np.arange(atoms * 3, dtype=np.float32).reshape(-1, 3),
            charge=np.zeros(8),
            spin=np.ones(8),
            omol_index=ids[source * 8 : source * 8 + 8],
            pair_targets=np.zeros(pp[-1], dtype=np.float32),
            diagonal_targets=np.zeros(atoms * 6, dtype=np.float32),
            asymmetry_mse=np.zeros(8, dtype=np.float32),
        )
        name = f"bundle_{source:06d}.hip"
        write_packed(tmp_path / name, values, {})
        markers.append(name)
    from fairchem.core.frozen_hip.cache import FeatureCacheWriter

    cache_root = tmp_path / "features"
    writer = FeatureCacheWriter(cache_root)
    seen = []
    mixed = False
    for rank in range(2):
        for worker in range(2):
            monkeypatch.setattr(
                module,
                "get_worker_info",
                lambda worker=worker: SimpleNamespace(id=worker, num_workers=2),
            )
            stream = module.PackedHessianStream(
                tmp_path,
                markers=markers,
                backbone_sha256="unused",
                rank=rank,
                world=2,
                max_batch=8,
                max_atoms=32,
                pair_budget=128,
                window_files=4,
                prefetch_window=prefetch_window,
                max_passes=1,
                feature_cache=cache_root,
            )
            for microbatches in stream:
                logical = []
                for data in microbatches:
                    seen.extend(int(i) for i in data.sid)
                    logical.extend(data.natoms.tolist())
                    features = (
                        torch.tensor([int(i) for i in data.sid], dtype=torch.float32)[
                            :, None, None
                        ]
                        .expand(-1, 9, 4)
                        .repeat_interleave(data.natoms, dim=0)
                    )
                    writer.store(data, features)
                    assert int((data.natoms**2).sum()) <= 128
                    assert data.hip_pair_targets.numel() == int(data.hip_pair_ptr[-1])
                assert len(logical) <= 8
                assert sum(logical) <= 32
                mixed |= len(set(logical)) > 1
            for microbatches in stream:
                for data in microbatches:
                    expected = (
                        torch.tensor([int(i) for i in data.sid], dtype=torch.float32)[
                            :, None, None
                        ]
                        .expand(-1, 9, 4)
                        .repeat_interleave(data.natoms, dim=0)
                    )
                    torch.testing.assert_close(data.hip_features, expected)
            before = [data for batch in stream for data in batch]
            stream.prefetch_window = not prefetch_window
            after = [data for batch in stream for data in batch]
            assert len(before) == len(after)
            for original, prefetched in zip(before, after, strict=True):
                assert original.sid == prefetched.sid
                for key in (
                    "natoms",
                    "pos",
                    "hip_features",
                    "hip_pair_targets",
                    "hip_diagonal_targets",
                ):
                    torch.testing.assert_close(
                        getattr(original, key), getattr(prefetched, key), rtol=0, atol=0
                    )
    assert sorted(seen) == sorted(ids.tolist())
    assert mixed
    assert len(list(cache_root.glob("bundles/*.json"))) == 8


@pytest.mark.parametrize(
    "options, message",
    [
        (dict(backbone_graph_version=1), "neighbor graph"),
        (dict(fused_backbone=True), "backbone rotations"),
    ],
)
def test_embedded_features_reject_changed_backbone_operations(
    tmp_path, options, message
):
    from fairchem.core.frozen_hip.packed import PackedHessianStream

    path = tmp_path / "bundle.hip"
    write_packed(
        path,
        dict(features=np.zeros((1, 9, 4), dtype=np.float32)),
        dict(backbone_sha256="frozen"),
    )
    stream = PackedHessianStream(
        tmp_path,
        markers=[path.name],
        backbone_sha256="frozen",
        max_passes=1,
        **options,
    )
    with pytest.raises(ValueError, match=message):
        next(iter(stream))


@pytest.mark.parametrize("include_targets", [False, True])
@pytest.mark.parametrize("scales", [(1.0, 1.0), (0.5, 2.0)])
def test_single_pass_cached_packing_preserves_every_tensor(include_targets, scales):
    from fairchem.core.frozen_hip.packed import pack_cached_window, pack_packed_window

    window = []
    for source, ns in enumerate(([2, 1, 3], [1, 5, 2])):
        sizes = np.asarray(ns, dtype=np.int64)
        total = int(sizes.sum())
        pairs = np.r_[0, np.cumsum(9 * sizes * (sizes - 1) // 2)]
        values = dict(
            natoms=sizes,
            atom_ptr=np.r_[0, np.cumsum(sizes)],
            pair_ptr=pairs,
            atomic_numbers=np.arange(total, dtype=np.int64) + 1,
            coords=np.arange(total * 3, dtype=np.float32).reshape(total, 3) * 0.3,
            charge=np.arange(3, dtype=np.int64) - 1,
            spin=np.ones(3, dtype=np.int64),
            omol_index=np.arange(3) + 10 * source,
            features=np.arange(total * 36, dtype=np.float32).reshape(total, 9, 4),
        )
        if include_targets:
            values.update(
                pair_targets=np.arange(pairs[-1], dtype=np.float32),
                diagonal_targets=np.arange(6 * total, dtype=np.float32),
                asymmetry_mse=np.arange(3, dtype=np.float32),
            )
        window.append((values, dict(feature_scale=scales[source])))
    # Interleave sources and reorder rows to check the original grouping contract.
    references = [(1, 1), (0, 2), (1, 0), (0, 0), (1, 2)]
    expected = pack_packed_window(window, references, include_targets=include_targets)
    actual = pack_cached_window(window, references, include_targets=include_targets)
    assert set(actual.keys()) == set(expected.keys())
    for key, b in expected:
        a = actual[key]
        if torch.is_tensor(b):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        else:
            assert a == b


@pytest.mark.parametrize("budget", [0, 32, 256, 1024, 8192])
def test_window_staging_respects_byte_budget_and_preserves_payloads(budget):
    from fairchem.core.frozen_hip.packed import stage_packed_arrays

    values = dict(
        features=np.arange(128, dtype=np.float32),
        pair_targets=np.arange(64, dtype=np.float32),
        diagonal_targets=np.arange(32, dtype=np.float32),
        coords=np.arange(12, dtype=np.float32),
        atomic_numbers=np.arange(4, dtype=np.int64),
    )
    staged, used = stage_packed_arrays(values, budget)
    assert used <= budget
    actual = 0
    for key, value in values.items():
        np.testing.assert_array_equal(staged[key], value)
        if staged[key] is not value:
            assert not np.shares_memory(staged[key], value)
            actual += value.nbytes
    assert actual == used


def test_window_staging_skips_targets_when_preparing_features():
    from fairchem.core.frozen_hip.packed import stage_packed_arrays

    labels = np.ones(128, dtype=np.float32)
    staged, used = stage_packed_arrays(
        dict(pair_targets=labels), 4096, include_targets=False
    )
    assert used == 0
    assert staged["pair_targets"] is labels
