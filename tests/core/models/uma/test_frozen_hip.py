"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Check dense label packing, held-out membership, and balanced molecular losses.
"""

from __future__ import annotations

import numpy as np
import torch

from fairchem.core.frozen_hip.data import make_batch, validation_mask
from fairchem.core.frozen_hip.model import structure_mse
from fairchem.core.models.uma.hessian_pred_utils import (
    fully_connected_hessian_graph_batch,
)


def test_packed_batch_preserves_dense_labels_charge_and_multiplicity():
    n, count = 3, 4
    values = {
        "natoms": np.full(count, n),
        "atomic_numbers": np.arange(n * count) + 1,
        "coords": np.arange(n * count * 3, dtype=np.float32).reshape(-1, 3),
        "hessian_flat": np.arange(count * 9 * n * n, dtype=np.float32),
        "charge": np.array([-1, 0, 1, 2]),
        "spin": np.array([1, 2, 3, 4]),
        "omol_index": np.arange(count) + 100,
    }
    rows = np.array([3, 1])
    data = make_batch(values, rows)
    assert data.charge.tolist() == [2, 0]
    assert data.spin.tolist() == [4, 2]
    assert data.sid == ["103", "101"]
    expected = values["hessian_flat"].reshape(count, -1)[rows].reshape(-1)
    np.testing.assert_array_equal(data.hessian.numpy(), expected)
    assert data.batch.tolist() == [0, 0, 0, 1, 1, 1]


def test_validation_membership_is_stable_and_disjoint():
    indices = np.arange(100000, dtype=np.int64)
    mask = validation_mask(indices)
    assert 0.005 < mask.mean() < 0.015
    np.testing.assert_array_equal(validation_mask(indices[::-1]), mask[::-1])
    assert not set(indices[mask]) & set(indices[~mask])


def test_vectorized_centering_matches_ase_teacher_inputs():
    from ase import Atoms

    from fairchem.core.datasets.atomic_data import AtomicData

    positions = np.array(
        [[1000.0, -800.0, 2.0], [1000.7, -800.0, 2.0], [1000.0, -799.3, 2.0]],
        dtype=np.float32,
    )
    values = {
        "natoms": np.array([3]),
        "atomic_numbers": np.array([8, 1, 1]),
        "coords": positions,
        "hessian_flat": np.zeros(81, dtype=np.float32),
        "charge": np.array([0]),
        "spin": np.array([1]),
        "omol_index": np.array([7]),
    }
    batch = make_batch(values, np.array([0]))
    atoms = Atoms(numbers=[8, 1, 1], positions=positions.astype(np.float64))
    atoms.info.update(charge=0, spin=1)
    reference = AtomicData.from_ase(atoms, task_name="omol", molecule_cell_size=120.0)
    torch.testing.assert_close(batch.pos, reference.pos, rtol=0, atol=0)
    torch.testing.assert_close(batch.cell, reference.cell, rtol=0, atol=0)


def test_mixed_shard_uses_atom_and_hessian_pointers():
    values = {
        "natoms": np.array([2, 3, 2]),
        "atom_ptr": np.array([0, 2, 5, 7]),
        "hessian_ptr": np.array([0, 36, 117, 153]),
        "atomic_numbers": np.arange(7) + 1,
        "coords": np.arange(21, dtype=np.float32).reshape(7, 3),
        "hessian_flat": np.arange(153, dtype=np.float32),
        "charge": np.array([-1, 0, 1]),
        "spin": np.array([1, 2, 3]),
        "omol_index": np.arange(3),
    }
    data = make_batch(values, np.array([2, 0]))
    np.testing.assert_array_equal(data.atomic_numbers.numpy(), [6, 7, 1, 2])
    np.testing.assert_array_equal(
        data.hessian.numpy(), np.r_[np.arange(117, 153), np.arange(36)]
    )


def test_feature_cache_exposes_only_complete_training_rows(tmp_path):
    from types import SimpleNamespace

    from fairchem.core.frozen_hip.cache import FeatureCacheWriter, cached_features

    writer = FeatureCacheWriter(tmp_path)
    data = SimpleNamespace(
        hip_cache_key="shards/00/shard_00000",
        hip_data_sha256="source",
        hip_total_atoms=5,
        hip_row_count=3,
        hip_rows=[0],
        hip_atom_spans=[(0, 2)],
        hip_training_rows=[0, 2],
    )
    first = torch.randn(2, 9, 4)
    writer.store(data, first)
    assert cached_features(tmp_path, data.hip_cache_key, "source") is None
    data.hip_rows = [2]
    data.hip_atom_spans = [(3, 5)]
    last = torch.randn(2, 9, 4)
    writer.store(data, last)
    result = cached_features(tmp_path, data.hip_cache_key, "source")
    np.testing.assert_array_equal(result[:2], first.numpy())
    np.testing.assert_array_equal(result[3:], last.numpy())


def test_molecular_mse_weights_each_molecule_equally():
    pred = torch.cat((torch.full((36,), 2.0), torch.ones(81))).requires_grad_()
    result = structure_mse(pred, torch.zeros_like(pred), [2, 3])
    torch.testing.assert_close(result, torch.tensor(2.5))
    result.backward()
    torch.testing.assert_close(pred.grad[:36], torch.full((36,), 2 / 36))
    torch.testing.assert_close(pred.grad[36:], torch.full((81,), 1 / 81))


def test_ragged_graph_preserves_reference_order_and_position_gradients():
    from types import SimpleNamespace

    sizes = [1, 2, 5, 3]
    batch = torch.arange(len(sizes)).repeat_interleave(torch.tensor(sizes))
    pos = torch.randn(sum(sizes), 3, requires_grad=True)
    data = SimpleNamespace(pos=pos, natoms=torch.tensor(sizes), batch=batch)
    edge_index, distance, *_ = fully_connected_hessian_graph_batch(data)
    count = sum(sizes)
    src = torch.arange(count).repeat(count)
    dst = torch.arange(count).repeat_interleave(count)
    keep = (batch[src] == batch[dst]) & (src != dst)
    assert torch.equal(edge_index, torch.stack((src[keep], dst[keep])))
    reference = (pos[src[keep]] - pos[dst[keep]]).norm(dim=-1)
    torch.testing.assert_close(distance, reference)
    actual_grad = torch.autograd.grad(distance.sum(), pos, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(reference.sum(), pos)[0]
    torch.testing.assert_close(actual_grad, expected_grad)


def test_fixed_inventory_keeps_rank_samples_disjoint_during_downloads(tmp_path):
    import json

    import h5py

    from fairchem.core.frozen_hip.data import HessianStream

    candidate = np.arange(1000, dtype=np.int64)
    indices = candidate[~validation_mask(candidate)][:32]
    markers = []
    for sid in range(16):
        name = f"shards/00/shard_{sid:05d}.h5"
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as f:
            f.attrs.update(
                complete=True, model="uma-s-1p2", hessian_unit="eV/Angstrom^2"
            )
            values = {
                "natoms": np.array([2, 2]),
                "atom_ptr": np.array([0, 2, 4]),
                "hessian_ptr": np.array([0, 36, 72]),
                "atomic_numbers": np.ones(4, dtype=np.int16),
                "coords": np.arange(12, dtype=np.float32).reshape(4, 3),
                "hessian_flat": np.zeros(72, dtype=np.float32),
                "charge": np.zeros(2, dtype=np.int32),
                "spin": np.ones(2, dtype=np.int32),
                "omol_index": indices[sid * 2 : sid * 2 + 2],
            }
            for key, value in values.items():
                f[key] = value
        marker = path.with_suffix(".verified.json")
        marker.write_text(json.dumps({"path": name, "sha256": "a" * 64}))
        markers.append(str(marker.relative_to(tmp_path)))
    streams = [
        HessianStream(tmp_path, rank=rank, world=4, markers=markers, max_passes=1)
        for rank in range(4)
    ]
    # A new marker must not change a rank's partition until coordinated refresh.
    (tmp_path / "shards/00/shard_99999.verified.json").write_text("{}")
    sets = [{int(sid) for batch in stream for sid in batch.sid} for stream in streams]
    assert set.union(*sets) == set(indices.tolist())
    assert sum(map(len, sets)) == len(set.union(*sets))


def test_atom_balanced_sampler_covers_samples_and_limits_work():
    from fairchem.core.frozen_hip.data import BalancedBatchSampler

    sizes = np.array([350, 300, 200, 150, 100, 80, 60, 40, 20, 10] * 20)
    batches = list(
        BalancedBatchSampler(sizes, max_atoms=640, batch_size=16, pair_budget=131072)
    )
    assert sorted(i for batch in batches for i in batch) == list(range(len(sizes)))
    for batch in batches:
        assert len(batch) <= 16
        assert sizes[batch].sum() <= 640
        assert (sizes[batch] ** 2).sum() <= 131072
    # Quadratic memory limits can leave atom capacity unused for large molecules.
    assert np.mean([sizes[b].sum() for b in batches]) > 500
    atom_only = list(
        BalancedBatchSampler(sizes, max_atoms=640, batch_size=16, pair_budget=1000000)
    )
    assert np.mean([sizes[b].sum() for b in atom_only]) > 600


def test_atom_balanced_sampler_keeps_oversized_singletons():
    from fairchem.core.frozen_hip.data import BalancedBatchSampler

    batches = list(
        BalancedBatchSampler(
            [1000, 20, 20], max_atoms=640, batch_size=16, pair_budget=131072
        )
    )
    assert [0] in batches
    assert sorted(i for batch in batches for i in batch) == [0, 1, 2]


def test_cross_shard_ragged_batch_preserves_labels_and_feature_cache(tmp_path):
    from fairchem.core.frozen_hip.cache import FeatureCacheWriter, cached_features
    from fairchem.core.frozen_hip.data import pack_window_batch

    window = []
    ids = np.arange(1000)[~validation_mask(np.arange(1000))][:4]
    for source, sample_sizes in enumerate(([2, 3], [4, 2])):
        sizes = np.array(sample_sizes)
        atoms = int(sizes.sum())
        labels = int((9 * sizes**2).sum())
        values = dict(
            natoms=sizes,
            atom_ptr=np.r_[0, np.cumsum(sizes)],
            hessian_ptr=np.r_[0, np.cumsum(9 * sizes**2)],
            atomic_numbers=np.ones(atoms),
            coords=np.arange(atoms * 3, dtype=np.float32).reshape(-1, 3),
            hessian_flat=np.arange(labels, dtype=np.float32) + source * 1000,
            charge=np.zeros(2),
            spin=np.ones(2),
            omol_index=ids[source * 2 : source * 2 + 2],
        )
        meta = dict(path=f"shards/00/shard_{source:05d}.h5", sha256=str(source))
        features = np.arange(atoms * 9 * 4, dtype=np.float32).reshape(atoms, 9, 4)
        window.append((values, meta, features))
    data = pack_window_batch(window, [(1, [1, 0]), (0, [0, 1])])
    assert data.natoms.tolist() == [2, 4, 2, 3]
    assert data.batch.tolist() == [0] * 2 + [1] * 4 + [2] * 2 + [3] * 3
    np.testing.assert_array_equal(
        data.hessian.numpy(),
        np.r_[np.arange(144, 180) + 1000, np.arange(144) + 1000, np.arange(117)],
    )
    assert data.hip_features.shape == (11, 9, 4)
    writer = FeatureCacheWriter(tmp_path)
    writer.store(data, data.hip_features)
    for source, (_, meta, features) in enumerate(window):
        result = cached_features(
            tmp_path, f"shards/00/shard_{source:05d}", meta["sha256"]
        )
        np.testing.assert_array_equal(result, features)


def test_selective_precision_preserves_parameters_and_checkpoint_names():
    from fairchem.core.frozen_hip.precision import configure_head_precision

    class Head(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.block = torch.nn.Module()
            self.block.rad_func = torch.nn.Module()
            self.block.rad_func.net = torch.nn.Sequential(
                torch.nn.Linear(8, 16),
                torch.nn.LayerNorm(16),
                torch.nn.SiLU(),
                torch.nn.Linear(16, 8),
            )
            self.projection = torch.nn.Linear(8, 1)

    head = Head()
    original = dict(head.named_parameters())
    x = torch.randn(4, 8)
    reference = head.block.rad_func.net(x)
    assert configure_head_precision(head, "bf16-mlp") == 1
    assert head.block.rad_func.net.enabled
    assert dict(head.named_parameters()).keys() == original.keys()
    assert all(p is original[k] for k, p in head.named_parameters())
    assert type(head.projection) is torch.nn.Linear
    configure_head_precision(head, "fp32")
    actual = head.block.rad_func.net(x)
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)


def test_microbatch_accumulation_matches_full_ragged_molecular_gradient():
    from fairchem.core.frozen_hip.data import microbatch_indices

    sizes = [2, 5, 3, 4]
    weights = torch.tensor([0.7, -0.4], requires_grad=True)
    inputs = [torch.randn(9 * n * n, 2) for n in sizes]
    targets = [torch.randn(9 * n * n) for n in sizes]
    prediction = torch.cat([x @ weights for x in inputs])
    reference_loss = structure_mse(prediction, torch.cat(targets), sizes)
    reference_grad = torch.autograd.grad(reference_loss, weights)[0]
    batches = microbatch_indices(list(range(4)), sizes, 30)
    assert sorted(i for batch in batches for i in batch) == list(range(4))
    for batch in batches:
        assert sum(sizes[i] ** 2 for i in batch) <= 30
        loss = structure_mse(
            torch.cat([inputs[i] @ weights for i in batch]),
            torch.cat([targets[i] for i in batch]),
            [sizes[i] for i in batch],
        )
        (loss * len(batch) / len(sizes)).backward()
    torch.testing.assert_close(weights.grad, reference_grad)
