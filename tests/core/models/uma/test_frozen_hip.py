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
