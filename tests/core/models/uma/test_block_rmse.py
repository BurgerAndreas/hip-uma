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

from fairchem.core.frozen_hip.losses import block_rmse_values, structure_block_rmse
from fairchem.core.frozen_hip.packed import block_rmse, pack_hessian
from fairchem.core.models.uma.hessian_pred_utils import (
    add_hessian_graph_batch,
    blocks3x3_to_hessian,
)


@pytest.mark.parametrize("value", [0.0, 1e-30, 1e-7, 1e20])
def test_block_norm_is_stable_and_has_finite_zero_subgradient(value):
    error = torch.full((2, 3, 3), value, requires_grad=True)
    loss = block_rmse_values(error)
    torch.testing.assert_close(loss, torch.full((2,), value), rtol=2e-6, atol=0)
    loss.sum().backward()
    expected = torch.full_like(error, 0 if value == 0 else 1 / 9)
    torch.testing.assert_close(error.grad, expected)


def test_geometry_mean_of_block_norms_is_not_whole_hessian_rmse():
    a = torch.full((3, 3), 2.0)
    b = (
        torch.tensor([[1.0, 2.0], [2.0, 7.0]])
        .repeat_interleave(3, 0)
        .repeat_interleave(3, 1)
    )
    prediction = torch.cat((a.flatten(), b.flatten())).requires_grad_()
    loss = structure_block_rmse(prediction, torch.zeros_like(prediction), [1, 2])
    torch.testing.assert_close(loss, torch.tensor(2.5))


def test_packed_block_rmse_matches_dense_symmetrized_loss_and_gradients():
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
    labels = [torch.randn(3 * n, 3 * n) for n in sizes]
    packed = [pack_hessian(h.numpy()) for h in labels]
    data.hip_pair_targets = torch.from_numpy(np.concatenate([p[0] for p in packed]))
    data.hip_diagonal_targets = torch.from_numpy(np.concatenate([p[1] for p in packed]))
    data.hip_asymmetry_mse = torch.full((len(sizes),), 1e8)
    dense = blocks3x3_to_hessian(data.edge_index_hessian, data, edge, node)
    expected = structure_block_rmse(
        dense, torch.cat([h.flatten() for h in labels]), sizes
    )
    gradients = torch.autograd.grad(expected, (edge, node), retain_graph=True)
    actual = block_rmse(edge, node, data)
    actual_gradients = torch.autograd.grad(actual, (edge, node))
    torch.testing.assert_close(actual, expected)
    for a, b in zip(actual_gradients, gradients, strict=True):
        torch.testing.assert_close(a, b)


def test_block_loss_is_invariant_under_consistent_cartesian_rotation():
    sizes = [2, 5]
    predictions = []
    targets = []
    rotated_p = []
    rotated_t = []
    r, _ = torch.linalg.qr(torch.randn(3, 3))
    r[:, 0] *= torch.linalg.det(r)
    for n in sizes:
        p = torch.randn(3 * n, 3 * n)
        t = torch.randn_like(p)
        rotation = torch.kron(torch.eye(n), r.contiguous())
        predictions.append(p.flatten())
        targets.append(t.flatten())
        rotated_p.append((rotation @ p @ rotation.T).flatten())
        rotated_t.append((rotation @ t @ rotation.T).flatten())
    expected = structure_block_rmse(torch.cat(predictions), torch.cat(targets), sizes)
    actual = structure_block_rmse(torch.cat(rotated_p), torch.cat(rotated_t), sizes)
    torch.testing.assert_close(actual, expected)
