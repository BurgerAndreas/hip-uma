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

from fairchem.core.frozen_hip.packed import block_mse, pack_hessian
from fairchem.core.models.uma.hessian_pred_utils import (
    fully_connected_hessian_graph_batch,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def metadata(sizes):
    n = torch.tensor(sizes, device="cuda")
    ap = torch.cat((n.new_zeros(1), n.cumsum(0)))
    ep = torch.cat((n.new_zeros(1), (n * (n - 1)).cumsum(0)))
    pp = ep * 9 // 2
    return SimpleNamespace(
        hip_sizes=sizes,
        natoms=n,
        hip_atom_ptr=ap,
        hip_edge_ptr=ep,
        hip_pair_ptr=pp,
        pos=torch.randn(sum(sizes), 3, device="cuda"),
    )


def test_fused_graph_matches_ragged_reference_including_single_atoms():
    data = metadata([1, 4, 1, 9, 2, 1])
    expected = fully_connected_hessian_graph_batch(data)
    data.hip_fused_graph = True
    actual = fully_connected_hessian_graph_batch(data)
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b)


def test_fused_loss_and_gradients_match_asymmetric_dense_equivalent_reference():
    data = metadata([1, 4, 1, 9, 2, 1])
    packed = [
        pack_hessian(np.random.default_rng(n).normal(size=(3 * n, 3 * n)))
        for n in data.hip_sizes
    ]
    data.hip_pair_targets = torch.tensor(
        np.concatenate([p[0] for p in packed]), device="cuda"
    )
    data.hip_diagonal_targets = torch.tensor(
        np.concatenate([p[1] for p in packed]), device="cuda"
    )
    data.hip_asymmetry_mse = torch.tensor([p[2] for p in packed], device="cuda")
    e = sum(n * (n - 1) for n in data.hip_sizes)
    edge = torch.randn(e, 3, 3, device="cuda", requires_grad=True)
    node = torch.randn(sum(data.hip_sizes), 3, 3, device="cuda", requires_grad=True)
    reference = block_mse(edge, node, data)
    expected = torch.autograd.grad(reference, (edge, node))
    data.hip_fused_loss = True
    actual = block_mse(edge, node, data)
    gradients = torch.autograd.grad(actual, (edge, node))
    torch.testing.assert_close(actual, reference)
    for a, b in zip(gradients, expected, strict=True):
        torch.testing.assert_close(a, b)


@pytest.mark.parametrize("wigner_grad", [False, True])
def test_fused_rotations_preserve_feature_and_optional_geometry_gradients(wigner_grad):
    from fairchem.core.models.uma.triton import (
        UMASFastGPUNodeToEdgeWignerPermute,
        UMASFastGPUPermuteWignerInvEdgeToNode,
    )
    from fairchem.core.models.uma.triton.constants import M_TO_L_GATHER_IDX

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(17)
    x = torch.randn(13, 9, 32, device="cuda", requires_grad=True)
    edge = torch.randint(13, (2, 37), device="cuda")
    w = torch.zeros(37, 9, 9, device="cuda")
    for a, b in [(0, 1), (1, 4), (4, 9)]:
        w[:, a:b, a:b] = torch.randn(37, b - a, b - a, device="cuda")
    w.requires_grad_(wigner_grad)
    block_mask = torch.zeros_like(w)
    for a, b in [(0, 1), (1, 4), (4, 9)]:
        block_mask[:, a:b, a:b] = 1
    # Wigner matrices vary only inside the three irreducible blocks.
    physical_wigner = w * block_mask
    inverse = [0] * 9
    for i, j in enumerate(M_TO_L_GATHER_IDX):
        inverse[j] = i
    expected = torch.bmm(physical_wigner, torch.cat((x[edge[0]], x[edge[1]]), dim=2))[
        :, inverse, :
    ]
    actual = UMASFastGPUNodeToEdgeWignerPermute.apply(x, edge, w)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=5e-6)
    probe = torch.randn_like(expected)
    inputs = (x, w) if wigner_grad else (x,)
    eg = torch.autograd.grad((expected * probe).sum(), inputs, retain_graph=True)
    ag = torch.autograd.grad((actual * probe).sum(), inputs)
    for a, b in zip(ag, eg, strict=True):
        torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-5)
    message = torch.randn(37, 9, 32, device="cuda", requires_grad=True)
    expected = torch.bmm(w * block_mask, message[:, M_TO_L_GATHER_IDX, :])
    actual = UMASFastGPUPermuteWignerInvEdgeToNode.apply(message, w)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=5e-6)
    inputs = (message, w) if wigner_grad else (message,)
    probe = torch.randn_like(expected)
    eg = torch.autograd.grad((expected * probe).sum(), inputs, retain_graph=True)
    ag = torch.autograd.grad((actual * probe).sum(), inputs)
    for a, b in zip(ag, eg, strict=True):
        torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("half", [False, True])
@pytest.mark.parametrize("factor", [1, 2])
def test_grouped_frozen_linear_matches_independent_system_experts(half, factor):
    from fairchem.core.frozen_hip.grouped_mole import grouped_linear
    from fairchem.core.models.uma.nn.mole import MOLEGlobals

    torch.backends.cuda.matmul.allow_tf32 = False
    sizes = torch.tensor([0, 7, 63, 2, 97, 0])
    globals_ = MOLEGlobals(None, sizes)
    x = torch.randn(int(sizes.sum()), factor, 96, device="cuda")
    if factor == 1:
        x = x[:, 0]
    weights = torch.randn(len(sizes), 96, 80, device="cuda").transpose(1, 2)
    bias = torch.randn(80, device="cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=half):
        parts = x.split(sizes.tolist())
        reference = torch.cat(
            [
                torch.nn.functional.linear(part, w, bias)
                for part, w in zip(parts, weights, strict=True)
            ]
        )
        actual = grouped_linear(x, weights, bias, globals_)
    assert actual.dtype == reference.dtype
    torch.testing.assert_close(
        actual, reference, rtol=2e-3 if half else 2e-5, atol=0.02 if half else 2e-5
    )


def test_async_feature_cache_owns_cuda_transfer_until_atomic_publication(tmp_path):
    from fairchem.core.frozen_hip.cache import AsyncFeatureCacheWriter, cached_features

    expected = torch.arange(4 * 9 * 128, dtype=torch.float32).reshape(4, 9, 128)
    data = SimpleNamespace(
        hip_cache_key="bundle",
        hip_total_atoms=4,
        hip_row_count=1,
        hip_rows=[0],
        hip_atom_spans=[(0, 4)],
        hip_training_rows=[0],
        hip_data_sha256="source",
    )
    writer = AsyncFeatureCacheWriter(tmp_path, max_pending=2)
    values = expected.cuda()
    writer.store(data, values)
    del values
    temporary = torch.empty_like(expected, device="cuda")
    temporary.fill_(-1)
    writer.close()
    np.testing.assert_array_equal(
        cached_features(tmp_path, "bundle", "source"), expected.numpy()
    )
