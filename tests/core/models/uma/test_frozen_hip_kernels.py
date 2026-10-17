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


def test_fp16_expert_can_retain_fp32_output_above_half_range():
    from fairchem.core.frozen_hip.grouped_mole import grouped_linear
    from fairchem.core.models.uma.nn.mole import MOLEGlobals

    sizes = torch.tensor([2, 0, 3])
    globals_ = MOLEGlobals(None, sizes)
    x = torch.full((5, 384), 1990.0, device="cuda")
    weights = torch.ones((3, 128, 384), device="cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        result = grouped_linear(x, weights, None, globals_, fp32_output=True)
    assert result.dtype == torch.float32
    torch.testing.assert_close(result, torch.full_like(result, 1990.0 * 384))
    assert torch.isfinite(result).all()


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


@pytest.mark.parametrize("sizes", [[1], [1, 4, 1, 9, 2, 1]])
def test_fused_block_norm_matches_reference_values_and_gradients(sizes):
    from fairchem.core.frozen_hip.packed import block_rmse

    data = metadata(sizes)
    packed = [
        pack_hessian(np.random.default_rng(n).normal(size=(3 * n, 3 * n)))
        for n in sizes
    ]
    data.hip_pair_targets = torch.tensor(
        np.concatenate([p[0] for p in packed]), device="cuda"
    )
    data.hip_diagonal_targets = torch.tensor(
        np.concatenate([p[1] for p in packed]), device="cuda"
    )
    edge = torch.randn(
        sum(n * (n - 1) for n in sizes), 3, 3, device="cuda", requires_grad=True
    )
    node = torch.randn(sum(sizes), 3, 3, device="cuda", requires_grad=True)
    expected = block_rmse(edge, node, data)
    expected_gradients = torch.autograd.grad(expected, (edge, node), allow_unused=True)
    data.hip_fused_loss = True
    actual = block_rmse(edge, node, data)
    gradients = torch.autograd.grad(actual, (edge, node))
    torch.testing.assert_close(actual, expected)
    for got, wanted in zip(gradients, expected_gradients, strict=True):
        if wanted is None:
            assert got.numel() == 0
        else:
            torch.testing.assert_close(got, wanted)


@pytest.mark.parametrize("value", [0.0, 1e-30, 1e-7, 1e20])
def test_fused_block_norm_is_stable_at_zero_and_extreme_scales(value):
    from fairchem.core.frozen_hip.packed import block_rmse

    data = metadata([1])
    data.hip_fused_loss = True
    data.hip_pair_targets = torch.empty(0, device="cuda")
    data.hip_diagonal_targets = torch.zeros(6, device="cuda")
    edge = torch.empty(0, 3, 3, device="cuda", requires_grad=True)
    node = torch.full((1, 3, 3), value, device="cuda", requires_grad=True)
    loss = block_rmse(edge, node, data)
    loss.backward()
    torch.testing.assert_close(
        loss, torch.tensor(2 * value, device="cuda"), rtol=2e-5, atol=0
    )
    torch.testing.assert_close(
        node.grad, torch.full_like(node, 0 if value == 0 else 2 / 9)
    )


@pytest.mark.parametrize("pbc_device", ["cpu", "cuda"])
@pytest.mark.parametrize("max_neighbors", [8, 300])
def test_vectorized_backbone_graph_matches_grid_graph_for_nonperiodic_molecules(
    max_neighbors,
    pbc_device,
):
    from fairchem.core.graph.compute import generate_graph

    torch.manual_seed(121)
    data = metadata([2, 5, 27, 350])
    data.batch = torch.arange(len(data.hip_sizes), device="cuda").repeat_interleave(
        data.natoms
    )
    data.cell = torch.eye(3, device="cuda").mul(120).repeat(len(data.hip_sizes), 1, 1)
    data.pbc = torch.zeros(len(data.hip_sizes), 3, device=pbc_device, dtype=torch.bool)
    data.pos = data.pos * 2 + 120
    reference = generate_graph(data, 6.0, max_neighbors, False, 2, data.pbc)
    actual = generate_graph(data, 6.0, max_neighbors, False, 1, data.pbc)

    def canonical(graph):
        edges = graph["edge_index"]
        order = (edges[1] * len(data.pos) + edges[0]).argsort()
        return (
            edges[:, order],
            graph["edge_distance"][order],
            graph["edge_distance_vec"][order],
        )

    for got, wanted in zip(canonical(actual), canonical(reference), strict=True):
        torch.testing.assert_close(got, wanted)


@pytest.mark.parametrize("limit", [0, 8, 300])
@pytest.mark.parametrize("strict", [False, True])
def test_fused_radius_matches_reference_cutoff_and_neighbor_rules(limit, strict):
    from fairchem.core.graph.compute import generate_graph

    torch.manual_seed(623)
    data = metadata([1, 2, 7, 350])
    data.cell = torch.eye(3, device="cuda").repeat(len(data.hip_sizes), 1, 1)
    data.pbc = torch.zeros(len(data.hip_sizes), 3, dtype=torch.bool)
    # Include coincident atoms, excluded near-zero distances, and cutoff ties.
    data.pos[1:3] = torch.tensor([[0, 0, 0], [0.005, 0, 0]], device="cuda")
    data.pos[3:10] = torch.tensor(
        [[0, 0, 0], [0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1]],
        device="cuda",
    )
    expected = generate_graph(data, 1.0, limit, strict, 1, data.pbc)
    actual = generate_graph(data, 1.0, limit, strict, 4, data.pbc)
    for key in ("edge_index", "edge_distance", "edge_distance_vec", "neighbors"):
        torch.testing.assert_close(actual[key], expected[key])
    assert not actual["cell_offsets"].any()


@pytest.mark.parametrize("sizes", [[1], [1, 1], [2, 5, 27, 350]])
def test_fused_radius_handles_empty_graphs_and_large_pruned_molecules(sizes):
    from fairchem.core.graph.compute import generate_graph

    data = metadata(sizes)
    data.cell = torch.eye(3, device="cuda").repeat(len(sizes), 1, 1)
    data.pbc = torch.zeros(len(sizes), 3, dtype=torch.bool)
    expected = generate_graph(data, 6.0, 300, False, 1, data.pbc)
    actual = generate_graph(data, 6.0, 300, False, 4, data.pbc)
    for key in ("edge_index", "edge_distance", "edge_distance_vec", "neighbors"):
        torch.testing.assert_close(actual[key], expected[key])


def test_fused_radius_rejects_periodicity_and_position_gradients():
    from fairchem.core.frozen_hip.radius import nonperiodic_graph

    data = metadata([2])
    with pytest.raises(ValueError, match="nonperiodic"):
        nonperiodic_graph(data, 6, 300, torch.ones(1, 3, dtype=torch.bool))
    data.pos.requires_grad_(True)
    with pytest.raises(ValueError, match="frozen coordinates"):
        nonperiodic_graph(data, 6, 300, torch.zeros(1, 3, dtype=torch.bool))


def test_custom_atomic_data_pins_targets_and_keeps_cpu_cache_metadata():
    from fairchem.core.frozen_hip.data import make_batch

    values = dict(
        natoms=np.array([2]),
        atom_ptr=np.array([0, 2]),
        atomic_numbers=np.array([1, 1]),
        coords=np.zeros((2, 3)),
        charge=np.array([0]),
        spin=np.array([1]),
        omol_index=np.array([7]),
        hessian_ptr=np.array([0, 36]),
        hessian_flat=np.zeros(36, dtype=np.float32),
    )
    data = make_batch(values, np.array([0]))
    marker = SimpleNamespace(rows=[0])
    data.hip_cache_groups = [marker]
    data.pin_memory()
    assert data.pos.is_pinned()
    assert data.hessian.is_pinned()
    assert data.batch.is_pinned()
    assert data.hip_cache_groups[0] is marker
    assert marker.rows == [0]
