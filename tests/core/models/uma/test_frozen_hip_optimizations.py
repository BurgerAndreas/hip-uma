"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Check packed MAE, scaled derivatives, and fused rotation accumulation.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fairchem.core.frozen_hip.losses import structure_mae
from fairchem.core.frozen_hip.packed import block_mae, pack_hessian
from fairchem.core.models.uma.hessian_pred_utils import (
    add_hessian_graph_batch,
    blocks3x3_to_hessian,
)


@pytest.mark.parametrize("cuda", [False, True])
@pytest.mark.parametrize("sizes", [[1], [1, 2, 5, 3]])
def test_packed_mae_matches_symmetric_dense_value_and_gradients(cuda, sizes):
    if cuda and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = "cuda" if cuda else "cpu"
    n = torch.tensor(sizes, device=device)
    ap = torch.cat((n.new_zeros(1), n.cumsum(0)))
    ep = torch.cat((n.new_zeros(1), (n * (n - 1)).cumsum(0)))
    data = SimpleNamespace(
        pos=torch.randn(sum(sizes), 3, device=device),
        natoms=n,
        batch=torch.arange(len(sizes), device=device).repeat_interleave(n),
        hip_sizes=sizes,
        hip_atom_ptr=ap,
        hip_edge_ptr=ep,
        hip_pair_ptr=ep * 9 // 2,
    )
    add_hessian_graph_batch(data)
    edge = torch.randn(
        sum(s * (s - 1) for s in sizes), 3, 3, device=device, requires_grad=True
    )
    node = torch.randn(sum(sizes), 3, 3, device=device, requires_grad=True)
    labels = [torch.randn(3 * s, 3 * s) for s in sizes]
    packed = [pack_hessian(h.numpy()) for h in labels]
    data.hip_pair_targets = torch.tensor(
        np.concatenate([p[0] for p in packed]), device=device
    )
    data.hip_diagonal_targets = torch.tensor(
        np.concatenate([p[1] for p in packed]), device=device
    )
    dense = blocks3x3_to_hessian(data.edge_index_hessian, data, edge, node)
    expected = structure_mae(
        dense, torch.cat([h.flatten() for h in labels]).to(device), sizes
    )
    gradients = torch.autograd.grad(
        expected, (edge, node), retain_graph=True, allow_unused=True
    )
    data.hip_fused_loss = cuda
    actual = block_mae(edge, node, data)
    actual_gradients = torch.autograd.grad(actual, (edge, node), allow_unused=True)
    torch.testing.assert_close(actual, expected)
    for a, b in zip(actual_gradients, gradients, strict=True):
        if a is None or b is None:
            assert a is None or a.numel() == 0
            assert b is None or b.numel() == 0
        else:
            torch.testing.assert_close(a, b)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("offset", [0, 3])
@pytest.mark.parametrize("segmented", [False, True])
def test_fused_rotation_scatter_preserves_values_and_gradients(offset, segmented):
    from fairchem.core.frozen_hip.scatter import fused_rotate_scatter
    from fairchem.core.models.uma.nn.execution_backends import UMASFastGPUBackend

    torch.manual_seed(71)
    x = torch.randn(701, 9, 128, device="cuda", requires_grad=True)
    edge = torch.randint(31, (2, 701), device="cuda") + offset
    edge = edge[:, edge[1].argsort()].contiguous()
    w = torch.zeros(701, 9, 9, device="cuda")
    for start, end in [(0, 1), (1, 4), (4, 9)]:
        w[:, start:end, start:end] = torch.randn(
            701, end - start, end - start, device="cuda"
        )
    reference = UMASFastGPUBackend.permute_wigner_inv_edge_to_node(
        x, w, edge, 31, offset
    )
    actual = fused_rotate_scatter(x, w, edge, 31, offset, segmented=segmented)
    torch.testing.assert_close(actual, reference, rtol=2e-5, atol=2e-5)
    g = torch.randn_like(actual)
    expected_gradient = torch.autograd.grad(reference, x, g)[0]
    actual_gradient = torch.autograd.grad(actual, x, g)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=2e-5, atol=2e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("channels", [4, 128])
@pytest.mark.parametrize("count", [0, 701])
@pytest.mark.parametrize("tile", [64, 256])
def test_fused_radial_gather_matches_values_and_both_gradients(
    channels, count, tile, monkeypatch
):
    from fairchem.core.frozen_hip import radial as radial_module
    from fairchem.core.frozen_hip.radial import fused_radial_gather

    monkeypatch.setitem(radial_module._KERNEL_CONFIG, "channel_tile", tile)
    from fairchem.core.models.uma.nn.execution_backends import UMASFastGPUBackend

    torch.manual_seed(71)
    x = torch.randn(31, 9, channels, device="cuda", requires_grad=True)
    edge = torch.randint(31, (2, count), device="cuda")
    w = torch.zeros(count, 9, 9, device="cuda")
    for start, end in [(0, 1), (1, 4), (4, 9)]:
        w[:, start:end, start:end] = torch.randn(
            count, end - start, end - start, device="cuda"
        )
    radial = torch.randn(count, 6 * 2 * channels, device="cuda", requires_grad=True)
    unscaled = UMASFastGPUBackend.node_to_edge_wigner_permute(x, edge, w)
    expanded = radial.reshape(count, 6, 2 * channels)[:, [0, 1, 2, 3, 4, 3, 4, 5, 5]]
    expected = unscaled * expanded
    actual = fused_radial_gather(x, edge, w, radial)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
    gradient = torch.randn_like(actual)
    reference_grad = torch.autograd.grad(expected, (x, radial), gradient)
    actual_grad = torch.autograd.grad(actual, (x, radial), gradient)
    for a, b in zip(actual_grad, reference_grad, strict=True):
        torch.testing.assert_close(a, b, rtol=3e-5, atol=2e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("sx,sg", [(1.0, 1.0), (1e5, 1e-10), (1e-10, 1e5)])
def test_scaled_linear_preserves_range_and_gradient_accuracy(sx, sg):
    from fairchem.core.frozen_hip.linear import linear

    torch.manual_seed(71)
    torch.backends.cuda.matmul.allow_tf32 = False
    x = (torch.randn(701, 384, device="cuda") * sx).requires_grad_()
    w = torch.randn(128, 384, device="cuda", requires_grad=True)
    b = torch.randn(128, device="cuda", requires_grad=True)
    g = torch.randn(701, 128, device="cuda") * sg
    expected = torch.nn.functional.linear(x, w, b)
    actual = linear(x, w, b)
    expected_gradient = torch.autograd.grad(expected, (x, w, b), g)
    actual_gradient = torch.autograd.grad(actual, (x, w, b), g)
    for a, r in [
        (actual, expected),
        *zip(actual_gradient, expected_gradient, strict=True),
    ]:
        assert torch.isfinite(a).all()
        assert (a - r).double().norm() <= 0.001 * r.double().norm()


@pytest.mark.parametrize("budget", [3000, 524288, 1000000000])
def test_balanced_sampler_keeps_original_best_fit_order_and_all_samples(budget):
    from fairchem.core.frozen_hip.data import BalancedBatchSampler

    rng = np.random.default_rng(117)
    sizes = np.r_[rng.integers(1, 350, 1000), [1, 1, 3500, 3500]]
    expected = []
    for row in np.argsort(-sizes, kind="stable"):
        n = int(sizes[row])
        feasible = [
            i
            for i, batch in enumerate(expected)
            if len(batch) < 64
            and sum(sizes[batch]) + n <= 2560
            and sum(sizes[batch] ** 2) + n * n <= budget
        ]
        if feasible:
            i = max(feasible, key=lambda j: sum(sizes[expected[j]]))
            expected[i].append(int(row))
        else:
            expected.append([int(row)])
    actual = list(
        BalancedBatchSampler(sizes, max_atoms=2560, batch_size=64, pair_budget=budget)
    )
    assert actual == expected
    assert sorted(index for batch in actual for index in batch) == list(
        range(len(sizes))
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_device_prefetch_preserves_batches_and_stream_lifetimes():
    from fairchem.core.frozen_hip.prefetch import DeviceBatchPrefetcher

    class Data:
        def __init__(self, value):
            self.value = value

        def apply(self, operation):
            self.value = operation(self.value)

    def source():
        for i in range(32):
            yield [Data(torch.full((1024, 1024), float(i), pin_memory=True))]

    def move(data):
        data.value = data.value.to("cuda", non_blocking=True)
        return data

    stream = DeviceBatchPrefetcher(iter(source()), move, torch.device("cuda"))
    results = []
    for batch in stream:
        results.append(batch[0].value.square().mean())
        del batch
    stream._shutdown_workers()
    torch.testing.assert_close(
        torch.stack(results).cpu(), torch.arange(32).float().square()
    )
    assert stream.pending is None


def test_dense_head_omits_unused_routing_without_changing_predictions(monkeypatch):
    from fairchem.core.frozen_hip.data import make_batch
    from fairchem.core.models.uma.escn_md import Hessian_Head, eSCNMDBackbone

    backbone = eSCNMDBackbone(
        max_num_elements=100,
        sphere_channels=4,
        lmax=2,
        mmax=2,
        otf_graph=True,
        edge_channels=5,
        num_distance_basis=7,
        num_layers=1,
        hidden_channels=4,
        use_dataset_embedding=False,
        always_use_pbc=False,
    )
    head = Hessian_Head(backbone)
    calls = []
    monkeypatch.setattr(backbone, "set_MOLE_sizes", lambda **kw: calls.append(kw))
    data = make_batch(
        dict(
            natoms=np.array([2]),
            atomic_numbers=np.array([1, 1]),
            coords=np.array([[0, 0, 0], [0.8, 0, 0]]),
            charge=np.array([0]),
            spin=np.array([1]),
            omol_index=np.array([7]),
            hessian_flat=np.zeros(36, dtype=np.float32),
        ),
        np.array([0]),
    )
    emb = {"node_embedding": torch.randn(2, 9, 4), "batch": data.batch}
    torch.manual_seed(19)
    expected = head(data, emb)["hessian"]
    assert len(calls) == 1
    head.requires_mole_routing = False
    torch.manual_seed(19)
    actual = head(data, emb)["hessian"]
    assert len(calls) == 1
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fused_radial_head_preserves_prediction_and_parameter_gradients():
    from fairchem.core.frozen_hip.data import make_batch
    from fairchem.core.frozen_hip.radial import enable_fused_radial
    from fairchem.core.frozen_hip.readout import enable_projected_readout
    from fairchem.core.frozen_hip.rotations import enable_fused_rotations
    from fairchem.core.frozen_hip.so2 import enable_block_so2
    from fairchem.core.models.uma.escn_md import Hessian_Head, eSCNMDBackbone

    torch.manual_seed(71)
    torch.backends.cuda.matmul.allow_tf32 = False
    backbone = eSCNMDBackbone(
        max_num_elements=100,
        sphere_channels=4,
        lmax=2,
        mmax=2,
        edge_channels=5,
        num_distance_basis=7,
        num_layers=1,
        hidden_channels=4,
        use_dataset_embedding=False,
        always_use_pbc=False,
    ).cuda()
    backbone.requires_grad_(False)
    head = Hessian_Head(backbone).cuda()
    head.requires_mole_routing = False
    enable_fused_rotations(head)
    enable_block_so2(head)
    enable_projected_readout(head)
    data = make_batch(
        dict(
            natoms=np.array([2, 3]),
            atomic_numbers=np.array([1, 1, 6, 1, 1]),
            coords=np.array(
                [[0, 0, 0], [0.8, 0, 0], [0, 0, 0], [0, 0.8, 0], [0.8, 0, 0]]
            ),
            charge=np.array([0, 0]),
            spin=np.array([1, 1]),
            omol_index=np.array([7, 9]),
            hessian_flat=np.zeros(117, dtype=np.float32),
        ),
        np.array([0, 1]),
    ).cuda()
    emb = {"node_embedding": torch.randn(5, 9, 4, device="cuda"), "batch": data.batch}
    parameters = [p for p in head.parameters() if p.requires_grad]
    before = tuple(head.state_dict())
    torch.manual_seed(19)
    expected = head(data, emb)["hessian"]
    expected_grad = torch.autograd.grad(expected.square().mean(), parameters)
    enable_fused_radial(head)
    assert tuple(head.state_dict()) == before
    torch.manual_seed(19)
    actual = head(data, emb)["hessian"]
    actual_grad = torch.autograd.grad(actual.square().mean(), parameters)
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)
    for a, b in zip(actual_grad, expected_grad, strict=True):
        torch.testing.assert_close(a, b, rtol=3e-4, atol=2e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("count", [0, 701])
def test_compact_quartic_wigner_matches_both_charts_and_envelope(count):
    from fairchem.core.frozen_hip.wigner import enable_fused_wigner, frozen_wigner
    from fairchem.core.models.uma.common.quaternion.wigner_d_hybrid import (
        axis_angle_wigner_hybrid,
    )
    from fairchem.core.models.uma.escn_md import Hessian_Head, eSCNMDBackbone

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(17)
    backbone = eSCNMDBackbone(
        max_num_elements=100,
        sphere_channels=4,
        lmax=2,
        mmax=2,
        edge_channels=5,
        num_distance_basis=7,
        num_layers=1,
        hidden_channels=4,
        use_dataset_embedding=False,
        always_use_pbc=False,
    ).cuda()
    head = Hessian_Head(backbone).cuda()
    before = tuple(head.state_dict())
    enable_fused_wigner(head)
    assert tuple(head.state_dict()) == before
    vectors = torch.randn(count, 3, device="cuda")
    if count:
        vectors[:7] = vectors.new_tensor(
            [
                [0, 1, 0],
                [0, -1, 0],
                [0, 0, 0],
                [1e-6, 1, 1e-6],
                [1e-6, -1, 1e-6],
                [100, 0, 0],
                [101, 0, 0],
            ]
        )
    distances = vectors.norm(dim=-1)
    gamma = torch.rand(count, device="cuda") * (2 * torch.pi)
    reference, inverse = axis_angle_wigner_hybrid(
        vectors,
        2,
        gamma=gamma,
        coeffs=backbone.wigner_data.coeffs,
        U_blocks=backbone.wigner_data.U_blocks,
        custom_kernels=backbone.wigner_data.custom_kernels,
    )
    envelope = backbone.envelope(distances / head.cutoff_hessian).reshape(-1, 1, 1)
    w, inv = frozen_wigner(head, vectors, distances, gamma)
    torch.testing.assert_close(w, reference, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(inv, inverse * envelope, rtol=2e-5, atol=2e-5)
    assert torch.isfinite(w).all()
    assert torch.isfinite(inv).all()
    if count:
        with pytest.raises(ValueError, match="frozen"):
            frozen_wigner(head, vectors.requires_grad_(), distances, gamma)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_native_half_product_retains_fp32_range_during_long_reduction():
    x = torch.ones(131072, 32, device="cuda", dtype=torch.float16)
    output = torch.mm(x.t(), x, out_dtype=torch.float32)
    torch.testing.assert_close(output, torch.full_like(output, 131072))
    assert output.dtype == torch.float32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_scaled_linear_compiles_with_fp32_gradients_and_no_graph_breaks():
    from fairchem.core.frozen_hip.linear import linear

    torch.manual_seed(73)
    x = torch.randn(701, 384, device="cuda", requires_grad=True)
    weight = torch.randn(128, 384, device="cuda", requires_grad=True)
    gradient = torch.randn(701, 128, device="cuda") * 1e-10
    expected = linear(x, weight)
    expected_gradients = torch.autograd.grad(expected, (x, weight), gradient)
    compiled = torch.compile(linear, dynamic=True, fullgraph=True)
    actual = compiled(x, weight)
    actual_gradients = torch.autograd.grad(actual, (x, weight), gradient)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
    for a, b in zip(actual_gradients, expected_gradients, strict=True):
        assert a.dtype == torch.float32
        torch.testing.assert_close(a, b, rtol=2e-5, atol=1e-15)
