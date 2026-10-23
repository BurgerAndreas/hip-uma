"""Check selective FP16 outputs, FP32 gradients, and restored HIP dispatch."""

from __future__ import annotations

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("fail", [False, True])
def test_frozen_features_keep_their_original_reduction_policy(fail):
    from types import SimpleNamespace

    from fairchem.core.frozen_hip.model import FrozenHIP

    original = torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction

    def backbone(_):
        assert torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
        if fail:
            raise ValueError("feature failure")
        return {"node_embedding": torch.ones(1, 9, 4, device="cuda")}

    model = SimpleNamespace(
        backbone=backbone, backbone_fp16_reduced_precision_reduction=True
    )
    try:
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        if fail:
            with pytest.raises(ValueError, match="feature failure"):
                FrozenHIP.features(model, None, "fp16")
        else:
            assert (
                FrozenHIP.features(model, None, "fp16")["node_embedding"].dtype
                == torch.float32
            )
        assert not torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
    finally:
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = original


@pytest.mark.parametrize("out32", [False, True])
def test_edge_linear_retains_fp32_parameters_and_scaled_gradients(out32):
    from fairchem.core.frozen_hip.edge_precision import half_linear

    torch.manual_seed(19)
    x = torch.randn(3, 7, 16, device="cuda", requires_grad=True)
    w = torch.randn(32, 16, device="cuda", requires_grad=True)
    b = torch.randn(32, device="cuda", requires_grad=True)
    expected = torch.nn.functional.linear(x, w, b)
    gradient = torch.randn_like(expected) / expected.numel()
    reference = torch.autograd.grad((expected * gradient).sum(), (x, w, b))
    actual = half_linear(x, w, b, out32=out32)
    assert actual.dtype == (torch.float32 if out32 else torch.float16)
    grads = torch.autograd.grad((actual.float() * gradient).sum() * 1024, (x, w, b))
    assert (actual.float() - expected).norm() / expected.norm() < 0.002
    for value, ref in zip(grads, reference, strict=True):
        assert value.dtype == torch.float32
        assert torch.isfinite(value).all()
        assert (value / 1024 - ref).norm() / ref.norm() < 0.003


def test_edge_layer_norm_computes_variance_in_fp32():
    from fairchem.core.frozen_hip.edge_precision import norm_forward

    torch.manual_seed(13)
    norm = torch.nn.LayerNorm(32).cuda()
    x = (torch.randn(8, 32, device="cuda") * 500).half().requires_grad_()
    expected = torch.nn.functional.layer_norm(
        x.float(), norm.normalized_shape, norm.weight, norm.bias, norm.eps
    ).half()
    actual = norm_forward(norm, x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.float().square().mean().backward()
    assert norm.weight.grad.dtype == torch.float32
    assert torch.isfinite(x.grad).all()


def test_edge_parameter_gradient_keeps_long_reduction_in_fp32():
    from fairchem.core.frozen_hip.edge_precision import half_linear

    x = torch.ones(131072, 32, device="cuda")
    w = torch.ones(1, 32, device="cuda", requires_grad=True)
    output = half_linear(x, w)
    (output.float().sum() * 1024).backward()
    assert w.grad.dtype == torch.float32
    torch.testing.assert_close(w.grad / 1024, torch.full_like(w, 131072))


def test_edge_precision_preserves_head_state_and_cartesian_output():
    from fairchem.core.frozen_hip.data import make_batch
    from fairchem.core.frozen_hip.edge_precision import enable_edge_precision
    from fairchem.core.frozen_hip.radial import enable_fused_radial
    from fairchem.core.frozen_hip.readout import enable_projected_readout
    from fairchem.core.frozen_hip.rotations import (
        enable_fused_rotations,
        enable_fused_scatter,
    )
    from fairchem.core.frozen_hip.so2 import enable_block_so2
    from fairchem.core.models.uma.escn_md import Hessian_Head, eSCNMDBackbone

    torch.manual_seed(71)
    backbone = (
        eSCNMDBackbone(
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
        )
        .cuda()
        .requires_grad_(False)
    )
    head = Hessian_Head(backbone).cuda()
    head.requires_mole_routing = False
    enable_fused_rotations(head)
    enable_block_so2(head)
    enable_projected_readout(head)
    enable_fused_radial(head)
    enable_fused_scatter(head, segmented=True)
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
    params = list(head.parameters())
    state_keys = tuple(head.state_dict())
    torch.manual_seed(19)
    expected = head(data, emb)["hessian"]
    reference = torch.autograd.grad(expected.square().mean(), params)
    enable_edge_precision(head)
    torch.manual_seed(19)
    actual = head(data, emb)["hessian"]
    grads = torch.autograd.grad(actual.square().mean() * 1024, params)
    assert actual.dtype == torch.float32
    assert tuple(head.state_dict()) == state_keys
    assert all(p.dtype == torch.float32 for p in params)
    assert (actual - expected).norm() / expected.norm() < 0.003
    assert all(g.dtype == torch.float32 and torch.isfinite(g).all() for g in grads)
    ref = torch.cat([g.flatten() for g in reference])
    got = torch.cat([g.flatten() / 1024 for g in grads])
    assert (got - ref).norm() / ref.norm() < 0.02
    enable_edge_precision(head, False)
    torch.manual_seed(19)
    restored = head(data, emb)["hessian"]
    torch.testing.assert_close(restored, expected, rtol=2e-4, atol=2e-5)


def test_second_convolution_preserves_large_finite_outputs():
    from fairchem.core.frozen_hip.edge_precision import linear_forward

    layer = torch.nn.Linear(16, 4, bias=False).cuda()
    with torch.no_grad():
        layer.weight.fill_(4)
    layer.hip_fp32_output = True
    x = torch.full(
        (8, 16), 2048.0, device="cuda", dtype=torch.float16, requires_grad=True
    )
    y = linear_forward(layer, x)
    assert y.dtype == torch.float32
    assert y.min() > torch.finfo(torch.float16).max
    torch.testing.assert_close(y, torch.nn.functional.linear(x.float(), layer.weight))
    y.mean().backward()
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(layer.weight.grad).all()


def test_scaled_edge_linear_handles_large_activations_and_backward_operands():
    from fairchem.core.frozen_hip.edge_precision import half_linear

    x = torch.full((256, 16), 2.0**17, device="cuda", requires_grad=True)
    w = torch.full((32, 16), 0.25, device="cuda", requires_grad=True)
    expected = torch.nn.functional.linear(x, w)
    gradient = torch.full_like(expected, 2.0**17)
    reference = torch.autograd.grad(expected, (x, w), gradient)
    actual = half_linear(x, w, out32=True)
    grads = torch.autograd.grad(actual, (x, w), gradient)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for value, ref in zip(grads, reference, strict=True):
        assert torch.isfinite(value).all()
        torch.testing.assert_close(value, ref, rtol=0, atol=0)
