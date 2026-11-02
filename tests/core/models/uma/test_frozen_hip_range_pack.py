"""Validate fused packing, virtual concatenation, and FP32 gradient reductions."""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("rows,columns", [(1, 7), (257, 193), (8193, 384)])
@pytest.mark.parametrize("strided", [False, True])
def test_scaled_pack_preserves_operand_values_and_bias(rows, columns, strided):
    from fairchem.core.frozen_hip.linear import quantize
    from fairchem.core.frozen_hip.range_pack import pack_pair

    torch.manual_seed(71)
    raw = torch.randn(rows, columns * (2 if strided else 1), device="cuda") * 2**17
    x = raw[:, ::2] if strided else raw
    packed, scale, bias = pack_pair(x, bias=True)
    expected, reference_scale = quantize(x)
    torch.testing.assert_close(scale, reference_scale, rtol=0, atol=0)
    torch.testing.assert_close(packed, expected, rtol=0, atol=0)
    torch.testing.assert_close(bias, x.sum(0), rtol=3e-5, atol=64)
    assert packed.dtype == torch.float16 and bias.dtype == torch.float32


@pytest.mark.parametrize("split", [0, 13, 32])
def test_virtual_concatenation_and_broadcast_gradients(split):
    from fairchem.core.frozen_hip.linear import quantize
    from fairchem.core.frozen_hip.range_pack import pack_pair

    torch.manual_seed(8)
    a = torch.randn(1, split, device="cuda").expand(259, -1)
    b = torch.randn(259, 37, device="cuda")
    value = torch.cat((a, b), 1)
    packed, scale, bias = pack_pair(a, b, bias=True)
    ref, ref_scale = quantize(value)
    torch.testing.assert_close(packed, ref, rtol=0, atol=0)
    torch.testing.assert_close(scale, ref_scale, rtol=0, atol=0)
    torch.testing.assert_close(bias, value.sum(0), rtol=3e-5, atol=3e-4)


def test_half_input_bypass_and_zero_range_are_finite():
    from fairchem.core.frozen_hip.range_pack import pack_pair

    x = torch.tensor([[2.0**-24, 65504.0, -4.0]], device="cuda", dtype=torch.float16)
    value, scale, _ = pack_pair(x)
    assert value.data_ptr() == x.data_ptr()
    torch.testing.assert_close(value, x, rtol=0, atol=0)
    assert scale.item() == 1
    for dtype in (torch.float16, torch.float32):
        packed, scale, bias = pack_pair(
            torch.zeros(129, 7, device="cuda", dtype=dtype), bias=True
        )
        assert torch.isfinite(scale) and scale > 0
        assert torch.count_nonzero(packed) == torch.count_nonzero(bias) == 0


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_values_still_reach_the_amp_guard(value):
    from fairchem.core.frozen_hip.range_pack import pack_pair

    x = torch.ones(257, 17, device="cuda")
    x[241, 9] = value
    packed, scale, _ = pack_pair(x)
    assert not torch.isfinite(scale)
    assert not torch.isfinite(packed).all()


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_fused_linear_matches_scaled_reference_and_fp32_gradients(dtype):
    from fairchem.core.frozen_hip.edge_precision import half_linear

    torch.manual_seed(29)
    x = (
        (torch.randn(513, 32, device="cuda") * (2**17 if dtype == torch.float32 else 1))
        .to(dtype)
        .requires_grad_()
    )
    w = torch.randn(64, 32, device="cuda", requires_grad=True)
    bias = torch.randn(64, device="cuda", requires_grad=True)
    g = torch.randn(513, 64, device="cuda") * 128
    ref = half_linear(x, w, bias, out32=True)
    grads = torch.autograd.grad(ref, (x, w, bias), g)
    got = half_linear(x, w, bias, out32=True, fused=True)
    actual = torch.autograd.grad(got, (x, w, bias), g)
    assert torch.isfinite(got).all()
    assert (got - ref).norm() / ref.norm() < 8e-4
    for a, b in zip(actual, grads, strict=True):
        assert torch.isfinite(a).all()
        assert (a.float() - b.float()).norm() / b.float().norm() < 1e-3
    assert actual[1].dtype == actual[2].dtype == torch.float32


@pytest.mark.parametrize("use_both", [False, True])
def test_split_outputs_fuse_incoming_gradient_and_bias(use_both):
    from fairchem.core.frozen_hip.edge_precision import SplitHalfLinear, half_linear

    torch.manual_seed(37)
    x = torch.randn(1027, 32, device="cuda", requires_grad=True)
    w = torch.randn(53, 32, device="cuda", requires_grad=True)
    b = torch.randn(53, device="cuda", requires_grad=True)
    ref = half_linear(x, w, b, out32=True)
    loss = ref[:, :13].square().sum() + (ref[:, 13:].sum() if use_both else 0)
    expected = torch.autograd.grad(loss, (x, w, b))
    first, second = SplitHalfLinear.apply(x, w, b, 13)
    loss = first.square().sum() + (second.sum() if use_both else 0)
    got = torch.autograd.grad(loss, (x, w, b))
    torch.testing.assert_close(torch.cat((first, second), 1), ref, rtol=0, atol=0)
    for actual, reference in zip(got, expected, strict=True):
        assert (actual - reference).norm() / reference.norm() < 1e-4


def test_complex_outputs_and_weight_gradient_fold():
    from fairchem.core.frozen_hip.edge_precision import ComplexHalfLinear, half_linear

    torch.manual_seed(17)
    x = torch.randn(257, 2, 24, device="cuda", requires_grad=True)
    w = torch.randn(32, 24, device="cuda", requires_grad=True)
    real, imaginary = w.chunk(2)
    full = torch.cat(
        (torch.cat((real, -imaginary), 1), torch.cat((imaginary, real), 1))
    )
    reference = half_linear(x.flatten(1), full, out32=True).reshape(257, 2, 4, 4)
    expected = torch.autograd.grad(reference.square().sum(), (x, w))
    a, b = ComplexHalfLinear.apply(x, w, 4)
    actual = torch.stack((a, b), 1)
    got = torch.autograd.grad(actual.square().sum(), (x, w))
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    for gradient, ref in zip(got, expected, strict=True):
        assert (gradient - ref).norm() / ref.norm() < 1e-4


def test_fused_paths_support_compile_and_checkpoint_recomputation():
    from torch.utils.checkpoint import checkpoint

    from fairchem.core.frozen_hip.edge_precision import SplitHalfLinear

    torch.manual_seed(19)
    x = torch.randn(511, 32, device="cuda", requires_grad=True)
    w = torch.randn(47, 32, device="cuda", requires_grad=True)
    b = torch.randn(47, device="cuda", requires_grad=True)

    def operation(value):
        first, second = SplitHalfLinear.apply(value, w, b, 13)
        return first.square().mean() + second.square().mean()

    expected = torch.autograd.grad(operation(x), (x, w, b))
    compiled = torch.compile(operation, fullgraph=True)
    result = checkpoint(compiled, x, use_reentrant=False)
    got = torch.autograd.grad(result, (x, w, b))
    for actual, reference in zip(got, expected, strict=True):
        torch.testing.assert_close(actual, reference, rtol=3e-5, atol=3e-5)
