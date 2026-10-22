"""Compare trainable experimental fusions with PyTorch, including FP32 parameter gradients."""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def assert_relative(actual, expected, tolerance):
    assert torch.isfinite(actual).all()
    error = (actual.float() - expected.float()).norm()
    assert error / expected.float().norm().clamp_min(1e-20) < tolerance


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("shape", [(17, 96), (257, 128), (122150, 128)])
def test_norm_silu_retains_input_and_parameter_gradients(dtype, shape):
    from fairchem.core.frozen_hip.nitro_candidates import _NormSiLU

    torch.manual_seed(19)
    x = (torch.randn(*shape, device="cuda", dtype=dtype) * 3).requires_grad_()
    w = torch.randn(shape[-1], device="cuda", requires_grad=True)
    b = torch.randn_like(w, requires_grad=True)
    expected = torch.nn.functional.silu(
        torch.nn.functional.layer_norm(x.float(), (shape[-1],), w, b, 1e-5).to(dtype)
    )
    gradient = torch.randn_like(expected)
    reference = torch.autograd.grad(expected, (x, w, b), gradient)
    actual = _NormSiLU.apply(x, w, b, 1e-5)
    grads = torch.autograd.grad(actual, (x, w, b), gradient)
    tolerance = 0.004 if dtype == torch.float16 else 5e-5
    assert_relative(actual, expected, tolerance)
    for value, ref in zip(grads, reference, strict=True):
        assert_relative(value, ref, tolerance)
    assert grads[1].dtype == grads[2].dtype == torch.float32


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("m_prime", [False, True])
@pytest.mark.parametrize("channels", [96, 128])
def test_gate_matches_both_coefficient_orders(dtype, m_prime, channels):
    from fairchem.core.frozen_hip.nitro_candidates import _Gate
    from fairchem.core.models.uma.nn.activation import GateActivation

    torch.manual_seed(29)
    module = GateActivation(2, 2, channels, m_prime=m_prime).cuda()
    x = torch.randn(513, 9, channels, device="cuda", dtype=dtype, requires_grad=True)
    g = torch.randn(513, 2 * channels, device="cuda", dtype=dtype, requires_grad=True)
    gradient = torch.randn_like(x)
    expected = module(g, x)
    reference = torch.autograd.grad(expected, (g, x), gradient)
    actual = _Gate.apply(g, x, module.expand_index)
    grads = torch.autograd.grad(actual, (g, x), gradient)
    tolerance = 0.004 if dtype == torch.float16 else 2e-5
    assert_relative(actual, expected, tolerance)
    for value, ref in zip(grads, reference, strict=True):
        assert_relative(value, ref, tolerance)


def test_scaled_operands_preserve_tiny_values_and_fp32_weight_gradients(monkeypatch):
    from fairchem.core.frozen_hip import linear as linear_kernels
    from fairchem.core.frozen_hip.nitro_candidates import nitro_scaled_quantize

    monkeypatch.setattr(linear_kernels, "quantize", nitro_scaled_quantize)
    previous = torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    try:
        x = torch.ones(1024, 2, device="cuda")
        x[:, 1] = 2.0**-30
        x.requires_grad_()
        w = torch.tensor([[0.0, 1.0]], device="cuda", requires_grad=True)
        expected = torch.nn.functional.linear(x, w)
        reference = torch.autograd.grad(expected.sum(), (x, w))
        actual = linear_kernels._ScaledLinear.apply(x, w, None)
        grads = torch.autograd.grad(actual.sum(), (x, w))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for value, ref in zip(grads, reference, strict=True):
            assert value.dtype == torch.float32
            torch.testing.assert_close(value, ref, rtol=0, atol=0)
    finally:
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = previous


@pytest.mark.parametrize("chunk", [512, 2048, 8192])
@pytest.mark.parametrize("strided", [False, True])
def test_dense_weight_gradient_matches_vendor_with_row_strides(chunk, strided):
    from fairchem.core.frozen_hip.nitro_candidates import dense_weight_gradient

    torch.manual_seed(37)
    if strided:
        x = torch.randn(8193, 195, device="cuda", dtype=torch.float16)[:, 3:99]
        g = torch.randn(8193, 135, device="cuda", dtype=torch.float16)[:, 5:70]
    else:
        x = torch.randn(8193, 96, device="cuda", dtype=torch.float16)
        g = torch.randn(8193, 65, device="cuda", dtype=torch.float16)
    previous = torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    try:
        expected = torch.mm(g.T, x, out_dtype=torch.float32)
        actual = dense_weight_gradient(x, g, chunk)
        assert actual.dtype == torch.float32
        assert_relative(actual, expected, 1e-5)
        oracle = g.double().T @ x.double()
        assert_relative(actual, oracle, 1e-5)
    finally:
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = previous


def test_dense_weight_gradient_retains_large_fp32_reductions():
    from fairchem.core.frozen_hip.nitro_candidates import dense_weight_gradient

    x = torch.ones(131072, 16, device="cuda", dtype=torch.float16)
    g = torch.full((131072, 1), 1024, device="cuda", dtype=torch.float16)
    actual = dense_weight_gradient(x, g, 2048)
    torch.testing.assert_close(
        actual, torch.full_like(actual, 131072 * 1024), rtol=0, atol=0
    )
