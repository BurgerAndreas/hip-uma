"""Verify FP32 projection outputs and derivatives against FP64 references."""

from __future__ import annotations

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "shape", [(17, 33, 3), (513, 257, 4), (4097, 1536, 3), (703, 1024, 2)]
)
@pytest.mark.parametrize("gradient_scale", [1.0, 1e-5, 1000.0])
def test_skinny_projection_matches_double(shape, gradient_scale):
    from fairchem.core.frozen_hip.skinny import skinny_linear

    torch.manual_seed(33)
    m, k, n = shape
    x = torch.randn(m, k + 37, device="cuda", dtype=torch.float16)[
        :, :k
    ].requires_grad_()
    w = (torch.randn(n, k, device="cuda") / k**0.5).requires_grad_()
    b = torch.randn(n, device="cuda", requires_grad=True) if n == 3 else None
    g = torch.randn(m, n, device="cuda") * gradient_scale
    inputs = [x, w] + ([b] if b is not None else [])
    reference = [v.detach().double().requires_grad_() for v in inputs]
    expected = torch.nn.functional.linear(
        reference[0],
        reference[1],
        reference[2] if b is not None else None,
    )
    expected_grads = torch.autograd.grad(expected, reference, g.double())
    actual = skinny_linear(x, w, b)
    actual_grads = torch.autograd.grad(actual, inputs, g)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected.float(), atol=3e-6, rtol=3e-5)
    for index, (a, r) in enumerate(zip(actual_grads, expected_grads, strict=True)):
        assert torch.isfinite(a).all()
        expected_gradient = r.to(a.dtype)
        # The returned activation gradient rounds to FP16. Tiny gradients can
        # lie on different sides of a subnormal rounding boundary after FP32
        # accumulation. Parameter gradients must meet the tighter FP32 limit.
        atol = max(6e-8, 2e-7 * gradient_scale) if index == 0 else 0.003
        torch.testing.assert_close(a, expected_gradient, atol=atol, rtol=0.002)
        relative = (
            a.float() - expected_gradient.float()
        ).norm() / expected_gradient.float().norm().clamp_min(1e-20)
        assert relative < (0.0002 if index == 0 else 3e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_skinny_projection_compiles_with_backward():
    from fairchem.core.frozen_hip.skinny import skinny_linear

    compiled = torch.compile(skinny_linear, dynamic=True)
    for m in [71, 87]:
        x = torch.randn(m, 257, device="cuda", dtype=torch.float16, requires_grad=True)
        w = torch.randn(3, 257, device="cuda", requires_grad=True)
        loss = compiled(x, w).square().mean()
        loss.backward()
        assert torch.isfinite(x.grad).all()
        assert torch.isfinite(w.grad).all()


def reference_project(messages, weights, bias):
    parts = messages.split((3, 4, 2), 1)
    values = [
        torch.nn.functional.linear(
            part.flatten(1), weight, bias if index == 0 else None
        )
        for index, (part, weight) in enumerate(zip(parts, weights, strict=True))
    ]
    return torch.cat(values, 1).unsqueeze(-1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("m,channels", [(0, 32), (17, 33), (513, 128), (4097, 512)])
@pytest.mark.parametrize("activation_grad", [False, True])
def test_combined_skinny_matches_double(m, channels, activation_grad):
    from fairchem.core.frozen_hip.skinny import skinny_project

    torch.manual_seed(31)
    x = torch.randn(
        m,
        9,
        channels,
        device="cuda",
        dtype=torch.float16,
        requires_grad=activation_grad,
    )
    weights = tuple(
        (
            torch.randn(n, width * channels, device="cuda") / (width * channels) ** 0.5
        ).requires_grad_()
        for n, width in zip((3, 4, 2), (3, 4, 2), strict=True)
    )
    bias = torch.randn(3, device="cuda", requires_grad=True)
    reference_x = x.detach().double().requires_grad_(activation_grad)
    reference_w = tuple(w.detach().double().requires_grad_() for w in weights)
    reference_b = bias.detach().double().requires_grad_()
    expected = reference_project(reference_x, reference_w, reference_b)
    actual = skinny_project(x, weights, bias)
    g = torch.randn_like(actual) * 1024
    inputs = ((x,) if activation_grad else ()) + weights + (bias,)
    references = (
        ((reference_x,) if activation_grad else ()) + reference_w + (reference_b,)
    )
    expected_grads = torch.autograd.grad(expected, references, g.double())
    actual_grads = torch.autograd.grad(actual, inputs, g)
    torch.testing.assert_close(actual, expected.float(), atol=3e-6, rtol=3e-5)
    for a, r in zip(actual_grads, expected_grads, strict=True):
        expected_gradient = r.to(a.dtype)
        relative = (
            a.float() - expected_gradient.float()
        ).norm() / expected_gradient.float().norm().clamp_min(1e-20)
        assert relative < (0.0002 if a.dtype == torch.float16 else 3e-6)
        assert torch.isfinite(a).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_combined_skinny_compiles_with_backward():
    from fairchem.core.frozen_hip.skinny import skinny_project

    compiled = torch.compile(skinny_project, dynamic=True)
    for m in [71, 87]:
        x = torch.randn(
            m, 9, 32, device="cuda", dtype=torch.float16, requires_grad=True
        )
        weights = tuple(
            torch.randn(n, width * 32, device="cuda", requires_grad=True)
            for n, width in zip((3, 4, 2), (3, 4, 2), strict=True)
        )
        compiled(x, weights).square().mean().backward()
        assert torch.isfinite(x.grad).all()
        assert all(torch.isfinite(w.grad).all() for w in weights)
