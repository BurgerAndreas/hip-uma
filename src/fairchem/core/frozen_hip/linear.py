"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Use scaled FP16 tensor-core operands with FP32 outputs and derivatives.
"""

from __future__ import annotations

from types import MethodType

import torch
from torch.nn import functional as F


def quantize(value):
    """Use exact power-of-two scales to protect the FP16 exponent range."""
    maximum = value.detach().abs().amax()
    scale = torch.exp2(torch.floor(torch.log2(maximum.clamp_min(2.0**-100))))
    return (value / scale).half(), scale


@torch.library.custom_op("fairchem::hip_half_mm_float_output", mutates_args=())
def half_mm_float_output(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Isolate the CUDA output-dtype API from older compiler meta functions."""
    return torch.mm(a, b, out_dtype=torch.float32)


@half_mm_float_output.register_fake
def _half_mm_float_output_fake(a, b):
    return a.new_empty((a.shape[0], b.shape[1]), dtype=torch.float32)


def product(a, b, sa, sb, bias):
    """Retain a full FP32 accumulator and output through the vendor GEMM."""
    output = half_mm_float_output(a, b) * (sa * sb)
    return output + bias if bias is not None else output


def weight_product(x, gradient, sx, sg):
    """Keep the long reduction in FP32 while reading scaled FP16 operands."""
    return half_mm_float_output(gradient.t(), x) * (sx * sg)


class _ScaledLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias):
        flat = x.reshape(-1, x.shape[-1]).contiguous()
        xh, sx = quantize(flat)
        wh, sw = quantize(weight)
        ctx.save_for_backward(xh, wh, sx, sw)
        ctx.shape = x.shape
        ctx.has_bias = bias is not None
        return product(xh, wh.t(), sx, sw, bias).reshape(*x.shape[:-1], weight.shape[0])

    @staticmethod
    def backward(ctx, gradient):
        x, weight, sx, sw = ctx.saved_tensors
        gradient = gradient.reshape(-1, weight.shape[0]).contiguous()
        gh, sg = quantize(gradient)
        dx = (
            product(gh, weight, sg, sw, None).reshape(ctx.shape)
            if ctx.needs_input_grad[0]
            else None
        )
        dw = weight_product(x, gh, sx, sg) if ctx.needs_input_grad[1] else None
        db = gradient.sum(0) if ctx.has_bias and ctx.needs_input_grad[2] else None
        return dx, dw, db


def linear(x, weight, bias=None):
    """Keep small matrices on the standard path and preserve FP32 state."""
    if not x.is_cuda or x.dtype != torch.float32 or weight.dtype != torch.float32:
        return F.linear(x, weight, bias)
    if x.numel() // x.shape[-1] < 256 or weight.shape[0] < 32:
        return F.linear(x, weight, bias)
    return _ScaledLinear.apply(x, weight, bias)


def _forward(self, x):
    return linear(x, self.weight, self.bias)


def enable_scaled_linear(head, enabled=True):
    """Preserve parameter names and select tensor-core linear operations."""
    count = 0
    for module in head.modules():
        if isinstance(module, torch.nn.Linear):
            module.forward = MethodType(
                _forward if enabled else torch.nn.Linear.forward, module
            )
            count += 1
        if hasattr(module, "projected_readout") or hasattr(module, "so2_m_conv"):
            module.scaled_linear = enabled
    for module in head.modules():
        if hasattr(module, "out_channels_half"):
            module.scaled_linear = enabled
    head.scaled_linear = enabled
    return count
