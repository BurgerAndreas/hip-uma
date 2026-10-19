"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Use FP16 edge activations with FP32 parameters, reductions, and Cartesian outputs.
"""

from __future__ import annotations

from types import MethodType

import torch
import triton
from torch.library import triton_op, wrap_triton

from fairchem.core.frozen_hip.linear import half_mm_float_output
from fairchem.core.frozen_hip.radial import (
    _backward_node,
    _backward_radial,
    _forward,
    enable_fused_radial,
)
from fairchem.core.frozen_hip.rotations import FusedRotationBackend
from fairchem.core.frozen_hip.scatter import _bounds, _segment_rotate
from fairchem.core.frozen_hip.so2 import enable_block_so2
from fairchem.core.models.uma.nn.so2_layers import SO2_m_Conv
from fairchem.core.models.uma.triton.constants import BLOCK_C, GRID_E_STRIDE
from fairchem.core.models.uma.triton.kernels import (
    permute_wigner_inv_edge_to_node_bwd_dx_kernel,
    permute_wigner_inv_edge_to_node_kernel,
)


class HalfLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, b, out32):
        xh = x.reshape(-1, x.shape[-1]).half()
        wh = w.half()
        ctx.save_for_backward(xh, wh)
        ctx.shape = x.shape
        ctx.dtype = x.dtype
        ctx.bias = b is not None
        if out32:
            out = half_mm_float_output(xh, wh.t())
            if b is not None:
                out = out + b
        else:
            # Keep the bias inside the vendor GEMM epilogue.
            out = torch.nn.functional.linear(xh, wh, None if b is None else b.half())
        return out.reshape(*x.shape[:-1], w.shape[0])

    @staticmethod
    def backward(ctx, g):
        x, w = ctx.saved_tensors
        g = g.reshape(-1, w.shape[0])
        gh = g.half()
        dx = (
            (
                half_mm_float_output(gh, w) if ctx.dtype == torch.float32 else gh @ w
            ).reshape(ctx.shape)
            if ctx.needs_input_grad[0]
            else None
        )
        dw = half_mm_float_output(gh.t(), x) if ctx.needs_input_grad[1] else None
        db = g.float().sum(0) if ctx.bias and ctx.needs_input_grad[2] else None
        return dx, dw, db, None


def half_linear(x, w, b=None, *, out32=False):
    return HalfLinear.apply(x, w, b, out32)


def linear_forward(self, x):
    return half_linear(x, self.weight, self.bias)


def so2_forward(self, x):
    real, imag = self.fc.weight.split(self.out_channels_half, 0)
    weight = torch.cat((torch.cat((real, -imag), 1), torch.cat((imag, real), 1)), 0)
    out = half_linear(x.flatten(1), weight)
    return out.reshape(x.shape[0], 2, -1, self.m_output_channels).unbind(1)


@triton_op("fairchem::hip_edge_mixed_radial", mutates_args={})
def mixed_radial(
    x: torch.Tensor, w: torch.Tensor, edges: torch.Tensor, r: torch.Tensor
) -> torch.Tensor:
    e, c = edges.shape[1], x.shape[-1]
    out = torch.empty((e, 9, 2 * c), device=x.device, dtype=torch.float16)
    wrap_triton(_forward)[(min(max(e, 1), 8192), triton.cdiv(2 * c, 64))](
        x, w, edges, r, out, e, c, 64, 8192, num_warps=2
    )
    return out


@triton_op("fairchem::hip_edge_mixed_radial_grad", mutates_args={})
def mixed_radial_grad(
    x: torch.Tensor,
    w: torch.Tensor,
    edges: torch.Tensor,
    r: torch.Tensor,
    g: torch.Tensor,
    need_x: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    e, c = edges.shape[1], x.shape[-1]
    dr = torch.empty_like(r)
    dx = torch.zeros_like(x) if need_x else x.new_empty(0)
    grid = (min(max(e, 1), 8192), triton.cdiv(2 * c, 64))
    wrap_triton(_backward_radial)[grid](x, w, edges, g, dr, e, c, 64, 8192, num_warps=2)
    if need_x:
        wrap_triton(_backward_node)[grid](
            w, edges, r, g, dx, e, c, 64, 8192, num_warps=2
        )
    return dx, dr


class MixedRadial(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, edges, w, r):
        ctx.save_for_backward(x, edges, w, r)
        return mixed_radial(x, w, edges, r)

    @staticmethod
    def backward(ctx, g):
        x, edges, w, r = ctx.saved_tensors
        dx, dr = mixed_radial_grad(
            x, w, edges, r, g.contiguous(), ctx.needs_input_grad[0]
        )
        return dx if ctx.needs_input_grad[0] else None, None, None, dr


def radial_convolution(conv, x, edges, w, edge_features):
    r = conv.rad_func(edge_features)
    messages = MixedRadial.apply(x, edges, w, r)
    parts = messages.split(conv.m_split_sizes, 1)
    e = messages.shape[0]
    c = conv.m_output_channels
    scalar = conv.fc_m0(parts[0].flatten(1))
    extra = conv.extra_m0_output_channels
    gates, scalar = scalar.split((extra, scalar.shape[-1] - extra), -1)
    out = [scalar.reshape(e, 3, c)]
    for m, block in enumerate(conv.so2_m_conv, 1):
        out.extend(block(parts[m].reshape(e, 2, (3 - m) * conv.sphere_channels)))
    return torch.cat(out, 1), gates


def project_convolution(conv, projection, messages):
    channels = conv.m_output_channels
    degrees = conv.lmax + 1
    p = projection.weight[:, 0, :]
    original = conv.fc_m0
    weight = torch.einsum(
        "lc,lci->li", p, original.weight.reshape(degrees, channels, -1)
    )
    bias = (
        (original.bias.reshape(degrees, channels) * p).sum(-1)
        if original.bias is not None
        else None
    )
    parts = messages.split(conv.m_split_sizes, 1)
    e = messages.shape[0]
    out = [
        torch.nn.functional.linear(parts[0].flatten(1).float(), weight, bias).unsqueeze(
            -1
        )
    ]
    for m, block in enumerate(conv.so2_m_conv, 1):
        count = degrees - m
        original = block.fc.weight.reshape(2, count, channels, -1)
        effective = torch.einsum("lc,slci->sli", p[m:], original)
        real, imag = effective.unbind(0)
        weight = torch.cat((torch.cat((real, -imag), 1), torch.cat((imag, real), 1)), 0)
        value = torch.nn.functional.linear(parts[m].flatten(1).float(), weight).reshape(
            e, 2, count, 1
        )
        out.extend(value.unbind(1))
    return torch.cat(out, 1)


@triton_op("fairchem::hip_edge_mixed_scatter", mutates_args={})
def mixed_scatter(
    x: torch.Tensor,
    w: torch.Tensor,
    edges: torch.Tensor,
    nodes: int,
    offset: int,
    segmented: bool,
) -> torch.Tensor:
    e, c = edges.shape[1], x.shape[-1]
    if segmented:
        ptr = torch.empty(nodes + 1, device=x.device, dtype=torch.int32)
        wrap_triton(_bounds)[(triton.cdiv(nodes + 1, 32),)](
            edges, ptr, e, nodes, offset, 32, num_warps=1
        )
        out = torch.empty((nodes, 9, c), device=x.device, dtype=torch.float32)
        wrap_triton(_segment_rotate)[(nodes * 9, triton.cdiv(c, 32))](
            x, w, ptr, out, c, 32, 32, num_warps=4
        )
    else:
        out = torch.zeros((nodes, 9, c), device=x.device, dtype=torch.float32)
        wrap_triton(permute_wigner_inv_edge_to_node_kernel)[
            (min(max(e, 1), GRID_E_STRIDE), triton.cdiv(c, BLOCK_C))
        ](
            x,
            w,
            out,
            x,
            e,
            c,
            BLOCK_C,
            GRID_E_STRIDE,
            False,
            edges,
            True,
            offset,
            num_warps=1,
        )
    return out


@triton_op("fairchem::hip_edge_mixed_gather", mutates_args={})
def mixed_gather(
    g: torch.Tensor, w: torch.Tensor, edges: torch.Tensor, offset: int
) -> torch.Tensor:
    e, c = edges.shape[1], g.shape[-1]
    out = torch.empty((e, 9, c), device=g.device, dtype=torch.float16)
    wrap_triton(permute_wigner_inv_edge_to_node_bwd_dx_kernel)[
        (min(max(e, 1), GRID_E_STRIDE), triton.cdiv(c, BLOCK_C))
    ](g, w, out, e, c, BLOCK_C, GRID_E_STRIDE, edges, True, offset, num_warps=1)
    return out


class MixedScatter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, edges, nodes, offset, segmented):
        ctx.save_for_backward(w, edges)
        ctx.offset = offset
        return mixed_scatter(x.contiguous(), w, edges, nodes, offset, segmented)

    @staticmethod
    def backward(ctx, g):
        w, edges = ctx.saved_tensors
        return (
            mixed_gather(g.contiguous(), w, edges, ctx.offset),
            None,
            None,
            None,
            None,
            None,
        )


class MixedBackend(FusedRotationBackend):
    def __init__(self, segmented):
        self.segmented = segmented

    def permute_wigner_inv_edge_to_node(self, x, w, edges, nodes, node_offset=0):
        return MixedScatter.apply(x, w, edges, nodes, node_offset, self.segmented)


def norm_forward(self, x):
    return torch.nn.functional.layer_norm(
        x.float(), self.normalized_shape, self.weight, self.bias, self.eps
    ).to(x.dtype)


def enable_edge_precision(head, enabled=True):
    """Use FP16 only for edge hidden layers; retain FP32 parameters and outputs."""
    if enabled:
        if not getattr(head, "fully_connected_hessian", False):
            raise ValueError("FP16 edge kernels require complete HIP graphs")
        if any(p.dtype != torch.float32 for p in head.parameters()):
            raise ValueError("FP16 edge kernels require FP32 parameters")
        enable_block_so2(head)
        enable_fused_radial(head)
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    backend = MixedBackend(True) if enabled else FusedRotationBackend()
    for block in [block.edge_wise for block in head.blocks] + [head.edge_readout]:
        for module in block.modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.LayerNorm, SO2_m_Conv)):
                if not hasattr(module, "_hip_original_edge_forward"):
                    module._hip_original_edge_forward = module.forward
                if not enabled:
                    module.forward = module._hip_original_edge_forward
                elif isinstance(module, torch.nn.Linear):
                    module.forward = MethodType(linear_forward, module)
                elif isinstance(module, torch.nn.LayerNorm):
                    module.forward = MethodType(norm_forward, module)
                else:
                    module.forward = MethodType(so2_forward, module)
        block.so2_conv_1.edge_mixed_precision = enabled
        block.so2_conv_2.edge_mixed_precision = enabled
    for module in head.modules():
        if hasattr(module, "backend"):
            if enabled:
                if not hasattr(module, "_hip_original_edge_backend"):
                    module._hip_original_edge_backend = module.backend
                module.backend = backend
            elif hasattr(module, "_hip_original_edge_backend"):
                module.backend = module._hip_original_edge_backend
    head.edge_mixed_precision = enabled
