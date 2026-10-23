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

from fairchem.core.frozen_hip.linear import half_mm_float_output, quantize
from fairchem.core.frozen_hip.radial import (
    _KERNEL_CONFIG,
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
        flat = x.reshape(-1, x.shape[-1])
        if out32:
            # Learned message amplitudes can exceed the FP16 exponent range.
            # Power-of-two scales retain tensor-core products without clipping.
            xh, sx = quantize(flat)
            wh, sw = quantize(w)
            ctx.save_for_backward(xh, wh, sx, sw)
        else:
            xh, wh = flat.half(), w.half()
            ctx.save_for_backward(xh, wh)
        ctx.shape = x.shape
        ctx.dtype = x.dtype
        ctx.bias = b is not None
        ctx.out32 = out32
        if out32:
            out = half_mm_float_output(xh, wh.t()) * (sx * sw)
            if b is not None:
                out = out + b
        else:
            # Keep the bias inside the vendor GEMM epilogue.
            out = torch.nn.functional.linear(xh, wh, None if b is None else b.half())
        return out.reshape(*x.shape[:-1], w.shape[0])

    @staticmethod
    def backward(ctx, g):
        saved = ctx.saved_tensors
        x, w = saved[:2]
        g = g.reshape(-1, w.shape[0])
        if ctx.out32:
            sx, sw = saved[2:]
            gh, sg = quantize(g)
            dx = (
                (half_mm_float_output(gh, w) * (sg * sw)).reshape(ctx.shape)
                if ctx.needs_input_grad[0]
                else None
            )
            dw = (
                half_mm_float_output(gh.t(), x) * (sg * sx)
                if ctx.needs_input_grad[1]
                else None
            )
            db = g.float().sum(0) if ctx.bias and ctx.needs_input_grad[2] else None
            return dx, dw, db, None
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
    return half_linear(
        x, self.weight, self.bias, out32=getattr(self, "hip_fp32_output", False)
    )


def so2_forward(self, x):
    real, imag = self.fc.weight.split(self.out_channels_half, 0)
    weight = torch.cat((torch.cat((real, -imag), 1), torch.cat((imag, real), 1)), 0)
    out = half_linear(
        x.flatten(1), weight, out32=getattr(self, "hip_fp32_output", False)
    )
    return out.reshape(x.shape[0], 2, -1, self.m_output_channels).unbind(1)


@triton_op("fairchem::hip_edge_mixed_radial", mutates_args={})
def mixed_radial(
    x: torch.Tensor, w: torch.Tensor, edges: torch.Tensor, r: torch.Tensor
) -> torch.Tensor:
    e, c = edges.shape[1], x.shape[-1]
    out = torch.empty((e, 9, 2 * c), device=x.device, dtype=torch.float16)
    tile = _KERNEL_CONFIG["channel_tile"]
    wrap_triton(_forward)[(min(max(e, 1), 8192), triton.cdiv(2 * c, tile))](
        x, w, edges, r, out, e, c, tile, 8192, num_warps=min(4, tile // 32)
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
    tuned: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    e, c = edges.shape[1], x.shape[-1]
    dr = torch.empty_like(r)
    segmented = tuned and e >= 32768 and c >= 128
    dx = torch.zeros_like(x) if need_x and not segmented else x.new_empty(0)
    tile = _KERNEL_CONFIG["channel_tile"]
    warps = min(4, tile // 32)
    grid = (min(max(e, 1), 8192), triton.cdiv(2 * c, tile))
    wrap_triton(_backward_radial)[grid](
        x, w, edges, g, dr, e, c, tile, 8192, num_warps=warps
    )
    if need_x:
        if segmented:
            from fairchem.core.frozen_hip.radial_segmented import segmented_radial_dx

            bc, be = (256, 4) if c >= 256 else (128, 8)
            dx = segmented_radial_dx(w, edges, r, g, x.shape[0], bc, be)
        else:
            wrap_triton(_backward_node)[grid](
                w, edges, r, g, dx, e, c, tile, 8192, num_warps=warps
            )
    return dx, dr


class MixedRadial(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, edges, w, r, tuned):
        ctx.save_for_backward(x, edges, w, r)
        ctx.tuned = tuned
        return mixed_radial(x, w, edges, r)

    @staticmethod
    def backward(ctx, g):
        x, edges, w, r = ctx.saved_tensors
        dx, dr = mixed_radial_grad(
            x, w, edges, r, g.contiguous(), ctx.needs_input_grad[0], ctx.tuned
        )
        return dx if ctx.needs_input_grad[0] else None, None, None, dr, None


def radial_convolution(conv, x, edges, w, edge_features):
    r = conv.rad_func(edge_features)
    messages = MixedRadial.apply(
        x, edges, w, r, getattr(conv, "hip_tuned_edge_kernels", False)
    )
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
    from fairchem.core.frozen_hip.skinny import skinny_linear, skinny_project

    use_skinny = (
        getattr(conv, "hip_skinny_readout", False) and messages.shape[0] >= 16384
    )
    combined = use_skinny and getattr(conv, "hip_combined_skinny_readout", False)

    def project(x, weight, bias=None):
        if use_skinny:
            return skinny_linear(x, weight, bias)
        return torch.nn.functional.linear(x.float(), weight, bias)

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
    weights = [weight]
    out = [] if combined else [project(parts[0].flatten(1), weight, bias).unsqueeze(-1)]
    for m, block in enumerate(conv.so2_m_conv, 1):
        count = degrees - m
        original = block.fc.weight.reshape(2, count, channels, -1)
        effective = torch.einsum("lc,slci->sli", p[m:], original)
        real, imag = effective.unbind(0)
        weight = torch.cat((torch.cat((real, -imag), 1), torch.cat((imag, real), 1)), 0)
        weights.append(weight)
        if not combined:
            value = project(parts[m].flatten(1), weight).reshape(e, 2, count, 1)
            out.extend(value.unbind(1))
    if combined:
        return skinny_project(messages, weights, bias)
    return torch.cat(out, 1)


def enable_skinny_readout(head, enabled=True, *, combined=True):
    """Avoid wide FP32 casts in the final projection of large FP16 edge batches."""
    if enabled and not getattr(head, "edge_mixed_precision", False):
        raise ValueError("Skinny readout requires FP16 edge precision")
    head.edge_readout.so2_conv_2.hip_skinny_readout = enabled
    head.edge_readout.so2_conv_2.hip_combined_skinny_readout = combined


@triton_op("fairchem::hip_edge_mixed_scatter", mutates_args={})
def mixed_scatter(
    x: torch.Tensor,
    w: torch.Tensor,
    edges: torch.Tensor,
    nodes: int,
    offset: int,
    segmented: bool,
    tuned: bool = False,
) -> torch.Tensor:
    e, c = edges.shape[1], x.shape[-1]
    if segmented:
        ptr = torch.empty(nodes + 1, device=x.device, dtype=torch.int32)
        wrap_triton(_bounds)[(triton.cdiv(nodes + 1, 32),)](
            edges, ptr, e, nodes, offset, 32, num_warps=1
        )
        out = torch.empty((nodes, 9, c), device=x.device, dtype=torch.float32)
        tile = (
            min(triton.next_power_of_2(c), 256)
            if tuned and e >= 16384 and c >= 128
            else 32
        )
        edge_tile = 4 if tile >= 128 else 32
        wrap_triton(_segment_rotate)[(nodes * 9, triton.cdiv(c, tile))](
            x, w, ptr, out, c, tile, edge_tile, num_warps=4
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
    g: torch.Tensor,
    w: torch.Tensor,
    edges: torch.Tensor,
    offset: int,
    tuned: bool = False,
    output_fp32: bool = False,
) -> torch.Tensor:
    e, c = edges.shape[1], g.shape[-1]
    out = torch.empty(
        (e, 9, c),
        device=g.device,
        dtype=torch.float32 if output_fp32 else torch.float16,
    )
    tile = (
        min(triton.next_power_of_2(c), 256)
        if tuned and e >= 16384 and c >= 128
        else BLOCK_C
    )
    wrap_triton(permute_wigner_inv_edge_to_node_bwd_dx_kernel)[
        (min(max(e, 1), GRID_E_STRIDE), triton.cdiv(c, tile))
    ](
        g,
        w,
        out,
        e,
        c,
        tile,
        GRID_E_STRIDE,
        edges,
        True,
        offset,
        num_warps=4 if tile >= 128 else 1,
    )
    return out


class MixedScatter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, edges, nodes, offset, segmented, tuned):
        ctx.save_for_backward(w, edges)
        ctx.offset = offset
        ctx.tuned = tuned
        ctx.output_fp32 = x.dtype == torch.float32
        return mixed_scatter(x.contiguous(), w, edges, nodes, offset, segmented, tuned)

    @staticmethod
    def backward(ctx, g):
        w, edges = ctx.saved_tensors
        return (
            mixed_gather(
                g.contiguous(), w, edges, ctx.offset, ctx.tuned, ctx.output_fp32
            ),
            None,
            None,
            None,
            None,
            None,
            None,
        )


class MixedBackend(FusedRotationBackend):
    def __init__(self, segmented, tuned=False):
        self.segmented = segmented
        self.tuned = tuned

    def permute_wigner_inv_edge_to_node(self, x, w, edges, nodes, node_offset=0):
        return MixedScatter.apply(
            x, w, edges, nodes, node_offset, self.segmented, self.tuned
        )


def enable_tuned_edge_kernels(head, enabled=True):
    """Tune wide complete-graph reductions while retaining FP32 accumulation."""
    if enabled and not getattr(head, "edge_mixed_precision", False):
        raise ValueError("Tuned edge kernels require FP16 edge precision")
    for block in [block.edge_wise for block in head.blocks] + [head.edge_readout]:
        block.so2_conv_1.hip_tuned_edge_kernels = enabled
    backend = MixedBackend(True, tuned=enabled)
    for module in head.modules():
        if isinstance(getattr(module, "backend", None), MixedBackend):
            module.backend = backend
    head.tuned_edge_kernels = enabled


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
        # Both convolutions can exceed the FP16 range as the head learns.
        # Keep their outputs in FP32 and rescale GEMM operands before casting.
        # The radial MLP retains its original selective FP16 dispatch.
        for convolution in (block.so2_conv_1, block.so2_conv_2):
            for module in (convolution.fc_m0, *convolution.so2_m_conv):
                module.hip_fp32_output = enabled
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
    head.edge_range_scaling = enabled
