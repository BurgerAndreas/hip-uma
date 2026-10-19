"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Fuse frozen Wigner gathering with trainable radial message weights.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton

from fairchem.core.frozen_hip.scatter import _m_order

_KERNEL_CONFIG = {"channel_tile": 64}


def configure_radial_tile(tile):
    """Select an FP32 channel tile without changing tensor layouts or weights."""
    if tile not in (64, 128, 256):
        raise ValueError("Radial channel tile must be 64, 128, or 256")
    _KERNEL_CONFIG["channel_tile"] = tile


@triton.jit
def _radial_row(m):
    return tl.where(m < 5, m, tl.where(m < 7, m - 2, 5))


@triton.jit
def _dot(X, W, atom, edge, channel, mask, C: tl.constexpr, ROW):
    start = tl.where(ROW == 0, 0, tl.where(ROW < 4, 1, 4))
    width = tl.where(ROW == 0, 1, tl.where(ROW < 4, 3, 5))
    value = tl.full(channel.shape, 0, tl.float32)
    for j in tl.static_range(5):
        if j < width:
            w = tl.load(W + edge * 81 + ROW * 9 + start + j)
            x = tl.load(X + (atom * 9 + start + j) * C + channel, mask, 0)
            value += w * x
    return value


@triton.jit
def _forward(
    X, W, EDGES, R, OUT, E, C: tl.constexpr, BC: tl.constexpr, GRID: tl.constexpr
):
    e = tl.program_id(0).to(tl.int64)
    c = tl.program_id(1) * BC + tl.arange(0, BC)
    mask = c < 2 * C
    while e < E:
        source = tl.load(EDGES + e)
        target = tl.load(EDGES + E + e)
        atom = tl.where(c < C, source, target)
        channel = c % C
        for l in tl.static_range(9):
            m = _m_order(l)
            value = _dot(X, W, atom, e, channel, mask, C, l)
            radial = tl.load(R + (e * 6 + _radial_row(m)) * (2 * C) + c, mask, 0)
            tl.store(OUT + (e * 9 + m) * (2 * C) + c, value * radial, mask)
        e += GRID


@triton.jit
def _backward_radial(
    X, W, EDGES, G, DR, E, C: tl.constexpr, BC: tl.constexpr, GRID: tl.constexpr
):
    e = tl.program_id(0).to(tl.int64)
    c = tl.program_id(1) * BC + tl.arange(0, BC)
    mask = c < 2 * C
    while e < E:
        source = tl.load(EDGES + e)
        target = tl.load(EDGES + E + e)
        atom = tl.where(c < C, source, target)
        channel = c % C
        for r in tl.static_range(6):
            # Two Cartesian components share a radial weight for nonzero m.
            l = (0, 2, 6, 3, 7, 8)[r]
            m = _m_order(l)
            value = _dot(X, W, atom, e, channel, mask, C, l)
            grad = tl.load(G + (e * 9 + m) * (2 * C) + c, mask, 0)
            result = value * grad
            if r >= 3:
                other = (1, 5, 4)[r - 3]
                value = _dot(X, W, atom, e, channel, mask, C, other)
                grad = tl.load(G + (e * 9 + _m_order(other)) * (2 * C) + c, mask, 0)
                result += value * grad
            tl.store(DR + (e * 6 + r) * (2 * C) + c, result, mask)
        e += GRID


@triton.jit
def _backward_node(
    W, EDGES, R, G, DX, E, C: tl.constexpr, BC: tl.constexpr, GRID: tl.constexpr
):
    e = tl.program_id(0).to(tl.int64)
    c = tl.program_id(1) * BC + tl.arange(0, BC)
    mask = c < 2 * C
    while e < E:
        source = tl.load(EDGES + e)
        target = tl.load(EDGES + E + e)
        atom = tl.where(c < C, source, target)
        channel = c % C
        for l in tl.static_range(9):
            start = tl.where(l == 0, 0, tl.where(l < 4, 1, 4))
            width = tl.where(l == 0, 1, tl.where(l < 4, 3, 5))
            result = tl.full((BC,), 0, tl.float32)
            for j in tl.static_range(5):
                if j < width:
                    row = start + j
                    m = _m_order(row)
                    w = tl.load(W + e * 81 + row * 9 + l)
                    radial = tl.load(
                        R + (e * 6 + _radial_row(m)) * (2 * C) + c, mask, 0
                    ).to(tl.float32)
                    grad = tl.load(G + (e * 9 + m) * (2 * C) + c, mask, 0).to(
                        tl.float32
                    )
                    result += w * (radial * grad)
            tl.atomic_add(
                DX + (atom * 9 + l) * C + channel, result, mask, sem="relaxed"
            )
        e += GRID


@triton_op("fairchem::hip_radial_gather", mutates_args={})
def radial_gather(
    x: torch.Tensor, w: torch.Tensor, edges: torch.Tensor, radial: torch.Tensor
) -> torch.Tensor:
    e, c = edges.shape[1], x.shape[-1]
    out = x.new_empty((e, 9, 2 * c))
    tile = _KERNEL_CONFIG["channel_tile"]
    wrap_triton(_forward)[(min(max(e, 1), 8192), triton.cdiv(2 * c, tile))](
        x, w, edges, radial, out, e, c, tile, 8192, num_warps=tile // 32
    )
    return out


@triton_op("fairchem::hip_radial_gather_grad", mutates_args={})
def radial_gather_grad(
    x: torch.Tensor,
    w: torch.Tensor,
    edges: torch.Tensor,
    radial: torch.Tensor,
    gradient: torch.Tensor,
    need_x: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    e, c = edges.shape[1], x.shape[-1]
    dr = torch.empty_like(radial)
    dx = torch.zeros_like(x) if need_x else x.new_empty(0)
    tile = _KERNEL_CONFIG["channel_tile"]
    grid = (min(max(e, 1), 8192), triton.cdiv(2 * c, tile))
    wrap_triton(_backward_radial)[grid](
        x, w, edges, gradient, dr, e, c, tile, 8192, num_warps=tile // 32
    )
    if need_x:
        wrap_triton(_backward_node)[grid](
            w, edges, radial, gradient, dx, e, c, tile, 8192, num_warps=tile // 32
        )
    return dx, dr


class _RadialGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, edges, w, radial):
        ctx.save_for_backward(x, edges, w, radial)
        return radial_gather(x, w, edges, radial)

    @staticmethod
    def backward(ctx, gradient):
        x, edges, w, radial = ctx.saved_tensors
        dx, dr = radial_gather_grad(
            x, w, edges, radial, gradient.contiguous(), ctx.needs_input_grad[0]
        )
        return dx if ctx.needs_input_grad[0] else None, None, None, dr


def fused_radial_gather(x, edges, w, radial):
    """Recompute rotations in backward instead of saving a wide edge tensor."""
    if w.requires_grad or any(value.dtype != torch.float32 for value in (x, w, radial)):
        raise ValueError(
            "Fused radial gathering requires FP32 messages and frozen geometry"
        )
    if (
        x.ndim != 3
        or x.shape[1] != 9
        or edges.ndim != 2
        or edges.shape[0] != 2
        or w.shape != (edges.shape[1], 9, 9)
        or radial.shape != (edges.shape[1], 12 * x.shape[-1])
    ):
        raise ValueError("Fused radial gathering received incompatible tensor shapes")
    return _RadialGather.apply(
        x.contiguous(), edges.contiguous(), w.contiguous(), radial.contiguous()
    )


def radial_convolution(convolution, x, edges, w, edge_features):
    """Keep the original trainable convolution after fused radial gathering."""
    radial = convolution.rad_func(edge_features)
    messages = fused_radial_gather(x, edges, w, radial)
    parts = messages.split(convolution.m_split_sizes, dim=1)
    e, channels = messages.shape[0], convolution.m_output_channels
    scalar = convolution.fc_m0(parts[0].flatten(1))
    extra = convolution.extra_m0_output_channels
    gates, scalar = scalar.split((extra, scalar.shape[-1] - extra), dim=-1)
    outputs = [scalar.reshape(e, convolution.lmax + 1, channels)]
    for m, block in enumerate(convolution.so2_m_conv, 1):
        inputs = parts[m].reshape(
            e, 2, (convolution.lmax + 1 - m) * convolution.sphere_channels
        )
        outputs.extend(block(inputs))
    return torch.cat(outputs, dim=1), gates


def enable_fused_radial(head, enabled=True):
    """Select dense FP32 HIP convolutions without changing their parameters."""
    from fairchem.core.models.uma.nn.so2_layers import SO2_Convolution

    blocks = [block.edge_wise for block in head.blocks] + [head.edge_readout]
    if enabled:
        for block in blocks:
            convolution = block.so2_conv_1
            if (
                not isinstance(convolution, SO2_Convolution)
                or convolution.lmax != 2
                or convolution.mmax != 2
                or convolution.rad_func is None
                or convolution.extra_m0_output_channels is None
            ):
                raise ValueError(
                    "Fused radial gathering requires dense gated lmax=mmax=2 HIP convolutions"
                )
    for block in blocks:
        block.fused_radial = enabled
    head.fused_radial = enabled
