"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Fuse FP32 Wigner rotation with accumulation and its gradient gather.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton

from fairchem.core.models.uma.triton.constants import BLOCK_C, GRID_E_STRIDE
from fairchem.core.models.uma.triton.kernels import (
    permute_wigner_inv_edge_to_node_bwd_dx_kernel,
    permute_wigner_inv_edge_to_node_kernel,
)


@triton.jit
def _bounds(EDGES, PTR, E, N, OFFSET: tl.constexpr, BLOCK: tl.constexpr):
    atom = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    low = tl.full((BLOCK,), 0, tl.int32)
    high = tl.full((BLOCK,), E, tl.int32)
    for _ in range(32):
        mid = (low + high) // 2
        target = tl.load(EDGES + E + mid, (atom <= N) & (mid < E), 2147483647)
        right = target < atom + OFFSET
        low = tl.where(right, mid + 1, low)
        high = tl.where(right, high, mid)
    tl.store(PTR + atom, low, atom <= N)


@triton.jit
def _m_order(k):
    return tl.where(
        k == 0,
        0,
        tl.where(
            k == 1,
            5,
            tl.where(
                k == 2,
                1,
                tl.where(
                    k == 3,
                    3,
                    tl.where(
                        k == 4,
                        8,
                        tl.where(
                            k == 5, 6, tl.where(k == 6, 2, tl.where(k == 7, 4, 7))
                        ),
                    ),
                ),
            ),
        ),
    )


@triton.jit
def _segment_rotate(
    X, W, PTR, OUT, C: tl.constexpr, BC: tl.constexpr, BE: tl.constexpr
):
    atom = tl.program_id(0) // 9
    coefficient = tl.program_id(0) % 9
    channel = tl.program_id(1) * BC + tl.arange(0, BC)
    edge = tl.load(PTR + atom)
    end = tl.load(PTR + atom + 1)
    row = tl.arange(0, BE)
    low = tl.where(coefficient == 0, 0, tl.where(coefficient < 4, 1, 4))
    width = tl.where(coefficient == 0, 1, tl.where(coefficient < 4, 3, 5))
    total = tl.zeros((BE, BC), tl.float32)
    while edge < end:
        e = (edge + row).to(tl.int64)
        mask = (e[:, None] < end) & (channel[None, :] < C)
        value = tl.zeros((BE, BC), tl.float32)
        for j in tl.static_range(5):
            if j < width:
                k = low + j
                rotation = tl.load(W + e * 81 + coefficient * 9 + k, e < end, 0)
                x = tl.load(
                    X + e[:, None] * (9 * C) + _m_order(k) * C + channel[None, :],
                    mask,
                    0,
                )
                value += rotation[:, None] * x
        total += value
        edge += BE
    result = tl.sum(total, 0)
    tl.store(OUT + (atom * 9 + coefficient) * C + channel, result, channel < C)


@triton_op("fairchem::hip_segment_rotate", mutates_args={})
def segment_rotate(
    x: torch.Tensor, w: torch.Tensor, edges: torch.Tensor, nodes: int, offset: int
) -> torch.Tensor:
    """Use target-sorted edges to avoid atomic accumulation and wide scratch."""
    e, _, c = x.shape
    pointers = torch.empty(nodes + 1, device=x.device, dtype=torch.int32)
    wrap_triton(_bounds)[(triton.cdiv(nodes + 1, 32),)](
        edges, pointers, e, nodes, offset, 32, num_warps=1
    )
    out = x.new_empty((nodes, 9, c))
    wrap_triton(_segment_rotate)[(nodes * 9, triton.cdiv(c, 32))](
        x, w, pointers, out, c, 32, 32, num_warps=4
    )
    return out


@triton_op("fairchem::hip_rotate_scatter", mutates_args={})
def rotate_scatter(
    x: torch.Tensor, w: torch.Tensor, edges: torch.Tensor, nodes: int, offset: int
) -> torch.Tensor:
    e, _, c = x.shape
    out = x.new_zeros((nodes, 9, c))
    wrap_triton(permute_wigner_inv_edge_to_node_kernel)[
        (min(max(e, 1), GRID_E_STRIDE), (c + BLOCK_C - 1) // BLOCK_C)
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


@triton_op("fairchem::hip_rotate_gather", mutates_args={})
def rotate_gather(
    gradient: torch.Tensor, w: torch.Tensor, edges: torch.Tensor, offset: int
) -> torch.Tensor:
    e, c = edges.shape[1], gradient.shape[-1]
    result = gradient.new_empty((e, 9, c))
    wrap_triton(permute_wigner_inv_edge_to_node_bwd_dx_kernel)[
        (min(max(e, 1), GRID_E_STRIDE), (c + BLOCK_C - 1) // BLOCK_C)
    ](
        gradient,
        w,
        result,
        e,
        c,
        BLOCK_C,
        GRID_E_STRIDE,
        edges,
        True,
        offset,
        num_warps=1,
    )
    return result


class _RotateScatter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, edges, nodes, offset, segmented):
        ctx.save_for_backward(w, edges)
        ctx.offset = offset
        operation = segment_rotate if segmented else rotate_scatter
        return operation(x.contiguous(), w, edges.contiguous(), nodes, offset)

    @staticmethod
    def backward(ctx, gradient):
        w, edges = ctx.saved_tensors
        return (
            rotate_gather(gradient.contiguous(), w, edges, ctx.offset),
            None,
            None,
            None,
            None,
            None,
        )


def fused_rotate_scatter(x, w, edges, nodes, offset=0, *, segmented=False):
    """Use fused accumulation only for frozen geometry and FP32 messages."""
    if (
        w.requires_grad
        or x.dtype != torch.float32
        or torch.are_deterministic_algorithms_enabled()
    ):
        from fairchem.core.models.uma.nn.execution_backends import UMASFastGPUBackend

        return UMASFastGPUBackend.permute_wigner_inv_edge_to_node(
            x, w, edges, nodes, offset
        )
    return _RotateScatter.apply(x, w, edges, nodes, offset, segmented)
