"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Fuse ragged molecular graph construction and symmetric Hessian loss on CUDA.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _owner(index, pointers, B, LEVELS: tl.constexpr):
    lo = tl.full(index.shape, 0, tl.int32)
    hi = tl.full(index.shape, B, tl.int32)
    for _ in range(LEVELS):
        mid = (lo + hi) // 2
        end = tl.load(pointers + tl.minimum(mid + 1, B))
        right = index >= end
        lo = tl.where(right, mid + 1, lo)
        hi = tl.where(right, hi, mid)
    return tl.minimum(lo, B - 1)


@triton.jit
def _graph(P, AP, EP, EI, V, D, E, B, BLOCK: tl.constexpr, LEVELS: tl.constexpr):
    e = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = e < E
    sample = _owner(e, EP, B, LEVELS)
    start = tl.load(AP + sample)
    n = tl.load(AP + sample + 1) - start
    local = e - tl.load(EP + sample)
    width = tl.maximum(n - 1, 1)
    j = local // width
    i = local % width
    i += i >= j
    source, target = start + i, start + j
    tl.store(EI + e, source, mask)
    tl.store(EI + E + e, target, mask)
    distance = tl.full((BLOCK,), 0, tl.float32)
    for k in range(3):
        delta = tl.load(P + source * 3 + k, mask, 0) - tl.load(
            P + target * 3 + k, mask, 0
        )
        tl.store(V + e * 3 + k, delta, mask)
        distance += delta * delta
    tl.store(D + e, tl.sqrt(distance), mask)


def graph(data):
    """
    Build only within-molecule directed edges in a single GPU launch.
    """
    sizes = data.hip_sizes
    edges = sum(n * (n - 1) for n in sizes)
    edge_index = torch.empty((2, edges), dtype=torch.int64, device=data.pos.device)
    vectors = data.pos.new_empty((edges, 3))
    distances = data.pos.new_empty(edges)
    if edges:
        _graph[(triton.cdiv(edges, 256),)](
            data.pos,
            data.hip_atom_ptr,
            data.hip_edge_ptr,
            edge_index,
            vectors,
            distances,
            edges,
            len(sizes),
            256,
            (len(sizes) + 1).bit_length(),
        )
    offsets = data.pos.new_zeros((edges, 3))
    neighbors = data.natoms * (data.natoms - 1)
    return edge_index, distances, vectors, offsets, offsets, neighbors


@triton.jit
def _pair_loss(
    EDGE,
    TARGET,
    AP,
    EP,
    PP,
    GRAD,
    PARTIAL,
    E,
    B,
    BLOCK: tl.constexpr,
    LEVELS: tl.constexpr,
):
    entry = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = entry < E * 9
    edge, k = entry // 9, entry % 9
    sample = _owner(edge, EP, B, LEVELS)
    n = tl.load(AP + sample + 1) - tl.load(AP + sample)
    base = tl.load(EP + sample)
    local = edge - base
    width = tl.maximum(n - 1, 1)
    j, i = local // width, local % width
    i += i >= j
    reverse = base + i * width + j - (j > i)
    kt = (k % 3) * 3 + k // 3
    predicted = tl.load(EDGE + entry, mask, 0).to(tl.float32)
    predicted += tl.load(EDGE + reverse * 9 + kt, mask, 0).to(tl.float32)
    low, high = tl.minimum(i, j), tl.maximum(i, j)
    pair = high * (high - 1) // 2 + low
    canonical_k = tl.where(i < j, k, kt)
    target = tl.load(TARGET + tl.load(PP + sample) + pair * 9 + canonical_k, mask, 0)
    error = predicted - target
    norm = 1.0 / (B * 9.0 * n * n)
    tl.store(GRAD + entry, 4.0 * error * norm, mask)
    tl.store(
        PARTIAL + tl.program_id(0), tl.sum(tl.where(mask, error * error * norm, 0))
    )


@triton.jit
def _node_loss(
    NODE,
    TARGET,
    AP,
    GRAD,
    PARTIAL,
    N,
    B,
    BLOCK: tl.constexpr,
    LEVELS: tl.constexpr,
):
    entry = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = entry < N * 9
    atom, k = entry // 9, entry % 9
    sample = _owner(atom, AP, B, LEVELS)
    n = tl.load(AP + sample + 1) - tl.load(AP + sample)
    row, col = k // 3, k % 3
    low, high = tl.minimum(row, col), tl.maximum(row, col)
    upper_index = low * 3 - low * (low - 1) // 2 + high - low
    predicted = tl.load(NODE + entry, mask, 0).to(tl.float32)
    predicted += tl.load(NODE + atom * 9 + col * 3 + row, mask, 0).to(tl.float32)
    error = predicted - tl.load(TARGET + atom * 6 + upper_index, mask, 0)
    norm = 1.0 / (B * 9.0 * n * n)
    tl.store(GRAD + entry, 4.0 * error * norm, mask)
    tl.store(
        PARTIAL + tl.program_id(0), tl.sum(tl.where(mask, error * error * norm, 0))
    )


class _BlockMSE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, edge, node, targets, diagonal, asymmetry, ap, ep, pp):
        edge, node = edge.contiguous(), node.contiguous()
        e, n, b = edge.shape[0], node.shape[0], ap.numel() - 1
        ge = torch.empty_like(edge, dtype=torch.float32)
        gn = torch.empty_like(node, dtype=torch.float32)
        pe = edge.new_empty(triton.cdiv(e * 9, 256), dtype=torch.float32)
        pn = node.new_empty(triton.cdiv(n * 9, 256), dtype=torch.float32)
        if e:
            _pair_loss[(pe.numel(),)](
                edge, targets, ap, ep, pp, ge, pe, e, b, 256, (b + 1).bit_length()
            )
        _node_loss[(pn.numel(),)](
            node, diagonal, ap, gn, pn, n, b, 256, (b + 1).bit_length()
        )
        ctx.save_for_backward(ge, gn)
        ctx.edge_dtype, ctx.node_dtype = edge.dtype, node.dtype
        return pe.sum() + pn.sum() + asymmetry.float().mean()

    @staticmethod
    def backward(ctx, gradient):
        ge, gn = ctx.saved_tensors
        return (
            ge.mul(gradient).to(ctx.edge_dtype),
            gn.mul(gradient).to(ctx.node_dtype),
            None,
            None,
            None,
            None,
            None,
            None,
        )


def fused_block_mse(edge, node, data):
    """
    Compute dense-equivalent MSE and gradients without dense assembly.
    """
    return _BlockMSE.apply(
        edge,
        node,
        data.hip_pair_targets,
        data.hip_diagonal_targets,
        data.hip_asymmetry_mse,
        data.hip_atom_ptr,
        data.hip_edge_ptr,
        data.hip_pair_ptr,
    )


@triton.jit
def _block_norm_loss(
    PRED,
    TARGET,
    AP,
    EP,
    PP,
    GRAD,
    PARTIAL,
    COUNT,
    B,
    NODE: tl.constexpr,
    BLOCK: tl.constexpr,
    LEVELS: tl.constexpr,
    ABSOLUTE: tl.constexpr = False,
):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    k = tl.arange(0, 16)
    mask = (index[:, None] < COUNT) & (k[None, :] < 9)
    if NODE:
        sample = _owner(index, AP, B, LEVELS)
        row, col = k // 3, k % 3
        low, high = tl.minimum(row, col), tl.maximum(row, col)
        upper = low * 3 - low * (low - 1) // 2 + high - low
        reverse_index = index[:, None] * 9 + (col * 3 + row)[None, :]
        target_index = index[:, None] * 6 + upper[None, :]
    else:
        sample = _owner(index, EP, B, LEVELS)
        n = tl.load(AP + sample + 1) - tl.load(AP + sample)
        base = tl.load(EP + sample)
        local = index - base
        width = tl.maximum(n - 1, 1)
        j, i = local // width, local % width
        i += i >= j
        reverse = base + i * width + j - (j > i)
        kt = (k % 3) * 3 + k // 3
        reverse_index = reverse[:, None] * 9 + kt[None, :]
        low, high = tl.minimum(i, j), tl.maximum(i, j)
        pair = high * (high - 1) // 2 + low
        canonical_k = tl.where(i[:, None] < j[:, None], k[None, :], kt[None, :])
        target_index = tl.load(PP + sample)[:, None] + pair[:, None] * 9 + canonical_k
    value = tl.load(PRED + index[:, None] * 9 + k[None, :], mask, 0).to(tl.float32)
    value += tl.load(PRED + reverse_index, mask, 0).to(tl.float32)
    error = value - tl.load(TARGET + target_index, mask, 0).to(tl.float32)
    error = tl.where(mask, error, 0)
    n = tl.load(AP + sample + 1) - tl.load(AP + sample)
    weight = 1.0 / (B * n * n)
    if ABSOLUTE:
        sign = tl.where(error > 0, 1.0, tl.where(error < 0, -1.0, 0.0))
        grad = sign * (2.0 * weight / 9.0)[:, None]
        loss = tl.sum(tl.abs(error), 1) * weight / 9.0
    else:
        scale = tl.max(tl.abs(error), 1)
        scaled = error / tl.where(scale > 0, scale, 1)[:, None]
        norm = tl.sqrt(tl.sum(scaled * scaled, 1))
        grad = scaled * (2.0 * weight / (3.0 * tl.where(norm > 0, norm, 1)))[:, None]
        loss = (scale / 3.0) * norm * weight
    tl.store(GRAD + index[:, None] * 9 + k[None, :], grad, mask)
    tl.store(PARTIAL + tl.program_id(0), tl.sum(tl.where(index < COUNT, loss, 0)))


class _BlockRMSE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, edge, node, targets, diagonal, ap, ep, pp, absolute):
        edge, node = edge.contiguous(), node.contiguous()
        ge = torch.empty_like(edge, dtype=torch.float32)
        gn = torch.empty_like(node, dtype=torch.float32)
        partials = []
        b = ap.numel() - 1
        for values, labels, gradient, is_node in (
            (edge, targets, ge, False),
            (node, diagonal, gn, True),
        ):
            count = values.shape[0]
            partial = values.new_empty(triton.cdiv(count, 64), dtype=torch.float32)
            if count:
                _block_norm_loss[(partial.numel(),)](
                    values,
                    labels,
                    ap,
                    ep,
                    pp,
                    gradient,
                    partial,
                    count,
                    b,
                    is_node,
                    64,
                    (b + 1).bit_length(),
                    absolute,
                    num_warps=4,
                )
            partials.append(partial.sum())
        ctx.save_for_backward(ge, gn)
        ctx.edge_dtype, ctx.node_dtype = edge.dtype, node.dtype
        return partials[0] + partials[1]

    @staticmethod
    def backward(ctx, gradient):
        ge, gn = ctx.saved_tensors
        return (
            ge.mul(gradient).to(ctx.edge_dtype),
            gn.mul(gradient).to(ctx.node_dtype),
            None,
            None,
            None,
            None,
            None,
            None,
        )


def fused_block_rmse(edge, node, data):
    """
    Compute block norms and first derivatives without dense Hessian assembly.
    """
    return _BlockRMSE.apply(
        edge,
        node,
        data.hip_pair_targets,
        data.hip_diagonal_targets,
        data.hip_atom_ptr,
        data.hip_edge_ptr,
        data.hip_pair_ptr,
        False,
    )


def fused_block_mae(edge, node, data):
    """Compute geometry-averaged MAE without a dense Hessian intermediate."""
    return _BlockRMSE.apply(
        edge,
        node,
        data.hip_pair_targets,
        data.hip_diagonal_targets,
        data.hip_atom_ptr,
        data.hip_edge_ptr,
        data.hip_pair_ptr,
        True,
    )
