"""Fuse the translation projection and its self-adjoint backward on CUDA."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _rows(E, N, AP, EP, R, WIDTH: tl.constexpr):
    i, b = tl.program_id(0), tl.program_id(1)
    a = tl.load(AP + b)
    n = tl.load(AP + b + 1) - a
    if i < n:
        start = tl.load(EP + b)
        j = tl.arange(0, WIDTH)
        c = tl.arange(0, 16)
        ct = (c % 3) * 3 + c // 3
        mask = (j[:, None] < n) & (j[:, None] != i) & (c[None, :] < 9)
        outward = start + j * (n - 1) + i - (i > j)
        inward = start + i * (n - 1) + j - (j > i)
        v = tl.load(E + outward[:, None] * 9 + c[None, :], mask, 0).to(tl.float32)
        v += tl.load(E + inward[:, None] * 9 + ct[None, :], mask, 0).to(tl.float32)
        diagonal = tl.load(N + (a + i) * 9 + c, c < 9, 0).to(tl.float32)
        diagonal += tl.load(N + (a + i) * 9 + ct, c < 9, 0).to(tl.float32)
        tl.store(R + (a + i) * 9 + c, (tl.sum(v, 0) + diagonal) / n, c < 9)


@triton.jit
def _grand(R, AP, G, WIDTH: tl.constexpr):
    b = tl.program_id(0)
    a = tl.load(AP + b)
    n = tl.load(AP + b + 1) - a
    i = tl.arange(0, WIDTH)
    c = tl.arange(0, 16)
    v = tl.load(
        R + (a + i[:, None]) * 9 + c[None, :], (i[:, None] < n) & (c[None, :] < 9), 0
    )
    tl.store(G + b * 9 + c, tl.sum(v, 0) / n, c < 9)


@triton.jit
def _correct(E, N, EI, OWNER, R, G, OE, ON, NE, NA, BLOCK: tl.constexpr):
    x = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = x < (NE + NA) * 9
    block, c = x // 9, x % 9
    ct = (c % 3) * 3 + c // 3
    is_edge = block < NE
    s = tl.load(EI + block, valid & is_edge, 0)
    t = tl.load(EI + NE + block, valid & is_edge, 0)
    s = tl.where(is_edge, s, block - NE)
    t = tl.where(is_edge, t, block - NE)
    b = tl.load(OWNER + s, valid, 0)
    r = tl.load(R + s * 9 + c, valid, 0)
    col = tl.load(R + t * 9 + ct, valid, 0)
    grand = tl.load(G + b * 9 + c, valid, 0)
    old = tl.load(E + x, valid & is_edge, 0).to(tl.float32)
    old += tl.load(N + (block - NE) * 9 + c, valid & ~is_edge, 0).to(tl.float32)
    value = old + 0.5 * (grand - r - col)
    tl.store(OE + x, value, valid & is_edge)
    tl.store(ON + (block - NE) * 9 + c, value, valid & ~is_edge)


def _apply(edge, node, ap, ep, ei, owner, sizes):
    edge, node = edge.contiguous(), node.contiguous()
    row = torch.empty_like(node, dtype=torch.float32)
    grand = torch.empty((len(sizes), 3, 3), device=node.device, dtype=torch.float32)
    oe, on = torch.empty_like(edge), torch.empty_like(node)
    width = triton.next_power_of_2(max(sizes))
    _rows[(max(sizes), len(sizes))](edge, node, ap, ep, row, width)
    _grand[(len(sizes),)](row, ap, grand, width)
    _correct[(triton.cdiv((edge.shape[0] + node.shape[0]) * 9, 256),)](
        edge, node, ei, owner, row, grand, oe, on, edge.shape[0], node.shape[0], 256
    )
    return oe, on


class _Projection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, edge, node, ap, ep, ei, owner, sizes):
        ctx.save_for_backward(ap, ep, ei, owner)
        ctx.sizes = sizes
        return _apply(edge, node, ap, ep, ei, owner, sizes)

    @staticmethod
    def backward(ctx, ge, gn):
        ap, ep, ei, owner = ctx.saved_tensors
        if ge is None:
            ge = gn.new_zeros(sum(n * (n - 1) for n in ctx.sizes), 3, 3)
        if gn is None:
            gn = ge.new_zeros(sum(ctx.sizes), 3, 3)
        ge, gn = _apply(ge, gn, ap, ep, ei, owner, ctx.sizes)
        return ge, gn, None, None, None, None, None


def project(edge, node, data):
    if edge.dtype != torch.float32 or node.dtype != torch.float32:
        raise ValueError("Translation projection requires FP32 Cartesian blocks")
    return _Projection.apply(
        edge,
        node,
        data.hip_atom_ptr,
        data.hip_edge_ptr,
        data.edge_index_hessian,
        data.batch,
        tuple(data.hip_sizes),
    )
