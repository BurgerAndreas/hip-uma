"""Ragged grouped GEMMs for HIP experts, including FP32 weight gradients."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _forward(
    X,
    W,
    BIAS,
    PTR,
    Y,
    I: tl.constexpr,
    O: tl.constexpr,
    BIAS_ON: tl.constexpr,
    BM: tl.constexpr = 32,
    BN: tl.constexpr = 64,
    BK: tl.constexpr = 32,
):
    b = tl.program_id(1)
    start = tl.load(PTR + b)
    end = tl.load(PTR + b + 1)
    pm = tl.program_id(0) // tl.cdiv(O, BN)
    pn = tl.program_id(0) % tl.cdiv(O, BN)
    if start + pm * BM < end:
        rows = start + pm * BM + tl.arange(0, BM)
        cols = pn * BN + tl.arange(0, BN)
        k = tl.arange(0, BK)
        acc = tl.zeros((BM, BN), tl.float32)
        for tile in range(tl.cdiv(I, BK)):
            kk = tile * BK + k
            a = tl.load(
                X + rows[:, None] * I + kk[None, :],
                (rows[:, None] < end) & (kk[None, :] < I),
                0,
            )
            w = tl.load(
                W + b * O * I + cols[None, :] * I + kk[:, None],
                (cols[None, :] < O) & (kk[:, None] < I),
                0,
            )
            acc += tl.dot(a, w, input_precision="ieee")
        if BIAS_ON:
            acc += tl.load(BIAS + cols, cols < O, 0)[None, :]
        tl.store(
            Y + rows[:, None] * O + cols[None, :],
            acc,
            (rows[:, None] < end) & (cols[None, :] < O),
        )


@triton.jit
def _dx(
    G,
    W,
    PTR,
    DX,
    I: tl.constexpr,
    O: tl.constexpr,
    BM: tl.constexpr = 32,
    BN: tl.constexpr = 64,
    BK: tl.constexpr = 32,
):
    b = tl.program_id(1)
    start = tl.load(PTR + b)
    end = tl.load(PTR + b + 1)
    pm = tl.program_id(0) // tl.cdiv(I, BN)
    pn = tl.program_id(0) % tl.cdiv(I, BN)
    if start + pm * BM < end:
        rows = start + pm * BM + tl.arange(0, BM)
        cols = pn * BN + tl.arange(0, BN)
        k = tl.arange(0, BK)
        acc = tl.zeros((BM, BN), tl.float32)
        for tile in range(tl.cdiv(O, BK)):
            kk = tile * BK + k
            g = tl.load(
                G + rows[:, None] * O + kk[None, :],
                (rows[:, None] < end) & (kk[None, :] < O),
                0,
            )
            w = tl.load(
                W + b * O * I + kk[:, None] * I + cols[None, :],
                (kk[:, None] < O) & (cols[None, :] < I),
                0,
            )
            acc += tl.dot(g, w, input_precision="ieee")
        tl.store(
            DX + rows[:, None] * I + cols[None, :],
            acc,
            (rows[:, None] < end) & (cols[None, :] < I),
        )


@triton.jit
def _dw(
    X,
    G,
    PTR,
    DW,
    I: tl.constexpr,
    O: tl.constexpr,
    BM: tl.constexpr = 32,
    BN: tl.constexpr = 32,
    BK: tl.constexpr = 32,
    CHUNK: tl.constexpr = 512,
):
    b = tl.program_id(1)
    start = tl.load(PTR + b)
    end = tl.load(PTR + b + 1)
    tile = tl.program_id(0)
    split = tl.program_id(2)
    if start + split * CHUNK < end:
        oo = (tile // tl.cdiv(I, BN)) * BM + tl.arange(0, BM)
        ii = (tile % tl.cdiv(I, BN)) * BN + tl.arange(0, BN)
        k = tl.arange(0, BK)
        acc = tl.zeros((BM, BN), tl.float32)
        for part in range(CHUNK // BK):
            rows = start + split * CHUNK + part * BK + k
            g = tl.load(
                G + rows[None, :] * O + oo[:, None],
                (rows[None, :] < end) & (oo[:, None] < O),
                0,
            )
            x = tl.load(
                X + rows[:, None] * I + ii[None, :],
                (rows[:, None] < end) & (ii[None, :] < I),
                0,
            )
            acc += tl.dot(g, x, input_precision="ieee")
        tl.atomic_add(
            DW + b * O * I + oo[:, None] * I + ii[None, :],
            acc,
            (oo[:, None] < O) & (ii[None, :] < I),
            sem="relaxed",
        )


class _Grouped(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, pointers, sizes, mixed):
        dtype = torch.float16 if mixed else torch.float32
        xx = x.to(dtype).contiguous()
        ww = weight.to(dtype).contiguous()
        out = torch.empty((x.shape[0], weight.shape[1]), device=x.device, dtype=dtype)
        maximum = max(n * (n - 1) for n in sizes)
        i, o = x.shape[1], weight.shape[1]
        if maximum:
            _forward[(triton.cdiv(maximum, 32) * triton.cdiv(o, 64), len(sizes))](
                xx,
                ww,
                bias if bias is not None else xx,
                pointers,
                out,
                i,
                o,
                bias is not None,
            )
        ctx.save_for_backward(xx, ww, pointers)
        ctx.sizes = sizes
        ctx.bias = bias is not None
        ctx.input_dtype = x.dtype
        return out

    @staticmethod
    def backward(ctx, gradient):
        x, w, ptr = ctx.saved_tensors
        g = gradient.to(x.dtype).contiguous()
        maximum = max(n * (n - 1) for n in ctx.sizes)
        i, o = x.shape[1], w.shape[1]
        dx = torch.empty(x.shape, device=x.device, dtype=ctx.input_dtype)
        dw = torch.zeros(w.shape, device=w.device, dtype=torch.float32)
        if maximum:
            _dx[(triton.cdiv(maximum, 32) * triton.cdiv(i, 64), len(ctx.sizes))](
                g, w, ptr, dx, i, o
            )
            _dw[
                (
                    triton.cdiv(o, 32) * triton.cdiv(i, 32),
                    len(ctx.sizes),
                    triton.cdiv(maximum, 512),
                )
            ](x, g, ptr, dw, i, o)
        db = gradient.float().sum(0) if ctx.bias else None
        return dx, dw, db, None, None, None


def grouped(x, weight, bias, pointers, sizes, mixed):
    return _Grouped.apply(x, weight, bias, pointers, tuple(sizes), mixed)
