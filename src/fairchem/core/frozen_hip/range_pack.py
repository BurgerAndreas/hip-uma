"""Coalesced FP16 operand packing with FP32 range and gradient reductions."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton

from fairchem.core.frozen_hip.linear import quantize


@triton.jit
def _read_pair(
    A, B, row, col, M, K1: tl.constexpr, K2: tl.constexpr, SA0, SA1, SB0, SB1
):
    first = tl.load(
        A + row[:, None] * SA0 + col[None, :] * SA1,
        (row[:, None] < M) & (col[None, :] < K1),
        0,
    ).to(tl.float32)
    second = tl.load(
        B + row[:, None] * SB0 + (col[None, :] - K1) * SB1,
        (row[:, None] < M) & (col[None, :] >= K1) & (col[None, :] < K1 + K2),
        0,
    ).to(tl.float32)
    return first + second


@triton.jit
def _partials(
    A,
    B,
    MAX,
    SUM,
    M,
    K1: tl.constexpr,
    K2: tl.constexpr,
    SA0,
    SA1,
    SB0,
    SB1,
    NC: tl.constexpr,
    BIAS: tl.constexpr,
    BR: tl.constexpr,
    BC: tl.constexpr,
):
    row = (tl.program_id(0) * BR + tl.arange(0, BR)).to(tl.int64)
    col = (tl.program_id(1) * BC + tl.arange(0, BC)).to(tl.int64)
    value = _read_pair(A, B, row, col, M, K1, K2, SA0, SA1, SB0, SB1)
    maximum = tl.max(tl.max(tl.abs(value), 0), 0)
    # Explicitly retain NaNs; an invalid update must reach the AMP guard.
    bad = tl.sum(tl.sum((value != value).to(tl.int32), 0), 0) > 0
    maximum = tl.where(bad, float("nan"), maximum)
    tl.store(MAX + tl.program_id(0) * NC + tl.program_id(1), maximum)
    if BIAS:
        tl.store(
            SUM + tl.program_id(0) * (K1 + K2) + col, tl.sum(value, 0), col < K1 + K2
        )


@triton.jit
def _scale(MAX, SCALE, N, BLOCK: tl.constexpr):
    index = tl.arange(0, BLOCK)
    maximum = tl.full((BLOCK,), 0, tl.float32)
    bad = tl.full((BLOCK,), 0, tl.int32)
    start = 0
    while start < N:
        value = tl.load(MAX + start + index, start + index < N, 0)
        bad |= (value != value).to(tl.int32)
        maximum = tl.maximum(maximum, value)
        start += BLOCK
    result = tl.max(maximum, 0)
    result = tl.exp2(tl.floor(tl.log2(tl.maximum(result, 2.0**-100))))
    result = tl.where(tl.sum(bad, 0) > 0, float("nan"), result)
    tl.store(SCALE, result)


@triton.jit
def _pack(
    A,
    B,
    SCALE,
    OUT,
    M,
    K1: tl.constexpr,
    K2: tl.constexpr,
    SA0,
    SA1,
    SB0,
    SB1,
    BR: tl.constexpr,
    BC: tl.constexpr,
):
    row = (tl.program_id(0) * BR + tl.arange(0, BR)).to(tl.int64)
    col = (tl.program_id(1) * BC + tl.arange(0, BC)).to(tl.int64)
    value = _read_pair(A, B, row, col, M, K1, K2, SA0, SA1, SB0, SB1)
    scale = tl.load(SCALE)
    tl.store(
        OUT + row[:, None] * (K1 + K2) + col[None, :],
        value / scale,
        (row[:, None] < M) & (col[None, :] < K1 + K2),
    )


@triton.jit
def _bias(SUM, OUT, R, K: tl.constexpr, BR: tl.constexpr, BC: tl.constexpr):
    col = tl.program_id(0) * BC + tl.arange(0, BC)
    row = tl.arange(0, BR)
    total = tl.zeros((BR, BC), tl.float32)
    start = 0
    while start < R:
        total += tl.load(
            SUM + (start + row[:, None]) * K + col[None, :],
            (start + row[:, None] < R) & (col[None, :] < K),
            0,
        )
        start += BR
    tl.store(OUT + col, tl.sum(total, 0), col < K)


@triton_op("fairchem::hip_range_pack", mutates_args={})
def range_pack_pair(
    first: torch.Tensor, second: torch.Tensor, bias: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack a virtual column concatenation without an edge-sized temporary."""
    rows, k1 = first.shape
    k2 = second.shape[1]
    columns = k1 + k2
    packed = first.new_empty((rows, columns), dtype=torch.float16)
    scale = first.new_empty((), dtype=torch.float32)
    db = first.new_empty((columns if bias else 0,), dtype=torch.float32)
    br, bc = 128, 128
    groups = triton.cdiv(rows, br)
    nc = triton.cdiv(columns, bc)
    maxima = first.new_empty((groups * nc,), dtype=torch.float32)
    partial = first.new_empty((groups, columns if bias else 0), dtype=torch.float32)
    wrap_triton(_partials)[(groups, nc)](
        first,
        second,
        maxima,
        partial,
        rows,
        k1,
        k2,
        first.stride(0),
        first.stride(1),
        second.stride(0),
        second.stride(1),
        nc,
        bias,
        br,
        bc,
        num_warps=8,
    )
    wrap_triton(_scale)[(1,)](maxima, scale, groups * nc, 1024, num_warps=4)
    wrap_triton(_pack)[(groups, nc)](
        first,
        second,
        scale,
        packed,
        rows,
        k1,
        k2,
        first.stride(0),
        first.stride(1),
        second.stride(0),
        second.stride(1),
        br,
        bc,
        num_warps=8,
    )
    if bias:
        wrap_triton(_bias)[(triton.cdiv(columns, 32),)](
            partial,
            db,
            groups,
            columns,
            128,
            32,
            num_warps=4,
        )
    return packed, scale, db


def pack_pair(first, second=None, *, bias=False):
    """Retain FP16 operands directly; otherwise scale and pack with FP32 sums."""
    if first.ndim != 2 or (second is not None and second.ndim != 2):
        raise ValueError("Range packing requires matrices")
    if second is not None and (
        second.shape[0] != first.shape[0]
        or second.device != first.device
        or second.dtype != first.dtype
    ):
        raise ValueError("Packed pieces must share rows, dtype, and device")
    if first.dtype == torch.float16 and second is None and not bias:
        return (
            first,
            first.new_ones((), dtype=torch.float32),
            first.new_empty(0, dtype=torch.float32),
        )
    # Inductor already fuses contiguous FP32 scans efficiently. Reserve tiled
    # packing for strided operands and virtual concatenations.
    if (
        second is None
        and first.dtype == torch.float32
        and first.numel()
        and first.is_contiguous()
    ):
        packed, scale = quantize(first)
        return packed, scale, first.sum(0) if bias else first.new_empty(0)
    if (
        not first.is_cuda
        or not first.numel()
        or first.dtype not in (torch.float16, torch.float32)
    ):
        value = first if second is None else torch.cat((first, second), 1)
        maximum = (
            value.detach().float().abs().amax()
            if value.numel()
            else value.new_zeros((), dtype=torch.float32)
        )
        scale = maximum.clamp_min(2.0**-100).log2().floor().exp2()
        return (
            (value.float() / scale).half(),
            scale,
            value.float().sum(0) if bias else value.new_empty(0),
        )
    if second is None:
        second = first.new_empty((first.shape[0], 0))
    return range_pack_pair(first, second, bias)


@triton.jit
def _expand(W, OUT, O: tl.constexpr, K: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, col = index // (2 * K), index % (2 * K)
    imaginary = (row < O) ^ (col < K)
    source = tl.where(imaginary, O, 0) + row % O
    value = tl.load(W + source * K + col % K, row < 2 * O, 0)
    value = tl.where((row < O) & (col >= K), -value, value)
    tl.store(OUT + index, value, row < 2 * O)


@triton_op("fairchem::hip_complex_weight_pack", mutates_args={})
def expand_complex_weight(weight: torch.Tensor) -> torch.Tensor:
    rows, columns = weight.shape
    output = weight.new_empty((rows, 2 * columns))
    wrap_triton(_expand)[(triton.cdiv(rows * 2 * columns, 256),)](
        weight,
        output,
        rows // 2,
        columns,
        256,
    )
    return output


@triton.jit
def _fold(DW, OUT, O: tl.constexpr, K: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, col = index // K, index % K
    a = tl.load(DW + (row % O) * (2 * K) + col, row < 2 * O, 0)
    b = tl.load(DW + (row % O + O) * (2 * K) + col + K, row < 2 * O, 0)
    c = tl.load(DW + (row % O + O) * (2 * K) + col, row < 2 * O, 0)
    d = tl.load(DW + (row % O) * (2 * K) + col + K, row < 2 * O, 0)
    tl.store(OUT + index, tl.where(row < O, a + b, c - d), row < 2 * O)


@triton_op("fairchem::hip_complex_weight_gradient", mutates_args={})
def fold_complex_gradient(gradient: torch.Tensor) -> torch.Tensor:
    rows, columns = gradient.shape
    output = gradient.new_empty((rows, columns // 2))
    wrap_triton(_fold)[(triton.cdiv(rows * (columns // 2), 256),)](
        gradient,
        output,
        rows // 2,
        columns // 2,
        256,
    )
    return output
