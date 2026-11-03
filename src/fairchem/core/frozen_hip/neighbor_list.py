"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _cell_lookup(CELLS, POINTER, M, qx, qy, qz, STEPS: tl.constexpr):
    lo = 0
    hi = M
    for _ in range(STEPS):
        mid = (lo + hi) // 2
        x = tl.load(CELLS + 3 * mid, mid < M, 9223372036854775807)
        y = tl.load(CELLS + 3 * mid + 1, mid < M, 9223372036854775807)
        z = tl.load(CELLS + 3 * mid + 2, mid < M, 9223372036854775807)
        before = (x < qx) | ((x == qx) & ((y < qy) | ((y == qy) & (z < qz))))
        lo = tl.where(before, mid + 1, lo)
        hi = tl.where(before, hi, mid)
    x = tl.load(CELLS + 3 * lo, lo < M, 0)
    y = tl.load(CELLS + 3 * lo + 1, lo < M, 0)
    z = tl.load(CELLS + 3 * lo + 2, lo < M, 0)
    found = (lo < M) & (x == qx) & (y == qy) & (z == qz)
    start = tl.load(POINTER + lo, found, 0)
    end = tl.load(POINTER + lo + 1, found, 0)
    return start, end


@triton.jit
def _neighbors(
    P,
    CELLS,
    CELL_ID,
    ORDER,
    CELL_PTR,
    COUNTS,
    ROW_PTR,
    INDEX,
    M,
    K,
    R: tl.constexpr,
    FILL: tl.constexpr,
    B: tl.constexpr,
    STEPS: tl.constexpr,
):
    atom = tl.program_id(0).to(tl.int64)
    cell = tl.load(CELL_ID + atom).to(tl.int64)
    cx = tl.load(CELLS + 3 * cell)
    cy = tl.load(CELLS + 3 * cell + 1)
    cz = tl.load(CELLS + 3 * cell + 2)
    px = tl.load(P + 3 * atom)
    py = tl.load(P + 3 * atom + 1)
    pz = tl.load(P + 3 * atom + 2)
    count = tl.full((), 0, tl.int32)
    if FILL:
        row_start = tl.load(ROW_PTR + atom)
    lane = tl.arange(0, B)
    for dx in tl.static_range(-1, 2):
        for dy in tl.static_range(-1, 2):
            for dz in tl.static_range(-1, 2):
                begin, end = _cell_lookup(
                    CELLS, CELL_PTR, M, cx + dx, cy + dy, cz + dz, STEPS
                )
                while begin < end:
                    k = begin + lane
                    source = tl.load(ORDER + k, k < end, 0).to(tl.int64)
                    valid = (k < end) & (source > atom)
                    vx = tl.load(P + 3 * source, valid, 0) - px
                    vy = tl.load(P + 3 * source + 1, valid, 0) - py
                    vz = tl.load(P + 3 * source + 2, valid, 0) - pz
                    keep = valid & (vx * vx + vy * vy + vz * vz <= R * R)
                    flags = keep.to(tl.int32)
                    if FILL:
                        slot = row_start + count + tl.cumsum(flags) - 1
                        tl.store(INDEX + slot, atom, keep)
                        tl.store(INDEX + K + slot, source, keep)
                    count += tl.sum(flags)
                    begin += B
    if not FILL:
        tl.store(COUNTS + atom, count)


@triton.jit
def _sort_tiles(COLUMNS, PTR, B: tl.constexpr):
    atom = tl.program_id(0)
    start = tl.load(PTR + atom)
    end = tl.load(PTR + atom + 1)
    lane = tl.arange(0, B)
    while start < end:
        q = start + lane
        value = tl.load(COLUMNS + q, q < end, 2147483647)
        value = tl.sort(value, descending=False)
        tl.store(COLUMNS + q, value, q < end)
        start += B


@triton.jit
def _merge_runs(
    ROWS,
    COLUMNS,
    OUT,
    PTR,
    K,
    WIDTH: tl.constexpr,
    STEPS: tl.constexpr,
    B: tl.constexpr,
):
    q = tl.program_id(0).to(tl.int64) * B + tl.arange(0, B)
    valid = q < K
    atom = tl.load(ROWS + q, valid, 0).to(tl.int64)
    start = tl.load(PTR + atom)
    end = tl.load(PTR + atom + 1)
    local = q - start
    base = start + (local // (2 * WIDTH)) * (2 * WIDTH)
    right = (local % (2 * WIDTH)) >= WIDTH
    other = tl.minimum(base + tl.where(right, 0, WIDTH), end)
    lo = other
    hi = tl.minimum(other + WIDTH, end)
    value = tl.load(COLUMNS + q, valid, 0)
    for _ in range(STEPS):
        mid = (lo + hi) // 2
        probe = tl.load(COLUMNS + mid, valid & (mid < hi), 2147483647)
        before = (probe < value) | (right & (probe == value))
        lo = tl.where(before, mid + 1, lo)
        hi = tl.where(before, hi, mid)
    slot = base + (local % WIDTH) + (lo - other)
    tl.store(OUT + slot, value, valid)


@torch.no_grad()
def cutoff_pairs(pos, cutoff):
    """
    Build uncapped, sorted i<j pairs once from occupied spatial cells.

    Storage is O(N+K). Sorting uses one temporary int32 column array of size K.
    Preserve all distances at or below cutoff, including coincident atoms.
    """
    if pos.device.type != "cuda" or pos.dtype != torch.float32:
        raise ValueError("Cutoff pairs require CUDA FP32 positions")
    if len(pos) == 0 or len(pos) >= 2**31 or not cutoff > 0:
        raise ValueError("Require a positive cutoff and 1 through 2^31-1 atoms")
    pos = pos.contiguous()
    # Sparse occupied-cell coordinates avoid allocating a grid over empty space.
    cell_width = float(cutoff) * (1 + 8 * torch.finfo(torch.float32).eps)
    coordinate = torch.floor((pos.double() - pos.amin(0).double()) / cell_width).long()
    cells, inverse, counts = torch.unique(
        coordinate, dim=0, sorted=True, return_inverse=True, return_counts=True
    )
    del coordinate
    order = torch.argsort(inverse, stable=True).int()
    inverse = inverse.int()
    cell_ptr = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
    row_counts = torch.empty(len(pos), device=pos.device, dtype=torch.int32)
    m = len(cells)
    steps = (m + 1).bit_length()
    # Unused pointers in the count specialization are never dereferenced.
    _neighbors[(len(pos),)](
        pos,
        cells,
        inverse,
        order,
        cell_ptr,
        row_counts,
        row_counts,
        row_counts,
        m,
        0,
        float(cutoff),
        False,
        256,
        steps,
    )
    row_ptr = torch.cat(
        (torch.zeros(1, device=pos.device, dtype=torch.int64), row_counts.cumsum(0))
    )
    k = int(row_ptr[-1].item())
    indices = torch.empty((2, k), device=pos.device, dtype=torch.int32)
    if k:
        _neighbors[(len(pos),)](
            pos,
            cells,
            inverse,
            order,
            cell_ptr,
            row_counts,
            row_ptr,
            indices,
            m,
            k,
            float(cutoff),
            True,
            256,
            steps,
        )
        columns = indices[1]
        _sort_tiles[(len(pos),)](columns, row_ptr, 512)
        maximum = int(row_counts.max().item())
        width = 512
        if maximum > width:
            scratch = torch.empty_like(columns)
            target = scratch
            while width < maximum:
                _merge_runs[(triton.cdiv(k, 256),)](
                    indices[0],
                    columns,
                    target,
                    row_ptr,
                    k,
                    width,
                    width.bit_length() + 1,
                    256,
                )
                columns, target = target, columns
                width *= 2
            if columns.data_ptr() != indices[1].data_ptr():
                indices[1].copy_(columns)
    return indices, row_ptr


@triton.jit
def _listed_geometry(
    P, INDEX, EI, V, D, BEGIN, K, PAIRS: tl.constexpr, B: tl.constexpr
):
    q = tl.program_id(0) * B + tl.arange(0, B)
    local = q % PAIRS
    slot = BEGIN + local.to(tl.int64)
    i = tl.load(INDEX + slot, q < 2 * PAIRS, 0).to(tl.int64)
    j = tl.load(INDEX + K + slot, q < 2 * PAIRS, 0).to(tl.int64)
    source = tl.where(q < PAIRS, j, i)
    target = tl.where(q < PAIRS, i, j)
    valid = q < 2 * PAIRS
    dx = tl.load(P + source * 3, valid, 0) - tl.load(P + target * 3, valid, 0)
    dy = tl.load(P + source * 3 + 1, valid, 0) - tl.load(P + target * 3 + 1, valid, 0)
    dz = tl.load(P + source * 3 + 2, valid, 0) - tl.load(P + target * 3 + 2, valid, 0)
    tl.store(EI + q, source, valid)
    tl.store(EI + 2 * PAIRS + q, target, valid)
    tl.store(V + q * 3, dx, valid)
    tl.store(V + q * 3 + 1, dy, valid)
    tl.store(V + q * 3 + 2, dz, valid)
    tl.store(D + q, tl.sqrt(dx * dx + dy * dy + dz * dz), valid)


def listed_chunk(pos, indices, begin, end):
    """
    Generate both directed geometries for a contiguous canonical-pair slice.
    """
    count = end - begin
    edges = torch.empty((2, 2 * count), device=pos.device, dtype=torch.int64)
    vectors = pos.new_empty((2 * count, 3))
    distances = pos.new_empty(2 * count)
    if count:
        _listed_geometry[(triton.cdiv(2 * count, 256),)](
            pos, indices, edges, vectors, distances, begin, indices.shape[1], count, 256
        )
    return edges, vectors, distances


@triton.jit
def _retain_gamma(GAMMA, INDEX, OUT, K, N, B: tl.constexpr):
    q = tl.program_id(0).to(tl.int64) * B + tl.arange(0, B)
    valid = q < K
    i = tl.load(INDEX + q, valid, 0).to(tl.int64)
    j = tl.load(INDEX + K + q, valid, 0).to(tl.int64)
    a = tl.load(GAMMA + i * (N - 1) + j - 1, valid, 0)
    b = tl.load(GAMMA + j * (N - 1) + i, valid, 0)
    tl.store(OUT + q, a, valid)
    tl.store(OUT + K + q, b, valid)


@torch.no_grad()
def retained_angles(gamma, indices, natoms, chunk_pairs):
    """
    Retain the exact legacy angles for both directions, then release the full buffer.

    Gather once. Host storage remains optional and preserves the old CUDA RNG order.
    """
    k = indices.shape[1]
    result = torch.empty((2, k), device=gamma.device, dtype=gamma.dtype)
    if gamma.device.type == "cuda":
        if k:
            _retain_gamma[(triton.cdiv(k, 256),)](
                gamma, indices, result, k, natoms, 256
            )
        return result
    for begin in range(0, k, chunk_pairs):
        end = min(begin + chunk_pairs, k)
        pair = indices[:, begin:end].long()
        i, j = pair[0], pair[1]
        ids = torch.stack((i * (natoms - 1) + j - 1, j * (natoms - 1) + i))
        ids = ids.to(device=gamma.device)
        result[:, begin:end].copy_(gamma[ids])
    return result
