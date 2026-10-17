"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Construct compact nonperiodic neighbor graphs for frozen molecular inference.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from fairchem.core.frozen_hip.kernels import _owner


@triton.jit
def _select_neighbors(
    P,
    AP,
    MASK,
    COUNTS,
    B,
    CUTOFF: tl.constexpr,
    LIMIT: tl.constexpr,
    WIDTH: tl.constexpr,
    LEVELS: tl.constexpr,
):
    atom = tl.program_id(0)
    sample = _owner(atom, AP, B, LEVELS)
    start = tl.load(AP + sample)
    n = tl.load(AP + sample + 1) - start
    j = tl.arange(0, WIDTH)
    d2 = tl.full((WIDTH,), 0, tl.float32)
    for k in range(3):
        delta = tl.load(P + (start + j) * 3 + k, j < n, 0) - tl.load(P + atom * 3 + k)
        d2 += delta * delta
    keep = (j < n) & (d2 > 0.0001) & (d2 <= CUTOFF * CUTOFF)
    if LIMIT > 0 and WIDTH > LIMIT + 1:  # noqa: SIM102
        if tl.sum(keep.to(tl.int32), 0) > LIMIT:
            ordered = tl.sort(tl.where(keep, d2, float("inf")), descending=False)
            # Match the reference's zero-based LIMIT and squared-distance tolerance.
            threshold = tl.sum(tl.where(j == LIMIT, ordered, 0), 0) + 0.01
            keep = keep & (d2 <= threshold)
    tl.store(MASK + atom * WIDTH + j, keep.to(tl.uint8))
    tl.store(COUNTS + atom, tl.sum(keep.to(tl.int32), 0))


@triton.jit
def _write_neighbors(
    P, AP, MASK, PTR, EI, V, D, Z, E, B, WIDTH: tl.constexpr, LEVELS: tl.constexpr
):
    atom = tl.program_id(0)
    sample = _owner(atom, AP, B, LEVELS)
    start = tl.load(AP + sample)
    j = tl.arange(0, WIDTH)
    keep = tl.load(MASK + atom * WIDTH + j) != 0
    out = tl.load(PTR + atom) + tl.cumsum(keep.to(tl.int32), 0) - 1
    source = start + j
    tl.store(EI + out, source, keep)
    tl.store(EI + E + out, atom, keep)
    d2 = tl.full((WIDTH,), 0, tl.float32)
    for k in range(3):
        delta = tl.load(P + source * 3 + k, keep, 0) - tl.load(P + atom * 3 + k)
        d2 += delta * delta
        tl.store(V + out * 3 + k, delta, keep)
        tl.store(Z + out * 3 + k, 0, keep)
    tl.store(D + out, tl.sqrt(d2), keep)


def nonperiodic_graph(data, cutoff, max_neighbors, pbc):
    """
    Build an indexed radius graph with one host transfer for its compact size.

    This path supports frozen FP32 coordinates and non-strict neighbor limits.
    Small padded masks are scratch space only. Returned edges have no padding.
    """
    if data.pos.device.type != "cuda" or data.pos.dtype != torch.float32:
        raise ValueError("Fused radius graphs require CUDA FP32 coordinates")
    if data.pos.requires_grad:
        raise ValueError("Fused radius graphs require frozen coordinates")
    if pbc is None or bool(pbc.any()):
        raise ValueError("Fused radius graphs require explicit nonperiodic flags")
    sizes = data.hip_sizes
    if not sizes or min(sizes) < 1 or max(sizes) > 1024:
        raise ValueError(
            "Fused radius graphs support 1 through 1024 atoms per molecule"
        )
    atoms, batch = len(data.pos), len(sizes)
    if sum(sizes) != atoms:
        raise ValueError("CPU atom counts differ from the coordinate array")
    width = triton.next_power_of_2(max(sizes))
    ap = getattr(data, "hip_atom_ptr", None)
    if ap is None:
        ap = torch.tensor([0, *sizes], device=data.pos.device).cumsum(0)
    mask = torch.empty((atoms, width), device=data.pos.device, dtype=torch.uint8)
    counts = torch.empty(atoms, device=data.pos.device, dtype=torch.int64)
    _select_neighbors[(atoms,)](
        data.pos,
        ap,
        mask,
        counts,
        batch,
        cutoff,
        max_neighbors,
        width,
        (batch + 1).bit_length(),
        enable_fp_fusion=False,
    )
    ptr = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
    boundaries = ptr[ap]
    # One compact host read replaces repeated scalar reads in the generic path.
    edges = int(boundaries[-1].item())
    ei = torch.empty((2, edges), device=data.pos.device, dtype=torch.int64)
    vectors = data.pos.new_empty((edges, 3))
    distances = data.pos.new_empty(edges)
    offsets = data.pos.new_empty((edges, 3))
    _write_neighbors[(atoms,)](
        data.pos,
        ap,
        mask,
        ptr,
        ei,
        vectors,
        distances,
        offsets,
        edges,
        batch,
        width,
        (batch + 1).bit_length(),
        enable_fp_fusion=False,
    )
    return dict(
        edge_index=ei,
        edge_distance=distances,
        edge_distance_vec=vectors,
        cell_offsets=offsets,
        offset_distances=offsets,
        neighbors=boundaries[1:] - boundaries[:-1],
    )
