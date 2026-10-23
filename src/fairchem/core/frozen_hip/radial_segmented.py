"""Accumulate complete-graph radial input gradients without atomic writes."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton

from fairchem.core.frozen_hip.radial import _radial_row
from fairchem.core.frozen_hip.scatter import _bounds, _m_order


@triton.jit
def _node_gradient(
    W,
    EDGES,
    R,
    G,
    PTR,
    OUT,
    E,
    C: tl.constexpr,
    BC: tl.constexpr,
    BE: tl.constexpr,
):
    atom = tl.program_id(0) // 9
    coefficient = tl.program_id(0) % 9
    channel = tl.program_id(1) * BC + tl.arange(0, BC)
    rows = tl.arange(0, BE)
    edge = tl.load(PTR + atom)
    end = tl.load(PTR + atom + 1)
    first = tl.load(EDGES + edge, edge < end, atom).to(tl.int32)
    molecule_start = tl.minimum(first, atom)
    local_atom = atom - molecule_start
    low = tl.where(coefficient == 0, 0, tl.where(coefficient < 4, 1, 4))
    width = tl.where(coefficient == 0, 1, tl.where(coefficient < 4, 3, 5))
    total = tl.zeros((BE, BC), tl.float32)
    while edge < end:
        incoming = (edge + rows).to(tl.int64)
        active = incoming < end
        source = tl.load(EDGES + incoming, active, atom).to(tl.int32)
        # Target-major complete graphs contain each reciprocal edge. Recover its
        # address directly, rather than constructing an edge-sized inverse map.
        reverse = (tl.load(PTR + source) + local_atom - (atom > source)).to(tl.int64)
        mask = active[:, None] & (channel[None, :] < C)
        value = tl.zeros((BE, BC), tl.float32)
        for j in tl.static_range(5):
            if j < width:
                row = low + j
                m = _m_order(row)
                radial_row = _radial_row(m)
                wi = tl.load(W + incoming * 81 + row * 9 + coefficient, active, 0)
                wr = tl.load(W + reverse * 81 + row * 9 + coefficient, active, 0)
                ri = tl.load(
                    R
                    + (incoming[:, None] * 6 + radial_row) * (2 * C)
                    + C
                    + channel[None, :],
                    mask,
                    0,
                ).to(tl.float32)
                gi = tl.load(
                    G + (incoming[:, None] * 9 + m) * (2 * C) + C + channel[None, :],
                    mask,
                    0,
                ).to(tl.float32)
                rr = tl.load(
                    R
                    + (reverse[:, None] * 6 + radial_row) * (2 * C)
                    + channel[None, :],
                    mask,
                    0,
                ).to(tl.float32)
                gr = tl.load(
                    G + (reverse[:, None] * 9 + m) * (2 * C) + channel[None, :],
                    mask,
                    0,
                ).to(tl.float32)
                value += wi[:, None] * (ri * gi) + wr[:, None] * (rr * gr)
        total += value
        edge += BE
    tl.store(
        OUT + (atom * 9 + coefficient) * C + channel,
        tl.sum(total, axis=0),
        channel < C,
    )


@triton_op("fairchem::hip_segmented_radial_dx", mutates_args={})
def segmented_radial_dx(
    w: torch.Tensor,
    edges: torch.Tensor,
    radial: torch.Tensor,
    gradient: torch.Tensor,
    nodes: int,
    channel_tile: int = 32,
    edge_tile: int = 16,
) -> torch.Tensor:
    """Require target-major complete graphs with contiguous molecule atoms.

    Radial coefficients and incoming gradients may use FP16. All products,
    reductions, and returned node gradients use FP32.
    """
    e, c = edges.shape[1], radial.shape[-1] // 12
    ptr = torch.empty(nodes + 1, device=edges.device, dtype=torch.int32)
    wrap_triton(_bounds)[(triton.cdiv(nodes + 1, 32),)](
        edges,
        ptr,
        e,
        nodes,
        0,
        32,
        num_warps=1,
    )
    out = torch.empty((nodes, 9, c), device=radial.device, dtype=torch.float32)
    wrap_triton(_node_gradient)[(nodes * 9, triton.cdiv(c, channel_tile))](
        w,
        edges,
        radial,
        gradient,
        ptr,
        out,
        e,
        c,
        channel_tile,
        edge_tile,
        num_warps=4,
    )
    return out
