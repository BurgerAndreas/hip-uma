"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

from fairchem.core.frozen_hip.readout import edge_irreps
from fairchem.core.frozen_hip.translation import project_dense
from fairchem.core.frozen_hip.wigner import frozen_wigner
from fairchem.core.models.uma.hessian_pred_utils import irreps_to_cartesian_matrix


@triton.jit
def _pairs(
    P,
    EI,
    V,
    D,
    KEEP,
    BEGIN,
    N: tl.constexpr,
    E: tl.constexpr,
    CUTOFF: tl.constexpr,
    B: tl.constexpr,
):
    k = tl.program_id(0) * B + tl.arange(0, B)
    q = BEGIN + k
    target = q // (N - 1)
    local = q % (N - 1)
    source = local + (local >= target)
    valid = k < E
    dx = tl.load(P + source * 3, valid, 0) - tl.load(P + target * 3, valid, 0)
    dy = tl.load(P + source * 3 + 1, valid, 0) - tl.load(P + target * 3 + 1, valid, 0)
    dz = tl.load(P + source * 3 + 2, valid, 0) - tl.load(P + target * 3 + 2, valid, 0)
    d2 = dx * dx + dy * dy + dz * dz
    tl.store(EI + k, source, valid)
    tl.store(EI + E + k, target, valid)
    tl.store(V + k * 3, dx, valid)
    tl.store(V + k * 3 + 1, dy, valid)
    tl.store(V + k * 3 + 2, dz, valid)
    tl.store(D + k, tl.sqrt(d2), valid)
    tl.store(KEEP + k, d2 <= CUTOFF * CUTOFF, valid)


@triton.jit
def _assemble(EI, BLOCKS, H, E, N: tl.constexpr, B: tl.constexpr):
    k = tl.program_id(0) * B + tl.arange(0, B)
    edge, component = k // 9, k % 9
    source = tl.load(EI + edge, edge < E, 0).to(tl.int64)
    target = tl.load(EI + E + edge, edge < E, 0).to(tl.int64)
    row, col = component // 3, component % 3
    value = tl.load(BLOCKS + k, edge < E, 0)
    # Match upstream directed assembly: C + C.T.
    tl.atomic_add(H + (source * 3 + row) * (N * 3) + target * 3 + col, value, edge < E)
    tl.atomic_add(H + (target * 3 + col) * (N * 3) + source * 3 + row, value, edge < E)


def pair_chunk(pos, begin, end, cutoff=None):
    pos = pos.contiguous()
    n, e = len(pos), end - begin
    edges = torch.empty((2, e), dtype=torch.int64, device=pos.device)
    vectors = pos.new_empty((e, 3))
    distances = pos.new_empty(e)
    keep = torch.empty(e, dtype=torch.bool, device=pos.device)
    _pairs[(triton.cdiv(e, 256),)](
        pos,
        edges,
        vectors,
        distances,
        keep,
        begin,
        n,
        e,
        float("inf") if cutoff is None else float(cutoff),
        256,
    )
    selected = None
    if cutoff is not None:
        selected = keep.nonzero().flatten()
        edges = edges[:, selected].contiguous()
        vectors, distances = vectors[selected], distances[selected]
    return edges, vectors, distances, selected


def features(head, data, begin, end, gamma, cutoff):
    edges, vectors, distances, selected = pair_chunk(data.pos, begin, end, cutoff)
    if edges.shape[1] == 0:
        return None
    roll = gamma[begin:end]
    if selected is not None:
        roll = roll[selected]
    w, inv = frozen_wigner(head, vectors, distances, gamma=roll)
    coefficient_index = (
        head.backbone.coefficient_index
        if head.backbone.mmax != head.backbone.lmax
        else None
    )
    w, inv = head.backend.prepare_wigner(
        w, inv, head.backbone.mappingReduced, coefficient_index
    )
    numbers = data.get("atomic_numbers_full", data.atomic_numbers)
    radial = head.distance_expansion_hessian(distances)
    x_edge = torch.cat(
        (
            radial,
            head.backbone.source_embedding(numbers[edges[0]]),
            head.backbone.target_embedding(numbers[edges[1]]),
        ),
        dim=1,
    )
    return edges, x_edge, w, inv


@torch.no_grad()
def predict(
    head, data, emb, chunk_edges=65536, cutoff=None, project=True, output_format="dense"
):
    """
    Predict one molecular Hessian and return its retained directed pair count.

    Memory for wide activations scales with chunk_edges, not N squared.
    Keep the head's trained radial scale (100 A) even when filtering pairs.
    All node updates happen after every chunk's messages have been summed.
    """
    from fairchem.core.common import gp_utils

    if gp_utils.initialized() or getattr(data, "node_partition", None) is not None:
        raise NotImplementedError("Chunked Hessians do not support graph parallelism")
    if data.pos.device.type != "cuda" or data.pos.dtype != torch.float32:
        raise ValueError("Chunked Hessians require CUDA FP32 coordinates")
    if len(data.pos) == 0:
        raise ValueError("Chunked Hessians require at least one atom")
    if output_format not in ("dense", "blocks"):
        raise ValueError("Unknown output format")
    if head.training or torch.is_grad_enabled():
        raise ValueError("Streaming supports frozen inference only")
    if len(data.natoms) != 1 or bool(data.pbc.any()):
        raise ValueError("Streaming requires one nonperiodic molecule")
    if chunk_edges < 1 or (cutoff is not None and cutoff <= 0):
        raise ValueError("Chunk size and cutoff must be positive")
    if head.backbone.lmax != 2 or head.backbone.mmax != 2:
        raise ValueError("Streaming kernels require lmax=mmax=2")
    if not getattr(head, "fused_wigner", False):
        raise ValueError("Enable fused rotations before streaming")
    if getattr(head, "requires_mole_routing", True):
        raise ValueError("Streaming requires a head without MoLE routing")
    n = len(data.pos)
    total = n * (n - 1)
    # Small scalar buffer preserves exactly the complete-graph random roll order.
    gamma = torch.rand(total, device=data.pos.device) * (2 * math.pi)
    chunks = range(0, total, chunk_edges)
    x = emb["node_embedding"].clone()
    for block in head.blocks:
        normalized = block.norm_1(x)
        messages = torch.zeros_like(x, dtype=torch.float32)
        for begin in chunks:
            f = features(
                head, data, begin, min(begin + chunk_edges, total), gamma, cutoff
            )
            if f is None:
                continue
            edges, radial, w, inv = f
            partial = block.edge_wise.forward_chunk(
                normalized, n, radial, edges, w, inv
            )
            messages.add_(partial)
            del partial, f, edges, radial, w, inv
        x = x + messages
        x = x + block.atom_wise(block.norm_2(x))
    x = head.norm(x)
    node_irreps = head.node_projection(x[:, :9]).squeeze(-1)
    basis = head.cartesian_wigner_3j_basis.to(device=x.device, dtype=node_irreps.dtype)
    node = irreps_to_cartesian_matrix(node_irreps, basis)
    diagonal = node + node.transpose(-1, -2)
    if output_format == "blocks":
        from fairchem.core.frozen_hip.sparse_output import accumulate, allocate, finish

        indices, values, ptr, rows = allocate(data, diagonal, chunk_edges, cutoff)
    else:
        h = x.new_zeros((3 * n, 3 * n), dtype=torch.float32)
        h4 = h.view(n, 3, n, 3)
        ids = torch.arange(n, device=x.device)
        h4[ids, :, ids, :] = diagonal
    edge_count = 0
    for begin in chunks:
        f = features(head, data, begin, min(begin + chunk_edges, total), gamma, cutoff)
        if f is None:
            continue
        edges, radial, w, inv = f
        irreps = edge_irreps(head, x, radial, edges, w, inv)
        blocks = irreps_to_cartesian_matrix(irreps, basis).contiguous()
        e = edges.shape[1]
        edge_count += e
        if output_format == "blocks":
            accumulate(edges, blocks, indices, values, ptr, rows, n, cutoff is None)
        else:
            _assemble[(triton.cdiv(e * 9, 256),)](edges, blocks, h, e, n, 256)
        del f, edges, radial, w, inv, irreps, blocks
    if output_format == "blocks":
        return finish(indices, values, ptr, diagonal, rows, project), edge_count
    if project:
        h = project_dense(h, [n])
    return h, edge_count
