"""Project complete Cartesian Hessians onto the translation sum rule."""

from __future__ import annotations

import torch


def project_dense(hessian, sizes):
    """Apply P H P without building P; retain gradients and the input layout."""
    pieces = []
    for flat, n in zip(
        hessian.reshape(-1).split([9 * n * n for n in sizes]), sizes, strict=True
    ):
        blocks = flat.reshape(n, 3, n, 3).permute(0, 2, 1, 3)
        row = blocks.mean(1, keepdim=True)
        col = blocks.mean(0, keepdim=True)
        pieces.append(
            (blocks - row - col + row.mean(0, keepdim=True))
            .permute(0, 2, 1, 3)
            .reshape(-1)
        )
    return torch.cat(pieces).reshape_as(hessian)


def project_blocks_reference(edge, node, data):
    """Project redundant directed blocks on complete, target-sorted graphs.

    Assembly is C + C.T. Keep its antisymmetric null space unchanged and
    project the symmetric part. This map is self-adjoint and idempotent.
    """
    source, target = data.edge_index_hessian
    rows = node + node.transpose(-1, -2)
    rows = rows.index_add(0, source, edge).index_add(0, target, edge.transpose(-1, -2))
    sizes = data.hip_sizes
    row = torch.cat([r / n for r, n in zip(rows.split(sizes), sizes, strict=True)])
    grand = torch.stack([r.mean(0) for r in row.split(sizes)])
    owner = data.batch
    return (
        edge
        + 0.5 * (-row[source] - row[target].transpose(-1, -2) + grand[owner[source]]),
        node + 0.5 * (-row - row.transpose(-1, -2) + grand[owner]),
    )


def project_blocks(edge, node, data):
    """Use FP32 reductions without assembling or padding a dense Hessian."""
    if not edge.is_cuda:
        return project_blocks_reference(edge, node, data)
    from fairchem.core.frozen_hip.translation_kernels import project

    return project(edge, node, data)
