"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from e3nn import o3

from fairchem.core.datasets.atomic_data import hessian_nentries_from_natoms
from fairchem.core.graph.compute import generate_graph

if TYPE_CHECKING:
    from fairchem.core.datasets.atomic_data import AtomicData


def get_cartesian_wigner_3j_basis(
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Fixed basis mapping l=0,1,2 irreps to Cartesian 3x3 blocks."""
    return torch.cat(
        (
            o3.wigner_3j(1, 1, 0, dtype=dtype, device=device),
            o3.wigner_3j(1, 1, 1, dtype=dtype, device=device),
            o3.wigner_3j(1, 1, 2, dtype=dtype, device=device),
        ),
        dim=-1,
    )


def irreps_to_cartesian_matrix(
    irreps: torch.Tensor, basis: torch.Tensor | None = None
) -> torch.Tensor:
    """Convert l=0,1,2 irreps to Cartesian 3x3 matrices."""
    if basis is None:
        basis = get_cartesian_wigner_3j_basis(
            dtype=irreps.dtype, device=irreps.device
        )
    elif basis.dtype != irreps.dtype or basis.device != irreps.device:
        basis = basis.to(dtype=irreps.dtype, device=irreps.device)
    return torch.einsum("...k,ijk->...ij", irreps, basis)


def _reject_graph_parallel(data: AtomicData) -> None:
    if hasattr(data, "node_partition"):
        raise NotImplementedError("Hessian graph utilities do not support GP batches.")
    if hasattr(data, "gp_node_offset") and data.gp_node_offset not in (0, None):
        raise NotImplementedError("Hessian graph utilities do not support GP batches.")
    if hasattr(data, "atomic_numbers_full") and (
        data.atomic_numbers_full.shape[0] != data.atomic_numbers.shape[0]
    ):
        raise NotImplementedError("Hessian graph utilities do not support GP batches.")


def fully_connected_hessian_graph_batch(
    data: AtomicData,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build all directed, non-self edges within each AtomicData sample."""
    _reject_graph_parallel(data)
    device = data.pos.device
    dtype = data.pos.dtype
    num_nodes = data.pos.shape[0]
    source = torch.arange(num_nodes, device=device, dtype=torch.long).repeat(num_nodes)
    target = torch.arange(num_nodes, device=device, dtype=torch.long).repeat_interleave(
        num_nodes
    )
    keep = (data.batch[source] == data.batch[target]) & (source != target)
    source = source[keep]
    target = target[keep]

    edge_index = torch.stack((source, target), dim=0)
    edge_distance_vec = data.pos[source] - data.pos[target]
    edge_distance = edge_distance_vec.norm(dim=-1)
    cell_offsets = torch.zeros(edge_index.shape[1], 3, device=device, dtype=dtype)
    offset_distances = torch.zeros_like(cell_offsets)
    natoms = data.natoms.to(device=device, dtype=torch.long)
    neighbors = natoms * (natoms - 1)
    return (
        edge_index,
        edge_distance,
        edge_distance_vec,
        cell_offsets,
        offset_distances,
        neighbors,
    )


def radius_hessian_graph_batch(
    data: AtomicData,
    cutoff: float,
    use_pbc: bool = False,
    radius_pbc_version: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build an unpruned cutoff Hessian graph with FairChem graph generation."""
    _reject_graph_parallel(data)
    pbc = data.pbc if use_pbc else torch.zeros_like(data.pbc, dtype=torch.bool)
    graph = generate_graph(
        data,
        cutoff=cutoff,
        # FairChem treats max_neighbors <= 0 as unbounded. Hessian graphs should
        # include every edge within the cutoff, not a nearest-neighbor subset.
        max_neighbors=0,
        enforce_max_neighbors_strictly=False,
        radius_pbc_version=radius_pbc_version,
        pbc=pbc,
        node_partition=None,
    )
    return (
        graph["edge_index"],
        graph["edge_distance"],
        graph["edge_distance_vec"],
        graph["cell_offsets"],
        graph["offset_distances"],
        graph["neighbors"],
    )


def _set_hessian_ptrs_and_offsets(
    data: AtomicData,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = data.pos.device
    natoms = data.natoms.to(device=device, dtype=torch.long)
    node_cumsum = torch.cumsum(natoms, dim=0)
    node_offsets = torch.zeros_like(node_cumsum)
    if natoms.numel() > 1:
        node_offsets[1:] = node_cumsum[:-1]

    hessian_nentries = hessian_nentries_from_natoms(natoms)
    hess_cumsum = torch.cumsum(hessian_nentries, dim=0)
    hess_offsets = torch.zeros_like(hess_cumsum)
    if natoms.numel() > 1:
        hess_offsets[1:] = hess_cumsum[:-1]

    data.hessian_nentries = hessian_nentries
    data.ptr_1d_hessian = torch.empty(natoms.numel() + 1, device=device, dtype=torch.long)
    data.ptr_1d_hessian[0] = 0
    data.ptr_1d_hessian[1:] = hess_cumsum
    return natoms, node_offsets, hess_offsets


def _add_offdiagonal_indices(
    data: AtomicData,
    edge_index: torch.Tensor,
    natoms: torch.Tensor,
    node_offsets: torch.Tensor,
    hess_offsets: torch.Tensor,
) -> None:
    device = data.pos.device
    num_edges = edge_index.shape[1]
    if num_edges == 0:
        data.nedges_hessian = torch.zeros_like(natoms)
        data.message_idx_ij = torch.empty(0, device=device, dtype=torch.long)
        data.message_idx_ji = torch.empty(0, device=device, dtype=torch.long)
        return

    i_global = edge_index[0].to(dtype=torch.long)
    j_global = edge_index[1].to(dtype=torch.long)
    sample_by_edge = data.batch[i_global].to(device=device, dtype=torch.long)
    if not torch.equal(sample_by_edge, data.batch[j_global].to(device=device)):
        raise ValueError("Hessian graph contains edges that cross AtomicData samples.")

    data.nedges_hessian = torch.bincount(sample_by_edge, minlength=natoms.numel())
    edge_node_offset = node_offsets[sample_by_edge]
    i_local = i_global - edge_node_offset
    j_local = j_global - edge_node_offset
    n3_by_edge = natoms[sample_by_edge] * 3

    ci = torch.arange(3, device=device, dtype=torch.long).view(1, 3, 1)
    cj = torch.arange(3, device=device, dtype=torch.long).view(1, 1, 3)
    i_local = i_local.view(num_edges, 1, 1)
    j_local = j_local.view(num_edges, 1, 1)
    n3_by_edge = n3_by_edge.view(num_edges, 1, 1)

    idx_ij = (i_local * 3 + ci) * n3_by_edge + (j_local * 3 + cj)
    idx_ji = (j_local * 3 + ci) * n3_by_edge + (i_local * 3 + cj)
    edge_hess_offsets = hess_offsets[sample_by_edge].view(num_edges, 1, 1)
    data.message_idx_ij = (idx_ij + edge_hess_offsets).reshape(-1)
    data.message_idx_ji = (idx_ji + edge_hess_offsets).reshape(-1)


def _add_diagonal_indices(
    data: AtomicData,
    natoms: torch.Tensor,
    node_offsets: torch.Tensor,
    hess_offsets: torch.Tensor,
) -> None:
    device = data.pos.device
    total_nodes = data.pos.shape[0]
    if total_nodes == 0:
        data.diag_ij = torch.empty(0, device=device, dtype=torch.long)
        data.diag_ji = torch.empty(0, device=device, dtype=torch.long)
        data.node_transpose_idx = torch.empty(0, device=device, dtype=torch.long)
        return

    sample_by_node = data.batch.to(device=device, dtype=torch.long)
    global_node_index = torch.arange(total_nodes, device=device, dtype=torch.long)
    node_local = global_node_index - node_offsets[sample_by_node]
    n3_by_node = natoms[sample_by_node] * 3
    hess_offset_by_node = hess_offsets[sample_by_node]

    ci = torch.arange(3, device=device, dtype=torch.long).view(1, 3, 1)
    cj = torch.arange(3, device=device, dtype=torch.long).view(1, 1, 3)
    node_local = node_local.view(total_nodes, 1, 1)
    n3_by_node = n3_by_node.view(total_nodes, 1, 1)
    hess_offset_by_node = hess_offset_by_node.view(total_nodes, 1, 1)

    row = node_local * 3 + ci
    col = node_local * 3 + cj
    diag = row * n3_by_node + col + hess_offset_by_node
    data.diag_ij = diag.reshape(-1)
    data.diag_ji = data.diag_ij.clone()

    node_flat_base = (global_node_index * 9).view(total_nodes, 1, 1)
    data.node_transpose_idx = (node_flat_base + cj * 3 + ci).reshape(-1)


def add_hessian_graph_batch(
    data: AtomicData,
    cutoff: float = 16.0,
    use_pbc: bool = False,
    fully_connected: bool = True,
    radius_pbc_version: int = 2,
) -> AtomicData:
    """Attach a Hessian graph and flattened scatter indices to AtomicData."""
    if fully_connected:
        graph = fully_connected_hessian_graph_batch(data)
    else:
        graph = radius_hessian_graph_batch(
            data,
            cutoff=cutoff,
            use_pbc=use_pbc,
            radius_pbc_version=radius_pbc_version,
        )

    (
        edge_index,
        edge_distance,
        edge_distance_vec,
        cell_offsets,
        offset_distances,
        neighbors,
    ) = graph
    data.edge_index_hessian = edge_index
    data.edge_distance_hessian = edge_distance
    data.edge_distance_vec_hessian = edge_distance_vec
    data.cell_offsets_hessian = cell_offsets
    data.cell_offset_distances_hessian = offset_distances
    data.neighbors_hessian = neighbors

    natoms, node_offsets, hess_offsets = _set_hessian_ptrs_and_offsets(data)
    _add_offdiagonal_indices(data, edge_index, natoms, node_offsets, hess_offsets)
    _add_diagonal_indices(data, natoms, node_offsets, hess_offsets)
    return data


def _indexadd_offdiagonal_to_flat_hessian(
    edge_index: torch.Tensor, messages: torch.Tensor, data: AtomicData
) -> torch.Tensor:
    total_entries = int(data.ptr_1d_hessian[-1].item())
    hessian = torch.zeros(total_entries, device=messages.device, dtype=messages.dtype)
    messages_t = messages.transpose(-2, -1).reshape(-1)
    hessian.index_add_(0, data.message_idx_ij, messages.reshape(-1))
    hessian.index_add_(0, data.message_idx_ji, messages_t)
    return hessian


def _indexadd_diagonal_to_flat_hessian(
    hessian: torch.Tensor, node_blocks: torch.Tensor, data: AtomicData
) -> torch.Tensor:
    node_blocks_flat = node_blocks.reshape(-1)
    hessian.index_add_(0, data.diag_ij, node_blocks_flat)
    hessian.index_add_(0, data.diag_ji, node_blocks_flat[data.node_transpose_idx])
    return hessian


def blocks3x3_to_hessian(
    edge_index: torch.Tensor,
    data: AtomicData,
    edge_blocks: torch.Tensor,
    node_blocks: torch.Tensor,
) -> torch.Tensor:
    """Scatter edge and node 3x3 blocks into a concatenated flat Hessian."""
    assert edge_blocks.shape == (edge_index.shape[1], 3, 3)
    assert node_blocks.shape == (data.pos.shape[0], 3, 3)
    assert data.message_idx_ij.numel() == edge_index.shape[1] * 9
    assert data.message_idx_ji.numel() == edge_index.shape[1] * 9
    assert data.diag_ij.numel() == data.pos.shape[0] * 9
    assert data.diag_ji.numel() == data.pos.shape[0] * 9
    assert data.node_transpose_idx.numel() == data.pos.shape[0] * 9
    hessian = _indexadd_offdiagonal_to_flat_hessian(edge_index, edge_blocks, data)
    return _indexadd_diagonal_to_flat_hessian(hessian, node_blocks, data)


def blocks3x3_to_hessian_loops(
    edge_index: torch.Tensor,
    data: AtomicData,
    edge_blocks: torch.Tensor,
    node_blocks: torch.Tensor,
) -> torch.Tensor:
    """Slow reference assembly returning the same flat layout as the fast path."""
    device = edge_blocks.device if edge_blocks.numel() else node_blocks.device
    dtype = edge_blocks.dtype if edge_blocks.numel() else node_blocks.dtype
    natoms = data.natoms.to(device=edge_index.device, dtype=torch.long)
    node_offsets = torch.zeros_like(natoms)
    if natoms.numel() > 1:
        node_offsets[1:] = torch.cumsum(natoms, dim=0)[:-1]

    pieces = []
    for sample_idx, n_atoms_t in enumerate(natoms):
        n_atoms = int(n_atoms_t.item())
        start = node_offsets[sample_idx]
        stop = start + n_atoms_t
        edge_mask = (edge_index[0] >= start) & (edge_index[0] < stop)
        sample_edges = edge_index[:, edge_mask] - start
        sample_blocks = edge_blocks[edge_mask]

        hessian = torch.zeros(n_atoms, 3, n_atoms, 3, device=device, dtype=dtype)
        for edge_idx in range(sample_edges.shape[1]):
            i = int(sample_edges[0, edge_idx].item())
            j = int(sample_edges[1, edge_idx].item())
            hessian[i, :, j, :] += sample_blocks[edge_idx]
            hessian[j, :, i, :] += sample_blocks[edge_idx].T

        sample_node_blocks = node_blocks[start:stop]
        for node_idx in range(n_atoms):
            hessian[node_idx, :, node_idx, :] += sample_node_blocks[node_idx]
            hessian[node_idx, :, node_idx, :] += sample_node_blocks[node_idx].T

        pieces.append(hessian.reshape(n_atoms * 3, n_atoms * 3).reshape(-1))

    return torch.cat(pieces, dim=0)
