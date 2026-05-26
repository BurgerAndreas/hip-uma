"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import torch

from fairchem.core.datasets.atomic_data import AtomicData, atomicdata_list_to_batch
from fairchem.core.models.uma.hessian_pred_utils import (
    add_hessian_graph_batch,
    blocks3x3_to_hessian,
    blocks3x3_to_hessian_loops,
    irreps_to_cartesian_matrix,
)


def _atomic_data(num_atoms: int, dtype: torch.dtype = torch.float32) -> AtomicData:
    pos = torch.arange(num_atoms * 3, dtype=dtype).view(num_atoms, 3) / 10
    return AtomicData.from_dict(
        {
            "pos": pos,
            "atomic_numbers": torch.ones(num_atoms, dtype=torch.long),
            "cell": torch.eye(3, dtype=dtype).view(1, 3, 3) * 20,
            "pbc": torch.zeros(1, 3, dtype=torch.bool),
            "natoms": torch.tensor([num_atoms], dtype=torch.long),
            "edge_index": torch.empty(2, 0, dtype=torch.long),
            "cell_offsets": torch.empty(0, 3, dtype=dtype),
            "nedges": torch.tensor([0], dtype=torch.long),
            "charge": torch.zeros(1, dtype=torch.long),
            "spin": torch.zeros(1, dtype=torch.long),
            "fixed": torch.zeros(num_atoms, dtype=torch.long),
            "tags": torch.ones(num_atoms, dtype=torch.long),
        }
    )


def test_irreps_to_cartesian_matrix_shape_and_dtype():
    irreps = torch.randn(4, 9, dtype=torch.float64)

    out = irreps_to_cartesian_matrix(irreps)

    assert out.shape == (4, 3, 3)
    assert out.dtype == torch.float64
    assert out.device == irreps.device


def test_fully_connected_hessian_graph_batch_variable_natoms():
    batch = atomicdata_list_to_batch([_atomic_data(2), _atomic_data(3)])

    add_hessian_graph_batch(batch, fully_connected=True)

    assert batch.edge_index_hessian.shape == (2, 8)
    assert torch.equal(batch.nedges_hessian, torch.tensor([2, 6]))
    assert torch.equal(batch.neighbors_hessian, torch.tensor([2, 6]))
    assert torch.equal(batch.hessian_nentries, torch.tensor([36, 81]))
    assert torch.equal(batch.ptr_1d_hessian, torch.tensor([0, 36, 117]))
    assert batch.message_idx_ij.shape == (8 * 9,)
    assert batch.message_idx_ji.shape == (8 * 9,)
    assert batch.diag_ij.shape == (5 * 9,)
    assert batch.diag_ji.shape == (5 * 9,)
    assert batch.node_transpose_idx.shape == (5 * 9,)

    src, dst = batch.edge_index_hessian
    assert torch.all(batch.batch[src] == batch.batch[dst])
    assert torch.all(src != dst)


def test_radius_hessian_graph_uses_uncapped_fairchem_graph():
    batch = atomicdata_list_to_batch([_atomic_data(2), _atomic_data(3)])

    add_hessian_graph_batch(batch, fully_connected=False, cutoff=100.0, use_pbc=False)

    assert batch.edge_index_hessian.shape == (2, 8)
    assert torch.equal(batch.nedges_hessian, torch.tensor([2, 6]))
    assert torch.equal(batch.neighbors_hessian, torch.tensor([2, 6]))
    src, dst = batch.edge_index_hessian
    assert torch.all(batch.batch[src] == batch.batch[dst])
    assert torch.all(src != dst)


def test_blocks3x3_to_hessian_matches_loop_reference_for_atomicdata_batch():
    torch.manual_seed(7)
    batch = atomicdata_list_to_batch([_atomic_data(2), _atomic_data(3)])
    add_hessian_graph_batch(batch, fully_connected=True)
    edge_blocks = torch.randn(batch.edge_index_hessian.shape[1], 3, 3)
    node_blocks = torch.randn(batch.pos.shape[0], 3, 3)

    fast = blocks3x3_to_hessian(
        batch.edge_index_hessian, batch, edge_blocks, node_blocks
    )
    slow = blocks3x3_to_hessian_loops(
        batch.edge_index_hessian, batch, edge_blocks, node_blocks
    )

    assert torch.allclose(fast, slow)
    assert fast.shape == (117,)


def test_blocks3x3_to_hessian_outputs_symmetric_per_sample_blocks():
    torch.manual_seed(11)
    batch = atomicdata_list_to_batch([_atomic_data(2), _atomic_data(3)])
    add_hessian_graph_batch(batch, fully_connected=True)
    edge_blocks = torch.randn(batch.edge_index_hessian.shape[1], 3, 3)
    node_blocks = torch.randn(batch.pos.shape[0], 3, 3)

    hessian = blocks3x3_to_hessian(
        batch.edge_index_hessian, batch, edge_blocks, node_blocks
    )

    for sample_idx, natoms in enumerate(batch.natoms.tolist()):
        start = batch.ptr_1d_hessian[sample_idx]
        stop = batch.ptr_1d_hessian[sample_idx + 1]
        hessian_matrix = hessian[start:stop].view(natoms * 3, natoms * 3)
        assert torch.allclose(hessian_matrix, hessian_matrix.T)
