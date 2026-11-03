"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import copy

import pytest
import torch

from fairchem.core.frozen_hip.neighbor_list import (
    cutoff_pairs,
    listed_chunk,
    retained_angles,
)
from fairchem.core.frozen_hip.sparse_output import finish, write_paired_blocks
from fairchem.core.frozen_hip.streamed_head import pair_chunk, predict


def cuda_required():
    """
    Skip these independent kernel checks when no CUDA device is present.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")


@pytest.mark.parametrize(
    "n,cutoff", [(1, 0.1), (7, 1e-6), (73, 0.8), (1400, 1.1), (1200, 1000.0)]
)
@torch.no_grad()
def test_spatial_pairs_match_all_candidate_reference(n, cutoff):
    """
    Check exact inclusion, ordering, empty graphs, and multi-tile row sorting.
    """
    cuda_required()
    torch.manual_seed(91)
    pos = 1.5 * torch.rand(n, 3, device="cuda")
    indices, ptr = cutoff_pairs(pos, cutoff)
    pieces = []
    for begin in range(0, n * (n - 1), 65536):
        edges, _, _, _ = pair_chunk(pos, begin, min(begin + 65536, n * (n - 1)), cutoff)
        pieces.append(edges[:, edges[0] > edges[1]].flip(0).int())
    expected = torch.cat(pieces, 1) if pieces else indices.new_empty((2, 0))
    torch.testing.assert_close(indices, expected, rtol=0, atol=0)
    assert ptr.dtype == torch.int64
    assert int(ptr[-1]) == indices.shape[1]
    torch.testing.assert_close(
        ptr[1:] - ptr[:-1], torch.bincount(indices[0].long(), minlength=n)
    )
    if n == 1400:
        assert int((ptr[1:] - ptr[:-1]).max()) > 512
    if indices.shape[1]:
        stop = min(113, indices.shape[1])
        edges, vectors, distances = listed_chunk(pos, indices, 0, stop)
        torch.testing.assert_close(vectors, pos[edges[0]] - pos[edges[1]])
        torch.testing.assert_close(distances, torch.linalg.vector_norm(vectors, dim=1))
        torch.testing.assert_close(edges[:, stop:], edges[:, :stop].flip(0))


@torch.no_grad()
def test_cutoff_boundaries_negative_coordinates_and_exact_angles():
    """
    Preserve boundary pairs and the complete-graph random-angle assignment.
    """
    cuda_required()
    pos = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
        ],
        device="cuda",
    )
    indices, _ = cutoff_pairs(pos, 1.0)
    edges, _, _, _ = pair_chunk(pos, 0, 30, 1.0)
    expected = edges[:, edges[0] > edges[1]].flip(0).int()
    torch.testing.assert_close(indices, expected, rtol=0, atol=0)
    i, j = indices.long()
    ids = torch.stack((i * 5 + j - 1, j * 5 + i))
    for device in ["cpu", "cuda"]:
        gamma = torch.arange(30, device=device, dtype=torch.float32)
        expected = gamma[ids.to(device)]
        actual = retained_angles(gamma, indices, 6, 3)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@torch.no_grad()
def test_paired_output_assembly_matches_independent_dense():
    """
    Check direct-slot writes, row sums, projection, and block lookup.
    """
    cuda_required()
    torch.manual_seed(41)
    n = 23
    pos = torch.randn(n, 3, device="cuda")
    indices, ptr = cutoff_pairs(pos, 1.8)
    k = indices.shape[1]
    values = torch.full((k, 3, 3), float("nan"), device="cuda")
    diagonal = torch.randn(n, 3, 3, device="cuda")
    diagonal += diagonal.transpose(-1, -2).clone()
    rows = diagonal.clone()
    reference = torch.zeros(n, 3, n, 3, device="cuda")
    ids = torch.arange(n, device="cuda")
    reference[ids, :, ids, :] = diagonal
    for begin in range(0, k, 7):
        end = min(begin + 7, k)
        count = end - begin
        blocks = torch.randn(2 * count, 3, 3, device="cuda")
        write_paired_blocks(blocks, indices, values, rows, begin, count)
        for q in range(count):
            i, j = indices[:, begin + q].tolist()
            v = blocks[count + q] + blocks[q].T
            reference[i, :, j, :] = v
            reference[j, :, i, :] = v.T
    torch.testing.assert_close(rows, reference.sum(2))
    output = finish(indices, values, ptr, diagonal, rows, False)
    torch.testing.assert_close(output.to_dense(), reference.reshape(n * 3, n * 3))
    for i, j in [(0, 1), (1, 0), (n - 1, n - 2)]:
        torch.testing.assert_close(output.block(i, j), reference[i, :, j, :])


@torch.no_grad()
def test_direct_message_accumulation_matches_independent_rotation():
    """
    Accumulate unsorted chunks without clearing the pre-existing node buffer.
    """
    cuda_required()
    from fairchem.core.frozen_hip.scatter import accumulate_rotated_messages

    torch.manual_seed(42)
    n, e, c = 17, 97, 19
    edges = torch.randint(n, (2, e), device="cuda")
    x = torch.randn(e, 9, c, device="cuda")
    w = torch.zeros(e, 9, 9, device="cuda")
    for lo, hi in [(0, 1), (1, 4), (4, 9)]:
        w[:, lo:hi, lo:hi] = torch.randn(e, hi - lo, hi - lo, device="cuda")
    initial = torch.randn(n, 9, c, device="cuda")
    expected = initial.clone()
    rotated = torch.bmm(w, x[:, [0, 5, 1, 3, 8, 6, 2, 4, 7]])
    expected.index_add_(0, edges[1], rotated)
    actual = initial.clone()
    for begin in range(0, e, 13):
        accumulate_rotated_messages(
            actual,
            x[begin : begin + 13],
            w[begin : begin + 13],
            edges[:, begin : begin + 13],
        )
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("fast", [False, True])
@torch.no_grad()
def test_neighbor_head_preserves_reference_and_cuda_rng(fast):
    """
    Compare the unchanged network and exact random frames across both graph paths.
    """
    cuda_required()
    from ase.build import molecule

    from fairchem.core.datasets.atomic_data import AtomicData, atomicdata_list_to_batch
    from fairchem.core.frozen_hip.model import FrozenHIP
    from fairchem.core.frozen_hip.precision import configure_head_precision
    from fairchem.core.frozen_hip.wigner import enable_fused_wigner

    torch.manual_seed(29)
    model = FrozenHIP("cuda", layers=1, backbone_graph_version=4).eval()
    atoms = molecule("C6H6")
    atoms.info.update(charge=0, spin=1)
    data = atomicdata_list_to_batch(
        [AtomicData.from_ase(atoms, task_name="omol", r_data_keys=["charge", "spin"])]
    )
    pbc = data.pbc
    data.apply(lambda t: t if t is pbc else t.to("cuda"))
    data.hip_sizes = [len(atoms)]
    data.hip_atom_ptr = torch.tensor([0, len(atoms)], device="cuda")
    emb = model.features(copy.deepcopy(data), "fp16" if fast else "fp32")
    if fast:
        configure_head_precision(model.head, "fp16-edge")
    enable_fused_wigner(model.head)
    for cutoff in [None, 1e-6, 1.8, 6.0]:
        state = torch.cuda.get_rng_state()
        expected, count = predict(
            model.head,
            data,
            emb,
            chunk_edges=7,
            cutoff=cutoff,
            output_format="blocks",
            neighbor_list=False,
            direct_messages=False,
        )
        after = torch.cuda.get_rng_state()
        for storage in ["device", "cpu"]:
            torch.cuda.set_rng_state(state)
            actual, other = predict(
                model.head,
                data,
                emb,
                chunk_edges=7,
                cutoff=cutoff,
                output_format="blocks",
                angle_storage=storage,
            )
            assert other == count
            assert torch.equal(torch.cuda.get_rng_state(), after)
            dense = expected.to_dense()
            delta = actual.to_dense() - dense
            relative = torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(
                dense
            ).clamp_min(1e-20)
            assert float(relative) < (2e-3 if fast else 2e-5), (cutoff, float(relative))
