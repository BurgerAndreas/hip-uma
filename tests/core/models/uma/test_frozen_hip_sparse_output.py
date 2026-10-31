"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import tempfile

import torch

from fairchem.core.frozen_hip.sparse_output import (
    HessianBlocks,
    accumulate,
    allocate,
    finish,
)
from fairchem.core.frozen_hip.streamed_head import pair_chunk, predict
from fairchem.core.frozen_hip.translation import project_dense


@torch.no_grad()
def check_blocks():
    from types import SimpleNamespace

    torch.manual_seed(31)
    results = []
    for n in (1, 7, 19):
        data = SimpleNamespace(pos=torch.randn(n, 3, device="cuda"))
        for cutoff in (None, 0.000001, 1.8, 100):
            diagonal = torch.randn(n, 3, 3, device="cuda")
            diagonal = diagonal + diagonal.transpose(-1, -2)
            index, value, ptr, rows = allocate(data, diagonal, 5, cutoff)
            reference = torch.zeros(3 * n, 3 * n, device="cuda")
            ids = torch.arange(n, device="cuda")
            reference.view(n, 3, n, 3)[ids, :, ids, :] = diagonal
            for start in range(0, n * (n - 1), 5):
                edge, _, _, _ = pair_chunk(
                    data.pos, start, min(start + 5, n * (n - 1)), cutoff
                )
                if edge.shape[1] == 0:
                    continue
                blocks = torch.randn(edge.shape[1], 3, 3, device="cuda")
                accumulate(edge, blocks, index, value, ptr, rows, n, cutoff is None)
                for k in range(edge.shape[1]):
                    i, j = edge[:, k].tolist()
                    reference[3 * i : 3 * i + 3, 3 * j : 3 * j + 3] += blocks[k]
                    reference[3 * j : 3 * j + 3, 3 * i : 3 * i + 3] += blocks[k].T
            out = finish(index, value, ptr, diagonal, rows, True)
            reference = project_dense(reference, [n])
            torch.testing.assert_close(out.to_dense(), reference, rtol=1e-5, atol=1e-5)
            v = torch.randn(3 * n, device="cuda")
            torch.testing.assert_close(
                out.matvec(v), reference @ v, rtol=2e-5, atol=2e-5
            )
            torch.testing.assert_close(
                out.block(0, n - 1), reference[:3, -3:], rtol=1e-5, atol=1e-5
            )
            assert out.finite()
            with tempfile.NamedTemporaryFile(suffix=".pt") as f:
                out.save(f.name)
                restored = HessianBlocks.load(f.name, map_location="cuda")
                torch.testing.assert_close(restored.matvec(v), out.matvec(v))
            results.append(dict(n=n, cutoff=cutoff, pairs=len(value), passed=True))
    return results


@torch.no_grad()
def verify_head(head, data, emb):
    results = []
    for cutoff in (None, 20, 60):
        rng = torch.cuda.get_rng_state()
        dense, count = predict(head, data, emb, chunk_edges=65536, cutoff=cutoff)
        torch.cuda.set_rng_state(rng)
        sparse, other = predict(
            head, data, emb, chunk_edges=65536, cutoff=cutoff, output_format="blocks"
        )
        assert count == other
        actual = sparse.to_dense()
        error = float(
            torch.linalg.vector_norm(actual - dense)
            / torch.linalg.vector_norm(dense).clamp_min(1e-30)
        )
        assert error < 2e-5, error
        v = torch.randn(dense.shape[0], device="cuda")
        mv = sparse.matvec(v)
        ref = dense @ v
        mv_error = float(
            torch.linalg.vector_norm(mv - ref)
            / torch.linalg.vector_norm(ref).clamp_min(1e-30)
        )
        assert mv_error < 2e-5, mv_error
        results.append(
            dict(
                cutoff=cutoff,
                relative_frobenius=error,
                matvec_relative=mv_error,
                stored_pairs=len(sparse.values),
                storage_bytes=sparse.storage_bytes,
            )
        )
    return results


def test_sparse_block_kernels_and_serialization():
    """
    Compare independent dense assembly, block lookup, matvec, and serialization.
    """
    import pytest

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    assert len(check_blocks()) == 12


def test_public_chunked_api_matches_dense_and_sparse():
    """
    Exercise the package API on the cached pretrained backbone and a fresh head.
    """
    import pytest
    from ase.build import molecule

    from fairchem.core.datasets.atomic_data import (
        AtomicData,
        atomicdata_list_to_batch,
    )
    from fairchem.core.frozen_hip.model import FrozenHIP

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    model = FrozenHIP("cuda", layers=1, backbone_graph_version=4).eval()
    atoms = molecule("CH4")
    atoms.info.update(charge=0, spin=1)
    data = atomicdata_list_to_batch(
        [AtomicData.from_ase(atoms, task_name="omol", r_data_keys=["spin", "charge"])]
    )
    pbc = data.pbc
    data.apply(lambda t: t if t is pbc else t.to("cuda"))
    for cutoff in (None, 1.5):
        state = torch.cuda.get_rng_state()
        dense, count = model.predict_hessian(data, chunk_edges=5, pair_cutoff=cutoff)
        torch.cuda.set_rng_state(state)
        sparse, other = model.predict_hessian(
            data, chunk_edges=5, pair_cutoff=cutoff, output_format="blocks"
        )
        assert count == other
        torch.testing.assert_close(sparse.to_dense(), dense, rtol=2e-5, atol=2e-5)
        assert model.backbone.radius_pbc_version == 4
    with pytest.raises(ValueError, match="eval"):
        model.train().predict_hessian(data)
