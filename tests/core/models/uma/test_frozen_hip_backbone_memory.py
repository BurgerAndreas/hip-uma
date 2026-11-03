"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch

from fairchem.core.frozen_hip.backbone_memory import (
    chunked_backbone,
    enable_backbone_chunking,
    validate_charge_spin,
)
from fairchem.core.frozen_hip.model import FrozenHIP, parameter_hash
from fairchem.core.models.uma.nn.embedding import ChgSpinEmbedding


def test_charge_spin_validation_before_device_lookup():
    """
    Preserve supported embeddings and reject invalid physical inputs on the CPU.
    """
    backbone = SimpleNamespace(
        charge_embedding=ChgSpinEmbedding("rand_emb", "charge", 4, False),
        spin_embedding=ChgSpinEmbedding("rand_emb", "spin", 4, False),
    )
    data = SimpleNamespace(
        charge=torch.tensor([-100, 100]), spin=torch.tensor([0, 100])
    )
    before = backbone.charge_embedding(data.charge).clone()
    validate_charge_spin(backbone, data)
    torch.testing.assert_close(before, backbone.charge_embedding(data.charge))
    for values, field in [
        ([-101], "charge"),
        ([101], "charge"),
        ([1.5], "charge"),
        ([-1], "spin"),
        ([101], "spin"),
    ]:
        bad = SimpleNamespace(charge=torch.tensor([0]), spin=torch.tensor([1]))
        setattr(bad, field, torch.tensor(values))
        with pytest.raises(ValueError, match=f"UMA {field} embedding"):
            validate_charge_spin(backbone, bad)


def methane_data():
    """
    Build one charged/spin-labelled molecule with host-side PBC flags.
    """
    from ase.build import molecule

    from fairchem.core.datasets.atomic_data import AtomicData, atomicdata_list_to_batch

    atoms = molecule("CH4")
    atoms.info.update(charge=0, spin=1)
    data = atomicdata_list_to_batch(
        [AtomicData.from_ase(atoms, task_name="omol", r_data_keys=["spin", "charge"])]
    )
    pbc = data.pbc
    data.apply(lambda t: t if t is pbc else t.to("cuda"))
    data.hip_sizes = [len(atoms)]
    data.hip_atom_ptr = torch.tensor([0, len(atoms)], device="cuda")
    return data


@pytest.mark.parametrize("merged", [False, True])
@pytest.mark.parametrize("precision", ["fp32", "fp16"])
def test_backbone_chunking_preserves_features_and_weights(merged, precision):
    """
    Compare merged and unmerged backbones with partial chunks in both precisions.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from fairchem.core.frozen_hip.merged_backbone import merge_for_batch

    torch.manual_seed(39)
    model = FrozenHIP("cuda", layers=1, backbone_graph_version=4).eval()
    data = methane_data()
    if merged:
        template = copy.deepcopy(model.backbone).cpu()
        merge_for_batch(model, template, copy.deepcopy(data), tf32=False)
    before = parameter_hash(model.backbone)
    state = torch.cuda.get_rng_state()
    reference = model.features(copy.deepcopy(data), precision)["node_embedding"]
    torch.cuda.set_rng_state(state)
    with chunked_backbone(model.backbone, 3):
        actual = model.features(copy.deepcopy(data), precision)["node_embedding"]
    relative = torch.linalg.vector_norm(actual - reference) / torch.linalg.vector_norm(
        reference
    )
    assert float(relative) < (2e-3 if precision == "fp16" else 2e-5)
    assert parameter_hash(model.backbone) == before
    assert not hasattr(model.backbone.edge_degree_embedding, "_hip_edge_chunk_size")
    enable_backbone_chunking(model.backbone, 5)
    with (
        pytest.raises(RuntimeError, match="exercise restoration"),
        chunked_backbone(model.backbone, 3),
    ):
        raise RuntimeError("exercise restoration")
    assert model.backbone.edge_degree_embedding._hip_edge_chunk_size == 5
    enable_backbone_chunking(model.backbone, None)
    for invalid in [0, -1, 2.5, True]:
        with pytest.raises(ValueError, match="positive integer"):
            enable_backbone_chunking(model.backbone, invalid)


def test_cpu_angle_storage_preserves_hessian():
    """
    Copy exact random angles to host memory without changing the predicted blocks.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    model = FrozenHIP("cuda", layers=1, backbone_graph_version=4).eval()
    data = methane_data()
    for cutoff in [None, 1.5]:
        state = torch.cuda.get_rng_state()
        reference, count = model.predict_hessian(
            copy.deepcopy(data),
            chunk_edges=3,
            backbone_chunk_edges=3,
            pair_cutoff=cutoff,
            output_format="blocks",
            angle_storage="device",
        )
        torch.cuda.set_rng_state(state)
        actual, other = model.predict_hessian(
            copy.deepcopy(data),
            chunk_edges=3,
            backbone_chunk_edges=3,
            pair_cutoff=cutoff,
            output_format="blocks",
            angle_storage="cpu",
        )
        assert other == count
        torch.testing.assert_close(
            actual.to_dense(), reference.to_dense(), rtol=1e-5, atol=1e-5
        )


def test_inplace_angle_scaling_preserves_random_values():
    """
    Remove the duplicate random-angle allocation without changing its values.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    import math

    state = torch.cuda.get_rng_state()
    expected = torch.rand(65537, device="cuda") * (2 * math.pi)
    torch.cuda.set_rng_state(state)
    actual = torch.rand(65537, device="cuda")
    actual.mul_(2 * math.pi)
    assert torch.equal(actual, expected)
