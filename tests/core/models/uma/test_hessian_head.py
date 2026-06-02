"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import torch
from ase import Atoms

from fairchem.core.datasets.atomic_data import AtomicData, atomicdata_list_to_batch
from fairchem.core.models.base import HydraModel
from fairchem.core.models.uma.escn_md import Hessian_Head, eSCNMDBackbone


def _sample(symbols: str, positions: list[list[float]]) -> AtomicData:
    return AtomicData.from_ase(
        Atoms(symbols=symbols, positions=positions),
        task_name="horm",
        r_edges=False,
        r_data_keys=["spin", "charge"],
    )


def _rotation_matrix(dtype: torch.dtype = torch.float32) -> torch.Tensor:
    axis = torch.tensor([0.3, -0.7, 0.2], dtype=dtype)
    axis = axis / axis.norm()
    angle = torch.tensor(0.71, dtype=dtype)
    cross = torch.tensor(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ],
        dtype=dtype,
    )
    identity = torch.eye(3, dtype=dtype)
    return (
        identity
        + torch.sin(angle) * cross
        + (1.0 - torch.cos(angle)) * (cross @ cross)
    )


def _rotate_positions(
    positions: list[list[float]], rotation: torch.Tensor
) -> list[list[float]]:
    pos = torch.tensor(positions, dtype=rotation.dtype)
    return (pos @ rotation.T).tolist()


def _transform_hessian(
    hessian_flat: torch.Tensor, natoms: int, rotation: torch.Tensor
) -> torch.Tensor:
    transform = torch.kron(
        torch.eye(natoms, device=hessian_flat.device, dtype=hessian_flat.dtype),
        rotation.to(device=hessian_flat.device, dtype=hessian_flat.dtype),
    )
    hessian = hessian_flat.view(natoms * 3, natoms * 3)
    return (transform @ hessian @ transform.T).reshape(-1)


def _tiny_backbone() -> eSCNMDBackbone:
    return eSCNMDBackbone(
        max_num_elements=100,
        sphere_channels=4,
        lmax=2,
        mmax=2,
        otf_graph=True,
        edge_channels=5,
        num_distance_basis=7,
        num_layers=1,
        hidden_channels=4,
        use_dataset_embedding=False,
        always_use_pbc=False,
    )


def _tiny_hydra_model() -> HydraModel:
    return HydraModel(
        backbone={
            "model": "fairchem.core.models.uma.escn_md.eSCNMDBackbone",
            "max_num_elements": 100,
            "sphere_channels": 4,
            "lmax": 2,
            "mmax": 2,
            "otf_graph": True,
            "edge_channels": 5,
            "num_distance_basis": 7,
            "num_layers": 1,
            "hidden_channels": 4,
            "use_dataset_embedding": False,
            "always_use_pbc": False,
        },
        heads={
            "hessian": {
                "module": "fairchem.core.models.uma.escn_md.Hessian_Head",
                "num_layers_hessian": 1,
                "cutoff_hessian": 10.0,
                "fully_connected_hessian": True,
                "use_pbc_hessian": False,
            }
        },
    )


def test_hessian_head_forward_variable_natoms_batch():
    torch.manual_seed(5)
    backbone = _tiny_backbone()
    head = Hessian_Head(
        backbone,
        num_layers_hessian=1,
        cutoff_hessian=10.0,
        fully_connected_hessian=True,
        use_pbc_hessian=False,
    )
    batch = atomicdata_list_to_batch(
        [
            _sample("H2", [[0.0, 0.0, 0.0], [0.8, 0.0, 0.0]]),
            _sample(
                "H2O",
                [[0.0, 0.0, 0.0], [0.8, 0.0, 0.0], [0.0, 0.8, 0.0]],
            ),
        ]
    )

    emb = backbone(batch)
    out = head(batch, emb)

    assert len(head.blocks) == 1
    assert head.edge_readout is not backbone.blocks[-1].edge_wise
    assert head.blocks[0] is not backbone.blocks[-1]
    assert out["hessian"].shape == (117,)
    assert batch.edge_index_hessian.shape == (2, 8)
    assert batch.nedges_hessian.tolist() == [2, 6]
    for sample_idx, natoms in enumerate(batch.natoms.tolist()):
        start = batch.ptr_1d_hessian[sample_idx]
        stop = batch.ptr_1d_hessian[sample_idx + 1]
        hessian = out["hessian"][start:stop].view(natoms * 3, natoms * 3)
        assert torch.allclose(hessian, hessian.T)


def test_hydra_model_hessian_head_forward():
    torch.manual_seed(5)
    model = _tiny_hydra_model()
    batch = atomicdata_list_to_batch(
        [
            _sample("H2", [[0.0, 0.0, 0.0], [0.8, 0.0, 0.0]]),
            _sample(
                "H2O",
                [[0.0, 0.0, 0.0], [0.8, 0.0, 0.0], [0.0, 0.8, 0.0]],
            ),
        ]
    )

    out = model(batch)

    assert out["hessian"]["hessian"].shape == (117,)
    assert batch.edge_index_hessian.shape == (2, 8)
    assert batch.nedges_hessian.tolist() == [2, 6]
    for sample_idx, natoms in enumerate(batch.natoms.tolist()):
        start = batch.ptr_1d_hessian[sample_idx]
        stop = batch.ptr_1d_hessian[sample_idx + 1]
        hessian = out["hessian"]["hessian"][start:stop].view(natoms * 3, natoms * 3)
        assert torch.allclose(hessian, hessian.T)


def test_hessian_head_rotation_equivariance():
    torch.manual_seed(5)
    positions = [[0.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.2, 0.8, 0.3]]
    rotation = _rotation_matrix()
    backbone = _tiny_backbone()
    head = Hessian_Head(
        backbone,
        cutoff_hessian=10.0,
        fully_connected_hessian=True,
        use_pbc_hessian=False,
    )
    batch = atomicdata_list_to_batch([_sample("H2O", positions)])
    rotated_batch = atomicdata_list_to_batch(
        [_sample("H2O", _rotate_positions(positions, rotation))]
    )

    hessian = head(batch, backbone(batch))["hessian"]
    rotated_hessian = head(rotated_batch, backbone(rotated_batch))["hessian"]

    expected = _transform_hessian(hessian, natoms=3, rotation=rotation)
    assert torch.allclose(rotated_hessian, expected, atol=2e-4, rtol=2e-4)


def test_hydra_model_hessian_head_rotation_equivariance():
    torch.manual_seed(5)
    positions = [[0.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.2, 0.8, 0.3]]
    rotation = _rotation_matrix()
    model = _tiny_hydra_model()
    batch = atomicdata_list_to_batch([_sample("H2O", positions)])
    rotated_batch = atomicdata_list_to_batch(
        [_sample("H2O", _rotate_positions(positions, rotation))]
    )

    hessian = model(batch)["hessian"]["hessian"]
    rotated_hessian = model(rotated_batch)["hessian"]["hessian"]

    expected = _transform_hessian(hessian, natoms=3, rotation=rotation)
    assert torch.allclose(rotated_hessian, expected, atol=2e-4, rtol=2e-4)
