"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import torch

from fairchem.core.datasets.atomic_data import AtomicData, atomicdata_list_to_batch
from fairchem.core.modules.loss import HessianLoss
from fairchem.core.modules.normalization.normalizer import Normalizer
from fairchem.core.units.mlip_unit.mlip_unit import (
    OutputSpec,
    Task,
    compute_loss,
    compute_metrics,
)


def _hessian_data(num_atoms: int) -> AtomicData:
    dtype = torch.float32
    hessian = torch.eye(num_atoms * 3, dtype=dtype).reshape(-1)
    return AtomicData.from_dict(
        {
            "pos": torch.arange(num_atoms * 3, dtype=dtype).view(num_atoms, 3) / 10,
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
            "hessian": hessian,
        }
    )


def _hessian_task(metrics: list[str] | None = None) -> Task:
    return Task(
        name="hessian",
        level="system",
        property="hessian",
        out_spec=OutputSpec(dim=[1], dtype="float32"),
        normalizer=Normalizer(mean=0.0, rmsd=1.0),
        datasets=["horm"],
        loss_fn=HessianLoss(mode="mae", reduction="structure_mean"),
        metrics=metrics or [],
    )


def test_compute_loss_supports_flattened_hessian_targets():
    batch = atomicdata_list_to_batch([_hessian_data(2), _hessian_data(3)])
    batch.dataset_name = ["horm", "horm"]
    pred = (batch.hessian + 2.0).clone().requires_grad_(True)
    predictions = {"hessian": {"hessian": pred}}

    loss = compute_loss([_hessian_task()], predictions, batch)["hessian"]

    assert torch.allclose(loss, torch.tensor(2.0))
    loss.backward()
    assert pred.grad is not None


def test_compute_metrics_supports_hessian_entry_metrics():
    batch = atomicdata_list_to_batch([_hessian_data(2), _hessian_data(3)])
    batch.dataset_name = ["horm", "horm"]
    predictions = {"hessian": {"hessian": batch.hessian + 2.0}}
    task = _hessian_task(metrics=["hessian_mae", "hessian_mse", "hessian_rmse"])

    metrics = compute_metrics(task, predictions, batch)

    assert metrics["hessian_mae"].metric == 2.0
    assert metrics["hessian_mse"].metric == 4.0
    assert metrics["hessian_rmse"].metric == 2.0


def test_compute_metrics_supports_eckart_eigenspectrum_metrics():
    dtype = torch.float32
    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [0.9, 0.0, 0.0], [0.0, 1.1, 0.0]], dtype=dtype
    )
    hessian = torch.eye(9, dtype=dtype).reshape(-1)
    data = AtomicData.from_dict(
        {
            "pos": pos,
            "atomic_numbers": torch.tensor([8, 1, 1], dtype=torch.long),
            "cell": torch.eye(3, dtype=dtype).view(1, 3, 3) * 20,
            "pbc": torch.zeros(1, 3, dtype=torch.bool),
            "natoms": torch.tensor([3], dtype=torch.long),
            "edge_index": torch.empty(2, 0, dtype=torch.long),
            "cell_offsets": torch.empty(0, 3, dtype=dtype),
            "nedges": torch.tensor([0], dtype=torch.long),
            "charge": torch.zeros(1, dtype=torch.long),
            "spin": torch.zeros(1, dtype=torch.long),
            "fixed": torch.zeros(3, dtype=torch.long),
            "tags": torch.ones(3, dtype=torch.long),
            "hessian": hessian,
        }
    )
    batch = atomicdata_list_to_batch([data])
    batch.dataset_name = ["horm"]
    predictions = {"hessian": {"hessian": batch.hessian.clone()}}
    task = _hessian_task(
        metrics=[
            "hessian_eckart_eigenvalue_mae",
            "hessian_eckart_eigvec1_cos",
            "hessian_eckart_eigvec2_cos",
        ]
    )

    metrics = compute_metrics(task, predictions, batch)

    assert metrics["hessian_eckart_eigenvalue_mae"].metric == 0.0
    assert torch.isclose(
        torch.tensor(metrics["hessian_eckart_eigvec1_cos"].metric), torch.tensor(1.0)
    )
    assert torch.isclose(
        torch.tensor(metrics["hessian_eckart_eigvec2_cos"].metric), torch.tensor(1.0)
    )
