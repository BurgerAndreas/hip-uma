"""Regressions for upstream inference fixes on the HIP training fork."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from fairchem.core.graph import radius_graph_pbc_nvidia as graph
from fairchem.core.models.base import HydraInterfaceMixin
from fairchem.core.models.uma.escn_md import (
    GradRegressConfig,
    MLP_EFS_Head,
    eSCNMDBackbone,
)
from fairchem.core.models.uma.nn.execution_backends import UMASFastGPUBackend
from fairchem.core.units.mlip_unit import InferenceSettings
from fairchem.core.units.mlip_unit.predict import (
    MLIPPredictUnit,
    _prepare_inference_gradients,
)
from fairchem.core.units.mlip_unit.utils import tf32_context_manager


class QuarticEnergy(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float64))

    def forward(self, features):
        return self.scale * features.pow(4).sum(-1, keepdim=True)


@pytest.mark.parametrize("vmap", [False, True])
@pytest.mark.parametrize("stress", [False, True])
@pytest.mark.parametrize("training", [False, True])
def test_hessian_values_and_derivative_graph(vmap, stress, training):
    config = GradRegressConfig(
        forces=True, stress=stress, hessian=True, hessian_vmap=vmap
    )
    backbone = SimpleNamespace(
        sphere_channels=3, hidden_channels=4, regress_config=config
    )
    head = MLP_EFS_Head(backbone, wrap_property=False)
    head.energy_block = QuarticEnergy()
    head.train(training)
    pos = torch.tensor(
        [[0.2, -0.3, 0.4], [0.5, 0.7, -0.2]], dtype=torch.float64, requires_grad=True
    )
    cell = torch.eye(3, dtype=torch.float64).unsqueeze(0).requires_grad_(True)
    features = pos + 0.1 * cell.sum()
    batch = torch.zeros(2, dtype=torch.long)
    data = {
        "pos": pos,
        "cell": cell,
        "batch": batch,
        "batch_full": batch,
        "natoms": torch.tensor([2]),
    }
    output = head(data, {"node_embedding": features.unsqueeze(1)})
    expected = torch.diag(12 * features.detach().square().flatten())
    torch.testing.assert_close(output["hessian"].reshape(6, 6), expected)
    assert output["hessian"].requires_grad is training
    if training:
        # A Hessian loss must still train the energy head.
        (gradient,) = torch.autograd.grad(
            output["hessian"].sum(), head.energy_block.scale
        )
        torch.testing.assert_close(gradient, expected.sum())


@pytest.mark.parametrize("vmap", [False, True])
def test_hessian_setting_survives_task_configuration(vmap):
    settings = InferenceSettings(hessian_vmap=vmap)
    overrides = eSCNMDBackbone.build_inference_settings(settings)
    assert overrides["hessian_vmap"] is vmap
    config = GradRegressConfig(hessian_vmap=overrides["hessian_vmap"])
    model = SimpleNamespace(backbone=SimpleNamespace(regress_config=config))
    HydraInterfaceMixin._configure_gradient_for_task(
        model, SimpleNamespace(property="hessian")
    )
    assert config.hessian_vmap is vmap
    assert config.hessian
    assert config.forces


@pytest.mark.parametrize(
    "direct,forces,stress",
    [
        (False, False, False),
        (False, True, False),
        (False, True, True),
        (True, True, True),
    ],
)
def test_prepare_gradient_metadata(direct, forces, stress):
    backbone = SimpleNamespace(
        regress_config=GradRegressConfig(
            direct_forces=direct, forces=forces, stress=stress
        )
    )
    data = {"pos": torch.randn(2, 3), "cell": torch.eye(3).unsqueeze(0)}
    _prepare_inference_gradients(backbone, data)
    assert data["pos"].requires_grad is (not direct and (forces or stress))
    assert data["cell"].requires_grad is (not direct and stress)
    _prepare_inference_gradients(SimpleNamespace(), data)


@pytest.mark.parametrize("tf32", [False, True])
def test_predict_applies_precision_and_restores_after_failure(tf32):
    class CheckingModel:
        module = SimpleNamespace(
            backbone=SimpleNamespace(regress_config=SimpleNamespace(direct_forces=True))
        )

        def __call__(self, data):
            assert torch.get_float32_matmul_precision() == (
                "high" if tf32 else "highest"
            )
            assert torch.backends.cudnn.allow_tf32 is tf32
            raise RuntimeError("test failure")

    predictor = MLIPPredictUnit.__new__(MLIPPredictUnit)
    predictor.inference_settings = InferenceSettings(tf32=tf32)
    predictor.model = CheckingModel()
    initial = torch.get_float32_matmul_precision()
    initial_cudnn = torch.backends.cudnn.allow_tf32
    try:
        torch.set_float32_matmul_precision("highest" if tf32 else "high")
        torch.backends.cudnn.allow_tf32 = not tf32
        with pytest.raises(RuntimeError, match="test failure"):
            predictor._run_inference(None, False)
        assert torch.get_float32_matmul_precision() == ("highest" if tf32 else "high")
        assert torch.backends.cudnn.allow_tf32 is not tf32
        with tf32_context_manager():
            assert torch.get_float32_matmul_precision() == "high"
    finally:
        torch.set_float32_matmul_precision(initial)
        torch.backends.cudnn.allow_tf32 = initial_cudnn


@pytest.mark.parametrize("periodic", [False, True])
def test_neighbor_list_cell_contract(monkeypatch, periodic):
    called = []

    def neighbor_list(**kwargs):
        called.append(True)
        assert (kwargs["cell"] is not None) is periodic
        assert (kwargs["pbc"] is not None) is periodic
        # Return one edge in each direction for two atoms.
        kwargs["neighbor_matrix"][:, 0] = torch.tensor([1, 0], dtype=torch.int32)
        kwargs["num_neighbors"].fill_(1)

    monkeypatch.setattr(graph, "neighbor_list", neighbor_list)
    monkeypatch.setattr(graph, "estimate_max_neighbors", lambda **kwargs: 4)
    # This test checks the call contract without the optional CUDA dependency.
    implementation = inspect.unwrap(graph.get_neighbors_nvidia)
    result = implementation(
        torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        torch.eye(3).unsqueeze(0) * 10,
        torch.full((1, 3), periodic, dtype=torch.bool),
        cutoff=2.0,
        max_neigh=4,
    )
    assert called == [True]
    assert result[0].numel() == 2


@pytest.mark.parametrize("vmap", [False, True])
def test_fast_backend_rejects_unsupported_hessian_vmap(monkeypatch, vmap):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    settings = InferenceSettings(
        merge_mole=True,
        activation_checkpointing=False,
        predict_untrained_hessian={"omol"},
        hessian_vmap=vmap,
    )
    if vmap:
        with pytest.raises(ValueError, match="set hessian_vmap=False"):
            UMASFastGPUBackend.validate(2, 2, settings)
    else:
        UMASFastGPUBackend.validate(2, 2, settings)
