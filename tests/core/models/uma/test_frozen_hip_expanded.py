"""Check widened heads, exact graph accumulation, and release holdouts."""

from __future__ import annotations

import copy
import json

import pytest
import torch

from fairchem.core.datasets.atomic_data import atomicdata_list_to_batch
from fairchem.core.frozen_hip.graphs import TrainingGraphs
from fairchem.core.frozen_hip.validation import load_manifest, sample_fingerprint
from fairchem.core.models.uma.escn_md import Hessian_Head
from tests.core.models.uma.test_hessian_head import _sample, _tiny_backbone


@pytest.mark.parametrize("layers", [1, 2])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_wider_head_preserves_shape_and_backbone(layers):
    backbone = _tiny_backbone().cuda().eval().requires_grad_(False)
    original = {k: p.detach().clone() for k, p in backbone.named_parameters()}
    head = Hessian_Head(backbone, num_layers_hessian=layers, width_multiplier=2).cuda()
    data = atomicdata_list_to_batch([_sample("H2", [[0, 0, 0], [0, 0, 0.8]])]).cuda()
    with torch.no_grad():
        emb = backbone(data)
    output = head(data, emb)["hessian"]
    assert output.numel() == 36
    output.square().mean().backward()
    assert head.input_projection.weight.grad.abs().sum() > 0
    assert len(head.blocks) == layers
    for k, p in backbone.named_parameters():
        assert p.grad is None
        torch.testing.assert_close(p, original[k], rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_graph_inputs_accumulation_and_learning_rates():

    torch.manual_seed(7)
    model = torch.nn.Linear(3, 2).cuda()
    reference = copy.deepcopy(model)
    opt = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0, fused=True)
    refopt = torch.optim.AdamW(
        reference.parameters(), lr=0.01, weight_decay=0, fused=True
    )

    def loss(d):
        return (model(d.hip_features) - d.target).square().mean()

    runner = TrainingGraphs(model.parameters(), loss, capacity=1)
    for step in range(9):
        opt.zero_grad(set_to_none=True)
        refopt.zero_grad(set_to_none=True)
        for _micro in range(2):
            d = atomicdata_list_to_batch(
                [_sample("H2", [[0, 0, 0], [0, 0, 0.8]])]
            ).cuda()
            d.hip_features = torch.randn(4, 3, device="cuda")
            d.target = torch.randn(4, 2, device="cuda")
            d.hip_sizes = [2, 2]
            v = runner.backward(d, 512.0)
            if v is None:
                (loss(d) * 512.0).backward()
            ((reference(d.hip_features) - d.target).square().mean() * 512).backward()
        for p in model.parameters():
            p.grad.div_(1024)
        for p in reference.parameters():
            p.grad.div_(1024)
        for a, b in zip(model.parameters(), reference.parameters(), strict=True):
            torch.testing.assert_close(a.grad, b.grad, rtol=1e-5, atol=1e-6)
        for o in [opt, refopt]:
            o.param_groups[0]["lr"] = 0.01 / (step + 1)
            o.step()
    assert runner.captures == 1
    assert runner.hits >= 10
    for a, b in zip(model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
    restored = copy.deepcopy(model)
    restored.load_state_dict(model.state_dict())
    assert len(runner.graphs) == 1


def test_release_validation_namespace_and_fingerprint(tmp_path):
    ids = [(1 << 60) + 4, (1 << 60) + 100]
    obj = dict(
        selection="release-holdout-v2",
        samples=2,
        sample_ids=ids,
        sample_ids_sha256=sample_fingerprint(ids),
        dataset_revision="a" * 40,
        entries=[],
        complete=False,
    )
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(obj))
    assert load_manifest(path)["samples"] == 2
    obj["sample_ids"][0] = 4
    path.write_text(json.dumps(obj))
    with pytest.raises(ValueError):
        load_manifest(path)


def test_published_upper_labels_and_parent_namespaces(tmp_path):
    import h5py
    import numpy as np

    from tools.prepare_frozen_hip_release import selected_arrays

    path = tmp_path / "shard.h5"
    dense = np.arange(36, dtype=np.float32).reshape(6, 6)
    dense = (dense + dense.T) * 0.5
    with h5py.File(path, "w") as f:
        f.attrs.update(
            model="uma-s-1p2",
            hessian_unit="eV/Angstrom^2",
            parent_partition=0,
            split="train",
            complete=True,
        )
        for k, v in dict(
            omol_index=[4, 100],
            natoms=[2, 2],
            atom_ptr=[0, 2, 4],
            hessian_ptr=[0, 21, 42],
            charge=[0, 0],
            spin=[1, 1],
            atomic_numbers=[1, 1, 1, 1],
            coords=np.zeros((4, 3), np.float32),
            hessian_upper=np.tile(dense[np.triu_indices(6)], 2),
        ).items():
            f[k] = v
    record = dict(parent_partition=0, split="train", samples=2)
    a = selected_arrays(path, record, np.array([4]), np.array([4, 100]))
    assert a["omol_index"].tolist() == [100]
    np.testing.assert_array_equal(a["hessian_flat"].reshape(6, 6), dense)
    with h5py.File(path, "a") as f:
        f.attrs.update(parent_partition=1, split="validation")
    record.update(parent_partition=1, split="validation")
    a = selected_arrays(path, record, np.array([4]), np.array([4, 100]))
    assert a["omol_index"].tolist() == [(1 << 60) + 4, (1 << 60) + 100]
    with pytest.raises(ValueError):
        selected_arrays(path, record, np.array([4]), np.array([4]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_graph_replays_changed_molecular_partitions():
    import numpy as np

    from fairchem.core.frozen_hip.packed import block_rmse, make_packed_batch
    from fairchem.core.frozen_hip.precision import configure_head_precision
    from fairchem.core.frozen_hip.radial import enable_fused_radial
    from fairchem.core.frozen_hip.readout import enable_projected_readout
    from fairchem.core.frozen_hip.rotations import (
        enable_fused_rotations,
        enable_fused_scatter,
    )
    from fairchem.core.frozen_hip.so2 import enable_block_so2

    backbone = _tiny_backbone().cuda().eval().requires_grad_(False)
    head = Hessian_Head(backbone, width_multiplier=2).cuda()
    head.requires_mole_routing = False
    enable_fused_rotations(head)
    enable_fused_scatter(head, segmented=True)
    enable_block_so2(head)
    enable_projected_readout(head)
    enable_fused_radial(head)
    configure_head_precision(head, "fp16-edge")

    def batch(sizes):
        sizes = np.array(sizes)
        atoms = sizes.sum()
        pairs = 9 * sizes * (sizes - 1) // 2
        rng = np.random.default_rng(int(sizes[0]))
        arrays = dict(
            natoms=sizes,
            atom_ptr=np.r_[0, np.cumsum(sizes)],
            pair_ptr=np.r_[0, np.cumsum(pairs)],
            omol_index=np.arange(3),
            atomic_numbers=np.ones(atoms),
            coords=rng.normal(size=(atoms, 3)),
            charge=np.zeros(3),
            spin=np.ones(3),
            features=rng.normal(size=(atoms, 9, 4)).astype(np.float32),
            pair_targets=np.zeros(pairs.sum(), np.float32),
            diagonal_targets=np.zeros(6 * atoms, np.float32),
            asymmetry_mse=np.zeros(3, np.float32),
        )
        d = make_packed_batch(arrays, np.arange(3))
        d.hip_sizes = sizes.tolist()
        d = d.cuda()
        d.hip_fused_loss = True
        return d

    first, second = batch([2, 5, 5]), batch([3, 3, 6])

    def loss(d):
        p = head(d, {"node_embedding": d.hip_features, "batch": d.batch})
        return block_rmse(p["edge_blocks"], p["node_blocks"], d)

    runner = TrainingGraphs(head.parameters(), loss, capacity=1)
    for _ in range(3):
        head.zero_grad(set_to_none=True)
        runner.backward(first.clone(), 1024)
    head.zero_grad(set_to_none=True)
    expected_loss = loss(second.clone())
    (expected_loss * 1024).backward()
    expected = [
        p.grad.clone() if p.grad is not None else torch.zeros_like(p)
        for p in head.parameters()
    ]
    head.zero_grad(set_to_none=True)
    actual = runner.backward(second.clone(), 1024)
    assert actual is not None
    torch.testing.assert_close(actual, expected_loss, rtol=0.002, atol=1e-6)
    relative = (
        sum(
            (p.grad - g).square().sum()
            for p, g in zip(head.parameters(), expected, strict=True)
        )
        / sum(g.square().sum() for g in expected)
    ).sqrt()
    assert relative.item() < 0.01
    assert runner.captures == 1
