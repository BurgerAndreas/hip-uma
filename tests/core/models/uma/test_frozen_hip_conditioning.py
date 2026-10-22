"""Check invariant context, equivariance, identity initialization and gradients."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from fairchem.core.frozen_hip.conditioning import (
    ChargeSpinConditioning,
    enable_charge_spin_conditioning,
)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_identity_and_equivariance(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(2)
    module = ChargeSpinConditioning(4).to(device)
    x = torch.randn(5, 9, 4, device=device, requires_grad=True)
    batch = torch.tensor([0, 0, 1, 1, 1], device=device)
    q = torch.tensor([-2, 3], device=device)
    spin = torch.tensor([1, 3], device=device)
    torch.testing.assert_close(module(x, q, spin, batch), x, rtol=0, atol=0)
    nn.init.normal_(module.network[-1].weight, std=0.1)
    rotation = torch.eye(9, device=device)
    rotation[1:, 1:] = torch.linalg.qr(torch.randn(8, 8, device=device)).Q

    def rotate(f):
        return torch.einsum("ij,ajc->aic", rotation, f)

    torch.testing.assert_close(
        module(rotate(x), q, spin, batch),
        rotate(module(x, q, spin, batch)),
        rtol=2e-5,
        atol=2e-6,
    )
    perm = torch.tensor([4, 0, 3, 1, 2], device=device)
    torch.testing.assert_close(
        module(x[perm], q, spin, batch[perm]), module(x, q, spin, batch)[perm]
    )
    assert not torch.allclose(module(x, q + 1, spin, batch), module(x, q, spin, batch))
    assert not torch.allclose(module(x, q, spin + 2, batch), module(x, q, spin, batch))
    with torch.autocast(device, dtype=torch.bfloat16):
        y = module(x, q, spin, batch)
    assert y.dtype == torch.float32
    (y.abs().mean() * 1024).backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters()
    )


def test_enable_preserves_random_stream_and_is_idempotent():
    head = nn.Linear(3, 3)
    head.backbone = SimpleNamespace(sphere_channels=3)
    state = torch.get_rng_state()
    enable_charge_spin_conditioning(head)
    torch.testing.assert_close(torch.get_rng_state(), state)
    original = head.hip_charge_spin
    enable_charge_spin_conditioning(head)
    assert head.hip_charge_spin is original


def test_training_and_state_roundtrip():
    torch.manual_seed(7)
    module = ChargeSpinConditioning(4)
    opt = torch.optim.AdamW(module.parameters(), lr=1e-3, weight_decay=0)
    x = torch.randn(5, 9, 4)
    q, spin, batch = (
        torch.tensor([-1, 2]),
        torch.tensor([2, 3]),
        torch.tensor([0, 0, 1, 1, 1]),
    )
    target = torch.randn_like(x)
    for _ in range(3):
        opt.zero_grad()
        (module(x, q, spin, batch) - target).square().mean().backward()
        opt.step()
    assert module.network[0].weight.grad.norm() > 0
    assert module.network[-1].weight.grad.norm() > 0
    clone = ChargeSpinConditioning(4)
    clone.load_state_dict(module.state_dict())
    torch.testing.assert_close(clone(x, q, spin, batch), module(x, q, spin, batch))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("width", [1, 2])
def test_full_head_dense_conditioning_and_mixed_precision(width):
    import numpy as np

    from fairchem.core.frozen_hip.conditioning import enable_charge_spin_conditioning
    from fairchem.core.frozen_hip.data import make_batch
    from fairchem.core.frozen_hip.precision import configure_head_precision
    from fairchem.core.frozen_hip.radial import enable_fused_radial
    from fairchem.core.frozen_hip.readout import enable_projected_readout
    from fairchem.core.frozen_hip.rotations import (
        enable_fused_rotations,
        enable_fused_scatter,
    )
    from fairchem.core.models.uma.escn_md import Hessian_Head, eSCNMDBackbone

    torch.manual_seed(33)
    backbone = (
        eSCNMDBackbone(
            max_num_elements=100,
            sphere_channels=4,
            lmax=2,
            mmax=2,
            edge_channels=5,
            num_distance_basis=7,
            num_layers=1,
            hidden_channels=4,
            use_dataset_embedding=False,
            always_use_pbc=False,
        )
        .cuda()
        .requires_grad_(False)
    )
    head = Hessian_Head(backbone, width_multiplier=width).cuda()
    head.requires_mole_routing = False
    data = make_batch(
        dict(
            natoms=np.array([2, 3]),
            atomic_numbers=np.array([1, 1, 6, 1, 1]),
            coords=np.array(
                [[0, 0, 0], [0.8, 0, 0], [0, 0, 0], [0, 0.8, 0], [0.8, 0, 0]]
            ),
            charge=np.array([-1, 1]),
            spin=np.array([2, 3]),
            omol_index=np.array([7, 9]),
            hessian_flat=np.zeros(117, dtype=np.float32),
        ),
        np.array([0, 1]),
    ).cuda()
    data.hip_sizes = [2, 3]
    emb = {"node_embedding": torch.randn(5, 9, 4, device="cuda"), "batch": data.batch}
    torch.manual_seed(19)
    baseline = head(data, emb)["hessian"]
    enable_charge_spin_conditioning(head)
    torch.manual_seed(19)
    actual = head(data, emb)["hessian"]
    torch.testing.assert_close(actual, baseline, atol=2e-5, rtol=3e-4)
    enable_fused_rotations(head)
    enable_projected_readout(head)
    enable_fused_radial(head)
    enable_fused_scatter(head, segmented=True)
    configure_head_precision(head, "fp16-edge")
    torch.manual_seed(19)
    out = head(data, emb)["hessian"]
    assert out.dtype == torch.float32
    assert (out - baseline).norm() / baseline.norm() < 0.01
    ((out - torch.randn_like(out)).abs().mean() * 1024).backward()
    assert head.hip_charge_spin.network[-1].weight.grad.norm() > 0
    assert all(p.grad is None for p in backbone.parameters())
    assert all(
        torch.isfinite(p.grad).all() for p in head.parameters() if p.grad is not None
    )
