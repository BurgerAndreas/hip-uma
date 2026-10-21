"""Check HIP expert routing, dense initialization and ragged GEMM derivatives."""

from __future__ import annotations

import io
from types import SimpleNamespace

import pytest
import torch

from fairchem.core.frozen_hip.mole import (
    HIPRouter,
    MixtureSO2,
    grouped_linear,
    grouped_reference,
)
from fairchem.core.models.uma.common.so3 import CoefficientMapping
from fairchem.core.models.uma.nn.so2_layers import SO2_Convolution


def metadata(sizes, device="cpu"):
    n = torch.tensor(sizes, device=device)
    return SimpleNamespace(
        hip_sizes=sizes,
        natoms=n,
        pos=torch.zeros(sum(sizes), 3, device=device),
        batch=torch.arange(len(sizes), device=device).repeat_interleave(n),
        atomic_numbers=torch.tensor(
            ([1, 6, 8] * sum(sizes))[: sum(sizes)], device=device
        ),
        charge=torch.arange(len(sizes), device=device),
        spin=torch.ones(len(sizes), device=device),
    )


def test_router_invariance_and_context():
    torch.manual_seed(4)
    router = HIPRouter(4)
    # Break the deliberately uniform initialization to test actual input use.
    torch.nn.init.normal_(router.network[-1].weight, std=0.1)
    d = metadata([3, 3])
    d.atomic_numbers[3:] = d.atomic_numbers[:3]
    d.charge.zero_()
    a = router(d)
    torch.testing.assert_close(a[0], a[1])
    d.pos.normal_()
    d.atomic_numbers[:3] = d.atomic_numbers[:3].flip(0)
    torch.testing.assert_close(router(d), a)
    d.charge[1] = 5
    assert not torch.allclose(router(d)[1], a[1])
    torch.testing.assert_close(a.sum(-1), torch.ones(2))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_so2_dense_initialization_gradients_and_checkpoint(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(5)
    mapping = CoefficientMapping(2, 2).to(device)
    original = SO2_Convolution(4, 4, 2, 2, mapping, internal_weights=True).to(device)
    mixture = MixtureSO2(original, 4)
    d = metadata([2, 3], device)
    router = HIPRouter(4).to(device)
    ptr = torch.tensor([0, 2, 8], device=device)
    alpha = router(d)
    mixture.prepare(alpha, ptr, d.hip_sizes)
    x = torch.randn(8, 9, 4, device=device)
    expected = original(x)
    actual = mixture(x)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    loss = (actual * torch.randn_like(actual)).sum()
    loss.backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in mixture.parameters()
    )
    assert router.network[-1].weight.grad.norm() > 0
    for p in mixture.expert_weights:
        assert p.grad.norm() > 0
    opt = torch.optim.AdamW(
        list(mixture.parameters()) + list(router.parameters()), lr=1e-3, weight_decay=0
    )
    opt.step()
    buf = io.BytesIO()
    torch.save(
        {"m": mixture.state_dict(), "r": router.state_dict(), "o": opt.state_dict()},
        buf,
    )
    buf.seek(0)
    state = torch.load(buf, weights_only=False)
    clone = MixtureSO2(original, 4)
    clone.load_state_dict(state["m"])
    router.load_state_dict(state["r"])
    clone.prepare(router(d), ptr, d.hip_sizes)
    mixture.prepare(router(d), ptr, d.hip_sizes)
    torch.testing.assert_close(clone(x), mixture(x))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "sizes,inputs,outputs",
    [([1, 2, 7], 7, 11), ([1], 32, 64), ([3, 16, 2], 64, 96), ([350], 32, 64)],
)
@pytest.mark.parametrize("mixed", [False, True])
def test_ragged_grouped_gemm_forward_backward(sizes, inputs, outputs, mixed):
    torch.manual_seed(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    count = sum(n * (n - 1) for n in sizes)
    x = torch.randn(count, inputs, device="cuda", requires_grad=True)
    w = torch.randn(len(sizes), outputs, inputs, device="cuda", requires_grad=True)
    bias = torch.randn(outputs, device="cuda", requires_grad=True)
    n = torch.tensor(sizes, device="cuda")
    ptr = torch.cat((n.new_zeros(1), (n * (n - 1)).cumsum(0)))
    actual = grouped_linear(x, w, bias, ptr, sizes, mixed)
    ref = grouped_reference(x, w, bias, sizes)
    scale = ref.detach().abs().max().clamp_min(1) if ref.numel() else ref.new_tensor(1)
    torch.testing.assert_close(
        actual.float() / scale,
        ref / scale,
        atol=8e-4 if mixed else 2e-6,
        rtol=3e-3 if mixed else 2e-5,
    )
    g = torch.randn_like(actual)
    gradients = torch.autograd.grad(actual, (x, w, bias), g)
    expected = torch.autograd.grad(ref, (x, w, bias), g.float())
    for a, b in zip(gradients, expected, strict=True):
        assert torch.isfinite(a).all()
        s = b.abs().max().clamp_min(1) if b.numel() else b.new_tensor(1)
        torch.testing.assert_close(
            a / s, b / s, atol=1e-3 if mixed else 1e-5, rtol=5e-3 if mixed else 3e-5
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fp32_expert_weight_gradient_long_reduction():
    sizes = [350]
    count = 350 * 349
    x = torch.ones(count, 4, device="cuda", requires_grad=True)
    w = torch.ones(1, 8, 4, device="cuda", requires_grad=True)
    ptr = torch.tensor([0, count], device="cuda")
    y = grouped_linear(x, w, None, ptr, sizes, True)
    (y.float().sum() * 1024).backward()
    assert w.grad.dtype == torch.float32
    torch.testing.assert_close(w.grad / 1024, torch.full_like(w, count))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_full_head_mixed_precision_expert_and_router_gradients():
    import numpy as np

    from fairchem.core.frozen_hip.data import make_batch
    from fairchem.core.frozen_hip.mole import enable_head_mole
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
    head = Hessian_Head(backbone).cuda()
    head.requires_mole_routing = False
    data = make_batch(
        dict(
            natoms=np.array([2, 3]),
            atomic_numbers=np.array([1, 1, 6, 1, 1]),
            coords=np.array(
                [[0, 0, 0], [0.8, 0, 0], [0, 0, 0], [0, 0.8, 0], [0.8, 0, 0]]
            ),
            charge=np.array([0, 0]),
            spin=np.array([1, 1]),
            omol_index=np.array([7, 9]),
            hessian_flat=np.zeros(117, dtype=np.float32),
        ),
        np.array([0, 1]),
    ).cuda()
    data.hip_sizes = [2, 3]
    emb = {"node_embedding": torch.randn(5, 9, 4, device="cuda"), "batch": data.batch}
    torch.manual_seed(19)
    baseline = head(data, emb)["hessian"]
    enable_head_mole(head, 4)
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
    for block in head.blocks:
        assert all(
            p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm() > 0
            for p in block.edge_wise.so2_conv_2.expert_weights
        )
    assert head.hip_mole_router.network[-1].weight.grad.norm() > 0
    assert all(p.grad is None for p in backbone.parameters())
