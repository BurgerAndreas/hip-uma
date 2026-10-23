"""Check accumulation, overflow rejection, and AMP checkpoint recovery."""

from __future__ import annotations

import copy

import pytest
import torch

from fairchem.core.frozen_hip.edge_precision import half_linear
from fairchem.core.frozen_hip.loss_scaling import restore_loss_scale, scaled_update


def test_precision_recovery_resets_scale_once_without_mutating_checkpoint():
    original = torch.amp.GradScaler("cpu", init_scale=1, growth_interval=2000)
    checkpoint = original.state_dict()
    scaler = torch.amp.GradScaler("cpu")
    restore_loss_scale(scaler, checkpoint, initial_scale=1024)
    assert scaler.get_scale() == 1024
    assert scaler.state_dict()["_growth_tracker"] == 0
    assert checkpoint["scale"] == 1
    saved = scaler.state_dict()
    saved.update(scale=2048, _growth_tracker=37)
    restore_loss_scale(scaler, saved)
    assert scaler.get_scale() == 2048
    assert scaler.state_dict()["_growth_tracker"] == 37


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_accumulated_scaled_update_matches_fp32(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    p = torch.nn.Parameter(torch.tensor([1.0, -2.0], device=device))
    reference = torch.nn.Parameter(p.detach().clone())
    optimizer = torch.optim.AdamW([p], lr=0.01, weight_decay=0, fused=device == "cuda")
    expected = torch.optim.AdamW(
        [reference], lr=0.01, weight_decay=0, fused=device == "cuda"
    )
    scaler = torch.amp.GradScaler(device, init_scale=1024, growth_interval=2)
    for coefficient, weight in ((2.0, 0.25), (3.0, 0.75)):
        scaler.scale(p.square().sum() * coefficient * weight).backward()
        (reference.square().sum() * coefficient * weight).backward()
    torch.nn.utils.clip_grad_norm_([reference], 10)
    expected.step()
    norm, applied = scaled_update(optimizer, scaler, [p])
    assert applied
    assert norm > 10
    torch.testing.assert_close(p, reference, rtol=0, atol=0)
    torch.testing.assert_close(
        optimizer.state[p]["exp_avg"], expected.state[reference]["exp_avg"]
    )


@pytest.mark.parametrize("norm_only", [False, True])
def test_overflow_preserves_parameters_and_optimizer_and_resets_growth(norm_only):
    p = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    optimizer = torch.optim.AdamW([p], lr=0.01, weight_decay=0)
    scaler = torch.amp.GradScaler("cpu", init_scale=1024, growth_interval=2)
    scaler.scale(p.square().sum()).backward()
    assert scaled_update(optimizer, scaler, [p])[1]
    optimizer.zero_grad(set_to_none=True)
    before = p.detach().clone()
    before_optimizer = copy.deepcopy(optimizer.state_dict())
    scaler.scale(p.sum()).backward()
    p.grad.fill_(1e30 if norm_only else float("inf"))
    assert not scaled_update(optimizer, scaler, [p])[1]
    torch.testing.assert_close(p, before, rtol=0, atol=0)
    for key, value in before_optimizer["state"][0].items():
        torch.testing.assert_close(optimizer.state[p][key], value, rtol=0, atol=0)
    assert scaler.get_scale() == 512
    assert scaler.state_dict()["_growth_tracker"] == 0
    restored = torch.amp.GradScaler("cpu")
    restored.load_state_dict(scaler.state_dict())
    optimizer.zero_grad(set_to_none=True)
    restored.scale(p.square().sum()).backward()
    assert scaled_update(optimizer, restored, [p])[1]
    assert restored.get_scale() == 512


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_custom_half_kernel_recovers_after_loss_scale_backoff():
    p = torch.nn.Parameter(torch.eye(2, device="cuda"))
    x = torch.ones(1, 2, device="cuda")
    optimizer = torch.optim.AdamW([p], lr=0.01, weight_decay=0, fused=True)
    scaler = torch.amp.GradScaler("cuda", init_scale=1024, growth_interval=2000)
    before = p.detach().clone()
    rejected = 0
    for _ in range(16):
        optimizer.zero_grad(set_to_none=True)
        loss = half_linear(x, p).float().sum() * 100
        scaler.scale(loss).backward()
        _, applied = scaled_update(optimizer, scaler, [p])
        if applied:
            break
        rejected += 1
        torch.testing.assert_close(p, before, rtol=0, atol=0)
        assert not optimizer.state
    assert applied
    assert rejected > 0
    assert torch.isfinite(p).all()
    assert not torch.equal(p, before)
