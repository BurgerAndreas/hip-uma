"""Explicit invariant charge/spin conditioning for a dense HIP head."""

from __future__ import annotations

import torch
from torch import nn


class ChargeSpinConditioning(nn.Module):
    """Apply invariant channel gains and scalar-only offsets in FP32.

    A zero final layer starts at the unchanged dense function. Each gain is
    shared across every angular component of its channel. Only l=0 receives
    offsets, so the module preserves rotational equivariance.

    Modes:
        ``extensive``: legacy ``(charge / 10, (spin - 1) / 10)`` with unbounded gains.
        ``intensive_bounded``: size-invariant ``(charge / natoms, (spin - 1) / natoms)``
            with ``gain_scale * tanh(gains)`` so modulation stays bounded for
            arbitrarily large or highly charged macromolecules.
    """

    def __init__(self, channels, hidden=32, mode="extensive", gain_scale=1.0):
        super().__init__()
        if mode not in ("extensive", "intensive_bounded"):
            raise ValueError(f"Unsupported charge/spin conditioning mode: {mode}")
        self.mode = mode
        self.gain_scale = float(gain_scale)
        self.network = nn.Sequential(
            nn.Linear(2, hidden), nn.SiLU(), nn.Linear(hidden, 2 * channels)
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, features, charge, spin, batch):
        with torch.autocast(features.device.type, enabled=False):
            if self.mode == "intensive_bounded":
                natoms = torch.bincount(batch, minlength=charge.shape[0]).clamp(min=1).float()
                inputs = torch.stack(
                    (charge.float() / natoms, (spin.float() - 1.0) / natoms), dim=-1
                )
            else:
                inputs = torch.stack((charge.float(), spin.float() - 1), dim=-1) / 10
            gains, offsets = self.network(inputs).chunk(2, dim=-1)
            if self.mode == "intensive_bounded":
                gains = self.gain_scale * torch.tanh(gains)
            gains, offsets = gains[batch], offsets[batch]
            output = features.float() * (1 + gains[:, None, :])
            # Avoid in-place updates needed by autograd and compiled blocks.
            return torch.cat(
                (output[:, :1, :] + offsets[:, None, :], output[:, 1:, :]), dim=1
            )


def enable_charge_spin_conditioning(head, *, mode="extensive", gain_scale=1.0):
    """Preserve the dense initialization and random stream for matched runs."""
    if hasattr(head, "hip_charge_spin"):
        return
    with torch.random.fork_rng(devices=[]):
        module = ChargeSpinConditioning(
            head.backbone.sphere_channels, mode=mode, gain_scale=gain_scale
        )
    head.hip_charge_spin = module.to(next(head.parameters()).device)
