"""Explicit invariant charge/spin conditioning for a dense HIP head."""

from __future__ import annotations

import torch
from torch import nn


class ChargeSpinConditioning(nn.Module):
    """Apply invariant channel gains and scalar-only offsets in FP32.

    A zero final layer starts at the unchanged dense function. Each gain is
    shared across every angular component of its channel. Only l=0 receives
    offsets, so the module preserves rotational equivariance.
    """

    def __init__(self, channels, hidden=32):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2, hidden), nn.SiLU(), nn.Linear(hidden, 2 * channels)
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, features, charge, spin, batch):
        with torch.autocast(features.device.type, enabled=False):
            inputs = torch.stack((charge.float(), spin.float() - 1), dim=-1) / 10
            gains, offsets = self.network(inputs).chunk(2, dim=-1)
            gains, offsets = gains[batch], offsets[batch]
            output = features.float() * (1 + gains[:, None, :])
            # Avoid in-place updates needed by autograd and compiled blocks.
            return torch.cat(
                (output[:, :1, :] + offsets[:, None, :], output[:, 1:, :]), dim=1
            )


def enable_charge_spin_conditioning(head):
    """Preserve the dense initialization and random stream for matched runs."""
    if hasattr(head, "hip_charge_spin"):
        return
    with torch.random.fork_rng(devices=[]):
        module = ChargeSpinConditioning(head.backbone.sphere_channels)
    head.hip_charge_spin = module.to(next(head.parameters()).device)
