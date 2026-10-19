"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Evaluate rotation-invariant Cartesian block norms without numerical smoothing.
"""

from __future__ import annotations

from functools import lru_cache

import torch
from ase.data import atomic_masses


@lru_cache(maxsize=16)
def _inverse_mass_table(device):
    return torch.as_tensor(atomic_masses, dtype=torch.float32, device=device).rsqrt()


def inverse_sqrt_masses(numbers):
    """Return inverse square root atomic masses in amu on the input device."""
    return _inverse_mass_table(numbers.device)[numbers.long()]


def block_rmse_values(error):
    """
    Return one Frobenius norm divided by three, with a zero subgradient at zero.

    Detached scaling avoids square overflow and underflow without an epsilon.
    """
    error = error.float()
    scale = error.detach().abs().amax(dim=(-2, -1), keepdim=True)
    denominator = torch.where(scale > 0, scale, torch.ones_like(scale))
    normalized = error / denominator
    return (denominator.squeeze(-1).squeeze(-1) / 3) * torch.linalg.vector_norm(
        normalized, dim=(-2, -1)
    )


def structure_block_rmse(prediction, target, sizes, *, numbers=None):
    """
    Average block norms within each geometry against symmetrized target labels.
    """
    widths = [9 * n * n for n in sizes]
    inverse = None if numbers is None else inverse_sqrt_masses(numbers)
    atom_offset = 0
    values = []
    for p, t, n in zip(
        prediction.split(widths), target.split(widths), sizes, strict=True
    ):
        target_matrix = t.float().reshape(3 * n, 3 * n)
        target_matrix = (target_matrix + target_matrix.T) * 0.5
        error = (
            (p.float().reshape(3 * n, 3 * n) - target_matrix)
            .reshape(n, 3, n, 3)
            .permute(0, 2, 1, 3)
        )
        norms = block_rmse_values(error)
        if inverse is not None:
            mass = inverse[atom_offset : atom_offset + n]
            norms = norms * mass[:, None] * mass[None, :]
        values.append(norms.mean())
        atom_offset += n
    return torch.stack(values).mean()


def structure_mae(prediction, target, sizes):
    """Average entrywise absolute errors against symmetric labels per geometry."""
    values = []
    for p, t, n in zip(
        prediction.split([9 * n * n for n in sizes]),
        target.split([9 * n * n for n in sizes]),
        sizes,
        strict=True,
    ):
        matrix = t.float().reshape(3 * n, 3 * n)
        symmetric = (matrix + matrix.T) * 0.5
        values.append((p.float().reshape_as(symmetric) - symmetric).abs().mean())
    return torch.stack(values).mean()
