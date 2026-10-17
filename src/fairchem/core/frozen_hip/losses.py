"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Evaluate rotation-invariant Cartesian block norms without numerical smoothing.
"""

from __future__ import annotations

import torch


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


def structure_block_rmse(prediction, target, sizes):
    """
    Average block norms within each geometry against symmetrized target labels.
    """
    widths = [9 * n * n for n in sizes]
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
        values.append(block_rmse_values(error).mean())
    return torch.stack(values).mean()
