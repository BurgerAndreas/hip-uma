"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from ase.data import atomic_masses

if TYPE_CHECKING:
    from collections.abc import Hashable

    from fairchem.core.units.mlip_unit._metrics import Metrics


def eckart_projected_modes(
    hessian_flat: torch.Tensor,
    pos: torch.Tensor,
    atomic_numbers: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    n_atoms = pos.shape[0]
    if n_atoms == 0:
        return hessian_flat.new_empty(0), hessian_flat.new_empty(0, 0)

    hessian = hessian_flat.view(n_atoms * 3, n_atoms * 3)
    hessian = 0.5 * (hessian + hessian.T)
    mass_table = torch.as_tensor(
        atomic_masses, device=hessian.device, dtype=hessian.dtype
    )
    masses = mass_table[atomic_numbers.to(device=hessian.device, dtype=torch.long)]
    inv_sqrt_mass = torch.repeat_interleave(torch.rsqrt(masses.clamp_min(1.0e-12)), 3)
    mass_weighted_hessian = (
        inv_sqrt_mass[:, None] * hessian * inv_sqrt_mass[None, :]
    )

    sqrt_masses = torch.sqrt(masses)
    center = (pos.to(dtype=hessian.dtype) * masses[:, None]).sum(dim=0) / masses.sum()
    centered_pos = pos.to(dtype=hessian.dtype) - center

    basis = []
    for axis in range(3):
        translation = hessian.new_zeros(n_atoms, 3)
        translation[:, axis] = sqrt_masses
        basis.append(translation.reshape(-1))

    axes = torch.eye(3, device=hessian.device, dtype=hessian.dtype)
    for axis in axes:
        rotation = torch.cross(
            axis.expand_as(centered_pos), centered_pos, dim=1
        ) * sqrt_masses[:, None]
        basis.append(rotation.reshape(-1))

    rigid = torch.stack(basis, dim=1)
    u, singular_values, _ = torch.linalg.svd(rigid, full_matrices=True)
    if singular_values.numel() == 0:
        rank = 0
    else:
        tol = torch.finfo(hessian.dtype).eps * max(rigid.shape) * singular_values[0]
        rank = int((singular_values > tol).sum().item())
    vibrational_basis = u[:, rank:]
    if vibrational_basis.shape[1] == 0:
        return hessian_flat.new_empty(0), hessian_flat.new_empty(n_atoms * 3, 0)

    projected_hessian = vibrational_basis.T @ mass_weighted_hessian @ vibrational_basis
    eigvals, eigvecs_projected = torch.linalg.eigh(projected_hessian)
    eigvecs = vibrational_basis @ eigvecs_projected
    return eigvals, eigvecs


def hessian_eckart_metric(
    prediction: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    key: Hashable,
    mode: str,
    eigvec_idx: int | None = None,
) -> Metrics:
    from fairchem.core.units.mlip_unit._metrics import Metrics

    pred = prediction[key].reshape(-1)
    target_tensor = target[key].reshape(-1)
    ptr = target["ptr_1d_hessian"].to(device=pred.device, dtype=torch.long)
    natoms = target["natoms"].to(device=pred.device, dtype=torch.long)
    node_ptr = torch.empty(natoms.numel() + 1, device=pred.device, dtype=torch.long)
    node_ptr[0] = 0
    node_ptr[1:] = torch.cumsum(natoms, dim=0)
    entry_mask = target.get(
        "entry_mask", torch.ones_like(target_tensor, dtype=torch.bool)
    ).to(device=pred.device, dtype=torch.bool)

    values = []
    for sample_idx, (start, stop) in enumerate(zip(ptr[:-1], ptr[1:])):
        if not entry_mask[start:stop].all():
            continue
        node_start = node_ptr[sample_idx]
        node_stop = node_ptr[sample_idx + 1]
        pos = target["pos"][node_start:node_stop].to(device=pred.device)
        atomic_numbers = target["atomic_numbers"][node_start:node_stop].to(
            device=pred.device
        )
        pred_eigvals, pred_eigvecs = eckart_projected_modes(
            pred[start:stop], pos, atomic_numbers
        )
        target_eigvals, target_eigvecs = eckart_projected_modes(
            target_tensor[start:stop], pos, atomic_numbers
        )
        n_modes = min(pred_eigvals.numel(), target_eigvals.numel())
        if mode == "eigval_mae":
            if n_modes > 0:
                values.append(
                    (pred_eigvals[:n_modes] - target_eigvals[:n_modes]).abs().mean()
                )
        elif mode == "eigvec_cos":
            assert eigvec_idx is not None
            if n_modes > eigvec_idx:
                cos = torch.dot(
                    pred_eigvecs[:, eigvec_idx], target_eigvecs[:, eigvec_idx]
                ).abs()
                values.append(cos)
        else:
            raise ValueError(f"Unknown Hessian Eckart metric mode {mode}")

    if not values:
        return Metrics()
    values_t = torch.stack(values)
    return Metrics(
        metric=values_t.mean().item(),
        total=values_t.sum().item(),
        numel=values_t.numel(),
    )
