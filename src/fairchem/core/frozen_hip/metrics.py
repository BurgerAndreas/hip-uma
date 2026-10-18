"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Measure mass-weighted Hessians after removing geometric rigid modes.
"""

from __future__ import annotations

import math

import torch
from ase.data import atomic_masses

PROJECTED_METRIC_NAMES = (
    "mass_weighted_eckart_abs_cos_v1",
    "mass_weighted_eckart_eigenvalue_mae",
    "mass_weighted_eckart_hessian_mae",
    "mass_weighted_eckart_hessian_rmse",
)


def _reduce_hessian(hessian, inverse_mass, q):
    dimension = inverse_mass.numel()
    hessian = hessian.reshape(dimension, dimension)
    symmetric = (hessian + hessian.T) * 0.5
    weighted = inverse_mass[:, None] * symmetric * inverse_mass[None, :]
    reduced = q.T @ weighted @ q
    return (reduced + reduced.T) * 0.5


@torch.no_grad()
def mass_weighted_eckart_totals(prediction, target, positions, numbers, sizes):
    """Return four geometry sums, the valid count, and the rigid-only count.

    Hessian errors retain the original 3N Cartesian axes after projection.
    RMSE takes one square root per complete molecular Hessian. Spectral errors
    use the vibrational complement; v1 has the lowest algebraic eigenvalue.
    All calculations use FP64. Linear molecules retain 3N-5 modes.
    """
    sizes = [int(n) for n in sizes]
    widths = [9 * n * n for n in sizes]
    atoms = sum(sizes)
    if (
        any(n < 1 for n in sizes)
        or prediction.numel() != sum(widths)
        or target.numel() != sum(widths)
        or positions.shape != (atoms, 3)
        or numbers.shape != (atoms,)
    ):
        raise ValueError("Projected metrics received inconsistent geometry shapes")
    device = prediction.device
    prediction = prediction.detach().to(dtype=torch.float64).reshape(-1)
    target = target.detach().to(device=device, dtype=torch.float64).reshape(-1)
    positions = positions.to(device=device, dtype=torch.float64)
    numbers = numbers.to(device=device, dtype=torch.long)
    table = torch.as_tensor(atomic_masses, device=device, dtype=torch.float64)
    if not bool(((numbers > 0) & (numbers < table.numel())).all()):
        raise ValueError("Projected metrics require valid atomic numbers")
    masses = table[numbers]
    if not bool((torch.isfinite(masses) & (masses > 0)).all()) or not bool(
        torch.isfinite(positions).all()
    ):
        raise ValueError("Projected metrics require finite positions and masses")
    totals = torch.zeros(6, device=device, dtype=torch.float64)
    values = []
    axes = torch.eye(3, device=device, dtype=torch.float64)
    for p, t, pos, mass, n in zip(
        prediction.split(widths),
        target.split(widths),
        positions.split(sizes),
        masses.split(sizes),
        sizes,
        strict=True,
    ):
        sqrt_mass = mass.sqrt()
        centered = pos - (pos * mass[:, None]).sum(0) / mass.sum()
        translations = sqrt_mass[:, None, None] * axes[None, :, :]
        rotations = (
            torch.cross(axes[None, :, :].expand(n, 3, 3), centered[:, None, :], dim=-1)
            * sqrt_mass[:, None, None]
        )
        rigid = torch.cat(
            (
                translations.reshape(3 * n, 3),
                rotations.transpose(1, 2).reshape(3 * n, 3),
            ),
            dim=1,
        )
        u, singular, _ = torch.linalg.svd(rigid, full_matrices=True)
        tolerance = torch.finfo(torch.float64).eps * max(rigid.shape) * singular[0]
        rank = int((singular > tolerance).sum().item())
        q = u[:, rank:]
        if not q.shape[1]:
            continue
        inverse_mass = mass.rsqrt().repeat_interleave(3)

        p_reduced = _reduce_hessian(p, inverse_mass, q)
        t_reduced = _reduce_hessian(t, inverse_mass, q)
        p_eigenvalues, p_vectors = torch.linalg.eigh(p_reduced)
        t_eigenvalues, t_vectors = torch.linalg.eigh(t_reduced)
        error = q @ (p_reduced - t_reduced) @ q.T
        values.append(
            torch.stack(
                (
                    torch.dot(p_vectors[:, 0], t_vectors[:, 0]).abs().clamp(0, 1),
                    (p_eigenvalues - t_eigenvalues).abs().mean(),
                    error.abs().mean(),
                    error.square().mean().sqrt(),
                )
            )
        )
    if values:
        totals[:4] = torch.stack(values).sum(0)
    totals[4] = len(values)
    totals[5] = len(sizes) - len(values)
    return totals


def projected_metric_record(totals, expected_samples):
    """Normalize geometry sums after distributed reduction and verify coverage."""
    values = totals.detach().cpu().tolist()
    if len(values) != 6 or not all(math.isfinite(v) for v in values):
        raise ValueError("Projected metric totals must contain six finite values")
    count, empty = values[4:]
    if count + empty != expected_samples or count != int(count) or empty != int(empty):
        raise ValueError("Projected metric coverage differs from the evaluation cohort")
    if count <= 0 or empty < 0:
        raise ValueError("Projected metrics require at least one vibrational geometry")
    record = {
        name: value / count
        for name, value in zip(PROJECTED_METRIC_NAMES, values[:4], strict=True)
    }
    record.update(
        mass_weighted_eckart_samples=int(count),
        mass_weighted_eckart_samples_without_modes=int(empty),
        mass_weighted_eckart_precision="float64",
        mass_weighted_eckart_unit="eV/(Angstrom^2*amu)",
        mass_weighted_eckart_hessian_frame="full Cartesian 3N by 3N",
        mass_weighted_eckart_first_mode="lowest algebraic vibrational eigenvalue",
        mass_weighted_eckart_averaging="equal geometry means",
    )
    return record
