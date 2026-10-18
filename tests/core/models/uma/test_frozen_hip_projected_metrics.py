"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import pytest
import torch
from ase.data import atomic_masses

from fairchem.core.frozen_hip.metrics import (
    PROJECTED_METRIC_NAMES,
    mass_weighted_eckart_totals,
    projected_metric_record,
)
from fairchem.core.units.mlip_unit.hessian_metrics import eckart_projected_modes


def dimer():
    pos = torch.tensor([[0.0, 0.0, 0.0], [1.3, 0.0, 0.0]], dtype=torch.float64)
    z = torch.tensor([1, 17])
    h = torch.zeros(6, 6, dtype=torch.float64)
    h[0, 0] = h[3, 3] = 2.0
    h[0, 3] = h[3, 0] = -2.0
    return h, pos, z


def rigid_basis(pos, z):
    masses = torch.tensor(atomic_masses, dtype=torch.float64)[z]
    centered = pos - (masses[:, None] * pos).sum(0) / masses.sum()
    columns = [
        (masses.sqrt()[:, None] * a).flatten()
        for a in torch.eye(3, dtype=torch.float64)
    ]
    columns += [
        (
            masses.sqrt()[:, None] * torch.cross(a.expand_as(pos), centered, dim=1)
        ).flatten()
        for a in torch.eye(3, dtype=torch.float64)
    ]
    return torch.stack(columns, dim=1), masses


def test_unequal_mass_spring_has_analytical_errors_and_one_vibrational_mode():
    h, pos, z = dimer()
    values = mass_weighted_eckart_totals(1.1 * h, h, pos, z, [2])
    masses = torch.tensor(atomic_masses, dtype=torch.float64)[z]
    eig_error = 0.2 * masses.reciprocal().sum()
    matrix_mae = 0.2 * masses.rsqrt().sum().square() / 36
    expected = torch.stack(
        (
            eig_error.new_tensor(1.0),
            eig_error,
            matrix_mae,
            eig_error / 6,
            eig_error.new_tensor(1.0),
            eig_error.new_tensor(0.0),
        )
    )
    torch.testing.assert_close(values, expected, atol=1e-12, rtol=1e-12)
    assert eckart_projected_modes(h.flatten(), pos, z)[0].numel() == 1
    row = projected_metric_record(values, 1)
    assert row["mass_weighted_eckart_precision"] == "float64"
    assert row["mass_weighted_eckart_hessian_frame"] == "full Cartesian 3N by 3N"
    assert row["mass_weighted_eckart_unit"] == "eV/(Angstrom^2*amu)"


@pytest.mark.parametrize("linear", [False, True])
def test_metrics_match_independent_full_cartesian_projector_and_existing_modes(linear):
    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0 if linear else 0.7, 0.0]],
        dtype=torch.float64,
    )
    z = torch.tensor([1, 6, 8])
    generator = torch.Generator().manual_seed(3)
    h = torch.randn(9, 9, generator=generator, dtype=torch.float64)
    h = (h + h.T) * 0.5 - torch.eye(9, dtype=torch.float64)
    p = h + torch.diag(torch.linspace(-0.4, 0.3, 9, dtype=torch.float64))
    g, masses = rigid_basis(pos, z)
    projector = torch.eye(9, dtype=torch.float64) - g @ torch.linalg.pinv(g)
    inverse = masses.rsqrt().repeat_interleave(3)
    error = projector @ (inverse[:, None] * (p - h) * inverse[None, :]) @ projector
    pe, pv = eckart_projected_modes(p.flatten(), pos, z)
    te, tv = eckart_projected_modes(h.flatten(), pos, z)
    assert te.numel() == (4 if linear else 3)
    assert te[0] < 0  # v1 follows algebraic order, including unstable modes.
    expected = torch.stack(
        (
            (pv[:, 0] @ tv[:, 0]).abs(),
            (pe - te).abs().mean(),
            error.abs().mean(),
            error.square().mean().sqrt(),
        )
    )
    result = mass_weighted_eckart_totals(p, h, pos, z, [3])
    torch.testing.assert_close(result[:4], expected, atol=1e-10, rtol=1e-10)


def test_rigid_translation_and_rotation_perturbations_do_not_contribute():
    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [0.9, 0.0, 0.0], [0.0, 1.1, 0.2]], dtype=torch.float64
    )
    z = torch.tensor([8, 1, 1])
    g, masses = rigid_basis(pos, z)
    sqrt_mass = masses.sqrt().repeat_interleave(3)
    h = torch.diag(torch.arange(1, 10, dtype=torch.float64))
    perturbation = sqrt_mass[:, None] * (g @ g.T) * sqrt_mass[None, :]
    result = mass_weighted_eckart_totals(h + perturbation, h, pos, z, [3])
    torch.testing.assert_close(
        result[0], result.new_tensor(1.0), atol=1e-11, rtol=1e-11
    )
    torch.testing.assert_close(result[1:4], result.new_zeros(3), atol=1e-11, rtol=0)


def test_geometry_means_and_partitioned_totals_have_equal_weights():
    h, pos, z = dimer()
    pos3 = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float64
    )
    z3 = torch.tensor([6, 1, 8])
    h3 = torch.eye(9, dtype=torch.float64)
    a = mass_weighted_eckart_totals(1.1 * h, h, pos, z, [2])
    b = mass_weighted_eckart_totals(1.3 * h3, h3, pos3, z3, [3])
    combined = mass_weighted_eckart_totals(
        torch.cat(((1.1 * h).flatten(), (1.3 * h3).flatten())),
        torch.cat((h.flatten(), h3.flatten())),
        torch.cat((pos, pos3)),
        torch.cat((z, z3)),
        [2, 3],
    )
    torch.testing.assert_close(combined, a + b, atol=1e-12, rtol=1e-12)
    row = projected_metric_record(combined, 2)
    for i, name in enumerate(PROJECTED_METRIC_NAMES):
        assert row[name] == pytest.approx(float((a[i] + b[i]) / 2))


def test_identity_and_rigid_only_geometry_have_explicit_counts():
    h, pos, z = dimer()
    h.requires_grad_(True)
    result = mass_weighted_eckart_totals(h, h, pos, z, [2])
    assert result.dtype == torch.float64
    assert not result.requires_grad
    torch.testing.assert_close(
        result[:4], result.new_tensor([1.0, 0.0, 0.0, 0.0]), atol=1e-12, rtol=0
    )
    mono = mass_weighted_eckart_totals(
        torch.eye(3), torch.eye(3), torch.zeros(1, 3), torch.tensor([1]), [1]
    )
    torch.testing.assert_close(mono, mono.new_tensor([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]))
    mixed = projected_metric_record(result + mono, 2)
    assert mixed["mass_weighted_eckart_samples"] == 1
    assert mixed["mass_weighted_eckart_samples_without_modes"] == 1


@pytest.mark.parametrize(
    "totals, count",
    [
        ([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], 2),
        ([float("nan"), 0.0, 0.0, 0.0, 1.0, 0.0], 1),
        ([0.0, 0.0, 0.0, 0.0, 0.0, 1.0], 1),
    ],
)
def test_records_reject_invalid_coverage_or_nonfinite_metrics(totals, count):
    with pytest.raises(ValueError):
        projected_metric_record(torch.tensor(totals, dtype=torch.float64), count)


def test_invalid_atomic_numbers_are_rejected():
    h, pos, z = dimer()
    with pytest.raises(ValueError, match="atomic numbers"):
        mass_weighted_eckart_totals(h, h, pos, torch.zeros_like(z), [2])
