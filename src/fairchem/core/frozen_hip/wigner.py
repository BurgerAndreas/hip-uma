"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Evaluate frozen molecular rotations without the redundant quartic tensor.
"""

from __future__ import annotations

import itertools

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton


@triton.jit
def _power(x, exponent):
    square = x * x
    return tl.where(
        exponent == 0,
        1.0,
        tl.where(
            exponent == 1,
            x,
            tl.where(
                exponent == 2,
                square,
                tl.where(exponent == 3, square * x, square * square),
            ),
        ),
    )


@triton.jit
def _rotation(
    V, DIST, GAMMA, POWERS, COEFF, W, INV, E, CUTOFF: tl.constexpr, BM: tl.constexpr
):
    edge = tl.program_id(0) * BM + tl.arange(0, BM)
    valid = edge < E
    vx = tl.load(V + edge * 3, valid, 0)
    vy = tl.load(V + edge * 3 + 1, valid, 0)
    vz = tl.load(V + edge * 3 + 2, valid, 0)
    norm = tl.maximum(tl.sqrt(vx * vx + vy * vy + vz * vz), 1.0e-12)
    ex, ey, ez = vx / norm, vy / norm, vz / norm
    na = tl.sqrt(
        tl.maximum((1 + ey) * (1 + ey) + ez * ez + ex * ex, 1.1920928955078125e-7)
    )
    nb = tl.sqrt(
        tl.maximum(ez * ez + (1 - ey) * (1 - ey) + ex * ex, 1.1920928955078125e-7)
    )
    a0, a1, a3 = (1 + ey) / na, -ez / na, ex / na
    b0, b1, b2 = -ez / nb, (1 - ey) / nb, ex / nb
    sign = tl.where(a0 * b0 + a1 * b1 < 0, -1.0, 1.0)
    t = tl.minimum(1.0, tl.maximum(0.0, (ey + 0.9) / 1.8))
    arg = (2 * t - 1) / tl.maximum(t * (1 - t), 1.1920928955078125e-7)
    blend = 1 / (1 + tl.exp(-arg))
    blend = tl.where(
        t < 1.1920928955078125e-7,
        0.0,
        tl.where(t > 1 - 1.1920928955078125e-7, 1.0, blend),
    )
    q0 = (1 - blend) * b0 + blend * a0 * sign
    q1 = (1 - blend) * b1 + blend * a1 * sign
    q2 = (1 - blend) * b2
    q3 = blend * a3 * sign
    qn = tl.maximum(tl.sqrt(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3), 1.0e-12)
    q0, q1, q2, q3 = q0 / qn, q1 / qn, q2 / qn, q3 / qn
    angle = tl.load(GAMMA + edge, valid, 0) * 0.5
    cosine, sine = tl.cos(angle), tl.sin(angle)
    w, x, y, z = (
        cosine * q0 - sine * q2,
        cosine * q1 + sine * q3,
        cosine * q2 + sine * q0,
        cosine * q3 - sine * q1,
    )
    term = tl.arange(0, 64)
    p0 = tl.load(POWERS + term * 4, term < 35, 0)
    p1 = tl.load(POWERS + term * 4 + 1, term < 35, 0)
    p2 = tl.load(POWERS + term * 4 + 2, term < 35, 0)
    p3 = tl.load(POWERS + term * 4 + 3, term < 35, 0)
    basis = _power(w[:, None], p0[None, :]) * _power(x[:, None], p1[None, :])
    basis *= _power(y[:, None], p2[None, :]) * _power(z[:, None], p3[None, :])
    basis = tl.where((valid[:, None]) & (term[None, :] < 35), basis, 0)
    entry = tl.arange(0, 32)
    coefficients = tl.load(
        COEFF + term[:, None] * 25 + entry[None, :],
        (term[:, None] < 35) & (entry[None, :] < 25),
        0,
    )
    l2 = tl.dot(basis, coefficients, input_precision="tf32x3")
    distance = tl.load(DIST + edge, valid, 0) / CUTOFF
    d2 = distance * distance
    envelope = 1 + d2 * d2 * distance * (-21 + distance * (35 - 15 * distance))
    envelope = tl.where(distance < 1, envelope, 0)
    full = tl.arange(0, 128)
    rr, cc = full // 9, full % 9
    populated = ((rr >= 1) & (rr < 4) & (cc >= 1) & (cc < 4)) | ((rr >= 4) & (cc >= 4))
    mask = valid[:, None] & (full[None, :] < 81) & ~populated[None, :]
    initial = tl.where(full == 0, 1.0, 0.0)
    tl.store(W + edge[:, None] * 81 + full[None, :], initial[None, :], mask)
    tl.store(
        INV + edge[:, None] * 81 + full[None, :],
        initial[None, :] * envelope[:, None],
        mask,
    )
    r0, c0 = entry // 5 + 4, entry % 5 + 4
    block_mask = valid[:, None] & (entry[None, :] < 25)
    tl.store(W + edge[:, None] * 81 + (r0 * 9 + c0)[None, :], l2, block_mask)
    tl.store(
        INV + edge[:, None] * 81 + (c0 * 9 + r0)[None, :],
        l2 * envelope[:, None],
        block_mask,
    )
    # Write the l=1 block with the same Cartesian quaternion convention.
    tl.store(W + edge * 81 + 10, 1 - 2 * (y * y + z * z), valid)
    tl.store(W + edge * 81 + 11, 2 * (x * y - w * z), valid)
    tl.store(W + edge * 81 + 12, 2 * (x * z + w * y), valid)
    tl.store(W + edge * 81 + 19, 2 * (x * y + w * z), valid)
    tl.store(W + edge * 81 + 20, 1 - 2 * (x * x + z * z), valid)
    tl.store(W + edge * 81 + 21, 2 * (y * z - w * x), valid)
    tl.store(W + edge * 81 + 28, 2 * (x * z - w * y), valid)
    tl.store(W + edge * 81 + 29, 2 * (y * z + w * x), valid)
    tl.store(W + edge * 81 + 30, 1 - 2 * (x * x + y * y), valid)
    tl.store(INV + edge * 81 + 10, (1 - 2 * (y * y + z * z)) * envelope, valid)
    tl.store(INV + edge * 81 + 19, 2 * (x * y - w * z) * envelope, valid)
    tl.store(INV + edge * 81 + 28, 2 * (x * z + w * y) * envelope, valid)
    tl.store(INV + edge * 81 + 11, 2 * (x * y + w * z) * envelope, valid)
    tl.store(INV + edge * 81 + 20, (1 - 2 * (x * x + z * z)) * envelope, valid)
    tl.store(INV + edge * 81 + 29, 2 * (y * z - w * x) * envelope, valid)
    tl.store(INV + edge * 81 + 12, 2 * (x * z - w * y) * envelope, valid)
    tl.store(INV + edge * 81 + 21, 2 * (y * z + w * x) * envelope, valid)
    tl.store(INV + edge * 81 + 30, (1 - 2 * (x * x + y * y)) * envelope, valid)


@triton_op("fairchem::hip_frozen_wigner", mutates_args={})
def rotations(
    v: torch.Tensor,
    distances: torch.Tensor,
    gamma: torch.Tensor,
    powers: torch.Tensor,
    coefficients: torch.Tensor,
    cutoff: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    n = v.shape[0]
    w = v.new_empty((n, 9, 9))
    inverse = torch.empty_like(w)
    if n:
        wrap_triton(_rotation)[(triton.cdiv(n, 64),)](
            v,
            distances,
            gamma,
            powers,
            coefficients,
            w,
            inverse,
            n,
            cutoff,
            64,
            num_warps=4,
        )
    return w, inverse


def enable_fused_wigner(head, enabled=True):
    """Prepare 35 independent quartic monomials instead of 256 products."""
    if enabled and (head.backbone.lmax != 2 or not head.backbone.use_quaternion_wigner):
        raise ValueError("Fused geometry requires quaternion rotations at lmax=2")
    if enabled and head.backbone.envelope.p != 5.0:
        raise ValueError("Fused geometry requires the standard polynomial envelope")
    if enabled and not hasattr(head, "hip_wigner_coefficients"):
        c = (
            head.backbone.wigner_data.custom_kernels.C_l2.detach()
            .cpu()
            .double()
            .reshape(25, 4, 4, 4, 4)
        )
        terms = list(itertools.combinations_with_replacement(range(4), 4))
        powers = [tuple(t.count(k) for k in range(4)) for t in terms]
        lookup = {p: i for i, p in enumerate(powers)}
        coefficients = torch.zeros(35, 25, dtype=torch.float64)
        for indices in itertools.product(range(4), repeat=4):
            power = tuple(indices.count(k) for k in range(4))
            coefficients[lookup[power]] += c[(slice(None), *indices)]
        device = head.node_projection.weight.device
        head.register_buffer(
            "hip_wigner_powers",
            torch.tensor(powers, dtype=torch.int32, device=device),
            persistent=False,
        )
        head.register_buffer(
            "hip_wigner_coefficients", coefficients.float().to(device), persistent=False
        )
    head.fused_wigner = enabled


def frozen_wigner(head, vectors, distances, gamma=None):
    """Use FP32 geometry and retain the original random roll distribution."""
    if vectors.requires_grad or vectors.dtype != torch.float32:
        raise ValueError("Fused geometry requires frozen FP32 coordinates")
    if gamma is None:
        gamma = torch.rand(len(vectors), device=vectors.device) * (2 * torch.pi)
    return rotations(
        vectors,
        distances,
        gamma,
        head.hip_wigner_powers,
        head.hip_wigner_coefficients,
        float(head.cutoff_hessian),
    )
