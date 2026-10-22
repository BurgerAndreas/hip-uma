"""Experimental trainable fusions inspired by UMA Nitro; production stays unchanged.

These kernels include parameter gradients. They do not import Nitro's frozen
inference operators. FP16 cast boundaries match the current HIP implementation.
"""

from __future__ import annotations

from itertools import pairwise
from types import MethodType

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton

from fairchem.core.frozen_hip import edge_precision as ep
from fairchem.core.frozen_hip.linear import half_mm_float_output as vendor_mm
from fairchem.core.models.uma.nn.activation import GateActivation


def nitro_scaled_quantize(value):
    """Use power-of-two scaling with headroom below FP16's maximum exponent."""
    maximum = value.detach().abs().amax()
    exponent = torch.floor(torch.log2(maximum.clamp_min(2.0**-100))) - 14
    scale = torch.exp2(exponent)
    return (value / scale).half(), scale


@triton.jit
def _dense_dw(
    X,
    G,
    DW,
    R,
    I: tl.constexpr,
    O: tl.constexpr,
    SX: tl.constexpr,
    SG: tl.constexpr,
    CHUNK: tl.constexpr,
):
    tile, split = tl.program_id(0), tl.program_id(1)
    oo = (tile // tl.cdiv(I, 32)) * 32 + tl.arange(0, 32)
    ii = (tile % tl.cdiv(I, 32)) * 32 + tl.arange(0, 32)
    k = tl.arange(0, 32)
    total = tl.zeros((32, 32), tl.float32)
    # Bound the Tensor Core accumulation length, then combine partials in FP32.
    for group in range(CHUNK // 512):
        partial = tl.zeros((32, 32), tl.float32)
        for part in range(512 // 32):
            rows = split * CHUNK + group * 512 + part * 32 + k
            g = tl.load(
                G + rows[None, :] * SG + oo[:, None],
                (rows[None, :] < R) & (oo[:, None] < O),
                0,
            )
            x = tl.load(
                X + rows[:, None] * SX + ii[None, :],
                (rows[:, None] < R) & (ii[None, :] < I),
                0,
            )
            partial = tl.dot(g, x, partial)
        total += partial
    tl.atomic_add(
        DW + oo[:, None] * I + ii[None, :],
        total,
        (oo[:, None] < O) & (ii[None, :] < I),
        sem="relaxed",
    )


@triton_op("fairchem::hip_dense_split_weight_gradient", mutates_args={})
def dense_weight_gradient(x: torch.Tensor, g: torch.Tensor, chunk: int) -> torch.Tensor:
    """Split long weight-gradient reductions; keep row strides and FP32 outputs."""
    r, i, o = x.shape[0], x.shape[1], g.shape[1]
    result = torch.zeros((o, i), device=x.device, dtype=torch.float32)
    wrap_triton(_dense_dw)[
        (triton.cdiv(o, 32) * triton.cdiv(i, 32), triton.cdiv(r, chunk))
    ](x, g, result, r, i, o, x.stride(0), g.stride(0), chunk, num_warps=4)
    return result


_DENSE_CONFIG = {"chunk": 2048}


def split_weight_mm(a, b):
    """Select only transposed long-reduction FP16 products from HalfLinear backward."""
    if (
        a.is_cuda
        and a.dtype == b.dtype == torch.float16
        and a.ndim == b.ndim == 2
        and a.stride(0) == 1
        and b.stride(1) == 1
        and a.shape[1] >= 8192
    ):
        return dense_weight_gradient(b, a.t(), _DENSE_CONFIG["chunk"])
    return vendor_mm(a, b)


def enable_dense_split_dw(chunk=2048):
    """Keep forward and input-gradient GEMMs on the existing vendor implementation."""
    if chunk not in (512, 2048, 8192):
        raise ValueError("Split reduction chunk must be 512, 2048, or 8192")
    _DENSE_CONFIG["chunk"] = chunk
    ep.half_mm_float_output = split_weight_mm


@triton.jit
def _gate_fwd(G, X, I, Y, N: tl.constexpr, C: tl.constexpr, B: tl.constexpr):
    p = tl.program_id(0) * B + tl.arange(0, B)
    row, k, c = p // (9 * C), (p // C) % 9, p % C
    x = tl.load(X + p, p < N, 0).to(tl.float32)
    idx = tl.load(I + tl.maximum(k - 1, 0))
    g = tl.load(G + row * 2 * C + idx * C + c, p < N, 0).to(tl.float32)
    # The existing sigmoid materializes in the input dtype before multiplication.
    s = (1 / (1 + tl.exp(-g))).to(Y.dtype.element_ty).to(tl.float32)
    y = tl.where(k == 0, x / (1 + tl.exp(-x)), x * s)
    tl.store(Y + p, y, p < N)


@triton.jit
def _gate_bwd(G, X, I, DY, DX, DG, R: tl.constexpr, C: tl.constexpr, BC: tl.constexpr):
    row = tl.program_id(0)
    k = tl.arange(0, 16)[:, None]
    c0 = tl.program_id(1) * BC + tl.arange(0, BC)
    c = c0[None, :]
    mask = (k < 9) & (c < C)
    p = (row * 9 + k) * C + c
    x = tl.load(X + p, mask, 0).to(tl.float32)
    dy = tl.load(DY + p, mask, 0).to(tl.float32)
    idx = tl.load(I + tl.maximum(tl.minimum(k - 1, 7), 0))
    g = tl.load(G + row * 2 * C + idx * C + c, c < C, 0).to(tl.float32)
    sg = (1 / (1 + tl.exp(-g))).to(G.dtype.element_ty).to(tl.float32)
    sx = 1 / (1 + tl.exp(-x))
    dx = tl.where(k == 0, dy * sx * (1 + x * (1 - sx)), dy * sg)
    tl.store(DX + p, dx, mask)
    # Match the dtype of each multiplication and the reduction in index_select backward.
    q = (dy * x).to(G.dtype.element_ty).to(tl.float32)
    for degree in tl.static_range(2):
        ds = tl.sum(tl.where((k > 0) & (k < 9) & (idx == degree), q, 0), 0)
        ds = ds.to(G.dtype.element_ty).to(tl.float32)
        gv = tl.load(G + row * 2 * C + degree * C + c0, c0 < C, 0).to(tl.float32)
        s = (1 / (1 + tl.exp(-gv))).to(G.dtype.element_ty).to(tl.float32)
        tl.store(DG + row * 2 * C + degree * C + c0, ds * s * (1 - s), c0 < C)


class _Gate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, g, x, index):
        g, x = g.contiguous(), x.contiguous()
        y = torch.empty_like(x)
        _gate_fwd[(triton.cdiv(x.numel(), 256),)](
            g, x, index, y, x.numel(), x.shape[-1], 256
        )
        ctx.save_for_backward(g, x, index)
        return y

    @staticmethod
    def backward(ctx, dy):
        g, x, index = ctx.saved_tensors
        dx, dg = torch.empty_like(x), torch.empty_like(g)
        _gate_bwd[(x.shape[0], triton.cdiv(x.shape[-1], 64))](
            g,
            x,
            index,
            dy.contiguous(),
            dx,
            dg,
            x.shape[0],
            x.shape[-1],
            64,
            num_warps=4,
        )
        return dg, dx, None


@triton.jit
def _ln_fwd(
    X, W, BIAS, Y, Z, MEAN, INV, C: tl.constexpr, EPS: tl.constexpr, BC: tl.constexpr
):
    row = tl.program_id(0)
    c = tl.arange(0, BC)
    x = tl.load(X + row * C + c, c < C, 0).to(tl.float32)
    mean = tl.sum(x, 0) / C
    centered = tl.where(c < C, x - mean, 0)
    inv = tl.rsqrt(tl.sum(centered * centered, 0) / C + EPS)
    w = tl.load(W + c, c < C, 0)
    bias = tl.load(BIAS + c, c < C, 0)
    z = (centered * inv * w + bias).to(Z.dtype.element_ty).to(tl.float32)
    tl.store(Z + row * C + c, z, c < C)
    tl.store(Y + row * C + c, z / (1 + tl.exp(-z)), c < C)
    tl.store(MEAN + row, mean)
    tl.store(INV + row, inv)


@triton.jit
def _ln_dx(X, W, Z, MEAN, INV, DY, DX, C: tl.constexpr, BC: tl.constexpr):
    row = tl.program_id(0)
    c = tl.arange(0, BC)
    x = tl.load(X + row * C + c, c < C, 0).to(tl.float32)
    mean, inv = tl.load(MEAN + row), tl.load(INV + row)
    xhat = (x - mean) * inv
    z = tl.load(Z + row * C + c, c < C, 0).to(tl.float32)
    dy = tl.load(DY + row * C + c, c < C, 0).to(tl.float32)
    s = 1 / (1 + tl.exp(-z))
    dz = (dy * s * (1 + z * (1 - s))).to(Z.dtype.element_ty).to(tl.float32)
    w = tl.load(W + c, c < C, 0)
    v = dz * w
    v = tl.where(c < C, v, 0)
    a = tl.sum(v, 0) / C
    b = tl.sum(tl.where(c < C, v * xhat, 0), 0) / C
    tl.store(DX + row * C + c, inv * (v - a - xhat * b), c < C)


@triton.jit
def _ln_dw(
    X,
    Z,
    MEAN,
    INV,
    DY,
    DW,
    DB,
    R: tl.constexpr,
    C: tl.constexpr,
    BR: tl.constexpr,
    BC: tl.constexpr,
):
    r = tl.program_id(0) * BR + tl.arange(0, BR)[:, None]
    c0 = tl.program_id(1) * BC + tl.arange(0, BC)
    c = c0[None, :]
    mask = (r < R) & (c < C)
    x = tl.load(X + r * C + c, mask, 0).to(tl.float32)
    z = tl.load(Z + r * C + c, mask, 0).to(tl.float32)
    dy = tl.load(DY + r * C + c, mask, 0).to(tl.float32)
    mean = tl.load(MEAN + r, r < R, 0)
    inv = tl.load(INV + r, r < R, 0)
    s = 1 / (1 + tl.exp(-z))
    dz = (dy * s * (1 + z * (1 - s))).to(Z.dtype.element_ty).to(tl.float32)
    dz = tl.where(mask, dz, 0)
    dw = tl.sum(dz * (x - mean) * inv, 0)
    db = tl.sum(dz, 0)
    tl.atomic_add(DW + c0, dw, c0 < C, sem="relaxed")
    tl.atomic_add(DB + c0, db, c0 < C, sem="relaxed")


class _NormSiLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, b, eps):
        x = x.contiguous()
        c = x.shape[-1]
        rows = x.numel() // c
        y, z = torch.empty_like(x), torch.empty_like(x)
        mean = torch.empty(rows, device=x.device, dtype=torch.float32)
        inv = torch.empty_like(mean)
        _ln_fwd[(rows,)](x, w, b, y, z, mean, inv, c, eps, triton.next_power_of_2(c))
        ctx.save_for_backward(x, w, z, mean, inv)
        return y

    @staticmethod
    def backward(ctx, dy):
        x, w, z, mean, inv = ctx.saved_tensors
        c, rows = x.shape[-1], mean.numel()
        dx = torch.empty_like(x)
        dw, db = torch.zeros_like(w), torch.zeros_like(w)
        dy = dy.contiguous()
        _ln_dx[(rows,)](x, w, z, mean, inv, dy, dx, c, triton.next_power_of_2(c))
        _ln_dw[(triton.cdiv(rows, 128), triton.cdiv(c, 64))](
            x, z, mean, inv, dy, dw, db, rows, c, 128, 64, num_warps=8
        )
        return dx, dw, db, None


def gate_forward(self, g, x):
    return _Gate.apply(g, x, self.expand_index)


def sequential_forward(self, x):
    modules = list(self.children())
    i = 0
    while i < len(modules):
        m = modules[i]
        if (
            isinstance(m, torch.nn.LayerNorm)
            and i + 1 < len(modules)
            and isinstance(modules[i + 1], torch.nn.SiLU)
        ):
            x = _NormSiLU.apply(x, m.weight, m.bias, m.eps)
            i += 2
        else:
            x = m(x)
            i += 1
    return x


def enable_candidates(head, *, gate=False, norm=False):
    counts = {"gate": 0, "norm_silu": 0}
    for module in head.modules():
        if gate and isinstance(module, GateActivation):
            if module.lmax != 2 or module.mmax != 2:
                raise ValueError("Experimental gate requires lmax=mmax=2")
            module.forward = MethodType(gate_forward, module)
            counts["gate"] += 1
        if norm and isinstance(module, torch.nn.Sequential):
            parts = list(module.children())
            if any(
                isinstance(a, torch.nn.LayerNorm) and isinstance(b, torch.nn.SiLU)
                for a, b in pairwise(parts)
            ):
                module.forward = MethodType(sequential_forward, module)
                counts["norm_silu"] += 1
    return counts


def enable_wide_radial_grid():
    """Expose more edge parallelism without changing the FP32 reduction formulas."""

    @triton_op("fairchem::hip_nitro_wide_radial", mutates_args={})
    def forward(
        x: torch.Tensor, w: torch.Tensor, edges: torch.Tensor, r: torch.Tensor
    ) -> torch.Tensor:
        e, c = edges.shape[1], x.shape[-1]
        out = torch.empty((e, 9, 2 * c), device=x.device, dtype=torch.float16)
        wrap_triton(ep._forward)[(min(max(e, 1), 131072), triton.cdiv(2 * c, 64))](
            x, w, edges, r, out, e, c, 64, 131072, num_warps=2
        )
        return out

    @triton_op("fairchem::hip_nitro_wide_radial_grad", mutates_args={})
    def backward(
        x: torch.Tensor,
        w: torch.Tensor,
        edges: torch.Tensor,
        r: torch.Tensor,
        g: torch.Tensor,
        need_x: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        e, c = edges.shape[1], x.shape[-1]
        dr = torch.empty_like(r)
        dx = torch.zeros_like(x) if need_x else x.new_empty(0)
        grid = (min(max(e, 1), 131072), triton.cdiv(2 * c, 64))
        wrap_triton(ep._backward_radial)[grid](
            x, w, edges, g, dr, e, c, 64, 131072, num_warps=2
        )
        if need_x:
            wrap_triton(ep._backward_node)[grid](
                w, edges, r, g, dx, e, c, 64, 131072, num_warps=2
            )
        return dx, dr

    ep.mixed_radial = forward
    ep.mixed_radial_grad = backward
