"""FP32 skinny projections that read FP16 activations without wide casts."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton


@triton.jit
def _forward(
    X,
    W,
    B,
    Y,
    M,
    SX,
    SY,
    K: tl.constexpr,
    N: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    BIAS: tl.constexpr,
):
    row = (tl.program_id(0) * BM + tl.arange(0, BM)).to(tl.int64)
    k = tl.arange(0, BK)
    x = tl.load(
        X + row[:, None] * SX + k[None, :], (row[:, None] < M) & (k[None, :] < K), 0
    ).to(tl.float32)
    for n in tl.static_range(N):
        weight = tl.load(W + n * K + k, k < K, 0)
        value = tl.sum(x * weight[None, :], axis=1)
        if BIAS:
            value += tl.load(B + n)
        tl.store(Y + row * SY + n, value, row < M)


@triton.jit
def _input_gradient(
    G, W, DX, M, K: tl.constexpr, N: tl.constexpr, SG, SDX, BLOCK: tl.constexpr
):
    index = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    # Inductor sometimes specializes M to a Python integer. Promote through a
    # tensor so both specialized and runtime sizes retain wide arithmetic.
    mask = index < (tl.full((), 0, tl.int64) + M) * K
    row, channel = index // K, index % K
    value = tl.full((BLOCK,), 0, tl.float32)
    for n in tl.static_range(N):
        g = tl.load(G + row * SG + n, mask, 0)
        w = tl.load(W + n * K + channel, mask, 0)
        value += g * w
    tl.store(DX + row * SDX + channel, value, mask)


@triton.jit
def _weight_gradient(
    X,
    G,
    PARTIAL,
    M,
    SX,
    SG,
    K: tl.constexpr,
    N: tl.constexpr,
    BE: tl.constexpr,
    BC: tl.constexpr,
):
    channel = tl.program_id(0) * BC + tl.arange(0, BC)
    split = tl.program_id(1)
    row = (split * BE + tl.arange(0, BE)).to(tl.int64)
    x = tl.load(
        X + row[:, None] * SX + channel[None, :],
        (row[:, None] < M) & (channel[None, :] < K),
        0,
    ).to(tl.float32)
    for n in tl.static_range(N):
        g = tl.load(G + row * SG + n, row < M, 0)
        value = tl.sum(x * g[:, None], axis=0)
        tl.store(PARTIAL + (split * N + n) * K + channel, value, channel < K)


@triton.jit
def _reduce_weight_gradient(
    PARTIAL, DW, SPLITS, D: tl.constexpr, BR: tl.constexpr, BC: tl.constexpr
):
    channel = tl.program_id(0) * BC + tl.arange(0, BC)
    row = tl.arange(0, BR)
    total = tl.zeros((BR, BC), tl.float32)
    start = 0
    while start < SPLITS:
        split = start + row
        total += tl.load(
            PARTIAL + split[:, None] * D + channel[None, :],
            (split[:, None] < SPLITS) & (channel[None, :] < D),
            0,
        )
        start += BR
    tl.store(DW + channel, tl.sum(total, axis=0), channel < D)


@triton_op("fairchem::hip_skinny_forward", mutates_args={})
def skinny_forward(
    x: torch.Tensor,
    w: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    m, k = x.shape
    n = w.shape[0]
    out = torch.empty((m, n), device=x.device, dtype=torch.float32)
    wrap_triton(_forward)[(triton.cdiv(m, 2),)](
        x,
        w,
        bias if bias is not None else w,
        out,
        m,
        x.stride(0),
        n,
        k,
        n,
        2,
        triton.next_power_of_2(k),
        bias is not None,
        num_warps=4,
    )
    return out


@triton_op("fairchem::hip_skinny_backward", mutates_args={})
def skinny_backward(
    x: torch.Tensor,
    w: torch.Tensor,
    gradient: torch.Tensor,
    need_x: bool,
    need_w: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    m, k = x.shape
    n = w.shape[0]
    dx = (
        torch.empty((m, k), device=x.device, dtype=x.dtype)
        if need_x
        else x.new_empty(0)
    )
    dw = torch.empty_like(w) if need_w else w.new_empty(0)
    if need_x:
        wrap_triton(_input_gradient)[(triton.cdiv(m * k, 512),)](
            gradient,
            w,
            dx,
            m,
            k,
            n,
            gradient.stride(0),
            k,
            512,
            num_warps=4,
        )
    if need_w:
        splits = triton.cdiv(m, 128)
        partial = torch.empty((splits, n, k), device=x.device, dtype=torch.float32)
        wrap_triton(_weight_gradient)[(triton.cdiv(k, 32), splits)](
            x,
            gradient,
            partial,
            m,
            x.stride(0),
            gradient.stride(0),
            k,
            n,
            128,
            32,
            num_warps=4,
        )
        wrap_triton(_reduce_weight_gradient)[(triton.cdiv(n * k, 32),)](
            partial,
            dw,
            splits,
            n * k,
            64,
            32,
            num_warps=4,
        )
    return dx, dw


@triton_op("fairchem::hip_skinny_project_forward", mutates_args={})
def skinny_project_forward(
    messages: torch.Tensor,
    w0: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    m, _, channels = messages.shape
    out = torch.empty((m, 9), device=messages.device, dtype=torch.float32)
    flat = messages.flatten(1)
    x_start = y_start = 0
    for w in (w0, w1, w2):
        n, k = w.shape
        x, y = flat[:, x_start : x_start + k], out[:, y_start : y_start + n]
        b = bias if y_start == 0 else None
        wrap_triton(_forward)[(triton.cdiv(m, 2),)](
            x,
            w,
            b if b is not None else w,
            y,
            m,
            9 * channels,
            9,
            k,
            n,
            2,
            triton.next_power_of_2(k),
            b is not None,
            num_warps=4,
        )
        x_start += k
        y_start += n
    return out.unsqueeze(-1)


@triton_op("fairchem::hip_skinny_project_backward", mutates_args={})
def skinny_project_backward(
    messages: torch.Tensor,
    w0: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    gradient: torch.Tensor,
    need_x: bool,
    need_w0: bool,
    need_w1: bool,
    need_w2: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    m, _, channels = messages.shape
    flat = messages.flatten(1)
    g = gradient.reshape(m, 9)
    dx = torch.empty_like(messages) if need_x else messages.new_empty(0)
    flat_dx = dx.flatten(1) if need_x else dx
    weights = (w0, w1, w2)
    needs = (need_w0, need_w1, need_w2)
    gradients = []
    x_start = y_start = 0
    for w, need_w in zip(weights, needs, strict=True):
        n, k = w.shape
        x = flat[:, x_start : x_start + k]
        dy = g[:, y_start : y_start + n]
        if need_x:
            target = flat_dx[:, x_start : x_start + k]
            wrap_triton(_input_gradient)[(triton.cdiv(m * k, 512),)](
                dy,
                w,
                target,
                m,
                k,
                n,
                9,
                9 * channels,
                512,
                num_warps=4,
            )
        dw = torch.empty_like(w) if need_w else w.new_empty(0)
        if need_w:
            splits = triton.cdiv(m, 128)
            partial = torch.empty((splits, n, k), device=x.device, dtype=torch.float32)
            wrap_triton(_weight_gradient)[(triton.cdiv(k, 32), splits)](
                x,
                dy,
                partial,
                m,
                9 * channels,
                9,
                k,
                n,
                128,
                32,
                num_warps=4,
            )
            wrap_triton(_reduce_weight_gradient)[(triton.cdiv(n * k, 32),)](
                partial,
                dw,
                splits,
                n * k,
                64,
                32,
                num_warps=4,
            )
        gradients.append(dw)
        x_start += k
        y_start += n
    return dx, gradients[0], gradients[1], gradients[2]


class _SkinnyProject(torch.autograd.Function):
    @staticmethod
    def forward(ctx, messages, w0, w1, w2, bias):
        ctx.save_for_backward(messages, w0, w1, w2)
        ctx.has_bias = bias is not None
        return skinny_project_forward(messages, w0, w1, w2, bias)

    @staticmethod
    def backward(ctx, gradient):
        messages, w0, w1, w2 = ctx.saved_tensors
        gradient = gradient.contiguous()
        needs = ctx.needs_input_grad
        dx, dw0, dw1, dw2 = skinny_project_backward(
            messages,
            w0,
            w1,
            w2,
            gradient,
            *needs[:4],
        )
        db = gradient[:, :3, 0].sum(0) if ctx.has_bias and needs[4] else None
        return (
            dx if needs[0] else None,
            dw0 if needs[1] else None,
            dw1 if needs[2] else None,
            dw2 if needs[3] else None,
            db,
        )


def skinny_project(messages, weights, bias=None):
    """Project all SO2 components with one shared activation-gradient buffer."""
    if messages.ndim != 3 or messages.shape[1] != 9 or not messages.is_contiguous():
        raise ValueError("Skinny projection requires contiguous degree-two messages")
    if messages.dtype != torch.float16 or not messages.is_cuda:
        raise ValueError("Skinny projection requires FP16 CUDA activations")
    channels = messages.shape[-1]
    for weight, n, width in zip(weights, (3, 4, 2), (3, 4, 2), strict=True):
        if weight.shape != (n, width * channels) or weight.dtype != torch.float32:
            raise ValueError("Skinny projection weights have incompatible shapes")
    if messages.shape[0] == 0:
        parts = messages.split((3, 4, 2), dim=1)
        values = [
            torch.nn.functional.linear(
                p.flatten(1).float(), w, bias if i == 0 else None
            )
            for i, (p, w) in enumerate(zip(parts, weights, strict=True))
        ]
        return torch.cat(values, dim=1).unsqueeze(-1)
    return _SkinnyProject.apply(messages, *(w.contiguous() for w in weights), bias)


class _SkinnyLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias):
        ctx.save_for_backward(x, weight)
        ctx.has_bias = bias is not None
        return skinny_forward(x, weight, bias)

    @staticmethod
    def backward(ctx, gradient):
        x, weight = ctx.saved_tensors
        gradient = gradient.contiguous()
        dx, dw = skinny_backward(
            x,
            weight,
            gradient,
            ctx.needs_input_grad[0],
            ctx.needs_input_grad[1],
        )
        db = gradient.sum(0) if ctx.has_bias and ctx.needs_input_grad[2] else None
        return (
            dx if ctx.needs_input_grad[0] else None,
            dw if ctx.needs_input_grad[1] else None,
            db,
        )


def skinny_linear(x, weight, bias=None):
    """Keep output, weight products, and weight gradients in FP32.

    This operation supports strided rows, including views of the packed SO2
    messages. It never casts the trainable projection weights to FP16.
    """
    if (
        not x.is_cuda
        or x.dtype != torch.float16
        or weight.dtype != torch.float32
        or x.ndim != 2
        or x.stride(1) != 1
        or weight.shape[0] > 4
        or x.shape[0] == 0
        or weight.shape[0] == 0
    ):
        return torch.nn.functional.linear(x.float(), weight, bias)
    return _SkinnyLinear.apply(x, weight.contiguous(), bias)
