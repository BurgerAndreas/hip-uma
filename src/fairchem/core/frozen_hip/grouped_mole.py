"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Execute frozen, composition-dependent expert layers with grouped tensor cores.
"""

from __future__ import annotations

from types import MethodType

import torch
import triton
import triton.language as tl

from fairchem.core.frozen_hip.kernels import _owner
from fairchem.core.models.uma.nn.mole import MOLE


@triton.jit
def _grouped_mm(
    X,
    W,
    BIAS,
    ROWS,
    TILES,
    Y,
    K: tl.constexpr,
    OUT: tl.constexpr,
    B,
    WS0,
    WS1,
    WS2,
    HALF: tl.constexpr,
    BIAS_PRESENT: tl.constexpr,
    PRECISION: tl.constexpr,
    LEVELS: tl.constexpr,
    BM: tl.constexpr = 32,
    BN: tl.constexpr = 64,
    BK: tl.constexpr = 32,
):
    tile = tl.program_id(0)
    group = _owner(tile, TILES, B, LEVELS)
    first, last = tl.load(ROWS + group), tl.load(ROWS + group + 1)
    local = tile - tl.load(TILES + group)
    row = first + local * BM + tl.arange(0, BM)
    col = tl.program_id(1) * BN + tl.arange(0, BN)
    reduction = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), tl.float32)
    for start in range(tl.cdiv(K, BK)):
        kk = start * BK + reduction
        a = tl.load(
            X + row[:, None] * K + kk[None, :],
            (row[:, None] < last) & (kk[None, :] < K),
            0,
        )
        w = tl.load(
            W + group * WS0 + col[None, :] * WS1 + kk[:, None] * WS2,
            (col[None, :] < OUT) & (kk[:, None] < K),
            0,
        )
        if HALF:
            a = a.to(tl.float16)
            w = w.to(tl.float16)
        acc += tl.dot(a, w, input_precision=PRECISION)
    if BIAS_PRESENT:
        bias = tl.load(BIAS + col, col < OUT, 0)
        if HALF:
            bias = bias.to(tl.float16).to(tl.float32)
        acc += bias[None, :]
    tl.store(
        Y + row[:, None] * OUT + col[None, :],
        acc,
        (row[:, None] < last) & (col[None, :] < OUT),
    )


def grouped_linear(x, weights, bias, globals_, *, fp32_output=False):
    """
    Apply each system's own merged weights without padding or Python GEMM loops.
    """
    if torch.is_grad_enabled() and (x.requires_grad or weights.requires_grad):
        raise ValueError("Grouped frozen expert inference does not support gradients")
    half = (
        torch.is_autocast_enabled("cuda")
        and torch.get_autocast_dtype("cuda") == torch.float16
    )
    if torch.is_autocast_enabled("cuda") and not half:
        raise ValueError("Use FP32 or FP16 with grouped frozen experts")
    bm, bn, bk = (64, 128, 64) if half else (32, 64, 32)
    factor = 1 if x.ndim == 2 else x.shape[1]
    flat = x.reshape(-1, x.shape[-1]).contiguous()
    sizes = globals_.mole_sizes
    if int(sizes.sum()) * factor != flat.shape[0] or globals_.ac_start_idx != 0:
        raise ValueError(
            "Grouped inference requires complete, unchunked system segments"
        )
    previous = getattr(globals_, "_hip_grouped_cache", None)
    if previous is None or previous[0] is not sizes:
        previous = (sizes, {})
        globals_._hip_grouped_cache = previous
    key = (factor, bm)
    if key not in previous[1]:
        rows = torch.cat(
            (torch.zeros(1, dtype=torch.int64), sizes.long().cumsum(0) * factor)
        )
        tile_counts = triton.cdiv(sizes.long() * factor, bm)
        tiles = torch.cat((torch.zeros(1, dtype=torch.int64), tile_counts.cumsum(0)))
        previous[1][key] = (
            rows.to(x.device, non_blocking=True),
            tiles.to(x.device, non_blocking=True),
            int(tiles[-1]),
        )
    rows, tiles, count = previous[1][key]
    if half:
        flat = flat.to(torch.float16)
        # Expert contraction returns interleaved batch strides. Pack once rather
        # than repeating strided FP32 loads and casts in every row tile.
        weights = weights.to(dtype=torch.float16, memory_format=torch.contiguous_format)
    out = torch.empty(
        (*x.shape[:-1], weights.shape[1]),
        device=x.device,
        dtype=torch.float32 if fp32_output else torch.float16 if half else x.dtype,
    )
    if flat.shape[0]:
        _grouped_mm[(count, triton.cdiv(weights.shape[1], bn))](
            flat,
            weights,
            bias if bias is not None else flat,
            rows,
            tiles,
            out,
            flat.shape[1],
            weights.shape[1],
            len(sizes),
            *weights.stride(),
            half,
            bias is not None,
            "tf32x3" if torch.backends.cuda.matmul.allow_tf32 else "ieee",
            (len(sizes) + 1).bit_length(),
            BM=bm,
            BN=bn,
            BK=bk,
            num_warps=4,
        )
    return out


def _forward(self, x):
    with torch.autocast(device_type=x.device.type, enabled=False):
        weights = torch.einsum(
            "eoi,be->boi",
            self.weights,
            self.global_mole_tensors.expert_mixing_coefficients,
        )
    return grouped_linear(
        x,
        weights,
        self.bias,
        self.global_mole_tensors,
        fp32_output=self._hip_fp32_output,
    )


def enable_grouped_mole(backbone, enabled=True):
    """
    Keep expert mixing and parameters intact; replace frozen GEMM dispatch only.
    """
    count = 0
    for name, module in backbone.named_modules():
        if isinstance(module, MOLE):
            # The second convolution can exceed FP16 range before its node sum.
            # Keep tensor-core inputs in FP16 and retain its FP32 accumulator.
            module._hip_fp32_output = ".so2_conv_2." in name
            module.forward = MethodType(_forward if enabled else MOLE.forward, module)
            count += 1
    return count
