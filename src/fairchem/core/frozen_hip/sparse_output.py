"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl


@triton.jit
def _upper_indices(INDEX, N: tl.constexpr, K: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0).to(tl.int64)
    start = i * (2 * N - i - 1) // 2
    for offset in range(tl.cdiv(N - i - 1, B)):
        j = i + 1 + offset * B + tl.arange(0, B)
        slot = start + j - i - 1
        tl.store(INDEX + slot, i, j < N)
        tl.store(INDEX + K + slot, j, j < N)


@triton.jit
def _accumulate(
    EI,
    BLOCKS,
    INDEX,
    PTR,
    VALUES,
    ROWS,
    E,
    N: tl.constexpr,
    K,
    FULL: tl.constexpr,
    B: tl.constexpr,
):
    q = tl.program_id(0) * B + tl.arange(0, B)
    e, c = q // 9, q % 9
    valid = e < E
    s = tl.load(EI + e, valid, 0).to(tl.int64)
    t = tl.load(EI + E + e, valid, 0).to(tl.int64)
    i, j = tl.minimum(s, t), tl.maximum(s, t)
    if FULL:
        slot = i * (2 * N - i - 1) // 2 + j - i - 1
    else:
        low = tl.load(PTR + i)
        high = tl.load(PTR + i + 1)
        for _ in range(32):
            mid = (low + high) // 2
            col = tl.load(INDEX + K + mid, valid & (mid < high), 2147483647)
            right = col < j
            low = tl.where(right, mid + 1, low)
            high = tl.where(right, high, mid)
        slot = low
    transposed = (c % 3) * 3 + c // 3
    out_c = tl.where(s < t, c, transposed)
    value = tl.load(BLOCKS + q, valid, 0)
    tl.atomic_add(VALUES + slot * 9 + out_c, value, valid)
    tl.atomic_add(ROWS + s * 9 + c, value, valid)
    tl.atomic_add(ROWS + t * 9 + transposed, value, valid)


@dataclass
class HessianBlocks:
    """
    Store int32 pair indices and FP32 Cartesian blocks with implicit projection.

    For projection: H_ij = raw_ij - row_mean[i] - row_mean[j].T + grand_mean.
    Missing raw off-diagonal pairs are zero. Diagonal raw blocks are separate.
    """

    indices: torch.Tensor
    values: torch.Tensor
    row_ptr: torch.Tensor
    diagonal: torch.Tensor
    row_mean: torch.Tensor
    grand_mean: torch.Tensor
    projected: bool = True

    @property
    def natoms(self):
        return len(self.diagonal)

    @property
    def storage_bytes(self):
        return sum(
            t.numel() * t.element_size()
            for t in (
                self.indices,
                self.values,
                self.row_ptr,
                self.diagonal,
                self.row_mean,
                self.grand_mean,
            )
        )

    def block(self, i, j):
        n = self.natoms
        if not (0 <= i < n and 0 <= j < n):
            raise IndexError((i, j))
        if i == j:
            out = self.diagonal[i].clone()
        else:
            a, b = min(i, j), max(i, j)
            start, end = int(self.row_ptr[a]), int(self.row_ptr[a + 1])
            columns = self.indices[1, start:end]
            k = int(torch.searchsorted(columns.contiguous(), b))
            present = k < len(columns) and int(columns[k]) == b
            out = (
                self.values[start + k].clone()
                if present
                else torch.zeros_like(self.diagonal[0])
            )
            if i > j:
                out = out.T
        if self.projected:
            out = out - self.row_mean[i] - self.row_mean[j].T + self.grand_mean
        return out

    def as_dict(self):
        return dict(
            format="hip_symmetric_block_coo_v1",
            natoms=self.natoms,
            indices=self.indices,
            values=self.values,
            row_ptr=self.row_ptr,
            diagonal=self.diagonal,
            row_mean=self.row_mean,
            grand_mean=self.grand_mean,
            projected=self.projected,
        )

    def save(self, path):
        torch.save(self.as_dict(), path)

    @classmethod
    def load(cls, path, map_location=None):
        record = torch.load(path, map_location=map_location, weights_only=True)
        if record.pop("format") != "hip_symmetric_block_coo_v1":
            raise ValueError("Unknown format")
        n = record.pop("natoms")
        out = cls(**record)
        if out.natoms != n:
            raise ValueError("Invalid atom count")
        return out

    def matvec(self, vector, chunk_pairs=65536):
        original = vector.shape
        v = vector.reshape(self.natoms, 3)
        if self.projected:
            v = v - v.mean(0, keepdim=True)
        out = torch.bmm(self.diagonal, v[:, :, None]).squeeze(-1)
        for begin in range(0, len(self.values), chunk_pairs):
            index = self.indices[:, begin : begin + chunk_pairs].long()
            blocks = self.values[begin : begin + chunk_pairs]
            out.index_add_(
                0, index[0], torch.bmm(blocks, v[index[1], :, None]).squeeze(-1)
            )
            out.index_add_(
                0,
                index[1],
                torch.bmm(blocks.transpose(-1, -2), v[index[0], :, None]).squeeze(-1),
            )
        if self.projected:
            out = out - out.mean(0, keepdim=True)
        return out.reshape(original)

    def to_dense(self):
        """
        Explicitly construct the full Hessian for consumers or small reference tests.
        """
        n = self.natoms
        h = self.diagonal.new_zeros((n, 3, n, 3))
        ids = torch.arange(n, device=h.device)
        h[ids, :, ids, :] = self.diagonal
        i, j = self.indices.long()
        h[i, :, j, :] = self.values
        h[j, :, i, :] = self.values.transpose(-1, -2)
        if self.projected:
            h = (
                h
                - self.row_mean[:, :, None, :]
                - self.row_mean.transpose(-1, -2)[None].permute(0, 2, 1, 3)
                + self.grand_mean[None, :, None, :]
            )
        return h.reshape(3 * n, 3 * n)

    def finite(self, chunk_pairs=65536):
        ok = (
            torch.isfinite(self.diagonal).all()
            & torch.isfinite(self.row_mean).all()
            & torch.isfinite(self.grand_mean).all()
        )
        for begin in range(0, len(self.values), chunk_pairs):
            ok = ok & torch.isfinite(self.values[begin : begin + chunk_pairs]).all()
        return bool(ok.item())


def allocate(data, diagonal, chunk_edges, cutoff):
    from fairchem.core.frozen_hip.streamed_head import pair_chunk

    n = len(data.pos)
    if cutoff is None:
        k = n * (n - 1) // 2
        indices = torch.empty((2, k), device=data.pos.device, dtype=torch.int32)
        _upper_indices[(n,)](indices, n, k, 256)
        rows = torch.arange(n + 1, device=data.pos.device, dtype=torch.int64)
        ptr = rows * (2 * n - rows - 1) // 2
    else:
        pieces = []
        counts = torch.zeros(n, device=data.pos.device, dtype=torch.int64)
        for begin in range(0, n * (n - 1), chunk_edges):
            edges, _, _, _ = pair_chunk(
                data.pos, begin, min(begin + chunk_edges, n * (n - 1)), cutoff
            )
            keep = edges[0] > edges[1]
            upper = edges[:, keep].flip(0).to(torch.int32)
            if upper.shape[1]:
                pieces.append(upper)
                counts.add_(torch.bincount(upper[0].long(), minlength=n))
        indices = (
            torch.cat(pieces, dim=1)
            if pieces
            else torch.empty((2, 0), device=data.pos.device, dtype=torch.int32)
        )
        del pieces
        k = indices.shape[1]
        ptr = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
    values = diagonal.new_zeros((k, 3, 3), dtype=torch.float32)
    rows = diagonal.clone().float()
    return indices, values, ptr, rows


def accumulate(edges, blocks, indices, values, ptr, rows, n, full):
    e = edges.shape[1]
    k = indices.shape[1]
    _accumulate[(triton.cdiv(e * 9, 256),)](
        edges, blocks, indices, ptr, values, rows, e, n, k, full, 256
    )


def finish(indices, values, ptr, diagonal, rows, project):
    mean = rows / len(diagonal)
    return HessianBlocks(indices, values, ptr, diagonal, mean, mean.mean(0), project)


@triton.jit
def _write_paired(
    BLOCKS, INDEX, VALUES, ROWS, BEGIN, K, P: tl.constexpr, B: tl.constexpr
):
    q = tl.program_id(0) * B + tl.arange(0, B)
    pair, component = q // 9, q % 9
    valid = pair < P
    transposed = (component % 3) * 3 + component // 3
    # First half is j->i; second half is i->j for canonical i<j.
    reverse = tl.load(BLOCKS + pair * 9 + transposed, valid, 0)
    forward = tl.load(BLOCKS + (P + pair) * 9 + component, valid, 0)
    value = forward + reverse
    slot = BEGIN + pair.to(tl.int64)
    i = tl.load(INDEX + slot, valid, 0).to(tl.int64)
    j = tl.load(INDEX + K + slot, valid, 0).to(tl.int64)
    tl.store(VALUES + slot * 9 + component, value, valid)
    tl.atomic_add(ROWS + i * 9 + component, value, valid)
    tl.atomic_add(ROWS + j * 9 + transposed, value, valid)


def write_paired_blocks(blocks, indices, values, rows, begin, pairs):
    """
    Write each pair once at its known slot, with no sparse-index binary searches.
    """
    if pairs:
        _write_paired[(triton.cdiv(pairs * 9, 256),)](
            blocks, indices, values, rows, begin, indices.shape[1], pairs, 256
        )
