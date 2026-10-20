"""Check translation projection values, constraints, covariance and derivatives."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from fairchem.core.frozen_hip.translation import (
    project_blocks,
    project_blocks_reference,
    project_dense,
)


def metadata(sizes, device):
    n = torch.tensor(sizes, device=device)
    ap = torch.cat((n.new_zeros(1), n.cumsum(0)))
    ep = torch.cat((n.new_zeros(1), (n * (n - 1)).cumsum(0)))
    edges = []
    a = 0
    for k in sizes:
        edges.extend((a + s, a + t) for t in range(k) for s in range(k) if s != t)
        a += k
    ei = (
        torch.tensor(edges, device=device, dtype=torch.long)
        .reshape(-1, 2)
        .T.contiguous()
    )
    return SimpleNamespace(
        hip_sizes=sizes,
        hip_atom_ptr=ap,
        hip_edge_ptr=ep,
        edge_index_hessian=ei,
        batch=torch.arange(len(sizes), device=device).repeat_interleave(n),
    )


def assemble(edge, node, data):
    src, dst = data.edge_index_hessian
    a = node.shape[0]
    blocks = node.new_zeros(a * a, 3, 3)
    blocks = blocks.index_add(0, src * a + dst, edge)
    blocks = blocks.index_add(0, dst * a + src, edge.transpose(-1, -2))
    i = torch.arange(a, device=node.device)
    blocks = blocks.index_add(0, i * a + i, node + node.transpose(-1, -2)).reshape(
        a, a, 3, 3
    )
    offset = 0
    parts = []
    for n in data.hip_sizes:
        parts.append(
            blocks[offset : offset + n, offset : offset + n]
            .permute(0, 2, 1, 3)
            .reshape(-1)
        )
        offset += n
    return torch.cat(parts)


@pytest.mark.parametrize("sizes", [[1], [1, 4, 2, 1], [17, 3], [350]])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_projection_assembly_gradients_and_sum_rule(sizes, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(4)
    data = metadata(sizes, device)
    e = torch.randn(
        sum(n * (n - 1) for n in sizes), 3, 3, device=device, requires_grad=True
    )
    n = torch.randn(sum(sizes), 3, 3, device=device, requires_grad=True)
    p = project_blocks(e, n, data)
    ref = project_blocks_reference(e, n, data)
    for x, y in zip(p, ref, strict=True):
        torch.testing.assert_close(x, y, atol=2e-6, rtol=2e-5)
    dense = assemble(*p, data)
    expected = project_dense(assemble(e, n, data), sizes)
    torch.testing.assert_close(dense, expected, atol=3e-6, rtol=3e-5)
    for flat, k in zip(dense.split([9 * k * k for k in sizes]), sizes, strict=True):
        h = flat.reshape(k, 3, k, 3)
        assert h.sum(2).norm() / h.norm().clamp_min(1) < 3e-5
        torch.testing.assert_close(h, h.permute(2, 3, 0, 1), atol=2e-6, rtol=2e-5)
    weights = [torch.randn_like(v) for v in p]
    g = torch.autograd.grad(
        sum((x * w).sum() for x, w in zip(p, weights, strict=True)),
        (e, n),
        retain_graph=True,
    )
    gr = torch.autograd.grad(
        sum((x * w).sum() for x, w in zip(ref, weights, strict=True)), (e, n)
    )
    for x, y in zip(g, gr, strict=True):
        torch.testing.assert_close(x, y, atol=3e-6, rtol=3e-5)
    twice = project_blocks(*p, data)
    for x, y in zip(twice, p, strict=True):
        torch.testing.assert_close(x, y, atol=2e-6, rtol=3e-5)


def test_dense_projection_explicit_matrix_and_rotation():
    torch.manual_seed(3)
    n = 7
    h = torch.randn(3 * n, 3 * n, dtype=torch.float64)
    h = (h + h.T) / 2
    p = torch.kron(
        torch.eye(n, dtype=h.dtype) - torch.ones(n, n, dtype=h.dtype) / n,
        torch.eye(3, dtype=h.dtype),
    )
    actual = project_dense(h, [n])
    torch.testing.assert_close(actual, p @ h @ p)
    r, _ = torch.linalg.qr(torch.randn(3, 3, dtype=h.dtype))
    q = torch.kron(torch.eye(n, dtype=h.dtype), r.contiguous())
    torch.testing.assert_close(project_dense(q @ h @ q.T, [n]), q @ actual @ q.T)
    assert torch.autograd.gradcheck(
        lambda x: project_dense(x, [n]), (h.requires_grad_(),)
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("scale", [0.0, 1e-12, 1e12])
def test_cuda_projection_scales_and_diagonal_only_gradient(scale):
    data = metadata([1, 2, 8], "cuda")
    e = (torch.randn(58, 3, 3, device="cuda") * scale).requires_grad_()
    n = (torch.randn(11, 3, 3, device="cuda") * scale).requires_grad_()
    p = project_blocks(e, n, data)
    r = project_blocks_reference(e, n, data)
    for a, b in zip(p, r, strict=True):
        torch.testing.assert_close(a, b, atol=max(1e-30, scale * 3e-6), rtol=3e-5)
    g = torch.autograd.grad(p[1].sum(), (e, n))
    rg = torch.autograd.grad(r[1].sum(), (e, n))
    for a, b in zip(g, rg, strict=True):
        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)
