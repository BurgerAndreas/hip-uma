"""Check reciprocal-edge radial gradients on complete ragged graphs."""

from __future__ import annotations

import pytest
import torch


def complete_edges(sizes, device):
    parts = []
    start = 0
    for n in sizes:
        source = torch.arange(n, device=device).repeat(n)
        target = torch.arange(n, device=device).repeat_interleave(n)
        mask = source != target
        parts.append(torch.stack((source[mask] + start, target[mask] + start)))
        start += n
    return torch.cat(parts, dim=1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("sizes", [(1, 1), (1, 2, 1, 7), (2, 3, 5), (27,) * 8, (350,)])
@pytest.mark.parametrize("channels", [64, 256])
def test_segmented_node_gradient_matches_atomic(sizes, channels):
    import triton

    from fairchem.core.frozen_hip.radial import _backward_node
    from fairchem.core.frozen_hip.radial_segmented import segmented_radial_dx

    torch.manual_seed(17)
    edges = complete_edges(sizes, "cuda")
    e = edges.shape[1]
    w = torch.randn(e, 9, 9, device="cuda")
    r = torch.randn(e, 12 * channels, device="cuda", dtype=torch.float16)
    g = torch.randn(e, 9, 2 * channels, device="cuda", dtype=torch.float16)
    reference = torch.zeros(sum(sizes), 9, channels, device="cuda")
    _backward_node[(min(max(e, 1), 8192), triton.cdiv(2 * channels, 64))](
        w, edges, r, g, reference, e, channels, 64, 8192, num_warps=2,
    )
    for channel_tile, edge_tile in [(32, 16), (64, 8), (128, 16)]:
        actual = segmented_radial_dx(w, edges, r, g, sum(sizes), channel_tile, edge_tile)
        torch.testing.assert_close(actual, reference, atol=0.003, rtol=0.0002)
        relative = (actual - reference).norm() / reference.norm().clamp_min(1)
        assert relative < 3e-6
