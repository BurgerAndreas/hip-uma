"""Check ragged reductions and addresses beyond signed 32-bit element offsets."""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def complete_edges(sizes):
    pieces, start = [], 0
    for n in sizes:
        source = torch.arange(n, device="cuda").repeat(n)
        target = torch.arange(n, device="cuda").repeat_interleave(n)
        mask = source != target
        pieces.append(torch.stack((source[mask] + start, target[mask] + start)))
        start += n
    return torch.cat(pieces, 1)


@pytest.mark.parametrize("channels", [128, 192, 256])
def test_tuned_scatter_and_gather_match_reference(channels):
    from fairchem.core.frozen_hip.edge_precision import mixed_gather, mixed_scatter

    torch.manual_seed(71)
    edges = complete_edges([1, 2, 182])
    e, nodes = edges.shape[1], 185
    w = torch.randn(e, 9, 9, device="cuda")
    x = torch.randn(e, 9, channels, device="cuda", dtype=torch.float16)
    expected = mixed_scatter(x, w, edges, nodes, 0, True, False)
    actual = mixed_scatter(x, w, edges, nodes, 0, True, True)
    assert (actual - expected).norm() / expected.norm() < 3e-6
    g = torch.randn_like(actual)
    expected = mixed_gather(g, w, edges, 0, False)
    actual = mixed_gather(g, w, edges, 0, True)
    assert (actual.float() - expected.float()).norm() / expected.float().norm() < 2e-6


@pytest.mark.parametrize("channels", [128, 256])
def test_tuned_radial_gradient_matches_reference(channels):
    from fairchem.core.frozen_hip.edge_precision import mixed_radial_grad

    torch.manual_seed(73)
    edges = complete_edges([1, 2, 182])
    e = edges.shape[1]
    x = torch.randn(185, 9, channels, device="cuda")
    w = torch.randn(e, 9, 9, device="cuda")
    r = torch.randn(e, 12 * channels, device="cuda", dtype=torch.float16)
    g = torch.randn(e, 9, 2 * channels, device="cuda", dtype=torch.float16)
    expected_x, expected_r = mixed_radial_grad(x, w, edges, r, g, True, False)
    actual_x, actual_r = mixed_radial_grad(x, w, edges, r, g, True, True)
    torch.testing.assert_close(actual_r, expected_r, rtol=0, atol=0)
    assert (actual_x - expected_x).norm() / expected_x.norm() < 3e-6


def require_large_memory():
    if torch.cuda.get_device_properties(0).total_memory < 20 * (1 << 30):
        pytest.skip("Wide-address regression needs a GPU with at least 20 GiB")


def test_skinny_projection_crosses_32_bit_element_address():
    from fairchem.core.frozen_hip.skinny import skinny_project

    require_large_memory()
    channels = 512
    m = (1 << 31) // (9 * channels) + 17
    x = torch.zeros(m, 9, channels, device="cuda", dtype=torch.float16)
    selected = torch.tensor([0, m // 2, m - 1], device="cuda")
    x[selected] = 1
    x.requires_grad_()
    weights = tuple(
        torch.full(
            (n, width * channels),
            1 / (width * channels),
            device="cuda",
            requires_grad=True,
        )
        for n, width in zip((3, 4, 2), (3, 4, 2), strict=True)
    )
    bias = torch.zeros(3, device="cuda", requires_grad=True)
    out = skinny_project(x, weights, bias)
    torch.testing.assert_close(
        out[selected], torch.ones_like(out[selected]), rtol=0, atol=3e-6
    )
    out[selected].sum().backward()
    torch.testing.assert_close(
        x.grad[selected], torch.full_like(x.grad[selected], 1 / 512), rtol=0, atol=0
    )
    for weight in weights:
        torch.testing.assert_close(
            weight.grad, torch.full_like(weight, 3), rtol=0, atol=0
        )
    torch.testing.assert_close(bias.grad, torch.full_like(bias, 3), rtol=0, atol=0)


def test_radial_gradient_crosses_32_bit_element_address():
    from fairchem.core.frozen_hip.radial_segmented import segmented_radial_dx

    require_large_memory()
    channels = 256
    edges = complete_edges([350] * 4)
    e = edges.shape[1]
    assert e * 9 * 2 * channels > 1 << 31
    w = torch.zeros(e, 9, 9, device="cuda")
    w.diagonal(dim1=1, dim2=2).fill_(1)
    r = torch.ones(e, 12 * channels, device="cuda", dtype=torch.float16)
    g = torch.zeros(e, 9, 2 * channels, device="cuda", dtype=torch.float16)
    g[-1] = 2
    actual = segmented_radial_dx(w, edges, r, g, 1400, 256, 4)
    torch.testing.assert_close(
        actual[-2:], torch.full_like(actual[-2:], 2), rtol=0, atol=0
    )
    assert torch.count_nonzero(actual[:-2]) == 0
