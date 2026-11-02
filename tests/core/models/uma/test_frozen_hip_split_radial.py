"""Compare direct angular gradient reads with the existing radial backward."""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def inputs(n, c, need_x=True):
    torch.manual_seed(61)
    x = torch.randn(n, 9, c, device="cuda", requires_grad=need_x)
    target = torch.arange(n, device="cuda").repeat_interleave(n)
    source = torch.arange(n, device="cuda").repeat(n)
    active = source != target
    edges = torch.stack((source[active], target[active]))
    e = edges.shape[1]
    w = torch.randn(e, 9, 9, device="cuda")
    r = (torch.randn(e, 12 * c, device="cuda") * 0.01).half().requires_grad_()
    return x, edges, w, r


@pytest.mark.parametrize("n,c,tuned", [(5, 17, False), (9, 64, True), (193, 128, True)])
def test_split_matches_contiguous_radial_forward_and_backward(n, c, tuned):
    from fairchem.core.frozen_hip.edge_precision import MixedRadial, SplitMixedRadial

    x, edges, w, r = inputs(n, c)
    ref = MixedRadial.apply(x, edges, w, r, tuned)
    g = torch.randn_like(ref) * 0.01
    expected = torch.autograd.grad(ref, (x, r), g)
    pieces = SplitMixedRadial.apply(x, edges, w, r, tuned)
    torch.testing.assert_close(torch.cat(pieces, 1), ref, rtol=0, atol=0)
    actual = torch.autograd.grad(pieces, (x, r), tuple(g.split((3, 4, 2), 1)))
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b, rtol=3e-5, atol=3e-5)


def test_split_handles_unused_modes_and_frozen_node_inputs():
    from fairchem.core.frozen_hip.edge_precision import MixedRadial, SplitMixedRadial

    x, edges, w, r = inputs(5, 17, False)
    ref = MixedRadial.apply(x, edges, w, r, False)
    expected = torch.autograd.grad(ref[:, :3].float().square().sum(), r)[0]
    modes = SplitMixedRadial.apply(x, edges, w, r, False)
    got = torch.autograd.grad(modes[0].float().square().sum(), r)[0]
    torch.testing.assert_close(got, expected, rtol=0, atol=0)


@pytest.mark.parametrize("split", [False, True])
def test_split_compiles_and_recomputes_checkpoint(split):
    from torch.utils.checkpoint import checkpoint

    from fairchem.core.frozen_hip.edge_precision import MixedRadial, SplitMixedRadial

    x, edges, w, r = inputs(5, 17)

    def operation(x, r):
        result = (SplitMixedRadial if split else MixedRadial).apply(
            x, edges, w, r, False
        )
        parts = result if split else (result,)
        return sum(part.float().square().mean() for part in parts)

    expected = torch.autograd.grad(operation(x, r), (x, r))
    compiled = torch.compile(operation, fullgraph=True)
    got = torch.autograd.grad(checkpoint(compiled, x, r, use_reentrant=False), (x, r))
    for a, b in zip(got, expected, strict=True):
        torch.testing.assert_close(a, b, rtol=3e-5, atol=3e-5)
