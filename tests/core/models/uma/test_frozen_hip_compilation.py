"""Check compiled dispatch boundaries without changing parameter ownership."""

import pytest
import torch

from fairchem.core.frozen_hip.compilation import bounded_compile


def test_wide_dispatch_uses_current_parameters_and_preserves_arguments(monkeypatch):
    calls = []
    weight = torch.nn.Parameter(torch.tensor(2.0))

    def original(x, x_edge, edge_index, *, offset):
        return x * weight + offset

    def compile_stub(function, **kwargs):
        def compiled(*args, **kwargs):
            calls.append("compiled")
            return function(*args, **kwargs)

        return compiled

    monkeypatch.setattr(torch, "compile", compile_stub)
    forward = bounded_compile(original, max_edges=4)
    for edges, value in [(4, 2.0), (5, 3.0), (3, 4.0)]:
        with torch.no_grad():
            weight.fill_(value)
        x = torch.tensor(3.0, requires_grad=True)
        result = forward(x, None, torch.empty(2, edges), offset=1)
        assert result == 3 * value + 1
        assert torch.autograd.grad(result, weight)[0] == 3
    assert calls == ["compiled", "compiled"]
    with pytest.raises(ValueError, match="positive"):
        bounded_compile(original, max_edges=0)
