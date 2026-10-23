"""Bound compiled HIP graphs while retaining eager kernels for wide batches."""

from __future__ import annotations

from functools import wraps

import torch


def bounded_compile(function, *, max_edges=393216, mode="default"):
    """Dispatch before tracing; large batches retain the same eager operation.

    Both HIP blocks and readouts receive edge_index as their third argument.
    The eager path retains custom Triton operations and mixed precision.
    """
    if max_edges < 1:
        raise ValueError("Compiled edge limit must be positive")
    compiled = torch.compile(function, dynamic=True, mode=mode)

    @wraps(function)
    def forward(x, x_edge, edge_index, *args, **kwargs):
        operation = compiled if edge_index.shape[1] <= max_edges else function
        return operation(x, x_edge, edge_index, *args, **kwargs)

    return forward
