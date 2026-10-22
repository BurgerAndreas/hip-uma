"""Bounded CUDA graphs for repeated cached-feature training batches.

Capture forward and backward only. The ordinary optimizer handles changing
learning rates, gradient clipping, and logical-batch gradient accumulation.
"""

from __future__ import annotations

from collections import Counter

import torch


class TrainingGraphs:
    """Keep a fixed number of small, recurring shapes; use eager fallback."""

    def __init__(
        self, parameters, loss, *, capacity=2, max_edges=100000, loss_scale=1024.0
    ):
        if capacity < 0:
            raise ValueError("Graph capacity must be nonnegative")
        self.parameters = list(parameters)
        self.loss = loss
        self.loss_scale = loss_scale
        self.capacity, self.max_edges = capacity, max_edges
        self.seen = Counter()
        self.graphs = {}
        self.hits = 0
        self.captures = 0

    def backward(self, data, scale):
        sizes = tuple(data.hip_sizes)
        if (
            not hasattr(data, "hip_features")
            or sum(n * (n - 1) for n in sizes) > self.max_edges
        ):
            return None
        tensors = {
            k: data[k]
            for k in sorted(data.keys())
            if torch.is_tensor(data[k]) and data[k].is_cuda
        }
        shape = (
            (len(sizes), sum(sizes), sum(n * (n - 1) for n in sizes))
            if getattr(data, "hip_fused_graph", False)
            and getattr(data, "hip_fused_loss", False)
            else sizes
        )
        key = (
            shape,
            float(getattr(data, "hip_feature_scale", 1.0))
            if not torch.is_tensor(getattr(data, "hip_feature_scale", None))
            else None,
            tuple(
                (k, tuple(v.shape), v.dtype, tuple(v.stride()))
                for k, v in sorted(tensors.items())
            ),
        )
        if key not in self.graphs:
            if len(self.graphs) >= self.capacity:
                return None
            # Count only bounded candidates. Never retain all variable shapes.
            if len(self.seen) >= 256 and key not in self.seen:
                return None
            self.seen[key] += 1
            if self.seen[key] < 3:
                return None
            static = data.clone()
            original = [p.grad for p in self.parameters]
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            try:
                with torch.cuda.stream(stream):
                    # Finish lazy compilation before capture. Allocate gradient
                    # outputs inside the graph, as required by CUDA autograd.
                    for _ in range(2):
                        for p in self.parameters:
                            p.grad = None
                        (self.loss(static) * self.loss_scale).backward()
                torch.cuda.current_stream().wait_stream(stream)
                torch.cuda.synchronize()
                for p in self.parameters:
                    p.grad = None
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    value = self.loss(static)
                    (value * self.loss_scale).backward()
                private = [
                    p.grad if p.grad is not None else torch.zeros_like(p)
                    for p in self.parameters
                ]
            finally:
                for p, g in zip(self.parameters, original, strict=True):
                    p.grad = g
            self.graphs[key] = (graph, static, private, value)
            self.captures += 1
        graph, static, private, value = self.graphs[key]
        # Copy fresh inputs, including targets, pointers, positions and features.
        # Python shape metadata is part of the exact capture key.
        for k, tensor in tensors.items():
            static[k].copy_(tensor)
        graph.replay()
        destinations, gradients = [], []
        for p, g in zip(self.parameters, private, strict=True):
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            destinations.append(p.grad)
            gradients.append(g)
        torch._foreach_add_(destinations, gradients, alpha=scale / self.loss_scale)
        self.hits += 1
        return value.detach()
