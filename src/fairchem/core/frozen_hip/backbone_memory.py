"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from contextlib import contextmanager
from types import MethodType

import torch

from fairchem.core.common import gp_utils
from fairchem.core.models.uma.escn_md_block import Edgewise, set_mole_ac_start_index
from fairchem.core.models.uma.nn.embedding import EdgeDegreeEmbedding


def validate_charge_spin(backbone, data):
    """
    Reject unsupported charge or spin values before a CUDA embedding lookup.
    """
    for field in ("charge", "spin"):
        embedding = getattr(backbone, f"{field}_embedding", None)
        if embedding is None or embedding.embedding_type != "rand_emb":
            continue
        low = -embedding._index_offset
        high = low + embedding._num_embeddings - 1
        values = getattr(data, field).detach().cpu().reshape(-1).tolist()
        if any(not low <= value <= high or int(value) != value for value in values):
            raise ValueError(
                f"UMA {field} embedding supports integers {low} through {high}; "
                f"received {values}. Prepare a supported physical system; "
                "do not clamp its charge or extend untrained embedding rows."
            )


def _degree_forward(self, x, x_edge, edge_index, wigner_inv_envelope, node_offset=0):
    if self.training or torch.is_grad_enabled():
        raise ValueError("Backbone edge chunking requires frozen inference")
    try:
        for start in range(0, edge_index.shape[1], self._hip_edge_chunk_size):
            end = start + self._hip_edge_chunk_size
            set_mole_ac_start_index(self, start)
            x = self.forward_chunk(
                x,
                x_edge[start:end],
                edge_index[:, start:end],
                wigner_inv_envelope[start:end],
                node_offset,
            )
    finally:
        set_mole_ac_start_index(self, 0)
    return x


def _edge_forward(
    self,
    x,
    x_edge,
    edge_index,
    wigner,
    wigner_inv_envelope,
    total_atoms_across_gp_ranks,
    node_offset=0,
):
    if self.training or torch.is_grad_enabled():
        raise ValueError("Backbone edge chunking requires frozen inference")
    if gp_utils.initialized():
        raise NotImplementedError(
            "Backbone chunking does not support graph parallelism"
        )
    output = None
    for start in range(0, edge_index.shape[1], self._hip_edge_chunk_size):
        end = start + self._hip_edge_chunk_size
        part = self.forward_chunk(
            x,
            x.shape[0],
            x_edge[start:end],
            edge_index[:, start:end],
            wigner[start:end],
            wigner_inv_envelope[start:end],
            node_offset,
            start,
        )
        if output is None:
            output = part
        else:
            output.add_(part)
    return torch.zeros_like(x) if output is None else output


def enable_backbone_chunking(backbone, chunk_edges=65536):
    """
    Bound wide edge activations and accumulate messages before each node update.

    Pass None to restore ordinary dispatch. Model weights remain unchanged.
    """
    if chunk_edges is not None and (
        isinstance(chunk_edges, bool)
        or chunk_edges < 1
        or int(chunk_edges) != chunk_edges
    ):
        raise ValueError("Backbone chunk size must be a positive integer or None")
    count = 0
    for module in backbone.modules():
        if not isinstance(module, (EdgeDegreeEmbedding, Edgewise)):
            continue
        if chunk_edges is None:
            if hasattr(module, "_hip_original_forward"):
                module.forward = module._hip_original_forward
                del module._hip_original_forward
                del module._hip_edge_chunk_size
            continue
        if not hasattr(module, "_hip_original_forward"):
            module._hip_original_forward = module.forward
        module._hip_edge_chunk_size = int(chunk_edges)
        module.forward = MethodType(
            _degree_forward
            if isinstance(module, EdgeDegreeEmbedding)
            else _edge_forward,
            module,
        )
        count += 1
    return count


@contextmanager
def chunked_backbone(backbone, chunk_edges):
    """
    Temporarily select bounded edge dispatch and restore previous settings.
    """
    previous = [
        (m, getattr(m, "_hip_edge_chunk_size", None))
        for m in backbone.modules()
        if isinstance(m, (EdgeDegreeEmbedding, Edgewise))
    ]
    try:
        if chunk_edges is not None:
            enable_backbone_chunking(backbone, chunk_edges)
        yield
    finally:
        for module, old_size in previous:
            enable_backbone_chunking(module, old_size)
