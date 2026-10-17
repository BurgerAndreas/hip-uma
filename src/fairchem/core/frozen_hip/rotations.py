"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Use fused rotation kernels without changing mixture-of-experts routing.
"""

from __future__ import annotations

from fairchem.core.models.uma.nn.execution_backends import (
    ExecutionBackend,
    UMASFastGPUBackend,
)


class FusedRotationBackend(ExecutionBackend):
    """
    Retain general SO(2) layers and composition-dependent expert weights.
    """

    prepare_wigner = staticmethod(UMASFastGPUBackend.prepare_wigner)
    node_to_edge_wigner_permute = staticmethod(
        UMASFastGPUBackend.node_to_edge_wigner_permute
    )
    permute_wigner_inv_edge_to_node = staticmethod(
        UMASFastGPUBackend.permute_wigner_inv_edge_to_node
    )
    edge_wigner_inv_permute = staticmethod(UMASFastGPUBackend.edge_wigner_inv_permute)
    edge_degree_scatter = staticmethod(UMASFastGPUBackend.edge_degree_scatter)


class FusedScatterBackend(FusedRotationBackend):
    """Accumulate rotated messages without a large per-edge output tensor."""

    @staticmethod
    def permute_wigner_inv_edge_to_node(x, w, edges, nodes, node_offset=0):
        from fairchem.core.frozen_hip.scatter import fused_rotate_scatter

        return fused_rotate_scatter(x, w, edges, nodes, node_offset)


class SegmentedScatterBackend(FusedRotationBackend):
    """Reduce the sorted complete molecular graph without atomic writes."""

    @staticmethod
    def permute_wigner_inv_edge_to_node(x, w, edges, nodes, node_offset=0):
        from fairchem.core.frozen_hip.scatter import fused_rotate_scatter

        return fused_rotate_scatter(x, w, edges, nodes, node_offset, segmented=True)


def enable_fused_scatter(module, enabled=True, *, segmented=False):
    """Select frozen-geometry scatter kernels while preserving every weight."""
    if enabled and segmented and not getattr(module, "fully_connected_hessian", False):
        raise ValueError(
            "Segmented scatter requires a complete, target-sorted HIP graph"
        )
    backend = (
        (SegmentedScatterBackend() if segmented else FusedScatterBackend())
        if enabled
        else FusedRotationBackend()
    )
    for child in module.modules():
        if hasattr(child, "backend"):
            child.backend = backend


def enable_fused_rotations(module, enabled=True):
    """
    Change parameterless operation dispatch; retain every model parameter.
    """
    backbone = getattr(module, "backbone", module)
    if (
        enabled
        and hasattr(backbone, "lmax")
        and (backbone.lmax != 2 or backbone.mmax != 2)
    ):
        raise ValueError("Fused rotations require lmax=mmax=2")
    backend = FusedRotationBackend() if enabled else ExecutionBackend()
    count = 0
    for child in module.modules():
        if hasattr(child, "backend"):
            child.backend = backend
            count += 1
    return count
