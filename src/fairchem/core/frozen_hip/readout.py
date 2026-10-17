"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Contract the final HIP convolution and projection before processing all edges.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F


def project_convolution(convolution, projection, messages):
    """
    Preserve both trainable factors while eliminating their wide intermediate.
    """
    channels = convolution.m_output_channels
    degrees = convolution.lmax + 1
    p = projection.weight[:, 0, :]
    original = convolution.fc_m0
    weight = torch.einsum(
        "lc,lci->li", p, original.weight.reshape(degrees, channels, -1)
    )
    bias = (
        (original.bias.reshape(degrees, channels) * p).sum(-1)
        if original.bias is not None
        else None
    )
    parts = messages.split(convolution.m_split_sizes, dim=1)
    edges = messages.shape[0]
    outputs = [F.linear(parts[0].flatten(1), weight, bias).unsqueeze(-1)]
    for m, block in enumerate(convolution.so2_m_conv, 1):
        count = degrees - m
        original = block.fc.weight.reshape(2, count, channels, -1)
        effective = torch.einsum("lc,slci->sli", p[m:], original)
        real, imaginary = effective.unbind(0)
        weight = torch.cat(
            (torch.cat((real, -imaginary), dim=1), torch.cat((imaginary, real), dim=1)),
            dim=0,
        )
        output = F.linear(parts[m].flatten(1), weight).reshape(edges, 2, count, 1)
        outputs.extend(output.unbind(1))
    return torch.cat(outputs, dim=1)


def edge_irreps(head, node_embedding, x_edge, edge_index, wigner, wigner_inv):
    """
    Commute the degree-wise projection with rotation and keep its bias outside.
    """
    readout = head.edge_readout
    messages = readout.backend.node_to_edge_wigner_permute(
        node_embedding, edge_index, wigner
    )
    messages, gates = readout.so2_conv_1(messages, x_edge)
    messages = readout.act(gates, messages)
    messages = project_convolution(readout.so2_conv_2, head.edge_projection, messages)
    irreps = readout.backend.edge_wigner_inv_permute(messages, wigner_inv).squeeze(-1)
    return torch.cat((irreps[:, :1] + head.edge_projection.bias, irreps[:, 1:]), dim=1)


def enable_projected_readout(head, enabled=True):
    """
    Retain all parameters and their optimizer state; select an equivalent path.
    """
    if enabled:
        conv = head.edge_readout.so2_conv_2
        if (
            conv.lmax != 2
            or conv.mmax != 2
            or conv.rad_func is not None
            or conv.extra_m0_output_channels is not None
        ):
            raise ValueError(
                "Projected readout requires an internal lmax=mmax=2 convolution"
            )
        if head.edge_projection.out_features != 1:
            raise ValueError(
                "Projected readout requires a scalar per irrep coefficient"
            )
    head.projected_readout = enabled
