"""Molecule-conditioned linear experts in the trainable HIP message block."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def grouped_reference(x, weight, bias, sizes):
    pieces = []
    offset = 0
    for b, n in enumerate(sizes):
        count = n * (n - 1)
        pieces.append(F.linear(x[offset : offset + count], weight[b], bias))
        offset += count
    return torch.cat(pieces)


def grouped_linear(x, weight, bias, pointers, sizes, mixed):
    if not x.is_cuda:
        return grouped_reference(x, weight, bias, sizes)
    from fairchem.core.frozen_hip.mole_kernels import grouped

    return grouped(x, weight, bias, pointers, sizes, mixed)


class HIPRouter(nn.Module):
    """Smooth scalar routing, invariant to geometry and atom permutations."""

    def __init__(self, experts):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(121, 32), nn.SiLU(), nn.Linear(32, experts)
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, data):
        b = len(data.hip_sizes)
        counts = torch.zeros(b * 119, device=data.pos.device)
        index = data.batch * 119 + data.atomic_numbers
        counts.scatter_add_(0, index, torch.ones_like(index, dtype=torch.float32))
        counts = counts.reshape(b, 119) / data.natoms.float().reshape(b, 1)
        extra = torch.stack((data.charge, data.spin), -1).to(counts) / 10
        return self.network(torch.cat((counts, extra), -1)).softmax(-1)


class MixtureSO2(nn.Module):
    """Mix weights once per geometry; apply them to ragged complete edge groups."""

    def __init__(self, convolution, experts):
        super().__init__()
        self.m_output_channels = convolution.m_output_channels
        self.m_split_sizes = convolution.m_split_sizes
        self.edge_mixed_precision = False
        self.expert_weights = nn.ParameterList()
        self.bias = nn.Parameter(convolution.fc_m0.bias.detach().clone())
        linears = [convolution.fc_m0] + [m.fc for m in convolution.so2_m_conv]
        for linear in linears:
            base = linear.weight.detach()
            noise = (
                torch.randn((experts, *base.shape), device=base.device)
                * base.std()
                * 0.01
            )
            noise -= noise.mean(0, keepdim=True)
            self.expert_weights.append(nn.Parameter(base.unsqueeze(0) + noise))
        self.mixture = None
        self.pointers = None
        self.sizes = None

    def prepare(self, alpha, pointers, sizes):
        self.mixture = [
            torch.einsum("bk,koi->boi", alpha, w) for w in self.expert_weights
        ]
        self.pointers, self.sizes = pointers, tuple(sizes)

    def forward(self, x, x_edge=None):
        if self.mixture is None:
            raise RuntimeError("Prepare HIP expert routing before the head forward")
        parts = x.split(self.m_split_sizes, dim=1)
        e, c = x.shape[0], self.m_output_channels
        scalar = grouped_linear(
            parts[0].flatten(1),
            self.mixture[0],
            self.bias,
            self.pointers,
            self.sizes,
            self.edge_mixed_precision,
        )
        out = [scalar.reshape(e, -1, c)]
        for part, weight in zip(parts[1:], self.mixture[1:], strict=True):
            real, imag = weight.chunk(2, dim=1)
            block = torch.cat(
                (torch.cat((real, -imag), -1), torch.cat((imag, real), -1)), 1
            )
            value = grouped_linear(
                part.flatten(1),
                block,
                None,
                self.pointers,
                self.sizes,
                self.edge_mixed_precision,
            )
            out.extend(value.reshape(e, 2, -1, c).unbind(1))
        return torch.cat(out, dim=1)


def enable_head_mole(head, experts):
    if experts < 2:
        raise ValueError("HIP MoLE requires at least two experts")
    if not head.fully_connected_hessian:
        raise ValueError("HIP MoLE requires complete molecular graphs")
    device = next(head.parameters()).device
    head.hip_mole_router = HIPRouter(experts).to(device)
    for block in head.blocks:
        block.edge_wise.so2_conv_2 = MixtureSO2(block.edge_wise.so2_conv_2, experts)
    if not len(head.blocks):
        raise ValueError("HIP MoLE requires a message-passing block")
    head.hip_experts = experts


def prepare_head_mole(head, data):
    if not hasattr(data, "hip_sizes"):
        data.hip_sizes = data.natoms.tolist()
    alpha = head.hip_mole_router(data)
    if hasattr(data, "hip_edge_ptr"):
        pointers = data.hip_edge_ptr
    else:
        n = data.natoms
        pointers = torch.cat((n.new_zeros(1), (n * (n - 1)).cumsum(0)))
    for block in head.blocks:
        block.edge_wise.so2_conv_2.prepare(alpha, pointers, data.hip_sizes)
