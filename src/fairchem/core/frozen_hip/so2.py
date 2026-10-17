"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Use differentiable complex block matrices with the original HIP parameters.
"""

from __future__ import annotations

from types import MethodType

import torch
from torch.nn import functional as F

from fairchem.core.models.uma.nn.so2_layers import SO2_m_Conv


def _forward(self, inputs):
    real, imaginary = self.fc.weight.split(self.out_channels_half, dim=0)
    # Rebuild through autograd on every call. Inference weight caches are unsafe
    # when the optimizer changes these trainable parameters.
    weight = torch.cat(
        (torch.cat((real, -imaginary), dim=1), torch.cat((imaginary, real), dim=1)),
        dim=0,
    )
    operation = F.linear
    if getattr(self, "scaled_linear", False):
        from fairchem.core.frozen_hip.linear import linear

        operation = linear
    output = operation(inputs.flatten(1), weight)
    return output.reshape(inputs.shape[0], 2, -1, self.m_output_channels).unbind(1)


def enable_block_so2(head, enabled=True):
    """
    Halve intermediate complex-linear outputs without changing model weights.
    """
    count = 0
    for module in head.modules():
        if isinstance(module, SO2_m_Conv):
            module.forward = MethodType(
                _forward if enabled else SO2_m_Conv.forward, module
            )
            count += 1
    return count
