"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Restrict reduced precision to the HIP head's scalar multilayer perceptrons.
"""

from __future__ import annotations

import torch
from torch import nn


class MixedPrecisionMLP(nn.Sequential):
    """
    Run dense scalar layers in BF16 and return FP32 equivariant coefficients.
    """

    enabled = True

    def forward(self, inputs):
        if not self.enabled:
            return super().forward(inputs)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.enabled):
            output = super().forward(inputs)
        return output.float()


def configure_head_precision(head, mode):
    """
    Preserve state names while selecting scalar MLPs for optional BF16 execution.

    Rotations, tensor products, equivariant projections, and reductions stay
    outside this autocast region. Parameters and optimizer states remain FP32.
    """
    if mode not in ("fp32", "bf16", "bf16-mlp"):
        raise ValueError(f"Unknown HIP precision mode: {mode}")
    count = 0
    for name, original in list(head.named_modules()):
        module = original
        if not name.endswith(("rad_func.net", "scalar_mlp")):
            continue
        if not isinstance(module, MixedPrecisionMLP):
            if mode != "bf16-mlp":
                continue
            parent_name, child_name = name.rsplit(".", 1)
            parent = head.get_submodule(parent_name)
            module = MixedPrecisionMLP(*module.children())
            setattr(parent, child_name, module)
        module.enabled = mode == "bf16-mlp"
        count += 1
    return count
