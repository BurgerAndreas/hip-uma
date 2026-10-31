"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Use composition-specific merged experts with FP32 output accumulation.
"""

from __future__ import annotations

import copy
import weakref
from types import MethodType

import torch
from torch.nn import functional as F

from fairchem.core.frozen_hip.edge_precision import half_linear
from fairchem.core.frozen_hip.grouped_mole import enable_grouped_mole
from fairchem.core.frozen_hip.rotations import enable_fused_rotations
from fairchem.core.models.uma.nn.mole import MOLE
from fairchem.core.units.mlip_unit import InferenceSettings


def _merged_fp32_output(self, inputs):
    # Reuse the production HIP half-input GEMM. The original grouped expert
    # path keeps these second-convolution outputs in FP32 to avoid overflow.
    if (
        torch.is_autocast_enabled("cuda")
        and torch.get_autocast_dtype("cuda") == torch.float16
    ):
        with torch.autocast("cuda", enabled=False):
            return half_linear(
                inputs,
                self.weight,
                None if self.bias is None else self.bias.half().float(),
                out32=True,
            )
    return F.linear(inputs, self.weight, self.bias)


def set_backbone(model, backbone):
    model.backbone = backbone.eval().requires_grad_(False)
    object.__setattr__(model.head, "_backbone_ref", weakref.ref(model.backbone))
    model.backbone.radius_pbc_version = 4
    model.backbone.regress_config.forces = False
    model.backbone.regress_config.stress = False
    model.backbone.regress_config.hessian = False
    enable_fused_rotations(model.backbone)


def make_unmerged(model, cpu_template):
    set_backbone(model, copy.deepcopy(cpu_template).to("cuda"))
    enable_grouped_mole(model.backbone)


def merge_for_batch(model, cpu_template, data, *, tf32):
    candidate = copy.deepcopy(cpu_template).to("cuda").eval().requires_grad_(False)
    settings = InferenceSettings(
        merge_mole=True,
        activation_checkpointing=False,
        compile=False,
        execution_mode="general",
        tf32=tf32,
    )
    with torch.no_grad():
        merged = candidate.prepare_for_inference(data, settings).to("cuda")
    assert not any(isinstance(m, MOLE) for m in merged.modules())
    set_backbone(model, merged)
    count = 0
    for name, module in model.backbone.named_modules():
        if isinstance(module, torch.nn.Linear) and ".so2_conv_2." in name:
            module.forward = MethodType(_merged_fp32_output, module)
            count += 1
    model.backbone.on_predict_check(data)
    return count
