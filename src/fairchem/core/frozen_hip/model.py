"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Keep the pretrained UMA backbone frozen and train only its additional HIP head.
"""

from __future__ import annotations

import hashlib

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from fairchem.core.calculate import pretrained_mlip
from fairchem.core.models.uma.escn_md import Hessian_Head
from fairchem.core.models.uma.nn.mole import MOLE
from fairchem.core.units.mlip_unit import InferenceSettings


def parameter_hash(module):
    """Hash every parameter, excluding mutable inference routing buffers."""
    digest = hashlib.sha256()
    for name, value in module.named_parameters():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class FrozenHIP(nn.Module):
    """Use the existing equivariant HIP architecture on frozen UMA features."""

    def __init__(
        self,
        device,
        *,
        layers=1,
        head_width_multiplier=1,
        execution_mode="general",
        checkpoint_edges=20000,
        backbone_graph_version=None,
        head_experts=0,
        head_charge_spin=False,
    ):
        super().__init__()
        settings = InferenceSettings(
            activation_checkpointing=False, execution_mode=execution_mode, tf32=False
        )
        predictor = pretrained_mlip.get_predict_unit(
            "uma-s-1p2",
            inference_settings=settings,
            device="cuda" if str(device).startswith("cuda") else "cpu",
        )
        original = getattr(predictor.model, "module", predictor.model)
        self.backbone = original.backbone.to(device).eval().requires_grad_(False)
        self.backbone_fp16_reduced_precision_reduction = (
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
        )
        if backbone_graph_version is not None:
            self.backbone.radius_pbc_version = backbone_graph_version
        self.backbone.regress_config.forces = False
        self.backbone.regress_config.stress = False
        self.backbone.regress_config.hessian = False
        self.head = Hessian_Head(
            self.backbone,
            num_layers_hessian=layers,
            width_multiplier=head_width_multiplier,
        ).to(device)
        if head_experts:
            from fairchem.core.frozen_hip.mole import enable_head_mole

            enable_head_mole(self.head, head_experts)
        if head_charge_spin:
            from fairchem.core.frozen_hip.conditioning import (
                enable_charge_spin_conditioning,
            )

            enable_charge_spin_conditioning(self.head)
        # Dense HIP layers do not consume the backbone's expert routing sizes.
        # Avoid copying those sizes to the host after each complete HIP graph.
        self.head.requires_mole_routing = any(
            isinstance(module, MOLE) for module in self.head.modules()
        )
        self.checkpoint_edges = checkpoint_edges

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def features(self, data, precision="fp32"):
        """
        Keep weights frozen while optionally accelerating dense inference math.
        """
        if precision not in ("fp32", "fp16"):
            raise ValueError("Frozen backbone precision must be fp32 or fp16")
        previous = torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
        try:
            # A head precision setting must not change frozen feature inference.
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = (
                self.backbone_fp16_reduced_precision_reduction
            )
            with (
                torch.no_grad(),
                torch.autocast(
                    "cuda", dtype=torch.float16, enabled=precision == "fp16"
                ),
            ):
                emb = self.backbone(data)
        finally:
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = previous
        emb["node_embedding"] = emb["node_embedding"].float()
        return emb

    def forward(self, data):
        # Energy and force heads are deliberately absent. No backbone activations
        # are saved for backward, and only HIP parameters enter the optimizer.
        with torch.no_grad():
            emb = self.backbone(data)
        edges = sum(n * (n - 1) for n in data.hip_sizes)
        if self.training and edges >= self.checkpoint_edges:
            return checkpoint(
                lambda x: self.head(data, {**emb, "node_embedding": x})["hessian"],
                emb["node_embedding"],
                use_reentrant=False,
            )
        return self.head(data, emb)["hessian"]


def structure_mse(prediction, target, sizes):
    """Weight each molecular Hessian equally, independent of its atom count."""
    if len(set(sizes)) == 1:
        return (
            (prediction.float() - target.float())
            .square()
            .reshape(len(sizes), -1)
            .mean(1)
            .mean()
        )
    widths = [9 * n * n for n in sizes]
    return torch.stack(
        [
            (p.float() - t.float()).square().mean()
            for p, t in zip(prediction.split(widths), target.split(widths), strict=True)
        ]
    ).mean()
