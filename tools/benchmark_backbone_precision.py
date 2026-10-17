"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Measure frozen-backbone autocast speed and its effect on HIP gradients.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import time
from pathlib import Path

import torch

from fairchem.core.frozen_hip.data import HessianStream
from fairchem.core.frozen_hip.grouped_mole import enable_grouped_mole
from fairchem.core.frozen_hip.model import FrozenHIP, parameter_hash, structure_mse
from fairchem.core.frozen_hip.rotations import enable_fused_rotations
from fairchem.core.models.uma.nn.execution_backends import ExecutionBackend


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument(
        "--modes", help="Optional comma-separated precision configurations"
    )
    parser.add_argument("--strict-reference", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True
    model = FrozenHIP("cuda").train()
    model.head.load_state_dict(
        torch.load(args.checkpoint, map_location="cuda", weights_only=False)["head"]
    )
    frozen = parameter_hash(model.backbone)
    stream = HessianStream(
        args.data,
        max_atoms=2560,
        max_batch=64,
        pair_budget=196608,
        seed=args.seed,
        split=args.split,
    )
    host = list(itertools.islice(stream, args.batches))
    results = []
    for index, source in enumerate(host):
        data = copy.deepcopy(source)
        data.hip_sizes = data.natoms.tolist()
        data.to("cuda")
        reference = None
        for mode, dtype in [
            ("fp32", torch.float32),
            ("fp16", torch.float16),
            ("bf16", torch.bfloat16),
            ("fp32-fused", torch.float32),
            ("fp16-fused", torch.float16),
            ("fp32-grouped", torch.float32),
            ("fp16-grouped", torch.float16),
            ("fp16-fused-grouped", torch.float16),
            ("fp32-fused-grouped", torch.float32),
        ]:
            if args.modes and mode not in args.modes.split(","):
                continue
            for child in model.backbone.modules():
                if hasattr(child, "backend"):
                    child.backend = ExecutionBackend()
            grouped_count = enable_grouped_mole(
                model.backbone, enabled="grouped" in mode
            )
            if "fused" in mode:
                enable_fused_rotations(model.backbone)

            def infer(dtype=dtype, data=data, mode=mode):
                previous_tf32 = torch.backends.cuda.matmul.allow_tf32
                if args.strict_reference and mode == "fp32":
                    torch.backends.cuda.matmul.allow_tf32 = False
                with torch.no_grad(), torch.autocast(
                    "cuda", dtype=dtype, enabled=dtype != torch.float32
                ):
                    output = model.backbone(data)
                    result = {
                        "node_embedding": output["node_embedding"].float(),
                        "batch": data.batch,
                    }
                torch.backends.cuda.matmul.allow_tf32 = previous_tf32
                return result

            emb = infer()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = time.monotonic()
            for _ in range(3):
                emb = infer()
            torch.cuda.synchronize()
            seconds = (time.monotonic() - started) / 3
            model.head.zero_grad(set_to_none=True)
            out = model.head(data, emb)["hessian"]
            loss = structure_mse(out, data.hessian, data.hip_sizes)
            loss.backward()
            grad = torch.cat(
                [
                    p.grad.flatten().float()
                    for p in model.head.parameters()
                    if p.grad is not None
                ]
            )
            values = (
                emb["node_embedding"].clone(),
                out.detach().clone(),
                grad.clone(),
                float(loss),
            )
            if reference is None:
                reference = values
            errors = {
                name: float((actual - expected).norm() / expected.norm())
                for name, actual, expected in zip(
                    ("features", "predictions", "gradients"),
                    values[:3],
                    reference[:3],
                    strict=True,
                )
            }
            row = dict(
                batch=index,
                precision=mode,
                grouped_layers=grouped_count if "grouped" in mode else 0,
                strict_reference=args.strict_reference,
                sizes=data.hip_sizes,
                seconds=seconds,
                samples_s=len(data.hip_sizes) / seconds,
                loss=float(loss),
                peak_allocated_GB=torch.cuda.max_memory_allocated() / 1e9,
                **errors,
            )
            results.append(row)
            print(json.dumps(row), flush=True)
            (args.output / "summary.json").write_text(
                json.dumps(results, indent=2) + "\n"
            )
            del out, loss, grad, values, emb
        del reference, data
    assert parameter_hash(model.backbone) == frozen


if __name__ == "__main__":
    main()
