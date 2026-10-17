"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Compare HIP kernels with identical molecular batches, features, and parameters.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import statistics
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from fairchem.core.frozen_hip.grouped_mole import enable_grouped_mole
from fairchem.core.frozen_hip.linear import enable_scaled_linear
from fairchem.core.frozen_hip.model import FrozenHIP, parameter_hash
from fairchem.core.frozen_hip.packed import PackedHessianStream, block_mae, block_rmse
from fairchem.core.frozen_hip.radial import enable_fused_radial
from fairchem.core.frozen_hip.readout import enable_projected_readout
from fairchem.core.frozen_hip.rotations import (
    enable_fused_rotations,
    enable_fused_scatter,
)
from fairchem.core.frozen_hip.so2 import enable_block_so2
from fairchem.core.frozen_hip.wigner import enable_fused_wigner


def move(raw):
    """Retain immutable host samples while preparing one device batch."""
    data = copy.copy(raw)
    data.__keys__ = raw.__keys__.copy()
    sizes, pbc = raw.natoms.tolist(), data.pbc
    data.apply(
        lambda value: value if value is pbc else value.to("cuda", non_blocking=True)
    )
    data.hip_sizes = sizes
    data.hip_fused_loss = True
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packed-data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batches", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--loss", choices=("mae", "block-rmse"), default="block-rmse")
    parser.add_argument("--max-batch", type=int, default=64)
    parser.add_argument("--max-atoms", type=int, default=2560)
    parser.add_argument("--microbatch-pairs", type=int, default=524288)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=(
            "baseline",
            "scatter",
            "segmented",
            "scaled",
            "wigner",
            "wigner-segmented",
            "wigner-scaled",
            "wigner-segmented-scaled",
            "compiled-geometry",
            "radial",
            "radial-segmented",
        ),
        default=["baseline", "wigner", "wigner-segmented", "wigner-scaled", "baseline"],
    )
    args = parser.parse_args()
    if min(args.batches, args.repeats) < 1 or args.workers < 0:
        parser.error(
            "Batch and repeat counts must be positive; worker count cannot be negative"
        )
    if args.modes[0] != "baseline":
        parser.error("The first mode must provide the baseline")
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch._dynamo.config.recompile_limit = 16
    model = FrozenHIP("cuda", backbone_graph_version=4).train()
    frozen = parameter_hash(model.backbone)
    state = torch.load(args.checkpoint, map_location="cuda", weights_only=False)
    if frozen != state["backbone_sha256"]:
        raise ValueError("Benchmark checkpoint uses another backbone")
    enable_grouped_mole(model.backbone)
    enable_fused_rotations(model.backbone)
    enable_fused_rotations(model.head)
    enable_block_so2(model.head)
    enable_projected_readout(model.head)
    markers = sorted(p.name for p in args.packed_data.glob("*.hip"))
    stream = PackedHessianStream(
        args.packed_data,
        markers=markers,
        seed=args.seed,
        backbone_sha256=frozen,
        backbone_precision="fp16",
        grouped_mole=True,
        backbone_graph_version=4,
        fused_backbone=True,
        tf32=True,
        max_batch=args.max_batch,
        max_atoms=args.max_atoms,
        pair_budget=args.microbatch_pairs,
        max_passes=1,
    )
    loader = DataLoader(
        stream,
        batch_size=None,
        num_workers=args.workers,
        **(
            {"multiprocessing_context": "spawn", "prefetch_factor": 2}
            if args.workers
            else {}
        ),
    )
    iterator = iter(loader)
    try:
        host = list(itertools.islice(iterator, args.batches))
    finally:
        if args.workers:
            iterator._shutdown_workers()
    if not host:
        raise ValueError("No training batches found")
    prepared = time.monotonic()
    with torch.no_grad():
        for logical in host:
            for raw in logical:
                data = move(raw)
                raw.hip_features = (
                    model.features(data, "fp16")["node_embedding"].cpu().pin_memory()
                )
                raw.pin_memory()
    torch.cuda.synchronize()
    samples = sum(len(raw.natoms) for logical in host for raw in logical)
    identities = [sid for logical in host for raw in logical for sid in raw.sid]
    report = dict(
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        samples=samples,
        updates=len(host),
        loss=args.loss,
        sample_order_sha256=hashlib.sha256("\n".join(identities).encode()).hexdigest(),
        feature_preparation_seconds=time.monotonic() - prepared,
        checkpoint_step=state["step"],
        backbone_sha256=frozen,
        cases=[],
    )
    original_geometry = model.backbone._get_rotmat_and_wigner
    compiled_geometry = torch.compile(original_geometry, dynamic=True)
    baseline = None
    operation = block_mae if args.loss == "mae" else block_rmse
    for name in args.modes:
        torch._dynamo.reset()
        torch.manual_seed(args.seed)
        enable_fused_scatter(
            model.head,
            "scatter" in name or "segmented" in name,
            segmented="segmented" in name,
        )
        enable_scaled_linear(model.head, "scaled" in name)
        enable_fused_radial(model.head, "radial" in name)
        enable_fused_wigner(model.head, name.startswith("wigner"))
        model.backbone._get_rotmat_and_wigner = (
            compiled_geometry if name == "compiled-geometry" else original_geometry
        )
        for block in model.head.blocks:
            block.compile(dynamic=True)
        model.head.projected_edge_irreps = torch.compile(
            model.head.projected_edge_irreps, dynamic=True
        )
        first = None
        optimizer = torch.optim.AdamW(
            model.head.parameters(), lr=1e-3, weight_decay=1e-4, fused=True
        )

        def reset(optimizer=optimizer):
            model.head.load_state_dict(state["head"])
            optimizer.state.clear()
            optimizer.zero_grad(set_to_none=True)

        def update_all(optimizer=optimizer):
            nonlocal first
            for logical in host:
                optimizer.zero_grad(set_to_none=True)
                count = sum(len(raw.natoms) for raw in logical)
                for raw in logical:
                    data = move(raw)
                    result = model.head(
                        data, {"node_embedding": data.hip_features, "batch": data.batch}
                    )
                    loss = operation(result["edge_blocks"], result["node_blocks"], data)
                    (loss * (len(data.hip_sizes) / count)).backward()
                    if first is None:
                        first = (
                            loss.detach().cpu(),
                            torch.cat(
                                [
                                    p.grad.flatten()
                                    for p in model.head.parameters()
                                    if p.grad is not None
                                ]
                            ).cpu(),
                        )
                norm = torch.nn.utils.clip_grad_norm_(
                    model.head.parameters(), 10.0, error_if_nonfinite=False
                )
                torch._assert_async(
                    torch.isfinite(norm), "Nonfinite benchmark gradient"
                )
                optimizer.step()
            torch.cuda.synchronize()

        reset()
        torch.cuda.reset_peak_memory_stats()
        update_all()
        timings = []
        for _ in range(args.repeats):
            reset()
            torch.cuda.synchronize()
            started = time.monotonic()
            update_all()
            timings.append(time.monotonic() - started)
        if baseline is None:
            baseline = first
        row = dict(
            mode=name,
            seconds=timings,
            samples_s=samples / statistics.median(timings),
            step_seconds=statistics.median(timings) / len(host),
            peak_GB=torch.cuda.max_memory_allocated() / 1e9,
            loss=float(first[0]),
            loss_relative_error=float(
                (first[0] - baseline[0]).abs() / baseline[0].abs().clamp_min(1e-30)
            ),
            gradient_relative_l2=float(
                (first[1] - baseline[1]).norm() / baseline[1].norm().clamp_min(1e-30)
            ),
        )
        report["cases"].append(row)
        print(json.dumps(row), flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    if parameter_hash(model.backbone) != frozen:
        raise RuntimeError("Benchmark changed frozen parameters")


if __name__ == "__main__":
    main()
