"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Compare equivalent dense, compact, fused, and reduced-precision HIP updates.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from fairchem.core.frozen_hip.data import HessianStream
from fairchem.core.frozen_hip.grouped_mole import enable_grouped_mole
from fairchem.core.frozen_hip.model import FrozenHIP, parameter_hash, structure_mse
from fairchem.core.frozen_hip.packed import (
    arrays_from_batches,
    block_mse,
    make_packed_batch,
    open_packed,
    write_packed,
)
from fairchem.core.frozen_hip.rotations import enable_fused_rotations
from fairchem.core.frozen_hip.so2 import enable_block_so2
from fairchem.core.models.uma.nn.execution_backends import ExecutionBackend


class BenchmarkStream(HessianStream):
    """
    Close worker queues after a finite workload instead of canceling prefetch.
    """

    def __iter__(self):
        yield from itertools.islice(super().__iter__(), self.benchmark_batches)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batches", type=int, default=24)
    parser.add_argument("--optimized-only", action="store_true")
    parser.add_argument("--projected-readout", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True
    model = FrozenHIP("cuda", checkpoint_edges=200000)
    state = torch.load(args.checkpoint, map_location="cuda", weights_only=False)
    model.head.load_state_dict(state["head"])
    initial = copy.deepcopy(model.head)
    frozen = parameter_hash(model.backbone)
    stream = BenchmarkStream(
        args.data, max_batch=64, max_atoms=2560, pair_budget=196608, seed=29
    )
    stream.benchmark_batches = args.batches
    loader = DataLoader(
        stream,
        batch_size=None,
        num_workers=2,
        multiprocessing_context="spawn",
        prefetch_factor=2,
    )
    iterator = iter(loader)
    host = list(iterator)[: args.batches]
    features = []
    started = time.monotonic()
    with torch.no_grad():
        for source in host:
            data = copy.deepcopy(source)
            data.hip_sizes = data.natoms.tolist()
            data.to("cuda")
            features.append(
                {
                    k: v.cpu() if torch.is_tensor(v) else v
                    for k, v in model.backbone(data).items()
                }
            )
    torch.cuda.synchronize()
    preprocessing = time.monotonic() - started
    examples = {}
    for dtype in ("float32", "float16"):
        values, scale = arrays_from_batches(host, features, feature_dtype=dtype)
        path = args.output / f"{dtype}.hip"
        write_packed(path, values, dict(backbone_sha256=frozen, feature_scale=scale))
        mapped, metadata = open_packed(path)
        bp = mapped["batch_ptr"]
        started = time.monotonic()
        examples[dtype] = [
            make_packed_batch(mapped, np.arange(a, b), metadata["feature_scale"])
            for a, b in zip(bp[:-1], bp[1:], strict=True)
        ]
        print(
            json.dumps(
                dict(
                    storage=dtype,
                    bytes=path.stat().st_size,
                    loading_seconds=time.monotonic() - started,
                    feature_scale=scale,
                )
            ),
            flush=True,
        )
    examples["dense"] = host
    results = dict(
        preprocessing_seconds=preprocessing,
        molecules=sum(len(d.natoms) for d in host),
        cases=[],
    )
    telemetry_file = (args.output / "gpu-utilization.csv").open("w")
    telemetry = subprocess.Popen(
        [
            "nvidia-smi",
            "-i",
            os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0],
            "--query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw",
            "--format=csv,noheader,nounits",
            "-lms",
            "200",
        ],
        stdout=telemetry_file,
        env={**os.environ, "TZ": "UTC"},
    )
    reference = None
    try:
        cases = (
            ("dense", False, False, "fp32"),
            ("dense", False, True, "fp32"),
            ("float32", False, True, "fp32"),
            ("float32", True, False, "fp32"),
            ("float32", True, True, "fp32"),
            ("float16", True, True, "fp32"),
            ("dense", False, True, "online-fp32"),
            ("dense", True, True, "online-fp16"),
            ("float32", True, True, "online-fp16"),
            ("float32", True, True, "fp32-block"),
            ("float32", True, True, "online-fp32-block"),
            ("float32", True, True, "online-fp16-block"),
            ("float32", True, True, "online-fp32-grouped-block"),
            ("float32", True, True, "online-fp16-grouped-block"),
        )

        if args.optimized_only:
            selected = {
                ("dense", False, False, "fp32"),
                ("dense", False, True, "fp32"),
                ("dense", False, True, "online-fp32"),
                ("float32", True, True, "fp32-block"),
                ("float32", True, True, "online-fp16-block"),
                ("float32", True, True, "online-fp32-grouped-block"),
            }
            cases = [case for case in cases if case in selected]
        for storage, fused, compiled, precision in cases:
            torch._dynamo.reset()
            model.head = copy.deepcopy(initial)
            model.backbone.backend = ExecutionBackend()
            enable_grouped_mole(model.backbone, enabled="grouped" in precision)
            if precision.endswith("-block"):
                enable_block_so2(model.head)
            if fused:
                enable_fused_rotations(model.head)
            if args.projected_readout and fused:
                from fairchem.core.frozen_hip.readout import enable_projected_readout

                enable_projected_readout(model.head)
            if compiled:
                torch._dynamo.config.recompile_limit = 16
                for block in model.head.blocks:
                    block.compile(dynamic=True)
                if args.projected_readout and fused:
                    model.head.projected_edge_irreps = torch.compile(
                        model.head.projected_edge_irreps, dynamic=True
                    )
                else:
                    model.head.edge_readout.forward_messages = torch.compile(
                        model.head.edge_readout.forward_messages, dynamic=True
                    )
            optimizer = torch.optim.AdamW(model.head.parameters(), lr=1e-3, fused=True)
            data_list = []
            for index, source in enumerate(examples[storage]):
                data = copy.deepcopy(source)
                data.hip_sizes = data.natoms.tolist()
                data.to("cuda")
                data.hip_fused_loss = fused
                emb = (
                    {
                        "node_embedding": data.hip_features.float()
                        * data.hip_feature_scale,
                        "batch": data.batch,
                    }
                    if storage != "dense"
                    else {
                        k: v.to("cuda") if torch.is_tensor(v) else v
                        for k, v in features[index].items()
                    }
                )
                data_list.append((data, emb))

            def loss_for(data, emb, precision=precision, storage=storage):
                if precision.startswith("online-"):
                    emb = model.features(
                        data, "fp16" if precision.startswith("online-fp16") else "fp32"
                    )
                out = model.head(data, emb)
                return (
                    structure_mse(out["hessian"], data.hessian, data.hip_sizes)
                    if storage == "dense"
                    else block_mse(out["edge_blocks"], out["node_blocks"], data)
                )

            optimizer.zero_grad(set_to_none=True)
            loss = loss_for(*data_list[0])
            scale = 128
            (loss * scale).backward()
            gradient = torch.cat(
                [
                    p.grad.flatten().float() / scale
                    for p in model.head.parameters()
                    if p.grad is not None
                ]
            )
            if reference is None:
                reference = (float(loss.detach()), gradient.clone())
            numerical = dict(
                loss=float(loss.detach()),
                loss_relative_error=abs(float(loss.detach()) - reference[0])
                / abs(reference[0]),
                gradient_relative_error=float(
                    (gradient - reference[1]).norm() / reference[1].norm()
                ),
                finite_gradients=bool(torch.isfinite(gradient).all()),
            )
            del gradient

            if not numerical["finite_gradients"]:
                raise ValueError(
                    "Reject configurations with nonfinite parameter gradients"
                )

            def update(data, emb, optimizer=optimizer, loss_for=loss_for):
                optimizer.zero_grad(set_to_none=True)
                loss = loss_for(data, emb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.head.parameters(), 10, error_if_nonfinite=True
                )
                optimizer.step()

            warmup_started = time.monotonic()
            for data, emb in data_list:
                update(data, emb)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            warmup_seconds = time.monotonic() - warmup_started
            started = time.monotonic()
            wall_start = time.time()
            for data, emb in data_list:
                update(data, emb)
            torch.cuda.synchronize()
            seconds = time.monotonic() - started
            row = dict(
                storage=storage,
                fused=fused,
                compiled=compiled,
                precision=precision,
                projected_readout=args.projected_readout and fused,
                seconds=seconds,
                warmup_seconds=warmup_seconds,
                samples_s=results["molecules"] / seconds,
                wall_start=wall_start,
                wall_end=time.time(),
                peak_allocated_GB=torch.cuda.max_memory_allocated() / 1e9,
                peak_reserved_GB=torch.cuda.max_memory_reserved() / 1e9,
                **numerical,
            )
            print(json.dumps(row), flush=True)
            results["cases"].append(row)
            (args.output / "summary.json").write_text(
                json.dumps(results, indent=2) + "\n"
            )
            del data_list, optimizer
        assert parameter_hash(model.backbone) == frozen
        results["frozen_verified"] = True
        (args.output / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    finally:
        telemetry.terminate()
        telemetry.wait(timeout=10)
        telemetry_file.close()


if __name__ == "__main__":
    main()
