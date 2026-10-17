"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Measure loading, frozen inference, and HIP training on identical molecular data.
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

import torch
from torch.profiler import ProfilerActivity, profile, record_function
from torch.utils.data import DataLoader

from fairchem.core.frozen_hip.data import HessianStream
from fairchem.core.frozen_hip.model import FrozenHIP, parameter_hash, structure_mse


def update(model, optimizer, data, features):
    optimizer.zero_grad(set_to_none=True)
    with record_function("frozen_backbone"), torch.no_grad():
        emb = features if features is not None else model.backbone(data)
    with record_function("hip_head"):
        prediction = model.head(data, emb)["hessian"]
    with record_function("hessian_loss"):
        loss = structure_mse(prediction, data.hessian, data.hip_sizes)
    with record_function("head_backward"):
        loss.backward()
    torch.nn.utils.clip_grad_norm_(
        model.head.parameters(), 10.0, error_if_nonfinite=True
    )
    optimizer.step()


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
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    model = FrozenHIP("cuda", checkpoint_edges=200000)
    state = torch.load(args.checkpoint, map_location="cuda", weights_only=False)
    model.head.load_state_dict(state["head"])
    initial = copy.deepcopy(model.head.state_dict())
    frozen = parameter_hash(model.backbone)
    torch.backends.cuda.matmul.allow_tf32 = True
    model.train()
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
    started = time.monotonic()
    first = next(iterator)
    startup_seconds = time.monotonic() - started
    started = time.monotonic()
    host = ([first] + list(iterator))[: args.batches]
    loader_seconds = time.monotonic() - started
    properties = torch.cuda.get_device_properties(0)
    gpu_id = str(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
    telemetry_path = args.output / "gpu-utilization.csv"
    telemetry_file = telemetry_path.open("w")
    telemetry = subprocess.Popen(
        [
            "nvidia-smi",
            "-i",
            gpu_id,
            "--query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw",
            "--format=csv,noheader,nounits",
            "-lms",
            "200",
        ],
        stdout=telemetry_file,
        env={**os.environ, "TZ": "UTC"},
        stderr=subprocess.DEVNULL,
    )
    results = dict(
        gpu=properties.name,
        batches=len(host),
        loader_seconds=loader_seconds,
        loader_startup_seconds=startup_seconds,
        molecules=sum(len(x.natoms) for x in host),
        atoms=sum(int(x.natoms.sum()) for x in host),
        cases=[],
    )
    try:
        for cached, compiled in ((False, False), (True, False), (True, True)):
            model.head.load_state_dict(initial)
            if compiled:
                torch._dynamo.config.recompile_limit = 16
                for block in model.head.blocks:
                    block.compile(dynamic=True)
                model.head.edge_readout.forward_messages = torch.compile(
                    model.head.edge_readout.forward_messages, dynamic=True
                )
            optimizer = torch.optim.AdamW(model.head.parameters(), lr=1e-3, fused=True)
            examples = []
            for batch in host:
                # Reuse host labels, but avoid retaining graph metadata on host inputs.
                data = copy.deepcopy(batch)
                data.hip_sizes = data.natoms.tolist()
                data.to("cuda")
                examples.append(data)
            features = []
            if cached:
                with torch.no_grad():
                    features = [model.backbone(data) for data in examples]

            for index in range(min(4, len(examples))):
                update(
                    model,
                    optimizer,
                    examples[index],
                    features[index] if cached else None,
                )
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = time.monotonic()
            wall_start = time.time()
            for index in range(len(examples)):
                update(
                    model,
                    optimizer,
                    examples[index],
                    features[index] if cached else None,
                )
            torch.cuda.synchronize()
            elapsed = time.monotonic() - started
            row = dict(
                cached=cached,
                compiled=compiled,
                seconds=elapsed,
                wall_start=wall_start,
                wall_end=time.time(),
                samples_s=results["molecules"] / elapsed,
                atoms_s=results["atoms"] / elapsed,
                peak_GB=torch.cuda.max_memory_allocated() / 1e9,
            )
            print(json.dumps(row), flush=True)
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]
            ) as prof:
                for index in range(min(4, len(examples))):
                    update(
                        model,
                        optimizer,
                        examples[index],
                        features[index] if cached else None,
                    )
            prof.export_chrome_trace(
                str(args.output / f"trace-cached{cached}-compiled{compiled}.json")
            )
            table = prof.key_averages().table(
                sort_by="self_cuda_time_total", row_limit=30
            )
            (
                args.output / f"operators-cached{cached}-compiled{compiled}.txt"
            ).write_text(table)
            results["cases"].append(row)
            (args.output / "summary.json").write_text(
                json.dumps(results, indent=2) + "\n"
            )
            del features, examples
        assert parameter_hash(model.backbone) == frozen
        results["frozen_verified"] = True
        (args.output / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    finally:
        telemetry.terminate()
        telemetry.wait(timeout=10)
        telemetry_file.close()


if __name__ == "__main__":
    main()
