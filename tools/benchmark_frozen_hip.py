"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Compare training precision and batch budgets on identical molecular inputs.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import h5py
import torch

from fairchem.core.frozen_hip.data import make_batch
from fairchem.core.frozen_hip.model import FrozenHIP, parameter_hash, structure_mse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--sizes", type=int, nargs="+", default=[16, 64, 128, 256, 350])
    args = parser.parse_args()
    torch.set_num_threads(2)
    device = "cuda"
    model = FrozenHIP(device)
    original_hash = parameter_hash(model.backbone)
    initial = copy.deepcopy(model.head.state_dict())
    candidates = []
    markers = sorted(args.data.glob("shards/*/*.verified.json"))
    largest = args.data / "shards/41/shard_41910.verified.json"
    cohort = markers[:: max(1, len(markers) // 256)]
    if largest.exists():
        cohort.append(largest)
    for path in cohort:
        record = json.loads(path.read_text())
        with h5py.File(args.data / record["path"]) as f:
            candidates.append((int(f["natoms"][0]), record["path"]))
    selected = sorted(
        {min(candidates, key=lambda x: abs(x[0] - n)) for n in args.sizes}
    )
    report = {"gpu": torch.cuda.get_device_name(), "cases": []}
    for n, path in selected:
        with h5py.File(args.data / path) as f:
            values = {
                k: f[k][:]
                for k in (
                    "natoms",
                    "atomic_numbers",
                    "coords",
                    "hessian_flat",
                    "charge",
                    "spin",
                    "omol_index",
                    "atom_ptr",
                    "hessian_ptr",
                )
            }
        for budget, precision, tf32 in [
            (8192, "fp32", False),
            (32768, "fp32", True),
            (65536, "fp32", True),
            (32768, "bf16", True),
        ]:
            import numpy as np

            rows = np.flatnonzero(values["natoms"] == n)
            b = min(64, len(rows), max(1, budget // (n * n)))
            data = make_batch(values, rows[:b]).to(device)
            data.hip_sizes = [n] * b
            model.head.load_state_dict(initial)
            model.train()
            torch.backends.cuda.matmul.allow_tf32 = tf32
            optimizer = torch.optim.AdamW(model.head.parameters(), lr=1e-4, fused=True)
            row = {
                "natoms": n,
                "batch": b,
                "pair_budget": budget,
                "precision": precision,
                "tf32": tf32,
            }
            try:
                torch.cuda.reset_peak_memory_stats()
                with torch.no_grad():
                    emb = model.backbone(data)
                    reference = model.head(data, emb)["hessian"].float()
                    with torch.autocast(
                        "cuda", dtype=torch.bfloat16, enabled=precision == "bf16"
                    ):
                        actual = model.head(data, emb)["hessian"].float()
                    row["head_relative_error"] = (
                        (actual - reference).norm() / reference.norm().clamp_min(1e-12)
                    ).item()
                for step in range(args.steps + 2):
                    if step == 2:
                        torch.cuda.synchronize()
                        started = time.monotonic()
                    optimizer.zero_grad(set_to_none=True)
                    with torch.no_grad():
                        emb = model.backbone(data)
                    with torch.autocast(
                        "cuda", dtype=torch.bfloat16, enabled=precision == "bf16"
                    ):
                        out = model.head(data, emb)["hessian"]
                    loss = structure_mse(out, data.hessian, [n] * b)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.head.parameters(), 10.0, error_if_nonfinite=True
                    )
                    optimizer.step()
                torch.cuda.synchronize()
                row.update(
                    samples_s=b * args.steps / (time.monotonic() - started),
                    peak_GB=torch.cuda.max_memory_allocated() / 1e9,
                    loss=loss.item(),
                )
                # Same head architecture and FP32 feature values, now reused.
                with torch.no_grad():
                    emb = model.backbone(data)
                torch.cuda.synchronize()
                cached_start = time.monotonic()
                for _ in range(args.steps):
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(
                        "cuda", dtype=torch.bfloat16, enabled=precision == "bf16"
                    ):
                        out = model.head(data, emb)["hessian"]
                    structure_mse(out, data.hessian, [n] * b).backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.head.parameters(), 10.0, error_if_nonfinite=True
                    )
                    optimizer.step()
                torch.cuda.synchronize()
                row["cached_samples_s"] = (
                    b * args.steps / (time.monotonic() - cached_start)
                )
            except torch.cuda.OutOfMemoryError:
                row["oom"] = True
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
            report["cases"].append(row)
            print(json.dumps(row), flush=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
    assert parameter_hash(model.backbone) == original_hash
    assert all(p.grad is None for p in model.backbone.parameters())
    report["frozen_backbone_verified"] = True
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
