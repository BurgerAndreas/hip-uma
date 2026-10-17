"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Compare HIP precision modes with identical samples, parameters, and features.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from fairchem.core.frozen_hip.data import make_batch
from fairchem.core.frozen_hip.model import FrozenHIP, parameter_hash, structure_mse
from fairchem.core.frozen_hip.precision import configure_head_precision


def loss_and_output(model, data, emb, mode):
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=mode == "bf16"):
        out = model.head(data, emb)["hessian"]
    return structure_mse(out, data.hessian, data.hip_sizes), out.float()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--steps", default=6, type=int)
    parser.add_argument("--pair-budget", default=196608, type=int)
    parser.add_argument("--sizes", nargs="+", default=[16, 72, 124, 350], type=int)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(7)
    model = FrozenHIP("cuda", checkpoint_edges=200000)
    frozen = parameter_hash(model.backbone)
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cuda", weights_only=False)
        assert state["backbone_sha256"] == frozen
        model.head.load_state_dict(state["head"])
    initial = copy.deepcopy(model.head.state_dict())
    markers = sorted(args.data.glob("shards/*/*.verified.json"))
    candidates = []
    for marker in markers[:: max(1, len(markers) // 256)] + markers[-1:]:
        row = json.loads(marker.read_text())
        with h5py.File(args.data / row["path"]) as f:
            candidates.append((int(f["natoms"][0]), row["path"]))
    selected = sorted(
        {min(candidates, key=lambda p: abs(p[0] - n)) for n in args.sizes}
    )
    report = dict(gpu=torch.cuda.get_device_name(), cases=[])
    for n, path in selected:
        with h5py.File(args.data / path) as f:
            values = {
                key: f[key][:]
                for key in (
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
        rows = np.flatnonzero(values["natoms"] == n)
        b = min(32, len(rows), max(1, args.pair_budget // (n * n)))
        data = make_batch(values, rows[:b]).to("cuda")
        data.hip_sizes = [n] * b
        torch.backends.cuda.matmul.allow_tf32 = False
        with torch.no_grad():
            reference_emb = model.backbone(data)
        reference, reference_grad, reference_loss = None, None, None
        for mode, tf32 in (
            ("fp32", False),
            ("fp32", True),
            ("bf16-mlp", True),
            ("bf16", True),
        ):
            emb = reference_emb
            configure_head_precision(model.head, mode)
            model.head.load_state_dict(initial)
            torch.backends.cuda.matmul.allow_tf32 = tf32
            optimizer = torch.optim.AdamW(model.head.parameters(), lr=1e-4, fused=True)

            optimizer.zero_grad(set_to_none=True)
            loss, actual = loss_and_output(model, data, emb, mode)
            loss.backward()
            grads = torch.cat(
                [p.grad.detach().flatten() for p in model.head.parameters()]
            )
            if reference is None:
                reference = actual.detach().clone()
                reference_grad = grads.clone()
                reference_loss = loss.detach().clone()
            record = dict(
                natoms=n,
                batch=b,
                precision=mode,
                tf32=tf32,
                output_relative_error=(
                    (actual.detach() - reference).norm()
                    / reference.norm().clamp_min(1e-12)
                ).item(),
                gradient_relative_error=(
                    (grads - reference_grad).norm()
                    / reference_grad.norm().clamp_min(1e-12)
                ).item(),
                loss_relative_error=(
                    (loss.detach() - reference_loss).abs()
                    / reference_loss.abs().clamp_min(1e-12)
                ).item(),
                gradients_finite=bool(grads.isfinite().all()),
            )
            del loss, actual, grads
            for cached in (True, False):
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.reset_peak_memory_stats()
                for step in range(args.steps + 2):
                    if step == 2:
                        torch.cuda.synchronize()
                        started = time.monotonic()
                    optimizer.zero_grad(set_to_none=True)
                    if not cached:
                        with torch.no_grad():
                            emb = model.backbone(data)
                    loss, out = loss_and_output(model, data, emb, mode)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.head.parameters(), 10.0, error_if_nonfinite=True
                    )
                    optimizer.step()
                    del loss, out
                torch.cuda.synchronize()
                prefix = "cached" if cached else "raw"
                record[f"{prefix}_samples_s"] = (
                    b * args.steps / (time.monotonic() - started)
                )
                record[f"{prefix}_peak_GB"] = torch.cuda.max_memory_allocated() / 1e9
            report["cases"].append(record)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(record), flush=True)
        del emb, reference_emb, data, reference, reference_grad
    assert parameter_hash(model.backbone) == frozen
    assert all(p.grad is None for p in model.backbone.parameters())
    report["frozen_backbone_verified"] = True
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
