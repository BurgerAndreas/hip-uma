"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Check frozen-backbone precision on every member of the fixed evaluation cohort.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from fairchem.core.frozen_hip.grouped_mole import enable_grouped_mole
from fairchem.core.frozen_hip.model import FrozenHIP, parameter_hash
from fairchem.core.frozen_hip.rotations import enable_fused_rotations
from fairchem.core.frozen_hip.so2 import enable_block_so2
from fairchem.core.frozen_hip.validation import (
    EvaluationStream,
    load_manifest,
    sample_fingerprint,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--projected-readout", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True
    model = FrozenHIP("cuda").eval()
    state = torch.load(args.checkpoint, map_location="cuda", weights_only=False)
    model.head.load_state_dict(state["head"])
    frozen = parameter_hash(model.backbone)
    if frozen != state["backbone_sha256"]:
        raise ValueError("Checkpoint backbone differs")
    cohort = load_manifest(args.manifest)
    stream = EvaluationStream(
        args.data, cohort, max_batch=64, max_atoms=2560, pair_budget=196608
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
    metrics = {
        mode: torch.zeros(4, device="cuda", dtype=torch.float64)
        for mode in ("fp32", "fp32-grouped", "fp16", "fp16-grouped")
    }
    seconds = {mode: 0.0 for mode in metrics}
    difference = {
        mode: torch.zeros(2, device="cuda", dtype=torch.float64)
        for mode in metrics
        if mode != "fp32"
    }
    seen = []
    started = time.monotonic()
    with torch.no_grad():
        for index, data in enumerate(loader):
            seen.extend(int(i) for i in data.sid)
            data.hip_sizes = data.natoms.tolist()
            data.to("cuda")
            reference = None
            for mode in metrics:
                torch.cuda.synchronize()
                timer = time.monotonic()
                enable_grouped_mole(model.backbone, enabled="grouped" in mode)
                enable_fused_rotations(model.head, enabled=mode != "fp32")
                enable_block_so2(model.head, enabled=mode != "fp32")
                from fairchem.core.frozen_hip.readout import enable_projected_readout

                enable_projected_readout(
                    model.head, enabled=args.projected_readout and mode != "fp32"
                )
                emb = model.features(
                    data, "fp16" if mode.startswith("fp16") else "fp32"
                )
                prediction = model.head(data, emb)["hessian"].float()
                torch.cuda.synchronize()
                seconds[mode] += time.monotonic() - timer
                error = prediction - data.hessian
                parts = error.split([9 * n * n for n in data.hip_sizes])
                metrics[mode] += torch.stack(
                    (
                        torch.stack([p.square().mean() for p in parts]).sum(),
                        torch.stack([p.abs().mean() for p in parts]).sum(),
                        error.new_tensor(len(parts)),
                        prediction.square().sum(),
                    )
                ).double()
                if reference is None:
                    reference = prediction
                else:
                    difference[mode][0] += (
                        (prediction - reference).double().square().sum()
                    )
                    difference[mode][1] += reference.double().square().sum()
            if (index + 1) % 20 == 0:
                print(
                    json.dumps(dict(batches=index + 1, samples=len(seen))), flush=True
                )
    fingerprint = sample_fingerprint(seen)
    if (
        len(seen) != cohort["samples"]
        or len(set(seen)) != len(seen)
        or fingerprint != cohort["sample_ids_sha256"]
    ):
        raise ValueError("Precision evaluation changed the fixed cohort")
    rows = {}
    for mode, values in metrics.items():
        mse, mae, count, _ = values.cpu().tolist()
        rows[mode] = dict(
            mse=mse / count,
            mae=mae / count,
            samples=int(count),
            compute_seconds=seconds[mode],
        )
    result = dict(
        samples=len(seen),
        projected_readout=args.projected_readout,
        sample_ids_sha256=fingerprint,
        metrics=rows,
        prediction_relative_l2={
            mode: float((v[0] / v[1]).sqrt()) for mode, v in difference.items()
        },
        checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        checkpoint_step=state["step"],
        backbone_sha256=frozen,
        wall_seconds=time.monotonic() - started,
    )
    if parameter_hash(model.backbone) != frozen:
        raise RuntimeError("Frozen parameters changed")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
