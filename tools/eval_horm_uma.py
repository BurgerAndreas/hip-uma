#!/usr/bin/env python3
"""Evaluate HIP-UMA checkpoints on a deterministic prefix of HORM validation.

This adapts ``hip/scripts/eval_horm.py`` to FAIRChem/UMA inference checkpoints.
The HIP model uses its learned Hessian head.  For an energy/force-only model,
the Hessian is the second derivative of the learned energy with respect to
coordinates (not the derivative of its direct-force head).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
import weakref
from pathlib import Path

import torch

from fairchem.core.datasets import create_dataset, data_list_collater
from fairchem.core.models.uma.outputs import compute_hessian
from fairchem.core.units.mlip_unit.utils import load_inference_model


FIELDS = [
    "sample_index",
    "natoms",
    "energy_abs_error",
    "forces_mae",
    "hessian_mae",
    "hessian_mse",
    "seconds",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--hessian-method", choices=("predict", "autograd"), required=True
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/horm_atomic/ts1x-val.atomic.lmdb"),
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/horm_atomic/ts1x-val.atomic.metadata.npz"),
    )
    parser.add_argument("--max-samples", type=int, default=10_000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--flush-every", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-ema", action="store_true")
    return parser.parse_args()


def unwrap(output: dict, head: str, prop: str) -> torch.Tensor:
    value = output[head]
    if isinstance(value, dict):
        value = value[prop]
    return value


def load_completed(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for expected_index, row in enumerate(rows):
        if int(row["sample_index"]) != expected_index:
            raise ValueError(f"Non-contiguous resume file at row {expected_index}: {path}")
    return rows


def summarize(
    rows: list[dict[str, str]], args: argparse.Namespace, elapsed_seconds: float
) -> dict[str, object]:
    means = {}
    for key in ("energy_abs_error", "forces_mae", "hessian_mae", "hessian_mse"):
        means[key] = sum(float(row[key]) for row in rows) / len(rows)
    return {
        "checkpoint": str(args.checkpoint),
        "dataset": str(args.dataset),
        "selection": f"indices [0, {len(rows)}) in LMDB order",
        "num_samples": len(rows),
        "hessian_method": args.hessian_method,
        "weights": "model" if args.no_ema else "ema",
        "energy_mae_eV": means["energy_abs_error"],
        "forces_mae_eV_per_A": means["forces_mae"],
        "hessian_mae_eV_per_A2": means["hessian_mae"],
        "hessian_rmse_eV_per_A2": math.sqrt(means["hessian_mse"]),
        "elapsed_seconds_this_invocation": elapsed_seconds,
    }


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary_path = args.output.with_suffix(".json")

    dataset = create_dataset(
        {
            "format": "atomic_lmdb",
            "src": str(args.dataset),
            "metadata_path": str(args.metadata),
            "transforms": {
                "common_transform": {"dataset_name": "omol"},
                "molecule_cell_transform": {"cell_size": 120.0},
            },
        },
        split="val",
    )
    n_samples = min(args.max_samples, len(dataset))
    if n_samples != args.max_samples:
        raise ValueError(
            f"Requested {args.max_samples} samples, but dataset has only {len(dataset)}"
        )

    model, checkpoint = load_inference_model(
        str(args.checkpoint), use_ema=not args.no_ema
    )
    model_module = model.module if hasattr(model, "module") else model
    model_module.setup_tasks(checkpoint.tasks_config)
    # AveragedModel deep-copies the module, but Hessian_Head deliberately keeps
    # its backbone as a weak reference. Rebind that reference to the EMA copy.
    if "hessian" in model_module.output_heads:
        object.__setattr__(
            model_module.output_heads["hessian"],
            "_backbone_ref",
            weakref.ref(model_module.backbone),
        )
    model = model.to(args.device).eval()

    completed = load_completed(args.output)
    if len(completed) > n_samples:
        raise ValueError(f"Resume file has {len(completed)} rows, expected <= {n_samples}")

    mode = "a" if completed else "w"
    start = time.perf_counter()
    with args.output.open(mode, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if not completed:
            writer.writeheader()

        for sample_index in range(len(completed), n_samples):
            sample_start = time.perf_counter()
            sample = dataset[sample_index]
            batch = data_list_collater([sample], otf_graph=True).to(args.device)
            batch.pos.requires_grad_(args.hessian_method == "autograd")

            if args.hessian_method == "predict":
                with torch.no_grad():
                    output = model(batch)
                hessian = unwrap(output, "hessian", "hessian")
            else:
                output = model(batch)
                energy_for_derivative = unwrap(output, "omol_energy", "energy")
                energy_forces = -torch.autograd.grad(
                    energy_for_derivative.sum(), batch.pos, create_graph=True
                )[0]
                hessian = compute_hessian(
                    energy_forces, batch.pos, vmap=True, training=False
                )

            energy = unwrap(output, "omol_energy", "energy").reshape(-1)
            forces = unwrap(output, "forces", "forces").reshape_as(batch.forces)
            hessian = hessian.reshape_as(batch.hessian)

            energy_error = (energy - batch.energy.reshape(-1)).abs().mean()
            force_error = (forces - batch.forces).abs().mean()
            hessian_delta = hessian - batch.hessian

            row = {
                "sample_index": sample_index,
                "natoms": int(batch.natoms.item()),
                "energy_abs_error": energy_error.item(),
                "forces_mae": force_error.item(),
                "hessian_mae": hessian_delta.abs().mean().item(),
                "hessian_mse": hessian_delta.square().mean().item(),
                "seconds": time.perf_counter() - sample_start,
            }
            writer.writerow(row)
            completed.append({key: str(value) for key, value in row.items()})

            if (sample_index + 1) % args.flush_every == 0:
                handle.flush()
                summary = summarize(completed, args, time.perf_counter() - start)
                summary_path.write_text(json.dumps(summary, indent=2) + "\n")
                print(json.dumps(summary, sort_keys=True), flush=True)

    summary = summarize(completed, args, time.perf_counter() - start)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
