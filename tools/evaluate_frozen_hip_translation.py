"""Compare unchanged last checkpoints before and after translation projection."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from fairchem.core.frozen_hip.grouped_mole import enable_grouped_mole
from fairchem.core.frozen_hip.losses import structure_block_rmse
from fairchem.core.frozen_hip.metrics import (
    mass_weighted_eckart_totals,
    projected_metric_record,
)
from fairchem.core.frozen_hip.model import FrozenHIP, parameter_hash
from fairchem.core.frozen_hip.precision import configure_head_precision
from fairchem.core.frozen_hip.radial import configure_radial_tile, enable_fused_radial
from fairchem.core.frozen_hip.readout import enable_projected_readout
from fairchem.core.frozen_hip.rotations import (
    enable_fused_rotations,
    enable_fused_scatter,
)
from fairchem.core.frozen_hip.so2 import enable_block_so2
from fairchem.core.frozen_hip.translation import project_dense
from fairchem.core.frozen_hip.validation import (
    EvaluationStream,
    load_manifest,
    sample_fingerprint,
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-samples", type=int, default=0)
    args = p.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(7)
    cohort = load_manifest(args.data / "validation_10000.json")
    model = FrozenHIP("cuda", backbone_graph_version=4).eval()
    frozen = parameter_hash(model.backbone)
    enable_fused_rotations(model.backbone)
    enable_grouped_mole(model.backbone)
    enable_fused_rotations(model.head)
    enable_block_so2(model.head)
    enable_projected_readout(model.head)
    enable_fused_scatter(model.head, segmented=True)
    enable_fused_radial(model.head)
    configure_radial_tile(64)
    torch.backends.cuda.matmul.allow_tf32 = True
    states = []
    for path in args.checkpoints:
        state = torch.load(path, map_location="cpu", weights_only=False)
        assert state["backbone_sha256"] == frozen
        state["evaluation_checkpoint_sha256"] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        states.append(state)
    configure_head_precision(model.head, "fp16-edge")
    totals = torch.zeros(len(states), 2, 6, device="cuda", dtype=torch.float64)
    raw = torch.zeros(len(states), 2, 4, device="cuda", dtype=torch.float64)
    residual = torch.zeros(len(states), 2, device="cuda", dtype=torch.float64)
    seen = []
    stream = EvaluationStream(
        args.data, cohort, max_batch=32, max_atoms=1280, pair_budget=524288
    )
    loader = DataLoader(
        stream,
        batch_size=None,
        num_workers=2,
        multiprocessing_context="spawn",
        prefetch_factor=2,
    )
    with torch.inference_mode():
        for index, data in enumerate(loader):
            sizes = data.natoms.tolist()
            seen.extend(int(s) for s in data.sid)
            pbc = data.pbc
            data.apply(lambda x, pbc=pbc: x if x is pbc else x.to("cuda"))
            data.hip_sizes = sizes
            data.hip_fused_loss = True
            emb = model.features(data, "fp16")
            for k, state in enumerate(states):
                model.head.load_state_dict(state["head"])
                base = model.head(data, emb)["hessian"].float()
                for j, pred in enumerate([base, project_dense(base, sizes)]):
                    totals[k, j] += mass_weighted_eckart_totals(
                        pred, data.hessian, data.pos, data.atomic_numbers, sizes
                    )
                    errors = (pred - data.hessian).split([9 * n * n for n in sizes])
                    raw[k, j, 0] += sum(e.abs().mean() for e in errors)
                    raw[k, j, 1] += sum(e.square().mean().sqrt() for e in errors)
                    raw[k, j, 2] += structure_block_rmse(
                        pred, data.hessian, sizes
                    ) * len(sizes)
                    raw[k, j, 3] += len(sizes)
                    for flat, n in zip(
                        pred.split([9 * n * n for n in sizes]), sizes, strict=True
                    ):
                        h = flat.reshape(n, 3, n, 3).double()
                        residual[k, j] += h.sum(2).norm() / (
                            n**0.5 * h.norm().clamp_min(1e-30)
                        )
            if index % 20 == 0:
                print(
                    json.dumps({"batches": index + 1, "samples": len(seen)}), flush=True
                )
            if args.max_samples and len(seen) >= args.max_samples:
                break
    count = len(seen)
    if not args.max_samples:
        assert count == cohort["samples"]
        assert sample_fingerprint(seen) == cohort["sample_ids_sha256"]
    result = {
        "samples": count,
        "sample_ids_sha256": sample_fingerprint(seen),
        "from_last_checkpoint": True,
        "runs": [],
    }
    for k, state in enumerate(states):
        entry = {
            "loss": state["loss_name"],
            "checkpoint_sha256": state["evaluation_checkpoint_sha256"],
            "step": state["step"],
            "weight_decay": state["weight_decay"],
        }
        for j, name in enumerate(["original", "translation_projected"]):
            entry[name] = projected_metric_record(totals[k, j], count)
            v = raw[k, j].cpu().tolist()
            entry[name].update(
                cartesian_mae=v[0] / count,
                cartesian_rmse=v[1] / count,
                cartesian_block_rmse=v[2] / count,
                translation_null_relative=float(residual[k, j] / count),
            )
        result["runs"].append(entry)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
