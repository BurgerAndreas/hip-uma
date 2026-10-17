"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Prepare resumable, portable HIP bundles with frozen UMA features and half labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from fairchem.core.frozen_hip.cache import feature_namespace
from fairchem.core.frozen_hip.data import HessianStream
from fairchem.core.frozen_hip.model import FrozenHIP, parameter_hash
from fairchem.core.frozen_hip.packed import (
    arrays_from_batches,
    open_packed,
    write_packed,
)
from fairchem.core.frozen_hip.validation import DATA_REVISION


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--feature-dtype", choices=("float32", "float16"), default="float32"
    )
    parser.add_argument("--group-shards", type=int, default=8)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world", type=int, default=1)
    parser.add_argument("--max-groups", type=int)
    parser.add_argument(
        "--group-indices", help="Optional comma-separated source group indices"
    )
    parser.add_argument("--feature-cache", type=Path)
    parser.add_argument("--targets-only", action="store_true")
    parser.add_argument("--grouped-mole", action="store_true")
    parser.add_argument("--fused-backbone", action="store_true")
    parser.add_argument(
        "--backbone-precision", choices=("fp32", "fp16"), default="fp32"
    )
    parser.add_argument(
        "--backbone-graph-version", type=int, choices=(1, 2, 4), default=2
    )
    parser.add_argument("--max-atoms", type=int, default=4096)
    args = parser.parse_args()
    if args.group_shards < 1 or not 0 <= args.rank < args.world:
        raise ValueError("Require positive shard groups and a valid worker rank")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    model = (
        None
        if args.targets_only
        else FrozenHIP(
            "cuda", backbone_graph_version=args.backbone_graph_version
        ).eval()
    )
    if model is not None:
        del model.head
    frozen = parameter_hash(model.backbone) if model is not None else None
    if model is not None and args.grouped_mole:
        from fairchem.core.frozen_hip.grouped_mole import enable_grouped_mole

        enable_grouped_mole(model.backbone)
    if model is not None and args.fused_backbone:
        from fairchem.core.frozen_hip.rotations import enable_fused_rotations

        enable_fused_rotations(model.backbone)
    torch.backends.cuda.matmul.allow_tf32 = True
    markers = sorted(args.data.glob("shards/*/*.verified.json"))
    if not markers:
        raise ValueError("No verified input shards")
    groups = list(
        enumerate(
            [
                markers[i : i + args.group_shards]
                for i in range(0, len(markers), args.group_shards)
            ]
        )
    )
    if args.group_indices:
        chosen = {int(i) for i in args.group_indices.split(",")}
        if not chosen or min(chosen) < 0 or max(chosen) >= len(groups):
            raise ValueError("Source group index is out of range")
        groups = [row for row in groups if row[0] in chosen]
    if args.max_groups is not None:
        groups = groups[: args.max_groups]
    for group_index, group in groups[args.rank :: args.world]:
        sources = [json.loads(p.read_text()) for p in group]
        identity = dict(
            dataset_revision=DATA_REVISION,
            backbone_sha256=frozen,
            tf32=True,
            geometry_version=1,
            feature_dtype=None if args.targets_only else args.feature_dtype,
            backbone_precision=None if args.targets_only else args.backbone_precision,
            sources=[dict(path=s["path"], sha256=s["sha256"]) for s in sources],
        )
        if not args.targets_only:
            identity["grouped_mole"] = args.grouped_mole
            if args.grouped_mole and args.backbone_precision == "fp16":
                identity["expert_fp32_output"] = True
            if args.fused_backbone:
                identity["fused_backbone"] = True
            if args.backbone_graph_version != 2:
                identity["backbone_graph_version"] = args.backbone_graph_version
        fingerprint = hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode()
        ).hexdigest()
        path = args.output / f"bundle_{group_index:06d}.hip"
        if path.exists():
            _, header = open_packed(path)
            if header.get("fingerprint") != fingerprint:
                raise ValueError(
                    "Existing bundle has different source or precision settings"
                )
            continue
        stream = HessianStream(
            args.data,
            markers=[str(p.relative_to(args.data)) for p in group],
            window_shards=len(group),
            max_batch=256,
            max_atoms=args.max_atoms,
            pair_budget=1000000000,
            split="train",
            max_passes=1,
            feature_cache=(
                args.feature_cache
                / feature_namespace(
                    frozen,
                    tf32=True,
                    backbone_precision=args.backbone_precision,
                    grouped_mole=args.grouped_mole,
                    backbone_graph_version=args.backbone_graph_version,
                    fused_backbone=args.fused_backbone,
                )
                if args.feature_cache and not args.targets_only
                else None
            ),
        )
        batches, features = [], []
        started = time.monotonic()
        with torch.inference_mode():
            for data in stream:
                data.hip_sizes = data.natoms.tolist()
                if args.targets_only:
                    features.append(None)
                else:
                    data.to("cuda")
                    emb = (
                        {"node_embedding": data.hip_features}
                        if hasattr(data, "hip_features")
                        else model.features(data, args.backbone_precision)
                    )
                    features.append(
                        {"node_embedding": emb["node_embedding"].float().cpu()}
                    )
                    del emb
                # Remove the backbone neighbor graph before retaining molecular data.
                batches.append(
                    SimpleNamespace(
                        **{
                            key: getattr(data, key).cpu()
                            if torch.is_tensor(getattr(data, key))
                            else getattr(data, key)
                            for key in (
                                "natoms",
                                "atomic_numbers",
                                "pos",
                                "charge",
                                "spin",
                                "sid",
                                "hessian",
                            )
                        }
                    )
                )
                del data
        values, scale = arrays_from_batches(
            batches, features, feature_dtype=args.feature_dtype
        )
        ids = values["omol_index"]
        if len(set(ids.tolist())) != len(ids):
            raise ValueError("Input source group contains duplicate sample identities")
        write_packed(
            path,
            values,
            dict(
                **identity,
                fingerprint=fingerprint,
                feature_scale=scale,
                samples=len(ids),
            ),
        )
        print(
            json.dumps(
                dict(
                    bundle=path.name,
                    samples=len(ids),
                    bytes=path.stat().st_size,
                    seconds=time.monotonic() - started,
                )
            ),
            flush=True,
        )
        del batches, features, values
    if model is not None and parameter_hash(model.backbone) != frozen:
        raise RuntimeError("Frozen UMA parameters changed during preprocessing")


if __name__ == "__main__":
    main()
