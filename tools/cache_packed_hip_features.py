"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Prepare reusable frozen features directly from existing compact training files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from fairchem.core.frozen_hip.cache import (
    AsyncFeatureCacheWriter,
    cached_features,
    feature_namespace,
)
from fairchem.core.frozen_hip.grouped_mole import enable_grouped_mole
from fairchem.core.frozen_hip.model import FrozenHIP, parameter_hash
from fairchem.core.frozen_hip.packed import PackedHessianStream, open_packed
from fairchem.core.frozen_hip.rotations import enable_fused_rotations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packed-data", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--duration", type=float, default=13200)
    parser.add_argument("--max-atoms", type=int, default=2560)
    parser.add_argument("--max-batch", type=int, default=64)
    parser.add_argument("--microbatch-pairs", type=int, default=524288)
    parser.add_argument("--window-files", type=int, default=8)
    parser.add_argument("--cache-memory-gb", type=float, default=16)
    parser.add_argument("--log-every", type=int, default=100000)
    parser.add_argument(
        "--backbone-precision", choices=("fp32", "fp16"), default="fp16"
    )
    parser.add_argument(
        "--backbone-graph-version", type=int, choices=(1, 2, 4), default=1
    )
    parser.add_argument("--tf32", action="store_true")
    parser.add_argument("--grouped-mole", action="store_true")
    parser.add_argument("--fused-backbone", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.rank < args.world:
        raise ValueError("Require 0 <= rank < world")
    if args.duration <= 0 or min(args.max_files, args.workers, args.log_every) < 0:
        raise ValueError("Duration must be positive and limits must be nonnegative")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    model = FrozenHIP("cuda", backbone_graph_version=args.backbone_graph_version).eval()
    del model.head
    frozen = parameter_hash(model.backbone)
    enable_fused_rotations(model.backbone, enabled=args.fused_backbone)
    enable_grouped_mole(model.backbone, enabled=args.grouped_mole)
    namespace = feature_namespace(
        frozen,
        tf32=args.tf32,
        backbone_precision=args.backbone_precision,
        grouped_mole=args.grouped_mole,
        backbone_graph_version=args.backbone_graph_version,
        fused_backbone=args.fused_backbone,
    )
    markers = (
        json.loads(args.inventory.read_text())
        if args.inventory
        else sorted(path.name for path in args.packed_data.glob("*.hip"))
    )
    if not isinstance(markers, list) or any(
        not isinstance(name, str)
        or Path(name).name != name
        or Path(name).suffix != ".hip"
        for name in markers
    ):
        raise ValueError(
            "Preparation inventory must contain relative compact filenames"
        )
    if len(set(markers)) != len(markers):
        raise ValueError("Preparation inventory contains duplicate files")
    if args.max_files:
        markers = markers[: args.max_files]
    if not markers:
        raise FileNotFoundError("No compact training files found")
    if len(markers) < args.world * max(1, args.workers):
        raise ValueError("Too few compact files for all ranks and loader workers")
    cache_root = args.feature_cache / namespace
    stream = PackedHessianStream(
        args.packed_data,
        markers=markers,
        backbone_sha256=frozen,
        backbone_precision=args.backbone_precision,
        backbone_graph_version=args.backbone_graph_version,
        fused_backbone=args.fused_backbone,
        tf32=args.tf32,
        grouped_mole=args.grouped_mole,
        feature_cache=cache_root,
        window_files=args.window_files,
        max_passes=1,
        rank=args.rank,
        world=args.world,
        max_atoms=args.max_atoms,
        max_batch=args.max_batch,
        pair_budget=args.microbatch_pairs,
        include_targets=False,
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
    writer = AsyncFeatureCacheWriter(
        cache_root, max_bytes=int(args.cache_memory_gb * (1 << 30))
    )
    started = time.monotonic()
    computed = reused = 0
    logged = 0
    completed = True
    iterator = iter(loader)
    try:
        with torch.no_grad():
            for logical in iterator:
                if time.monotonic() - started >= args.duration:
                    completed = False
                    break
                if args.log_every and computed + reused - logged >= args.log_every:
                    logged = computed + reused
                    elapsed = time.monotonic() - started
                    print(
                        json.dumps(
                            dict(
                                event="feature_preparation_progress",
                                processed_samples=logged,
                                computed_samples=computed,
                                reused_samples=reused,
                                seconds=elapsed,
                                computed_samples_s=computed / elapsed,
                            )
                        ),
                        flush=True,
                    )
                for data in logical:
                    count = len(data.natoms)
                    if hasattr(data, "hip_features"):
                        reused += count
                        continue
                    data.hip_sizes = data.natoms.tolist()
                    # Frozen inference does not need the large Cartesian targets.
                    for key in (
                        "pos",
                        "atomic_numbers",
                        "cell",
                        "natoms",
                        "charge",
                        "spin",
                        "batch",
                        "hip_atom_ptr",
                    ):
                        value = getattr(data, key, None)
                        if torch.is_tensor(value):
                            setattr(
                                data,
                                key,
                                value.pin_memory().to("cuda", non_blocking=True),
                            )
                    features = model.features(data, args.backbone_precision)
                    writer.store(data, features["node_embedding"])
                    computed += count
    finally:
        if args.workers and hasattr(iterator, "_shutdown_workers"):
            iterator._shutdown_workers()
        writer.close()
    if parameter_hash(model.backbone) != frozen:
        raise RuntimeError("Frozen backbone parameters changed")
    assigned = np.random.default_rng(7).permutation(len(markers))[
        args.rank :: args.world
    ]
    available = 0
    for index in assigned:
        path = args.packed_data / markers[index]
        values, header = open_packed(path)
        if "features" in values:
            available += 1
            continue
        digest = hashlib.sha256(str(header.get("fingerprint", "")).encode())
        digest.update(np.asarray(values["omol_index"]).tobytes())
        digest.update(np.asarray(values["natoms"]).tobytes())
        if (
            cached_features(cache_root, "bundles/" + path.stem, digest.hexdigest())
            is not None
        ):
            available += 1
    missing = len(assigned) - available
    seconds = time.monotonic() - started
    print(
        json.dumps(
            dict(
                event="feature_preparation_complete",
                computed_samples=computed,
                reused_samples=reused,
                seconds=seconds,
                computed_samples_s=computed / seconds,
                complete_pass=completed,
                assigned_bundles=len(assigned),
                available_bundles=available,
                missing_bundles=missing,
                frozen_backbone_verified=True,
                rank=args.rank,
                world=args.world,
                namespace=namespace,
            )
        ),
        flush=True,
    )
    if completed and missing:
        raise RuntimeError(
            "Feature pass left incomplete bundles; increase cache memory or reduce "
            "the window size and rerun"
        )


if __name__ == "__main__":
    main()
