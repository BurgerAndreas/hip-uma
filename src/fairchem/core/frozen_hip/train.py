"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Train a frozen UMA-s-1.2 plus HIP head with resumable distributed checkpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from fairchem.core.frozen_hip.cache import AsyncFeatureCacheWriter, feature_namespace
from fairchem.core.frozen_hip.data import HessianStream
from fairchem.core.frozen_hip.losses import structure_block_rmse, structure_mae
from fairchem.core.frozen_hip.model import FrozenHIP, parameter_hash, structure_mse
from fairchem.core.frozen_hip.packed import (
    PackedHessianStream,
    block_mae,
    block_mse,
    block_rmse,
)
from fairchem.core.frozen_hip.precision import configure_head_precision
from fairchem.core.frozen_hip.rotations import enable_fused_rotations
from fairchem.core.frozen_hip.validation import (
    EvaluationStream,
    load_last_head,
    load_manifest,
    sample_fingerprint,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=200000)
    p.add_argument("--duration", type=float, default=13200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument(
        "--loss", choices=("mse", "mae", "block-rmse", "mass-block-rmse"), default="mse"
    )
    p.add_argument("--profile-steps", type=int, default=0)
    p.add_argument("--profile-after", type=int, default=100)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--prefetch-factor", type=int, default=8)
    p.add_argument("--device-prefetch", action="store_true")
    p.add_argument("--threaded-prefetch", action="store_true")
    p.add_argument("--fast-packed-loader", action="store_true")
    p.add_argument("--packed-window-memory-gb", type=float, default=0.0)
    p.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--pair-budget",
        type=int,
        default=1000000000,
        help="Optional per-rank pair limit for the optimizer batch",
    )
    p.add_argument(
        "--microbatch-pairs",
        type=int,
        default=196608,
        help="Per-GPU quadratic work limit before gradient accumulation",
    )
    p.add_argument(
        "--max-batch", type=int, default=64, help="Global molecule cap across all ranks"
    )
    p.add_argument(
        "--max-atoms",
        type=int,
        default=2560,
        help="Global target atom budget across all ranks",
    )
    p.add_argument("--window-shards", type=int, default=8)
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--checkpoint-every", type=int, default=500)
    p.add_argument("--validate-every", type=int, default=400000)
    p.add_argument("--validation-manifest", type=Path)
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--code-revision")
    p.add_argument("--execution-mode", choices=("general",), default="general")
    p.add_argument(
        "--head-precision", choices=("fp32", "bf16", "bf16-mlp"), default="fp32"
    )
    p.add_argument(
        "--checkpoint-edges",
        type=int,
        help="Defaults to the microbatch pair limit plus one",
    )
    p.add_argument("--benchmark", action="store_true")
    p.add_argument("--resume", type=Path)
    p.add_argument("--feature-cache", type=Path)
    p.add_argument("--feature-cache-read-only", action="store_true")
    p.add_argument("--cache-memory-gb", type=float, default=16.0)
    p.add_argument("--packed-data", type=Path)
    p.add_argument("--fused-rotations", action="store_true")
    p.add_argument("--grouped-mole", action="store_true")
    p.add_argument("--fused-backbone", action="store_true")
    p.add_argument("--block-so2", action="store_true")
    p.add_argument("--projected-readout", action="store_true")
    p.add_argument("--backbone-precision", choices=("fp32", "fp16"), default="fp32")
    p.add_argument("--backbone-graph-version", type=int, choices=(1, 2, 4), default=2)
    p.add_argument("--tf32", action="store_true")
    p.add_argument("--compile-head", action="store_true")
    p.add_argument("--scaled-linear", action="store_true")
    p.add_argument("--fused-scatter", action="store_true")
    p.add_argument("--fused-wigner", action="store_true")
    p.add_argument("--fused-radial", action="store_true")
    p.add_argument("--radial-tile", type=int, choices=(64, 128, 256), default=64)
    p.add_argument("--segmented-scatter", action="store_true")
    p.add_argument("--async-grad-check", action="store_true")
    p.add_argument("--stop-file", type=Path)
    p.add_argument("--refresh-data-every", type=int, default=2000)
    args = p.parse_args()
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        p.error("--weight-decay must be finite and nonnegative")
    if (
        not math.isfinite(args.packed_window_memory_gb)
        or args.packed_window_memory_gb < 0
    ):
        p.error("Packed window memory must be finite and nonnegative")
    if args.threaded_prefetch and not args.device_prefetch:
        p.error("Threaded prefetch requires device prefetch")
    if args.prefetch_factor < 1:
        p.error("Prefetch count must be positive")
    if args.segmented_scatter:
        args.fused_scatter = True
    if args.scaled_linear and args.head_precision != "fp32":
        p.error("Scaled linear operations require FP32 head precision")
    if args.fused_wigner and not args.fused_rotations:
        p.error("Fused Wigner geometry requires fused rotations")
    if args.fused_scatter and not args.fused_rotations:
        p.error("Fused scatter requires fused rotations")
    if args.fused_radial and (
        not args.fused_rotations or args.head_precision != "fp32"
    ):
        p.error(
            "Fused radial operations require fused rotations and FP32 head precision"
        )
    if args.checkpoint_edges is None:
        args.checkpoint_edges = args.microbatch_pairs + 1
    return args


def main():
    args = parse_args()
    cohort_path = args.validation_manifest or args.data / "validation_10000.json"
    cohort = load_manifest(cohort_path)
    if args.eval_only and not (args.output / "latest.pt").exists():
        raise FileNotFoundError("Final evaluation requires the last saved checkpoint")
    evaluation_state = None
    if args.eval_only:
        evaluation_state = torch.load(
            args.output / "latest.pt", map_location="cpu", weights_only=False
        )
        args.layers = evaluation_state.get("head_layers", args.layers)
        for key, default in (
            ("loss_name", "mse"),
            ("head_precision", "fp32"),
            ("backbone_precision", "fp32"),
            ("backbone_graph_version", 2),
            ("tf32", False),
            ("grouped_mole", False),
            ("block_so2", False),
            ("projected_readout", False),
            ("fused_rotations", False),
            ("fused_backbone", False),
            ("scaled_linear", False),
            ("fused_scatter", False),
            ("fused_wigner", False),
            ("fused_radial", False),
            ("segmented_scatter", False),
        ):
            setattr(
                args,
                "loss" if key == "loss_name" else key,
                evaluation_state.get(key, default),
            )
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    if args.cache_memory_gb <= 0:
        raise ValueError("Feature cache memory budget must be positive")
    if args.microbatch_pairs < 1 or args.window_shards < 1:
        raise ValueError("Microbatch and shard-window limits must be positive")
    if args.max_batch < world or args.max_atoms < world:
        raise ValueError("Global batch limits must allow at least one item per rank")
    if args.max_batch % world or args.max_atoms % world:
        raise ValueError("Global batch limits must be divisible by the rank count")
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    if world > 1:
        dist.init_process_group("nccl", device_id=device)
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    args.output.mkdir(parents=True, exist_ok=True)
    model = FrozenHIP(
        device,
        layers=args.layers,
        execution_mode=args.execution_mode,
        checkpoint_edges=args.checkpoint_edges,
        backbone_graph_version=args.backbone_graph_version,
    )
    frozen_hash = parameter_hash(model.backbone)
    if args.fused_backbone:
        enable_fused_rotations(model.backbone)
    if args.grouped_mole:
        from fairchem.core.frozen_hip.grouped_mole import enable_grouped_mole

        enable_grouped_mole(model.backbone)
    if args.block_so2:
        from fairchem.core.frozen_hip.so2 import enable_block_so2

        enable_block_so2(model.head)
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    cache_root = (
        args.feature_cache
        / feature_namespace(
            frozen_hash,
            tf32=args.tf32,
            backbone_precision=args.backbone_precision,
            grouped_mole=args.grouped_mole,
            backbone_graph_version=args.backbone_graph_version,
            fused_backbone=args.fused_backbone,
        )
        if args.feature_cache
        else None
    )
    cache_writer = (
        AsyncFeatureCacheWriter(
            cache_root, max_bytes=int(args.cache_memory_gb * (1 << 30))
        )
        if cache_root and not args.feature_cache_read_only
        else None
    )
    if args.fused_rotations:
        enable_fused_rotations(model.head)
    if args.projected_readout:
        from fairchem.core.frozen_hip.readout import enable_projected_readout

        enable_projected_readout(model.head)
    if args.fused_wigner:
        from fairchem.core.frozen_hip.wigner import enable_fused_wigner

        enable_fused_wigner(model.head)
    if args.fused_scatter:
        from fairchem.core.frozen_hip.rotations import enable_fused_scatter

        enable_fused_scatter(model.head, segmented=args.segmented_scatter)
    if args.scaled_linear:
        from fairchem.core.frozen_hip.linear import enable_scaled_linear

        enable_scaled_linear(model.head)
    if args.fused_radial:
        from fairchem.core.frozen_hip.radial import (
            configure_radial_tile,
            enable_fused_radial,
        )

        configure_radial_tile(args.radial_tile)
        enable_fused_radial(model.head)
    head = model.head
    configure_head_precision(head, args.head_precision)
    if args.compile_head:
        # The small head fits one gradient bucket. Avoid DDP graph splitting,
        # whose symbolic reshape handling fails for variable molecular graphs.
        torch._dynamo.config.optimize_ddp = False
        torch._dynamo.config.recompile_limit = 16
        for block in head.blocks:
            block.compile(dynamic=True)
        if args.projected_readout:
            head.projected_edge_irreps = torch.compile(
                head.projected_edge_irreps, dynamic=True
            )
        else:
            head.edge_readout.forward_messages = torch.compile(
                head.edge_readout.forward_messages, dynamic=True
            )
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=args.lr, weight_decay=args.weight_decay, fused=True
    )
    initial_head_hash = parameter_hash(head)
    step = 0
    resume = (
        args.output / "latest.pt"
        if args.eval_only
        else args.resume or args.output / "latest.pt"
    )
    if resume.exists():
        state = (
            evaluation_state
            if args.eval_only
            else torch.load(resume, map_location=device, weights_only=False)
        )
        if state["backbone_sha256"] != frozen_hash:
            raise ValueError("Checkpoint backbone differs from the frozen UMA model")
        if state.get("validation_sample_ids_sha256") not in (
            None,
            cohort["sample_ids_sha256"],
        ):
            raise ValueError("Checkpoint evaluation cohort differs from this run")
        saved_loss = state.get("loss_name", "mse")
        if args.eval_only:
            args.loss = saved_loss
        elif saved_loss != args.loss and not args.benchmark:
            raise ValueError(
                "Checkpoint loss differs; use a separate output for a new loss"
            )
        head.load_state_dict(state["head"])
        initial_head_hash = state.get("initial_head_sha256")
        if (
            not args.eval_only
            and not args.benchmark
            and state.get("target_steps") not in (None, args.steps)
        ):
            raise ValueError("Checkpoint step target differs from this experiment")
        if not args.eval_only:
            optimizer.load_state_dict(state["optimizer"])
            if any(
                group["weight_decay"] != args.weight_decay
                for group in optimizer.param_groups
            ):
                raise ValueError("Checkpoint weight decay differs from this experiment")
        step = state["step"]
    if world > 1:
        # Synchronize the small trainable head, never the frozen backbone.
        model.head = DistributedDataParallel(
            head, device_ids=[local], broadcast_buffers=False
        )

    def synchronized_inventory():
        inventory_root = args.packed_data or args.data
        pattern = "*.hip" if args.packed_data else "shards/*/*.verified.json"
        # Partition one immutable list on every rank while downloads add files.
        entries = (
            [
                str(p.relative_to(inventory_root))
                for p in sorted(inventory_root.glob(pattern))
            ]
            if rank == 0
            else None
        )
        message = [entries]
        if world > 1:
            dist.broadcast_object_list(message, src=0, device=device)
        return message[0]

    inventory = synchronized_inventory()

    def loader():
        if args.packed_data:
            stream = PackedHessianStream(
                args.packed_data,
                rank=rank,
                world=world,
                seed=args.seed + step,
                max_batch=args.max_batch // world,
                max_atoms=args.max_atoms // world,
                pair_budget=args.microbatch_pairs,
                logical_pair_budget=args.pair_budget,
                markers=inventory,
                backbone_sha256=frozen_hash,
                backbone_precision=args.backbone_precision,
                backbone_graph_version=args.backbone_graph_version,
                fused_backbone=args.fused_backbone,
                tf32=args.tf32,
                grouped_mole=args.grouped_mole,
                window_files=args.window_shards,
                feature_cache=cache_root,
                include_cache_metadata=not args.feature_cache_read_only,
                fast_packing=args.fast_packed_loader,
                window_memory_bytes=int(args.packed_window_memory_gb * 1024**3),
            )
        else:
            stream = HessianStream(
                args.data,
                rank=rank,
                world=world,
                seed=args.seed + step,
                pair_budget=args.pair_budget,
                microbatch_pairs=args.microbatch_pairs,
                max_batch=args.max_batch // world,
                max_atoms=args.max_atoms // world,
                window_shards=args.window_shards,
                split="train",
                feature_cache=cache_root,
                markers=inventory,
            )
        kw = (
            {
                "multiprocessing_context": "spawn",
                "prefetch_factor": args.prefetch_factor,
                "persistent_workers": True,
            }
            if args.workers
            else {}
        )
        return DataLoader(
            stream,
            batch_size=None,
            num_workers=args.workers,
            pin_memory=args.pin_memory,
            **kw,
        )

    train_loader = None if args.eval_only else loader()
    train_iter = iter(train_loader) if train_loader is not None else None
    stopping = False

    def stop_requested(*_):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop_requested)
    signal.signal(signal.SIGUSR1, stop_requested)
    started = time.monotonic()
    window = started
    window_updates = 0
    metrics = torch.zeros(5, device=device, dtype=torch.float64)
    host_metrics = [0, 0, 0, 0]
    total_samples = 0
    model.train()

    profiler = None
    phase = nullcontext
    if args.profile_steps:
        if args.profile_steps < 0 or args.profile_after < 0:
            raise ValueError("Profile step limits must be nonnegative")
        from torch.profiler import ProfilerActivity, profile, record_function, schedule

        def write_trace(trace):
            trace.export_chrome_trace(str(args.output / f"profile-rank{rank}.json"))
            (args.output / f"profile-rank{rank}.txt").write_text(
                trace.key_averages().table(sort_by="self_cpu_time_total", row_limit=80)
            )

        profiler = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(
                wait=args.profile_after, warmup=1, active=args.profile_steps, repeat=1
            ),
            on_trace_ready=write_trace,
        )
        phase = record_function
        profiler.start()

    def move(data):
        sizes = data.natoms.tolist()
        pbc = data.pbc
        if args.packed_data and hasattr(data, "hip_features"):
            # Cached, complete HIP graphs need neither periodic cells nor CSD
            # inputs. Keep unused bookkeeping off the device transfer stream.
            for name in (
                "pos",
                "atomic_numbers",
                "atomic_numbers_full",
                "natoms",
                "batch",
                "batch_full",
                "hip_features",
                "hip_feature_scale",
                "hip_atom_ptr",
                "hip_edge_ptr",
                "hip_pair_ptr",
                "hip_pair_targets",
                "hip_diagonal_targets",
                "hip_asymmetry_mse",
                "hessian",
            ):
                value = getattr(data, name, None)
                if torch.is_tensor(value):
                    setattr(data, name, value.to(device, non_blocking=True))
        else:
            # Boolean periodic flags control Python branches in graph generation.
            data.apply(
                lambda value: value
                if value is pbc
                else value.to(device, non_blocking=True)
            )
        data.hip_sizes = sizes
        data.hip_fused_loss = args.fused_rotations
        return data

    def device_iterator(source):
        if not args.device_prefetch:
            return source
        from fairchem.core.frozen_hip.prefetch import (
            DeviceBatchPrefetcher,
            ThreadedDeviceBatchPrefetcher,
        )

        prefetcher = (
            ThreadedDeviceBatchPrefetcher
            if args.threaded_prefetch
            else DeviceBatchPrefetcher
        )
        return prefetcher(source, move, device)

    if train_iter is not None:
        train_iter = device_iterator(train_iter)

    def predict(data):
        # Frozen inference can use FP16; the head receives FP32 features.
        with phase("frozen_backbone"), torch.no_grad():
            if hasattr(data, "hip_features"):
                features = data.hip_features.float()
                scale = getattr(data, "hip_feature_scale", 1.0)
                if not isinstance(scale, (float, int)) or scale != 1.0:
                    features = features * scale
                emb = {
                    "node_embedding": features,
                    "batch": data.batch,
                }
            else:
                emb = model.features(data, args.backbone_precision)
                if cache_writer and model.training:
                    with phase("feature_cache_store"):
                        cache_writer.store(data, emb["node_embedding"])
        from torch.utils.checkpoint import checkpoint

        def forward(x):
            with (
                phase("hip_head"),
                torch.autocast(
                    "cuda", dtype=torch.bfloat16, enabled=args.head_precision == "bf16"
                ),
            ):
                module = model.head if model.training else head
                result = module(data, {**emb, "node_embedding": x})
                return (
                    result if hasattr(data, "hip_pair_targets") else result["hessian"]
                )

        edges = sum(n * (n - 1) for n in data.hip_sizes)
        if model.training and edges >= args.checkpoint_edges:
            return checkpoint(forward, emb["node_embedding"], use_reentrant=False)
        return forward(emb["node_embedding"])

    def save():
        if rank != 0 or args.benchmark:
            return
        current_hash = parameter_hash(model.backbone)
        assert current_hash == frozen_hash
        assert all(p.grad is None for p in model.backbone.parameters())
        checkpoint = {
            "step": step,
            "target_steps": args.steps,
            "weight_decay": args.weight_decay,
            "initial_head_sha256": initial_head_hash,
            "code_revision": args.code_revision,
            "head": head.state_dict(),
            "optimizer": optimizer.state_dict(),
            "backbone": "uma-s-1p2",
            "backbone_sha256": frozen_hash,
            "head_layers": args.layers,
            "dataset_revision": "7ce83e7425649619c9f85dad485bb2cfd7ee12bd",
            "head_precision": args.head_precision,
            "tf32": args.tf32,
            "backbone_precision": args.backbone_precision,
            "backbone_graph_version": args.backbone_graph_version,
            "fused_backbone": args.fused_backbone,
            "grouped_mole": args.grouped_mole,
            "expert_fp32_output": args.grouped_mole
            and args.backbone_precision == "fp16",
            "block_so2": args.block_so2,
            "projected_readout": args.projected_readout,
            "fused_rotations": args.fused_rotations,
            "fused_scatter": args.fused_scatter,
            "fused_wigner": args.fused_wigner,
            "fused_radial": args.fused_radial,
            "segmented_scatter": args.segmented_scatter,
            "scaled_linear": args.scaled_linear,
            "available_shards": len(inventory),
            "inventory_sha256": hashlib.sha256(
                "\n".join(inventory).encode()
            ).hexdigest(),
            "seed": args.seed,
            "world_size": world,
            "loss_name": args.loss,
            "loss": (
                "structure_mean_mse_eV2_Angstrom4"
                if args.loss == "mse"
                else "structure_mean_mae_symmetrized_eV_Angstrom2"
                if args.loss == "mae"
                else "structure_mean_mass_block_rmse_symmetrized_eV_Angstrom2_amu"
                if args.loss == "mass-block-rmse"
                else "structure_mean_block_rmse_symmetrized_eV_Angstrom2"
            ),
            "training_targets": "symmetrized"
            if args.loss != "mse"
            else "dense_original_mse",
            "global_batch_cap": args.max_batch,
            "global_atom_target": args.max_atoms,
            "sampler": "windowed_best_fit_atoms",
            "microbatch_pairs": args.microbatch_pairs,
            "validation_sample_ids_sha256": cohort["sample_ids_sha256"],
            "validation_samples": cohort["samples"],
            "validate_every": args.validate_every,
            "checkpoint_every": args.checkpoint_every,
        }
        temporary = args.output / "checkpoint.tmp"
        torch.save(checkpoint, temporary)
        temporary.replace(args.output / "latest.pt")
        print(
            json.dumps(
                {"event": "checkpoint", "step": step, "frozen_backbone_verified": True}
            ),
            flush=True,
        )

    def evaluate(*, final=False):
        from fairchem.core.frozen_hip.metrics import (
            mass_weighted_eckart_totals,
            projected_metric_record,
        )

        model.eval()
        stream = EvaluationStream(
            args.data,
            cohort,
            rank=rank,
            world=world,
            max_batch=args.max_batch // world,
            max_atoms=args.max_atoms // world,
            pair_budget=args.microbatch_pairs,
        )
        val = torch.zeros(4, device=device, dtype=torch.float64)
        projected = torch.zeros(6, device=device, dtype=torch.float64)
        seen = []
        with torch.no_grad():
            val_loader = DataLoader(
                stream,
                batch_size=None,
                num_workers=args.workers,
                pin_memory=args.pin_memory,
                **(
                    {"multiprocessing_context": "spawn", "prefetch_factor": 2}
                    if args.workers
                    else {}
                ),
            )
            for host_data in val_loader:
                seen.extend(int(sid) for sid in host_data.sid)
                data = move(host_data)
                prediction = predict(data).float().reshape(-1)
                projected += mass_weighted_eckart_totals(
                    prediction,
                    data.hessian,
                    data.pos,
                    data.atomic_numbers,
                    data.hip_sizes,
                )
                error = prediction - data.hessian
                parts = error.split([9 * n * n for n in data.hip_sizes])
                val += torch.stack(
                    (
                        torch.stack([part.square().mean() for part in parts]).sum(),
                        torch.stack([part.abs().mean() for part in parts]).sum(),
                        error.new_tensor(len(data.hip_sizes)),
                        structure_block_rmse(prediction, data.hessian, data.hip_sizes)
                        * len(data.hip_sizes),
                    )
                ).double()
        identities = [seen]
        if world > 1:
            dist.all_reduce(val)
            dist.all_reduce(projected)
            identities = [None] * world
            dist.all_gather_object(identities, seen)
        ids = [value for partition in identities for value in partition]
        fingerprint = sample_fingerprint(ids)
        if (
            len(ids) != cohort["samples"]
            or len(set(ids)) != len(ids)
            or fingerprint != cohort["sample_ids_sha256"]
        ):
            raise ValueError("Evaluated samples differ from the fixed cohort")
        projected_record = projected_metric_record(projected, cohort["samples"])
        values = val.cpu().tolist()
        if rank == 0:
            row = dict(
                event="final_evaluation" if final else "validation",
                step=step,
                loss_name=args.loss,
                mse=values[0] / values[2],
                mae=values[1] / values[2],
                block_rmse=values[3] / values[2],
                samples=int(values[2]),
                sample_ids_sha256=fingerprint,
                from_last_checkpoint=final,
            )
            row.update(projected_record)
            if final:
                row["checkpoint"] = "latest.pt"
                row["checkpoint_sha256"] = hashlib.sha256(
                    (args.output / "latest.pt").read_bytes()
                ).hexdigest()
                temporary = args.output / "final_evaluation.tmp"
                temporary.write_text(json.dumps(row, indent=2) + "\n")
                temporary.replace(args.output / "final_evaluation.json")
            print(json.dumps(row), flush=True)
            with (args.output / "evaluation.jsonl").open("a") as f:
                f.write(json.dumps(row) + "\n")
        model.train()

    if rank == 0:
        public_config = {k: v for k, v in vars(args).items() if not isinstance(v, Path)}
        public_config.update(
            backbone="uma-s-1p2",
            backbone_sha256=frozen_hash,
            initial_head_sha256=initial_head_hash,
            head_parameters=sum(p.numel() for p in head.parameters()),
            world_size=world,
            validation="fixed_parent_holdout_cohort_v1",
            validation_samples=cohort["samples"],
            validation_sample_ids_sha256=cohort["sample_ids_sha256"],
            available_shards_at_start=len(inventory),
        )
        (args.output / "config.json").write_text(
            json.dumps(public_config, indent=2) + "\n"
        )
        print(json.dumps({"event": "training_start", **public_config}), flush=True)
    try:
        while not args.eval_only and step < args.steps:
            if time.monotonic() - started >= args.duration:
                stopping = True
            if (
                rank == 0
                and step % args.log_every == 0
                and args.stop_file
                and args.stop_file.exists()
            ):
                stopping = True
            if world > 1:
                stop = torch.tensor(int(stopping), device=device)
                dist.all_reduce(stop, op=dist.ReduceOp.MAX)
                should_stop = bool(stop.item())
            else:
                should_stop = stopping
            if should_stop:
                break
            with phase("data_load"):
                host_batches = next(train_iter)
            local_count = sum(len(data.natoms) for data in host_batches)
            count = local_count
            if world > 1:
                count = torch.tensor(local_count, device=device, dtype=torch.float32)
                dist.all_reduce(count)
            optimizer.zero_grad(set_to_none=True)
            for micro_index, host_data in enumerate(host_batches):
                with phase("device_transfer"):
                    data = host_data if args.device_prefetch else move(host_data)
                b = len(data.hip_sizes)
                # Only the last microbatch synchronizes gradients. Every molecule
                # keeps its weight in the same global optimizer batch.
                context = (
                    model.head.no_sync()
                    if world > 1 and micro_index + 1 < len(host_batches)
                    else nullcontext()
                )
                with context:
                    prediction = predict(data)
                    with phase("hessian_loss"):
                        if hasattr(data, "hip_pair_targets"):
                            loss_function = {
                                "mse": block_mse,
                                "mae": block_mae,
                                "block-rmse": block_rmse,
                                "mass-block-rmse": block_rmse,
                            }[args.loss]
                            loss = loss_function(
                                prediction["edge_blocks"],
                                prediction["node_blocks"],
                                data,
                                **(
                                    {"mass_weighted": True}
                                    if args.loss == "mass-block-rmse"
                                    else {}
                                ),
                            )
                        else:
                            loss_function = {
                                "mse": structure_mse,
                                "mae": structure_mae,
                                "block-rmse": structure_block_rmse,
                                "mass-block-rmse": structure_block_rmse,
                            }[args.loss]
                            loss = loss_function(
                                prediction,
                                data.hessian,
                                data.hip_sizes,
                                **(
                                    {"numbers": data.atomic_numbers}
                                    if args.loss == "mass-block-rmse"
                                    else {}
                                ),
                            )
                    with phase("head_backward"):
                        (loss * (b * world / count)).backward()
                with phase("batch_metrics"):
                    metrics[0].add_(loss.detach(), alpha=b)
                    host_metrics[0] += b
                    host_metrics[1] += sum(data.hip_sizes)
                    host_metrics[2] += b if hasattr(data, "hip_features") else 0
                    host_metrics[3] += 1
                host_batches[micro_index] = None
                del prediction, loss, data, host_data
            with phase("grad_clip"):
                norm = torch.nn.utils.clip_grad_norm_(
                    head.parameters(),
                    max_norm=10.0,
                    error_if_nonfinite=not args.async_grad_check,
                )
                if args.async_grad_check:
                    torch._assert_async(torch.isfinite(norm), "Nonfinite HIP gradient")
            warmup = min(1.0, (step + 1) / 500)
            progress = max(0.0, (step - 500) / max(1, args.steps - 500))
            lr = args.lr * warmup * (0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2)
            for group in optimizer.param_groups:
                group["lr"] = lr
            with phase("optimizer_step"):
                optimizer.step()
            if profiler is not None:
                profiler.step()
            step += 1
            window_updates += 1
            total_samples += local_count
            if step % args.log_every == 0:
                torch.cuda.synchronize()
                metrics[1:].copy_(torch.tensor(host_metrics, dtype=torch.float64))
                if world > 1:
                    dist.all_reduce(metrics)
                elapsed = time.monotonic() - window
                values = metrics.cpu().tolist()
                if rank == 0:
                    row = {
                        "event": "train",
                        "wall_time": time.time(),
                        "step_seconds": elapsed / window_updates,
                        "window_updates": window_updates,
                        "mean_microbatches_per_gpu_step": values[4]
                        / (window_updates * world),
                        "mean_global_batch": values[1] / window_updates,
                        "mean_global_atoms": values[2] / window_updates,
                        "step": step,
                        "loss_name": args.loss,
                        args.loss.replace("-", "_"): values[0] / values[1],
                        "samples_s": values[1] / elapsed,
                        "atoms_s": values[2] / elapsed,
                        "lr": lr,
                        "peak_GB": torch.cuda.max_memory_allocated() / 1e9,
                        "peak_reserved_GB": torch.cuda.max_memory_reserved() / 1e9,
                        "cached_feature_fraction": values[3] / values[1],
                    }
                    print(json.dumps(row), flush=True)
                    with (args.output / "metrics.jsonl").open("a") as f:
                        f.write(json.dumps(row) + "\n")
                metrics.zero_()
                host_metrics = [0, 0, 0, 0]
                window_updates = 0
                window = time.monotonic()
            if step % args.checkpoint_every == 0:
                save()
                if world > 1:
                    dist.barrier()
            if (
                not args.benchmark
                and step % args.validate_every == 0
                and step < args.steps
            ):
                evaluate()
                window = time.monotonic()
            if args.refresh_data_every and step % args.refresh_data_every == 0:
                updated = synchronized_inventory()
                if updated != inventory:
                    if args.workers:
                        train_iter._shutdown_workers()
                    inventory = updated
                    train_loader = loader()
                    train_iter = device_iterator(iter(train_loader))
                    if rank == 0:
                        print(
                            json.dumps(
                                {
                                    "event": "data_inventory_refresh",
                                    "step": step,
                                    "verified_shards": len(inventory),
                                }
                            ),
                            flush=True,
                        )
    finally:
        if profiler is not None:
            profiler.stop()
        # Stop the infinite loader before interpreter and CUDA teardown. Held-out
        # evaluation is bounded and uses the main process rather than abandoned
        # multiprocessing prefetch queues.
        if not args.benchmark and not args.eval_only:
            save()
        if (args.workers or args.device_prefetch) and train_iter is not None:
            try:
                train_iter._shutdown_workers()
            except RuntimeError as error:
                # Workers can abort while discarding queued tensor transfers.
                # Only tolerate that failure after intentional loader shutdown.
                # Exceptions raised during training still propagate normally.
                if "DataLoader worker" not in str(error):
                    raise
                print(
                    json.dumps(
                        {"event": "loader_shutdown_warning", "error": str(error)}
                    ),
                    flush=True,
                )
        if cache_writer is not None:
            cache_writer.close()
    # Allocation timeouts only checkpoint. Final evaluation happens at the
    # training target, or through an explicit evaluation-only invocation.
    if not args.benchmark and (step >= args.steps or args.eval_only):
        if world > 1:
            dist.barrier()
        state = load_last_head(head, args.output / "latest.pt", frozen_hash, device)
        step = state["step"]
        torch.backends.cuda.matmul.allow_tf32 = state.get("tf32", False)
        configure_head_precision(head, state.get("head_precision", "fp32"))
        args.head_precision = state.get("head_precision", "fp32")
        args.backbone_precision = state.get("backbone_precision", "fp32")
        args.backbone_graph_version = state.get("backbone_graph_version", 2)
        model.backbone.radius_pbc_version = args.backbone_graph_version
        enable_fused_rotations(
            model.backbone, enabled=state.get("fused_backbone", False)
        )
        if state.get("grouped_mole", False) or args.grouped_mole:
            from fairchem.core.frozen_hip.grouped_mole import enable_grouped_mole

            enable_grouped_mole(
                model.backbone, enabled=state.get("grouped_mole", False)
            )
        enable_fused_rotations(head, enabled=state.get("fused_rotations", False))
        from fairchem.core.frozen_hip.so2 import enable_block_so2

        enable_block_so2(head, enabled=state.get("block_so2", False))
        from fairchem.core.frozen_hip.readout import enable_projected_readout

        enable_projected_readout(head, enabled=state.get("projected_readout", False))
        from fairchem.core.frozen_hip.linear import enable_scaled_linear
        from fairchem.core.frozen_hip.rotations import enable_fused_scatter

        enable_scaled_linear(head, enabled=state.get("scaled_linear", False))
        if state.get("fused_scatter", False):
            enable_fused_scatter(head, segmented=state.get("segmented_scatter", False))
        from fairchem.core.frozen_hip.wigner import enable_fused_wigner

        enable_fused_wigner(head, enabled=state.get("fused_wigner", False))
        from fairchem.core.frozen_hip.radial import enable_fused_radial

        enable_fused_radial(head, enabled=state.get("fused_radial", False))
        evaluate(final=True)
    if rank == 0:
        assert parameter_hash(model.backbone) == frozen_hash
        summary = {
            "step": step,
            "samples_rank0": total_samples,
            "seconds": time.monotonic() - started,
            "frozen_backbone_verified": True,
            "world_size": world,
        }
        (args.output / "session.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps({"event": "session_complete", **summary}), flush=True)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
