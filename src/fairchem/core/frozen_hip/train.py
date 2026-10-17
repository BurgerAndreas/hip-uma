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

from fairchem.core.frozen_hip.cache import AsyncFeatureCacheWriter
from fairchem.core.frozen_hip.data import HessianStream
from fairchem.core.frozen_hip.model import FrozenHIP, parameter_hash, structure_mse
from fairchem.core.frozen_hip.packed import PackedHessianStream, block_mse
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
    p.add_argument("--workers", type=int, default=2)
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
    p.add_argument("--validate-every", type=int, default=100000)
    p.add_argument("--validation-manifest", type=Path)
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--seed", type=int, default=7)
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
    p.add_argument("--cache-memory-gb", type=float, default=16.0)
    p.add_argument("--packed-data", type=Path)
    p.add_argument("--fused-rotations", action="store_true")
    p.add_argument("--grouped-mole", action="store_true")
    p.add_argument("--block-so2", action="store_true")
    p.add_argument("--projected-readout", action="store_true")
    p.add_argument("--backbone-precision", choices=("fp32", "fp16"), default="fp32")
    p.add_argument("--tf32", action="store_true")
    p.add_argument("--compile-head", action="store_true")
    p.add_argument("--stop-file", type=Path)
    p.add_argument("--refresh-data-every", type=int, default=2000)
    args = p.parse_args()
    if args.checkpoint_edges is None:
        args.checkpoint_edges = args.microbatch_pairs + 1
    return args


def main():
    args = parse_args()
    cohort_path = args.validation_manifest or args.data / "validation_10000.json"
    cohort = load_manifest(cohort_path)
    if args.eval_only and not (args.output / "latest.pt").exists():
        raise FileNotFoundError("Final evaluation requires the last saved checkpoint")
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
    )
    frozen_hash = parameter_hash(model.backbone)
    if args.grouped_mole:
        from fairchem.core.frozen_hip.grouped_mole import enable_grouped_mole

        enable_grouped_mole(model.backbone)
    if args.block_so2:
        from fairchem.core.frozen_hip.so2 import enable_block_so2

        enable_block_so2(model.head)
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    cache_root = (
        args.feature_cache
        / (
            frozen_hash
            + ("-tf32" if args.tf32 else "-fp32")
            + "-geometry1"
            + ("-backbonefp16" if args.backbone_precision == "fp16" else "")
            + ("-grouped" if args.grouped_mole else "")
        )
        if args.feature_cache
        else None
    )
    cache_writer = (
        AsyncFeatureCacheWriter(
            cache_root, max_bytes=int(args.cache_memory_gb * (1 << 30))
        )
        if cache_root
        else None
    )
    if args.fused_rotations:
        enable_fused_rotations(model.head)
    if args.projected_readout:
        from fairchem.core.frozen_hip.readout import enable_projected_readout

        enable_projected_readout(model.head)
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
        head.parameters(), lr=args.lr, weight_decay=1e-4, fused=True
    )
    step = 0
    resume = (
        args.output / "latest.pt"
        if args.eval_only
        else args.resume or args.output / "latest.pt"
    )
    if resume.exists():
        state = torch.load(resume, map_location=device, weights_only=False)
        if state["backbone_sha256"] != frozen_hash:
            raise ValueError("Checkpoint backbone differs from the frozen UMA model")
        if state.get("validation_sample_ids_sha256") not in (
            None,
            cohort["sample_ids_sha256"],
        ):
            raise ValueError("Checkpoint evaluation cohort differs from this run")
        head.load_state_dict(state["head"])
        optimizer.load_state_dict(state["optimizer"])
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
                tf32=args.tf32,
                grouped_mole=args.grouped_mole,
                window_files=args.window_shards,
                feature_cache=cache_root,
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
                "prefetch_factor": 2,
                "persistent_workers": True,
            }
            if args.workers
            else {}
        )
        return DataLoader(stream, batch_size=None, num_workers=args.workers, **kw)

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
    metrics = torch.zeros(5, device=device, dtype=torch.float64)
    total_samples = 0
    model.train()

    def move(data):
        sizes = data.natoms.tolist()
        data.to(device, non_blocking=True)
        data.hip_sizes = sizes
        data.hip_fused_loss = args.fused_rotations
        return data

    def predict(data):
        # Frozen inference can use FP16; the head receives FP32 features.
        with torch.no_grad():
            if hasattr(data, "hip_features"):
                emb = {
                    "node_embedding": data.hip_features.float()
                    * getattr(data, "hip_feature_scale", 1.0),
                    "batch": data.batch,
                }
            else:
                emb = model.features(data, args.backbone_precision)
                if cache_writer and model.training:
                    cache_writer.store(data, emb["node_embedding"])
        from torch.utils.checkpoint import checkpoint

        def forward(x):
            with torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=args.head_precision == "bf16"
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
            "head": head.state_dict(),
            "optimizer": optimizer.state_dict(),
            "backbone": "uma-s-1p2",
            "backbone_sha256": frozen_hash,
            "head_layers": args.layers,
            "dataset_revision": "7ce83e7425649619c9f85dad485bb2cfd7ee12bd",
            "head_precision": args.head_precision,
            "tf32": args.tf32,
            "backbone_precision": args.backbone_precision,
            "grouped_mole": args.grouped_mole,
            "block_so2": args.block_so2,
            "projected_readout": args.projected_readout,
            "fused_rotations": args.fused_rotations,
            "available_shards": len(inventory),
            "inventory_sha256": hashlib.sha256(
                "\n".join(inventory).encode()
            ).hexdigest(),
            "seed": args.seed,
            "world_size": world,
            "loss": "structure_mean_mse_eV2_Angstrom4",
            "global_batch_cap": args.max_batch,
            "global_atom_target": args.max_atoms,
            "sampler": "windowed_best_fit_atoms",
            "microbatch_pairs": args.microbatch_pairs,
            "validation_sample_ids_sha256": cohort["sample_ids_sha256"],
            "validation_samples": cohort["samples"],
            "validate_every": args.validate_every,
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
        val = torch.zeros(3, device=device, dtype=torch.float64)
        seen = []
        with torch.no_grad():
            val_loader = DataLoader(
                stream,
                batch_size=None,
                num_workers=args.workers,
                **(
                    {"multiprocessing_context": "spawn", "prefetch_factor": 2}
                    if args.workers
                    else {}
                ),
            )
            for host_data in val_loader:
                seen.extend(int(sid) for sid in host_data.sid)
                data = move(host_data)
                error = predict(data).float().reshape(-1) - data.hessian
                parts = error.split([9 * n * n for n in data.hip_sizes])
                val += torch.stack(
                    (
                        torch.stack([part.square().mean() for part in parts]).sum(),
                        torch.stack([part.abs().mean() for part in parts]).sum(),
                        error.new_tensor(len(data.hip_sizes)),
                    )
                ).double()
        identities = [seen]
        if world > 1:
            dist.all_reduce(val)
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
        values = val.cpu().tolist()
        if rank == 0:
            row = dict(
                event="final_evaluation" if final else "validation",
                step=step,
                mse=values[0] / values[2],
                mae=values[1] / values[2],
                samples=int(values[2]),
                sample_ids_sha256=fingerprint,
                from_last_checkpoint=final,
            )
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
            stop = torch.tensor(int(stopping), device=device)
            if world > 1:
                dist.all_reduce(stop, op=dist.ReduceOp.MAX)
            if stop.item():
                break
            host_batches = next(train_iter)
            local_count = sum(len(data.natoms) for data in host_batches)
            count = torch.tensor(local_count, device=device, dtype=torch.float32)
            if world > 1:
                dist.all_reduce(count)
            optimizer.zero_grad(set_to_none=True)
            batch_metrics = torch.zeros_like(metrics)
            for micro_index, host_data in enumerate(host_batches):
                data = move(host_data)
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
                    loss = (
                        block_mse(
                            prediction["edge_blocks"], prediction["node_blocks"], data
                        )
                        if hasattr(data, "hip_pair_targets")
                        else structure_mse(prediction, data.hessian, data.hip_sizes)
                    )
                    (loss * (b * world / count)).backward()
                batch_metrics += torch.stack(
                    (
                        loss.detach().double() * b,
                        count.new_tensor(b).double(),
                        count.new_tensor(sum(data.hip_sizes)).double(),
                        count.new_tensor(
                            b if hasattr(data, "hip_features") else 0
                        ).double(),
                        count.new_tensor(1).double(),
                    )
                )
                host_batches[micro_index] = None
                del prediction, loss, data, host_data
            torch.nn.utils.clip_grad_norm_(
                head.parameters(), max_norm=10.0, error_if_nonfinite=True
            )
            warmup = min(1.0, (step + 1) / 500)
            progress = max(0.0, (step - 500) / max(1, args.steps - 500))
            lr = args.lr * warmup * (0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.step()
            step += 1
            total_samples += local_count
            metrics += batch_metrics
            if step % args.log_every == 0:
                torch.cuda.synchronize()
                if world > 1:
                    dist.all_reduce(metrics)
                elapsed = time.monotonic() - window
                values = metrics.cpu().tolist()
                if rank == 0:
                    row = {
                        "event": "train",
                        "wall_time": time.time(),
                        "step_seconds": elapsed / args.log_every,
                        "mean_microbatches_per_gpu_step": values[4]
                        / (args.log_every * world),
                        "mean_global_batch": values[1] / args.log_every,
                        "mean_global_atoms": values[2] / args.log_every,
                        "step": step,
                        "mse": values[0] / values[1],
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
                    train_iter = iter(train_loader)
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
        # Stop the infinite loader before interpreter and CUDA teardown. Held-out
        # evaluation is bounded and uses the main process rather than abandoned
        # multiprocessing prefetch queues.
        if args.workers and train_iter is not None:
            train_iter._shutdown_workers()
        if not args.benchmark and not args.eval_only:
            save()
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
