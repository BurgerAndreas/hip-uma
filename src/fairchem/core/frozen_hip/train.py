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
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from fairchem.core.frozen_hip.cache import FeatureCacheWriter
from fairchem.core.frozen_hip.data import HessianStream
from fairchem.core.frozen_hip.model import FrozenHIP, parameter_hash, structure_mse


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=200000)
    p.add_argument("--duration", type=float, default=13200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--pair-budget", type=int, default=8192)
    p.add_argument("--max-batch", type=int, default=32)
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--checkpoint-every", type=int, default=500)
    p.add_argument("--validate-every", type=int, default=1000)
    p.add_argument("--validation-batches", type=int, default=16)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--execution-mode", choices=("general",), default="general")
    p.add_argument("--head-precision", choices=("fp32", "bf16"), default="fp32")
    p.add_argument("--checkpoint-edges", type=int, default=20000)
    p.add_argument("--benchmark", action="store_true")
    p.add_argument("--resume", type=Path)
    p.add_argument("--feature-cache", type=Path)
    p.add_argument("--tf32", action="store_true")
    p.add_argument("--compile-head", action="store_true")
    p.add_argument("--stop-file", type=Path)
    p.add_argument("--refresh-data-every", type=int, default=2000)
    return p.parse_args()


def main():
    args = parse_args()
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
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
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    cache_root = (
        args.feature_cache
        / (frozen_hash + ("-tf32" if args.tf32 else "-fp32") + "-geometry1")
        if args.feature_cache
        else None
    )
    cache_writer = FeatureCacheWriter(cache_root) if cache_root else None
    head = model.head
    if args.compile_head:
        torch._dynamo.config.recompile_limit = 16
        for block in head.blocks:
            block.compile(dynamic=True)
        head.edge_readout.forward_messages = torch.compile(
            head.edge_readout.forward_messages, dynamic=True
        )
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=args.lr, weight_decay=1e-4, fused=True
    )
    step = 0
    resume = args.resume or args.output / "latest.pt"
    if resume.exists():
        state = torch.load(resume, map_location=device, weights_only=False)
        if state["backbone_sha256"] != frozen_hash:
            raise ValueError("Checkpoint backbone differs from the frozen UMA model")
        head.load_state_dict(state["head"])
        optimizer.load_state_dict(state["optimizer"])
        step = state["step"]
    if world > 1:
        # Synchronize the small trainable head, never the frozen backbone.
        model.head = DistributedDataParallel(
            head, device_ids=[local], broadcast_buffers=False
        )

    def synchronized_inventory():
        # Partition one immutable list on every rank while downloads add files.
        entries = (
            [
                str(p.relative_to(args.data))
                for p in sorted(args.data.glob("shards/*/*.verified.json"))
            ]
            if rank == 0
            else None
        )
        message = [entries]
        if world > 1:
            dist.broadcast_object_list(message, src=0, device=device)
        return message[0]

    inventory = synchronized_inventory()

    def loader(split):
        worker_count = args.workers if split == "train" else 0
        stream = HessianStream(
            args.data,
            rank=rank,
            world=world,
            seed=args.seed if split == "val" else args.seed + step,
            pair_budget=args.pair_budget,
            max_batch=args.max_batch,
            split=split,
            max_passes=1 if split == "val" else None,
            feature_cache=cache_root,
            markers=inventory,
        )
        kw = (
            {
                "multiprocessing_context": "spawn",
                "prefetch_factor": 2,
                "persistent_workers": True,
            }
            if worker_count
            else {}
        )
        return DataLoader(stream, batch_size=None, num_workers=worker_count, **kw)

    train_loader = loader("train")
    train_iter = iter(train_loader)
    stopping = False

    def stop_requested(*_):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop_requested)
    signal.signal(signal.SIGUSR1, stop_requested)
    started = time.monotonic()
    window = started
    metrics = torch.zeros(4, device=device, dtype=torch.float64)
    total_samples = 0
    model.train()

    def move(data):
        sizes = data.natoms.tolist()
        data.to(device, non_blocking=True)
        data.hip_sizes = sizes
        return data

    def predict(data):
        # UMA features stay in FP32; autocast applies only to the trainable head.
        with torch.no_grad():
            if hasattr(data, "hip_features"):
                emb = {"node_embedding": data.hip_features, "batch": data.batch}
            else:
                emb = model.backbone(data)
                if cache_writer and model.training:
                    cache_writer.store(data, emb["node_embedding"])
        from torch.utils.checkpoint import checkpoint

        def forward(x):
            with torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=args.head_precision == "bf16"
            ):
                return model.head(data, {**emb, "node_embedding": x})["hessian"]

        edges = sum(n * (n - 1) for n in data.hip_sizes)
        if model.training and edges >= args.checkpoint_edges:
            return checkpoint(forward, emb["node_embedding"], use_reentrant=False)
        return forward(emb["node_embedding"])

    def save():
        if rank != 0:
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
            "available_shards": len(inventory),
            "inventory_sha256": hashlib.sha256(
                "\n".join(inventory).encode()
            ).hexdigest(),
            "seed": args.seed,
            "world_size": world,
            "loss": "structure_mean_mse_eV2_Angstrom4",
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

    if rank == 0:
        public_config = {k: v for k, v in vars(args).items() if not isinstance(v, Path)}
        public_config.update(
            backbone="uma-s-1p2",
            backbone_sha256=frozen_hash,
            head_parameters=sum(p.numel() for p in head.parameters()),
            world_size=world,
            validation="deterministic_parent_index_1pct",
            available_shards_at_start=len(inventory),
        )
        (args.output / "config.json").write_text(
            json.dumps(public_config, indent=2) + "\n"
        )
        print(json.dumps({"event": "training_start", **public_config}), flush=True)
    try:
        while step < args.steps:
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
            data = move(next(train_iter))
            b = len(data.hip_sizes)
            optimizer.zero_grad(set_to_none=True)
            prediction = predict(data)
            loss = structure_mse(prediction, data.hessian, data.hip_sizes)
            # Ranks can carry different molecule counts because their atom counts
            # differ. Match the exact global per-molecule mean under DDP averaging.
            count = torch.tensor(b, device=device, dtype=torch.float32)
            if world > 1:
                dist.all_reduce(count)
            (loss * (b * world / count)).backward()
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
            total_samples += b
            metrics += torch.stack(
                (
                    loss.detach().double() * b,
                    count.new_tensor(b).double(),
                    count.new_tensor(sum(data.hip_sizes)).double(),
                    count.new_tensor(
                        b if hasattr(data, "hip_features") else 0
                    ).double(),
                )
            )
            if step % args.log_every == 0:
                torch.cuda.synchronize()
                if world > 1:
                    dist.all_reduce(metrics)
                elapsed = time.monotonic() - window
                values = metrics.cpu().tolist()
                if rank == 0:
                    row = {
                        "event": "train",
                        "step": step,
                        "mse": values[0] / values[1],
                        "samples_s": values[1] / elapsed,
                        "atoms_s": values[2] / elapsed,
                        "lr": lr,
                        "peak_GB": torch.cuda.max_memory_allocated() / 1e9,
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
            if step % args.validate_every == 0:
                model.eval()
                val = torch.zeros(3, device=device, dtype=torch.float64)
                with torch.no_grad():
                    for i, host_data in enumerate(loader("val")):
                        if i >= args.validation_batches:
                            break
                        data = move(host_data)
                        error = (predict(data).float() - data.hessian).reshape(
                            len(data.hip_sizes), -1
                        )
                        val += torch.stack(
                            (
                                error.square().mean(1).sum(),
                                error.abs().mean(1).sum(),
                                error.new_tensor(len(data.hip_sizes)),
                            )
                        ).double()
                if world > 1:
                    dist.all_reduce(val)
                values = val.cpu().tolist()
                if rank == 0 and values[2]:
                    print(
                        json.dumps(
                            {
                                "event": "validation",
                                "step": step,
                                "mse": values[0] / values[2],
                                "mae": values[1] / values[2],
                                "samples": values[2],
                            }
                        ),
                        flush=True,
                    )
                model.train()
                window = time.monotonic()
            if args.refresh_data_every and step % args.refresh_data_every == 0:
                updated = synchronized_inventory()
                if updated != inventory:
                    if args.workers:
                        train_iter._shutdown_workers()
                    inventory = updated
                    train_loader = loader("train")
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
        if args.workers:
            train_iter._shutdown_workers()
        if not args.benchmark:
            save()
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
