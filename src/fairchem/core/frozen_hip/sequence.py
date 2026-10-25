"""Immutable data contracts and restartable, worker-independent batch plans."""

from __future__ import annotations

import hashlib
import json
import random
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from fairchem.core.frozen_hip.data import (
    BalancedBatchSampler,
    microbatch_indices,
    validation_mask,
)
from fairchem.core.frozen_hip.packed import (
    open_packed,
    pack_cached_window,
    pack_packed_window,
)

VERSION = 1


def digest_json(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def source_identity(values, header):
    """Bind order to scientific metadata rather than a machine-specific path."""
    keys = (
        "arrays",
        "fingerprint",
        "dataset_revision",
        "backbone_sha256",
        "backbone_precision",
        "backbone_graph_version",
        "fused_backbone",
        "tf32",
        "grouped_mole",
        "expert_fp32_output",
        "training_split",
    )
    digest = hashlib.sha256(
        json.dumps({k: header.get(k) for k in keys}, sort_keys=True).encode()
    )
    for key in ("omol_index", "natoms", "charge", "spin"):
        digest.update(np.asarray(values[key], dtype="<i8").tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class BatchRequest:
    sources: tuple
    identities: tuple
    references: tuple
    sizes: tuple
    before: tuple
    after: tuple
    update_index: int


class PackedSequence(Sampler):
    """Plan in the main process; workers only materialize explicit requests.

    The canonical sample order uses one fixed 64/2560 balanced packing policy.
    Logical batch limits regroup that order without changing sample order.
    Physical microbatch limits, workers, prefetch, and model RNG do not affect it.
    """

    def __init__(self, stream, *, seed=7, cursor=(0, 0, 0)):
        self.stream, self.seed = stream, int(seed)
        self.start_update = 0
        self.cursor = tuple(int(x) for x in cursor)
        if len(self.cursor) != 3 or min(self.cursor) < 0:
            raise ValueError("Invalid training data cursor")
        stream.markers = sorted(stream.markers)
        if len(set(stream.markers)) != len(stream.markers):
            raise ValueError("Training inventory contains duplicate sources")
        self.metadata = []
        for marker in stream.markers:
            path = Path(marker)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("Training inventory must use relative paths")
            values, header = open_packed(stream.root / marker)
            ids = np.asarray(values["omol_index"], dtype=np.int64).copy()
            sizes = np.asarray(values["natoms"], dtype=np.int64).copy()
            rows = (
                np.arange(len(ids))
                if header.get("training_split") == "release-v2"
                else np.flatnonzero(~validation_mask(ids))
            )
            self.metadata.append(
                dict(
                    marker=marker,
                    identity=source_identity(values, header),
                    ids=ids,
                    sizes=sizes,
                    rows=rows,
                )
            )
        if len(self.metadata) < stream.world:
            raise ValueError("Too few training sources for the rank count")
        eligible_ids = np.concatenate([m["ids"][m["rows"]] for m in self.metadata])
        ordered_ids = np.sort(eligible_ids)
        if len(ordered_ids) == 0 or np.any(ordered_ids[1:] == ordered_ids[:-1]):
            raise ValueError("Training sample membership is empty or duplicated")
        self.samples = len(ordered_ids)
        membership = hashlib.sha256(ordered_ids.astype("<i8").tobytes()).hexdigest()
        sources = [
            {"path": m["marker"], "sha256": m["identity"]} for m in self.metadata
        ]
        self.contract = dict(
            version=VERSION,
            update_rng="absolute-update-v1",
            data_seed=self.seed,
            training_samples=self.samples,
            training_sample_ids_sha256=membership,
            sources=sources,
            world_size=stream.world,
            window_files=stream.window_files,
            canonical_batch_cap=64,
            canonical_atom_target=2560,
        )
        self.order_sha256 = digest_json(self.contract)
        self.batching = dict(
            max_atoms=stream.max_atoms,
            max_batch=stream.max_batch,
            logical_pair_budget=stream.logical_pair_budget,
        )
        self.plan_sha256 = digest_json(
            dict(order_sha256=self.order_sha256, **self.batching)
        )

    def state(self, cursor, updates, prefix):
        return dict(
            version=VERSION,
            order_sha256=self.order_sha256,
            plan_sha256=self.plan_sha256,
            cursor=list(cursor),
            rank=self.stream.rank,
            updates=updates,
            prefix_sha256=prefix,
        )

    def validate_state(self, state):
        if (
            state.get("version") != VERSION
            or state.get("plan_sha256") != self.plan_sha256
            or state.get("rank") != self.stream.rank
        ):
            raise ValueError(
                "Training data, seed, rank topology, or logical batch policy changed"
            )
        self.cursor = tuple(state["cursor"])
        self.start_update = int(state["updates"])
        if len(self.cursor) != 3 or min(self.cursor) < 0:
            raise ValueError("Invalid saved training data cursor")

    def sources(self, epoch):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, epoch, 0]))
        return rng.permutation(len(self.metadata)).tolist()[
            self.stream.rank :: self.stream.world
        ]

    def plan_window(self, epoch, window, order):
        sources = tuple(
            order[
                window * self.stream.window_files : (window + 1)
                * self.stream.window_files
            ]
        )
        rng = np.random.default_rng(
            np.random.SeedSequence([self.seed, epoch, window, 1])
        )
        refs, sizes = [], []
        for source, index in enumerate(sources):
            m = self.metadata[index]
            rows = m["rows"].copy()
            rng.shuffle(rows)
            refs.extend((source, int(r)) for r in rows)
            sizes.extend(int(m["sizes"][r]) for r in rows)
        canonical = list(
            BalancedBatchSampler(
                sizes, max_atoms=2560, batch_size=64, pair_budget=1000000000
            )
        )
        rng.shuffle(canonical)
        ordered = [i for batch in canonical for i in batch]
        batches, current, atoms, pairs = [], [], 0, 0
        for i in ordered:
            n = sizes[i]
            if current and (
                len(current) >= self.stream.max_batch
                or atoms + n > self.stream.max_atoms
                or pairs + n * n > self.stream.logical_pair_budget
            ):
                batches.append(current)
                current, atoms, pairs = [], 0, 0
            current.append(i)
            atoms += n
            pairs += n * n
        if current:
            batches.append(current)
        return sources, refs, sizes, batches

    def seek_updates(self, updates):
        """Explicit one-time adoption of a checkpoint without an old data cursor."""
        self.start_update = int(updates)
        remaining, epoch = int(updates), 0
        while True:
            order = self.sources(epoch)
            windows = (
                len(order) + self.stream.window_files - 1
            ) // self.stream.window_files
            epoch_batches = 0
            for window in range(windows):
                *_, batches = self.plan_window(epoch, window, order)
                epoch_batches += len(batches)
                if remaining < len(batches):
                    self.cursor = (epoch, window, remaining)
                    return
                remaining -= len(batches)
            if not epoch_batches:
                raise ValueError("Training inventory contains no eligible samples")
            epoch += 1

    def __iter__(self):
        epoch, start_window, start_batch = self.cursor
        update_index = self.start_update
        while True:
            order = self.sources(epoch)
            windows = (
                len(order) + self.stream.window_files - 1
            ) // self.stream.window_files
            if start_window >= windows:
                raise ValueError("Saved training window exceeds the frozen inventory")
            epoch_batches = 0
            for window in range(start_window, windows):
                sources, refs, sizes, batches = self.plan_window(epoch, window, order)
                first = start_batch if window == start_window else 0
                if first > len(batches):
                    raise ValueError("Saved batch cursor exceeds its training window")
                epoch_batches += len(batches)
                for b in range(first, len(batches)):
                    after = (epoch, window, b + 1)
                    if b + 1 == len(batches):
                        after = (
                            (epoch, window + 1, 0)
                            if window + 1 < windows
                            else (epoch + 1, 0, 0)
                        )
                    indices = batches[b]
                    yield BatchRequest(
                        sources,
                        tuple(self.metadata[i]["identity"] for i in sources),
                        tuple(refs[i] for i in indices),
                        tuple(sizes[i] for i in indices),
                        (epoch, window, b),
                        after,
                        update_index,
                    )
                    update_index += 1
            if not epoch_batches and start_window == 0 and start_batch == 0:
                raise ValueError("Training inventory contains no eligible samples")
            epoch += 1
            start_window, start_batch = 0, 0


class PlannedPackedDataset(Dataset):
    """Cache one local window while honoring the main process's exact order."""

    def __init__(self, stream):
        self.stream = stream
        self.key, self.window = None, None

    def __getitem__(self, request):
        if request.sources != self.key:
            self.window = self.stream._load_window(request.sources)
            self.key = request.sources
            self.identities = tuple(source_identity(v, h) for v, h in self.window)
        if self.identities != request.identities:
            raise ValueError("Training source changed after sequence planning")
        operation = (
            pack_cached_window if self.stream.fast_packing else pack_packed_window
        )
        kw = dict(include_targets=self.stream.include_targets, preserve_order=True)
        if self.stream.fast_packing:
            kw["feature_dtype"] = np.float16 if self.stream.pack_features_fp16 else None
        result = []
        for indices in microbatch_indices(
            list(range(len(request.references))), request.sizes, self.stream.pair_budget
        ):
            data = operation(
                self.window, [request.references[i] for i in indices], **kw
            )
            data.hip_sequence_update = request.update_index
            data.hip_sequence_before = list(request.before)
            data.hip_sequence_after = list(request.after)
            result.append(data)
        return result


def capture_rng(device=None):
    state = dict(
        python=random.getstate(),
        numpy=np.random.get_state(),
        torch=torch.get_rng_state(),
    )
    if device is not None and torch.device(device).type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state(device)
    return state


def restore_rng(state, device=None):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state:
        torch.cuda.set_rng_state(state["cuda"].cpu(), device)


def seed_rng(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)


def seed_update(data_seed, update):
    """Select an update's random view independently of prior work or restarts."""
    value = int(digest_json(["hip-update-v1", int(data_seed), int(update)])[:16], 16)
    seed_rng(value % (2**63))


@contextmanager
def fixed_rng(seed, device=None):
    """Give evaluation its own repeatable random stream, including on errors."""
    state = capture_rng(device)
    try:
        seed_rng(seed)
        yield
    finally:
        restore_rng(state, device)


def pin_manifest(output, role, manifest):
    """Freeze membership immediately and freeze locations once labels complete."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{role}_data_contract.json"
    keys = (
        "selection",
        "dataset_revision",
        "samples",
        "sample_ids_sha256",
        "backbone_sha256",
    )
    identity = {k: manifest.get(k) for k in keys}
    content = digest_json(manifest) if manifest.get("complete", True) else None
    record = dict(identity=identity, manifest_sha256=content)
    if path.exists():
        previous = json.loads(path.read_text())
        if previous["identity"] != identity or (
            previous["manifest_sha256"] is not None
            and previous["manifest_sha256"] != content
        ):
            raise ValueError(f"The fixed {role} validation manifest changed")
        if previous["manifest_sha256"] is not None or content is None:
            return previous
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, sort_keys=True, indent=2) + "\n")
    temporary.replace(path)
    return record


def pin_validation_order(output, role, ids):
    """Reject a different validation batch order even with the same membership."""
    path = Path(output) / f"{role}_order.json"
    record = dict(
        samples=len(ids),
        sha256=hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest(),
    )
    if path.exists() and json.loads(path.read_text()) != record:
        raise ValueError(f"The fixed {role} validation sequence changed")
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, sort_keys=True) + "\n")
    temporary.replace(path)
    return record["sha256"]
