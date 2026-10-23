"""Stage an immutable published label snapshot and reuse existing frozen features.

Run numerical preparation in a GPU allocation. Use a pinned hub inventory.
Keep the credential in a private file outside the source checkout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import h5py
import numpy as np
import torch

from fairchem.core.frozen_hip.data import BalancedBatchSampler, make_batch
from fairchem.core.frozen_hip.grouped_mole import enable_grouped_mole
from fairchem.core.frozen_hip.model import FrozenHIP, parameter_hash
from fairchem.core.frozen_hip.packed import (
    arrays_from_batches,
    open_packed,
    write_packed,
)
from fairchem.core.frozen_hip.rotations import enable_fused_rotations
from fairchem.core.frozen_hip.validation import sample_fingerprint, write_manifest

OFFSET = 1 << 60


def membership(ids, sorted_ids):
    pos = np.searchsorted(sorted_ids, ids)
    return (pos < len(sorted_ids)) & (
        sorted_ids[np.minimum(pos, len(sorted_ids) - 1)] == ids
    )


def load_legacy_index(path, revision):
    """Load the CPU stage's verified ID index and portable bundle names."""
    metadata = json.loads(path.with_suffix(".json").read_text())
    if metadata["dataset_revision"] != revision:
        raise ValueError("Legacy index belongs to another release")
    ids = np.load(path, mmap_mode="r", allow_pickle=False)
    if ids.ndim != 1 or ids.dtype != np.int64 or len(ids) != metadata["samples"]:
        raise ValueError("Legacy index shape differs")
    if len(ids) > 1 and np.any(ids[1:] <= ids[:-1]):
        raise ValueError("Legacy index must contain sorted unique IDs")
    if sample_fingerprint(ids) != metadata["sample_ids_sha256"]:
        raise ValueError("Legacy index sample identities differ")
    if any(Path(name).name != name or not name.endswith(".hip") for name in metadata["files"]):
        raise ValueError("Legacy index has invalid bundle names")
    return ids, metadata["files"]


def download_file(task):
    row, args, inventory = task
    if args.sparse_base and row["parent_partition"] == 0:
        # Fetch only metadata and the missing label spans from immutable files.
        # The hub binds each byte range to the pinned commit and content hash.
        fs = args.hub_filesystem
        name = (
            "datasets/"
            + inventory["repository"]
            + "@"
            + inventory["revision"]
            + "/"
            + row["path"]
        )
        return fs.open(name, "rb", block_size=65536, size=row["bytes"])
    target = args.output / "downloads" / row["path"]
    target.parent.mkdir(parents=True, exist_ok=True)
    marker = target.with_suffix(".checked")
    if (
        marker.exists()
        and marker.read_text().strip() == row["sha256"]
        and target.is_file()
    ):
        return target
    url = (
        "https://huggingface.co/datasets/"
        + inventory["repository"]
        + "/resolve/"
        + inventory["revision"]
        + "/"
        + row["path"]
    )
    request = urllib.request.Request(
        url, headers={"Authorization": "Bearer " + args.token.read_text().strip()}
    )
    temporary = target.with_suffix(".download")
    digest = hashlib.sha256()
    with (
        urllib.request.urlopen(request, timeout=120) as response,
        temporary.open("wb") as output,
    ):
        while chunk := response.read(8 << 20):
            output.write(chunk)
            digest.update(chunk)
    if digest.hexdigest() != row["sha256"]:
        raise ValueError("Downloaded shard checksum differs")
    temporary.replace(target)
    marker.write_text(row["sha256"] + "\n")
    return target


def selected_arrays(path, record, legacy_ids, allowed):
    with h5py.File(path, "r") as f:
        if (
            f.attrs.get("model") != "uma-s-1p2"
            or f.attrs.get("hessian_unit") != "eV/Angstrom^2"
        ):
            raise ValueError("Teacher or Hessian units differ")
        partition = record["parent_partition"]
        if (
            int(f.attrs.get("parent_partition", -1)) != partition
            or f.attrs.get("split") != record["split"]
            or not f.attrs.get("complete")
            or f["hessian_upper"].dtype != np.float32
        ):
            raise ValueError("Published partition, split, or label format differs")
        ids = f["omol_index"][:].astype(np.int64)
        if len(ids) != record["samples"] or len(np.unique(ids)) != len(ids):
            raise ValueError("Published sample count or uniqueness differs")
        if not membership(ids, allowed).all():
            raise ValueError("Published rows leave the authoritative geometry split")
        rows = (
            np.flatnonzero(~membership(ids, legacy_ids))
            if partition == 0
            else np.arange(len(ids))
        )
        if not len(rows):
            return None
        sizes = f["natoms"][:][rows].astype(np.int64)
        ap, hp = f["atom_ptr"][:], f["hessian_ptr"][:]
        values = {
            key: np.concatenate([f[key][ap[i] : ap[i + 1]] for i in rows])
            for key in ["atomic_numbers", "coords"]
        }
        dense = []
        for i, n in zip(rows, sizes, strict=True):
            upper = f["hessian_upper"][hp[i] : hp[i + 1]]
            if not np.isfinite(upper).all():
                raise ValueError("Nonfinite teacher Hessian")
            h = np.empty((3 * n, 3 * n), np.float32)
            rr, cc = np.triu_indices(3 * n)
            h[rr, cc] = upper
            h[cc, rr] = upper
            dense.append(h.reshape(-1))
        values.update(
            natoms=sizes,
            atom_ptr=np.r_[0, np.cumsum(sizes)],
            omol_index=ids[rows] + partition * OFFSET,
            charge=f["charge"][:][rows],
            spin=f["spin"][:][rows],
            hessian_ptr=np.r_[0, np.cumsum(9 * sizes**2)],
            hessian_flat=np.concatenate(dense),
        )
        return values


def concatenate(parts):
    keys = [
        "natoms",
        "atomic_numbers",
        "coords",
        "omol_index",
        "charge",
        "spin",
        "hessian_flat",
    ]
    values = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    sizes = values["natoms"]
    values["atom_ptr"] = np.r_[0, np.cumsum(sizes)]
    values["hessian_ptr"] = np.r_[0, np.cumsum(9 * sizes**2)]
    return values


def finalize(args, inventory):
    receipts = [
        json.loads(p.read_text())
        for p in sorted((args.output / "receipts").glob("*.json"))
    ]
    if len(receipts) != args.world:
        raise ValueError("Not all preparation workers completed")
    entries = [e for r in receipts for e in r["entries"]]
    backbone_hashes = {r["backbone_sha256"] for r in receipts}
    if len(backbone_hashes) != 1:
        raise ValueError("Prepared features use different frozen backbones")
    backbone_hash = backbone_hashes.pop()
    train = []
    for p in sorted((args.output / "train").glob("*.hip")):
        a, _ = open_packed(p)
        train.append(np.asarray(a["omol_index"]))
    train_ids = np.concatenate(train)
    expected = inventory["counts"]["train"]
    if len(train_ids) != expected or len(np.unique(train_ids)) != expected:
        raise ValueError(
            f"Training coverage differs: {len(train_ids)} versus {expected}"
        )
    val_entries = [e for e in entries if e["split"] == "validation"]
    val_ids = np.concatenate([e["ids"] for e in val_entries]).astype(np.int64)
    if len(val_ids) != inventory["counts"]["validation"] or len(
        np.unique(val_ids)
    ) != len(val_ids):
        raise ValueError("Validation coverage differs from the pinned snapshot")
    if np.intersect1d(train_ids, val_ids).size:
        raise ValueError("Training and validation overlap")
    all_val = (
        np.load(args.manifests / "validation_200k.npz")["indices"].astype(np.int64)
        + OFFSET
    )
    if not membership(val_ids, np.sort(all_val)).all():
        raise ValueError("Validation leaves the held-out split")
    if len(val_ids) < 10000:
        raise ValueError("Too few labels for the training validation cohort")
    # Deterministic sample membership, independent of worker and file order.
    selected = np.random.default_rng(20261022).permutation(np.sort(val_ids))[:10000]

    def manifest(ids, name):
        chosen = {int(i) for i in ids}
        rows = []
        for entry in val_entries:
            keep = [i for i, sid in enumerate(entry["ids"]) if sid in chosen]
            if keep:
                rows.append(
                    dict(
                        path=entry["path"],
                        rows=keep,
                        omol_indices=[entry["ids"][i] for i in keep],
                    )
                )
        covered = sum(len(e["rows"]) for e in rows)
        obj = dict(
            selection="release-holdout-v2",
            backbone_sha256=backbone_hash,
            dataset_revision=inventory["revision"],
            samples=len(ids),
            sample_ids=[int(i) for i in ids],
            sample_ids_sha256=sample_fingerprint(ids),
            entries=rows,
            complete=covered == len(ids),
            available_samples=covered,
        )
        write_manifest(args.output / name, obj)
        return {
            k: obj[k]
            for k in ["samples", "available_samples", "complete", "sample_ids_sha256"]
        }

    monitor = manifest(selected, "validation_10000.json")
    final = manifest(all_val, "validation_200000.json")
    report = dict(
        dataset_revision=inventory["revision"],
        backbone_sha256=backbone_hash,
        training_samples=len(train_ids),
        training_sample_ids_sha256=sample_fingerprint(train_ids),
        monitoring_validation=monitor,
        final_validation=final,
        training_validation_overlap=0,
    )
    write_manifest(args.output / "READY.json", report)
    print(json.dumps(report), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inventory", type=Path, required=True)
    p.add_argument("--manifests", type=Path, required=True)
    p.add_argument("--legacy-packed", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--token", type=Path)
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--world", type=int, default=8)
    p.add_argument("--finalize", action="store_true")
    p.add_argument("--group-files", type=int, default=4)
    p.add_argument("--sparse-base", action="store_true")
    p.add_argument("--staged-input", type=Path)
    p.add_argument("--legacy-index", type=Path)
    p.add_argument("--group-indices", help="Finite smoke-test groups; does not write a worker receipt")
    args = p.parse_args()
    inventory = json.loads(args.inventory.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    if args.finalize:
        finalize(args, inventory)
        return
    if not 0 <= args.rank < args.world:
        p.error("Invalid worker rank")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True
    if args.sparse_base and args.staged_input is None:
        from huggingface_hub import HfFileSystem

        args.hub_filesystem = HfFileSystem(token=args.token.read_text().strip())
    print(json.dumps(dict(event="scan_legacy", worker=args.rank)), flush=True)
    legacy = []
    if args.legacy_index:
        legacy_ids, names = load_legacy_index(args.legacy_index, inventory["revision"])
        paths = [args.legacy_packed / name for name in names]
    else:
        paths = sorted(args.legacy_packed.glob("*.hip"))
    for path in paths:
        if args.legacy_index is None:
            values, _ = open_packed(path)
            legacy.append(np.asarray(values["omol_index"]).copy())
        target = args.output / "train" / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            target.symlink_to(path.resolve())
        except FileExistsError:
            pass
    print(json.dumps(dict(event="legacy_scanned", bundles=len(paths))), flush=True)
    if args.legacy_index is None:
        legacy_ids = np.sort(np.concatenate(legacy))
    del legacy
    allowed_base = np.sort(np.load(args.manifests / "sample_10m.npz")["indices"])
    allowed_add = np.sort(
        np.load(args.manifests / "training_additions_800k.npz")["indices"]
    )
    allowed_val = np.sort(np.load(args.manifests / "validation_200k.npz")["indices"])
    if not membership(legacy_ids, allowed_base).all():
        raise ValueError("Legacy data leave the new training split")
    model = FrozenHIP("cuda", backbone_graph_version=4).eval()
    del model.head
    print(json.dumps(dict(event="backbone_loaded")), flush=True)
    frozen = parameter_hash(model.backbone)
    enable_grouped_mole(model.backbone)
    enable_fused_rotations(model.backbone)
    records = sorted(
        inventory["files"], key=lambda x: (x["parent_partition"], x["split"], x["path"])
    )
    # Separate groups by split and parent; never merge geometry namespaces.
    groups = []
    for partition, split in [(0, "train"), (1, "train"), (1, "validation")]:
        subset = [
            r
            for r in records
            if r["parent_partition"] == partition and r["split"] == split
        ]
        groups.extend(
            subset[i : i + args.group_files]
            for i in range(0, len(subset), args.group_files)
        )
    indices = range(args.rank, len(groups), args.world)
    if args.group_indices:
        indices = [int(x) for x in args.group_indices.split(",")]
        if not indices or min(indices) < 0 or max(indices) >= len(groups):
            p.error("Invalid group selection")
    entries = []
    for index in indices:
        group = groups[index]
        split = group[0]["split"]
        partition = group[0]["parent_partition"]
        destination = (
            "train" if split == "train" else "validation"
        ) + f"/release_{index:06d}.hip"
        path = args.output / destination
        receipt = path.with_suffix(".json")
        if path.exists() and receipt.exists():
            entries.append(json.loads(receipt.read_text()))
            continue
        print(
            json.dumps(dict(event="read_group", group=index, files=len(group))),
            flush=True,
        )
        if args.staged_input is not None:
            marker = json.loads((args.staged_input / f"{index:06d}.json").read_text())
            if marker["dataset_revision"] != inventory["revision"]:
                raise ValueError("Staged dataset revision differs")
            if marker["group"] != index:
                raise ValueError("Staged group identity differs")
            if marker.get("already_packed"):
                raise ValueError("Packed output disappeared after CPU staging")
            if marker["samples"] == 0:
                continue
            with np.load(args.staged_input / f"{index:06d}.npz", allow_pickle=False) as f:
                values = {key: f[key] for key in f.files}
            if len(values["natoms"]) != marker["samples"]:
                raise ValueError("Staged sample count differs")
            if sample_fingerprint(values["omol_index"]) != marker["sample_ids_sha256"]:
                raise ValueError("Staged sample identities differ")
            parts = []
        else:
            parts = []
            with ThreadPoolExecutor(max_workers=4) as pool:
                for record, source in zip(
                    group,
                    pool.map(download_file, [(r, args, inventory) for r in group]),
                    strict=True,
                ):
                    allowed = (
                        allowed_base
                        if partition == 0
                        else allowed_add
                        if split == "train"
                        else allowed_val
                    )
                    try:
                        values = selected_arrays(source, record, legacy_ids, allowed)
                    finally:
                        if hasattr(source, "close"):
                            source.close()
                    if values is not None:
                        parts.append(values)
            if not parts:
                continue
            values = concatenate(parts)
        batches = []
        features = []
        sampler = BalancedBatchSampler(
            values["natoms"], max_atoms=4096, batch_size=128, pair_budget=1000000000
        )
        with torch.inference_mode():
            for rows in sampler:
                data = make_batch(values, np.asarray(rows))
                data.hip_sizes = data.natoms.tolist()
                device = data.clone().cuda()
                emb = model.features(device, "fp16")["node_embedding"].float().cpu()
                if not torch.isfinite(emb).all():
                    raise ValueError("Nonfinite frozen features")
                batches.append(data)
                features.append(dict(node_embedding=emb))
        arrays, scale = arrays_from_batches(batches, features, feature_dtype="float32")
        if split == "validation":
            # Reorder dense labels exactly like the balanced feature batches.
            arrays["hessian_flat"] = np.concatenate(
                [b.hessian.numpy() for b in batches]
            )
            arrays["hessian_ptr"] = np.r_[0, np.cumsum(9 * arrays["natoms"] ** 2)]
        metadata = dict(
            dataset_revision=inventory["revision"],
            backbone_sha256=frozen,
            backbone_precision="fp16",
            backbone_graph_version=4,
            fused_backbone=True,
            tf32=True,
            grouped_mole=True,
            expert_fp32_output=True,
            training_split="release-v2",
            feature_scale=scale,
        )
        write_packed(path, arrays, metadata)
        entry = dict(
            path=destination,
            split=split,
            ids=arrays["omol_index"].astype(np.int64).tolist(),
        )
        write_manifest(receipt, entry)
        entries.append(entry)
        print(
            json.dumps(
                dict(
                    group=index,
                    groups=len(groups),
                    samples=len(entry["ids"]),
                    split=split,
                )
            ),
            flush=True,
        )
        del values, parts, batches, features, arrays
    if parameter_hash(model.backbone) != frozen:
        raise ValueError("Frozen backbone changed")
    if args.group_indices:
        return
    write_manifest(
        args.output / "receipts" / f"{args.rank:03d}.json",
        dict(rank=args.rank, entries=entries, backbone_sha256=frozen),
    )


if __name__ == "__main__":
    main()
