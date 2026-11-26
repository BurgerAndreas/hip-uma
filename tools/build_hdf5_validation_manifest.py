"""Build fixed-cohort manifests for staged HIP-UMA validation shards."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from fairchem.core.frozen_hip.validation import sample_fingerprint, write_manifest

EXPECTED = "55ca478f8e3c44305e37147af79f36b4a0d1ceb205fe2e2446fc7a63947432cc"
REVISION = "8815b275a29149bd343ea6b3819bf4eb420c5efd"
SOURCE_REVISION = "3ca8ec13e704e5c5f090552241f8adb15a5ec9d5"
BACKBONE = "33f9b444901549d42998d40cec167a940e7b1547cfd45caf5080589590610bd2"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Root containing validation_200k/shards")
    parser.add_argument("output", type=Path, help="Directory for generated manifests")
    args = parser.parse_args()
    entries, ids = [], []
    for path in sorted((args.root / "validation_200k" / "shards").rglob("*.h5")):
        with h5py.File(path, "r") as f:
            if (not f.attrs.get("complete") or f.attrs.get("split") != "validation"
                or f.attrs.get("model") != "uma-s-1p2"
                or f.attrs.get("hessian_unit") != "eV/Angstrom^2"):
                raise ValueError(f"Unexpected shard metadata: {path}")
            partition = int(f.attrs["parent_partition"])
            sample_ids = (f["omol_index"][:].astype(np.int64) + partition * (1 << 60)).tolist()
        entries.append({"path": str(path.relative_to(args.root)), "rows": list(range(len(sample_ids))), "omol_indices": sample_ids})
        ids.extend(sample_ids)
    if len(ids) != 200000 or len(set(ids)) != 200000 or sample_fingerprint(ids) != EXPECTED:
        raise ValueError(f"Full validation cohort mismatch: {len(ids)} rows, fingerprint {sample_fingerprint(ids)}")
    args.output.mkdir(parents=True, exist_ok=True)
    name = "validation_200000.json"
    record = {
        "selection": "release-holdout-v2", "source_format": "release-hdf5",
        "backbone_sha256": BACKBONE, "dataset_revision": REVISION,
        "source_dataset_revision": SOURCE_REVISION,
        "samples": len(ids), "sample_ids": ids,
        "sample_ids_sha256": sample_fingerprint(ids), "entries": entries,
        "complete": True, "available_samples": len(ids),
    }
    write_manifest(args.output / name, record)
    print(json.dumps({"manifest": name, "samples": len(ids), "sample_ids_sha256": record["sample_ids_sha256"]}), flush=True)


if __name__ == "__main__":
    main()
