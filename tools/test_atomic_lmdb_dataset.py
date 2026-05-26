#!/usr/bin/env python3
"""Test FAIRChem's atomic_lmdb dataset reader."""

from __future__ import annotations

import argparse

from fairchem.core.datasets import create_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--first-n", type=int, default=8)
    parser.add_argument("--dataset-name", default="horm")
    args = parser.parse_args()

    dataset = create_dataset(
        {
            "format": "atomic_lmdb",
            "src": args.src,
            "metadata_path": args.metadata,
            "first_n": args.first_n,
            "transforms": {
                "common_transform": {
                    "dataset_name": args.dataset_name,
                },
            },
        },
        split="train",
    )
    sample = dataset[0]
    print("length", len(dataset))
    print("metadata_natoms0", int(dataset.get_metadata("natoms", 0)))
    print("dataset", sample.dataset)
    print("pos", tuple(sample.pos.shape), sample.pos.dtype)
    print("atomic_numbers", tuple(sample.atomic_numbers.shape), sample.atomic_numbers.dtype)
    print("energy", tuple(sample.energy.shape), sample.energy.dtype)
    print("forces", tuple(sample.forces.shape), sample.forces.dtype)
    print("hessian", tuple(sample.hessian.shape), sample.hessian.dtype)


if __name__ == "__main__":
    main()
