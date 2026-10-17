"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Cache the fixed HIP evaluation cohort without changing its dense labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fairchem.core.frozen_hip.validation import build_validation_cache, load_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    manifest = load_manifest(args.manifest or args.data / "validation_10000.json")
    output = build_validation_cache(args.data, manifest, args.output, args.workers)
    print(json.dumps(dict(samples=manifest["samples"], bytes=output.stat().st_size)))


if __name__ == "__main__":
    main()
