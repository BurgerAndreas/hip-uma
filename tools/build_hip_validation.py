"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Build the fixed approximately 10,000-molecule HIP evaluation manifest.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fairchem.core.frozen_hip.validation import build_manifest, write_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--completion", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    completion = args.completion or args.data.parent / "DOWNLOAD_COMPLETE.json"
    output = args.output or args.data / "validation_10000.json"
    manifest = build_manifest(args.data, json.loads(completion.read_text()))
    write_manifest(output, manifest)
    print(
        json.dumps({key: value for key, value in manifest.items() if key != "entries"})
    )


if __name__ == "__main__":
    main()
