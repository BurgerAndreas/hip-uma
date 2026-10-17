"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Summarize GPU kernel gaps and CPU phases from a bounded HIP training trace.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def merged_duration(intervals):
    """
    Measure a union of time intervals without counting overlapping streams twice.
    """
    if not intervals:
        return 0.0
    ordered = sorted(intervals)
    first, last = ordered[0]
    total = 0.0
    for begin, end in ordered[1:]:
        if begin > last:
            total += last - first
            first, last = begin, end
        else:
            last = max(last, end)
    return total + last - first


def summarize(events):
    """
    Report trace timing only; instrumentation can reduce measured GPU activity.
    """
    ranges = [
        row for row in events if row.get("cat") == "user_annotation" and "dur" in row
    ]
    steps = [row for row in ranges if row["name"].startswith("ProfilerStep")]
    if not steps:
        raise ValueError("Trace contains no captured training steps")
    begin = min(row["ts"] for row in steps)
    end = max(row["ts"] + row["dur"] for row in steps)
    devices = defaultdict(list)
    kernel_counts = defaultdict(int)
    for row in events:
        if row.get("cat") == "kernel" and "dur" in row:
            a, b = max(begin, row["ts"]), min(end, row["ts"] + row["dur"])
            if a < b:
                device = str(row.get("args", {}).get("device", 0))
                devices[device].append((a, b))
                kernel_counts[device] += 1
    phases = {}
    items = [
        row
        for row in events
        if row.get("cat") == "cpu_op" and row.get("name") == "aten::item"
    ]
    for name in (
        "data_load",
        "device_transfer",
        "frozen_backbone",
        "generate_graph",
        "hip_head",
        "hessian_loss",
        "head_backward",
        "batch_metrics",
        "grad_clip",
        "optimizer_step",
    ):
        sections = [row for row in ranges if row["name"] == name]
        extracted = sum(
            any(
                p["tid"] == row["tid"]
                and p["ts"] <= row["ts"]
                and row["ts"] + row["dur"] <= p["ts"] + p["dur"]
                for p in sections
            )
            for row in items
        )
        phases[name] = dict(
            calls=len(sections),
            cpu_ms=sum(row["dur"] for row in sections) / 1000,
            scalar_extractions=extracted,
        )
    return dict(
        steps=len(steps),
        wall_ms=(end - begin) / 1000,
        kernel_busy_percent={
            device: 100 * merged_duration(intervals) / (end - begin)
            for device, intervals in devices.items()
        },
        kernel_counts=dict(kernel_counts),
        device_to_host_copies=sum(
            row.get("cat") == "gpu_memcpy" and "DtoH" in row.get("name", "")
            for row in events
        ),
        phases=phases,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            summarize(json.loads(args.trace.read_text())["traceEvents"]), indent=2
        )
    )


if __name__ == "__main__":
    main()
