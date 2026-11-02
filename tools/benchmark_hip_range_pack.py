"""Compare compiled range reductions with coalesced FP16 operand packing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from fairchem.core.frozen_hip.linear import quantize
from fairchem.core.frozen_hip.range_pack import pack_pair


def measure(operation, repeats):
    for _ in range(5):
        operation()
    torch.cuda.synchronize()
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    start.record()
    for _ in range(repeats):
        operation()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repeats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=65536)
    parser.add_argument("--columns", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.rows, args.columns, args.repeats) < 1:
        parser.error("Dimensions and repeat count must be positive")
    torch.manual_seed(19)
    cases = []
    for dtype in (torch.float32, torch.float16):
        value = torch.randn(args.rows, args.columns, device="cuda", dtype=dtype)
        for split in (False, True):
            first, second = value.chunk(2, dim=1) if split else (value, None)
            for bias in (False, True):
                # Bind each case before compilation. Keep compiler overhead outside timing.
                def reference(first=first, second=second, bias=bias):
                    full = (
                        first if second is None else torch.cat((first, second), dim=1)
                    )
                    operand, scale = quantize(full)
                    return operand, scale, full.float().sum(0) if bias else None

                def candidate(first=first, second=second, bias=bias):
                    return pack_pair(first, second, bias=bias)

                baseline = torch.compile(reference, fullgraph=True)
                optimized = torch.compile(candidate, fullgraph=True)
                expected, actual = baseline(), optimized()
                # Compare represented values because bypassing FP16 normalization is deliberate.
                torch.testing.assert_close(
                    actual[0].float() * actual[1],
                    expected[0].float() * expected[1],
                    rtol=1e-3,
                    atol=1e-6,
                )
                if bias:
                    torch.testing.assert_close(
                        actual[2], expected[2], rtol=1e-4, atol=1e-3
                    )
                baseline_ms = measure(baseline, args.repeats)
                optimized_ms = measure(optimized, args.repeats)
                cases.append(
                    dict(
                        dtype=str(dtype),
                        split=split,
                        bias=bias,
                        baseline_ms=baseline_ms,
                        optimized_ms=optimized_ms,
                        speedup=baseline_ms / optimized_ms,
                    )
                )
    report = dict(
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        rows=args.rows,
        columns=args.columns,
        repeats=args.repeats,
        cases=cases,
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
