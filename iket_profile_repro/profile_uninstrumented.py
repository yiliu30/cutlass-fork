#!/usr/bin/env python3
"""Measure the uninstrumented GEMM kernel duration with torch.profiler.

This is a sanity check only. The guide recommends comparing IKET variants via
trace-reported kernel durations, not via external timing boundaries.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from torch.profiler import ProfilerActivity, profile


def parse_mnk(value: str) -> tuple[int, int, int]:
    parts = tuple(int(x.strip()) for x in value.split(","))
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("MNK must contain exactly 3 integers")
    return parts


def main() -> None:
    root = Path(__file__).resolve().parent
    repo_root = root.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--mnk", type=parse_mnk, default=(512, 1024, 64))
    parser.add_argument("--tolerance", type=float, default=0.1)
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()

    sys.path.insert(
        0, str(repo_root / "examples/python/CuTeDSL/dsl_tutorials")
    )
    import fp16_gemm_4_iket as example

    for _ in range(args.warmup):
        example.run_dense_gemm(args.mnk, args.tolerance)

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        example.run_dense_gemm(args.mnk, args.tolerance)

    rows = []
    for event in prof.events():
        if "cutlass" in event.name.lower() and event.device_type.name == "CUDA":
            rows.append((event.name, event.device_time_total))

    print(f"MATCHED_EVENTS {len(rows)}")
    for name, duration_us in rows:
        print(f"{duration_us:.3f} us  {name}")


if __name__ == "__main__":
    main()
