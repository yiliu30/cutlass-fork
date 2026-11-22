#!/usr/bin/env python3
"""Generate a one-range variant of the upstream IKET example.

The generated kernel keeps only the outer kernel_e2e range. This is the
minimal-instrumentation variant used for the IKET overhead A/B test.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent
    repo_root = root.parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=repo_root
        / "examples/python/CuTeDSL/dsl_tutorials/fp16_gemm_4_iket.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "build/fp16_gemm_4_iket_min.py",
    )
    args = parser.parse_args()

    lines = args.input.read_text().splitlines(keepends=True)
    iket_lines = [i for i, line in enumerate(lines) if "cute.experimental.iket." in line]
    compute_grid = next(
        i for i, line in enumerate(lines) if line.startswith("def compute_grid(")
    )
    before_compute = [i for i in iket_lines if i < compute_grid]
    if not iket_lines or not before_compute:
        raise SystemExit(f"No IKET calls found in {args.input}")

    keep = set()
    for i in iket_lines:
        if 'range_push("kernel_e2e")' in lines[i]:
            keep.add(i)
    keep.add(before_compute[-1])

    out = []
    for i, line in enumerate(lines):
        if i in iket_lines and i not in keep:
            out.append("# " + line)
        else:
            out.append(line)

    remaining = [
        out[i].strip()
        for i in iket_lines
        if not out[i].lstrip().startswith("#")
    ]
    if len(remaining) != 2 or 'range_push("kernel_e2e")' not in remaining[0]:
        raise SystemExit(f"Unexpected remaining IKET calls: {remaining}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(out))
    print(f"Wrote {args.output}")
    print("Remaining IKET calls:")
    for line in remaining:
        print(f"  {line}")


if __name__ == "__main__":
    main()
