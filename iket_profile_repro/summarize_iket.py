#!/usr/bin/env python3
"""Summarize trace-reported IKET kernel durations from a run directory."""
from __future__ import annotations

import argparse
import glob
import json
import statistics
from pathlib import Path


def find_trace(path: Path) -> Path:
    if path.is_file():
        return path
    matches = sorted(glob.glob(str(path / "*.trace.json")))
    if not matches:
        raise SystemExit(f"No *.trace.json found under {path}")
    return Path(matches[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("label")
    parser.add_argument("rep")
    parser.add_argument("trace", type=Path)
    args = parser.parse_args()

    trace = find_trace(args.trace)
    data = json.loads(trace.read_text())
    launch = data["launches"][0]
    strings = data["stringTable"]
    e2e = [
        r["endTs"] - r["startTs"]
        for r in launch["ranges"]
        if strings[r["rangeNameIdx"]] == "kernel_e2e"
    ]
    lifetimes = [w["endTs"] - w["startTs"] for w in launch["warpLifetimes"]]
    if not e2e:
        raise SystemExit(f"No kernel_e2e ranges found in {trace}")

    print(
        args.label,
        args.rep,
        "ranges",
        len(launch["ranges"]),
        "e2e_median_us",
        f"{statistics.median(e2e) / 1000:.3f}",
        "e2e_mean_us",
        f"{statistics.mean(e2e) / 1000:.3f}",
        "warp_median_us",
        f"{statistics.median(lifetimes) / 1000:.3f}",
    )


if __name__ == "__main__":
    main()
