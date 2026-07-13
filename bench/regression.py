#!/usr/bin/env python3
"""Fail a benchmark comparison when a candidate regresses beyond a threshold.

Usage: python bench/regression.py baseline.json candidate.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load_metrics(path: Path) -> dict:
    """Extract the metrics dict from smoke or thermal validation output."""
    data = json.loads(path.read_text())

    # smoke.py output: {"metrics": {"ttft_s": ..., "prefill_tps_wall": ..., ...}}
    if "metrics" in data:
        return data["metrics"]

    # smoke.py flat output (already has ttft_s, prefill_tps_wall)
    if "ttft_s" in data and "prefill_tps_wall" in data:
        return data

    # thermal_validation.py output: {"rounds": [...], ...}
    if "rounds" in data and data["rounds"]:
        # Average first two rounds (steady state)
        rounds = data["rounds"]
        tps_wall = sum(r["tps_wall"] for r in rounds[:2]) / min(2, len(rounds))
        tps_burn = sum(r["tps_burn"] for r in rounds[:2]) / min(2, len(rounds))
        return {
            "ttft_s": 0,  # thermal validation doesn't measure TTFT
            "prefill_tps_wall": tps_wall,
            "prefill_tps_burn": tps_burn,
        }

    raise ValueError(f"Unrecognized benchmark format: {path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline")
    parser.add_argument("candidate")
    parser.add_argument("--max-ttft-regression", type=float, default=0.10)
    parser.add_argument("--max-throughput-regression", type=float, default=0.10)
    args = parser.parse_args()

    baseline = _load_metrics(Path(args.baseline))
    candidate = _load_metrics(Path(args.candidate))

    failures = []

    if baseline.get("ttft_s", 0) > 0 and candidate.get("ttft_s", 0) > 0:
        if candidate["ttft_s"] > baseline["ttft_s"] * (1 + args.max_ttft_regression):
            failures.append(
                f"TTFT regressed: {baseline['ttft_s']:.2f}s -> {candidate['ttft_s']:.2f}s"
            )

    if "prefill_tps_wall" in baseline and "prefill_tps_wall" in candidate:
        if candidate["prefill_tps_wall"] < baseline["prefill_tps_wall"] * (
            1 - args.max_throughput_regression
        ):
            failures.append(
                f"prefill throughput regressed: {baseline['prefill_tps_wall']:.1f} -> "
                f"{candidate['prefill_tps_wall']:.1f} tok/s"
            )

    if failures:
        print(
            "Benchmark regression detected:\n- " + "\n- ".join(failures), file=sys.stderr
        )
        return 1

    print("benchmark gate passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())