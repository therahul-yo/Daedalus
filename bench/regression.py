"""Fail a benchmark comparison when a candidate regresses beyond a threshold.

Usage: python bench/regression.py baseline.json candidate.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline")
    parser.add_argument("candidate")
    parser.add_argument("--max-ttft-regression", type=float, default=0.10)
    parser.add_argument("--max-throughput-regression", type=float, default=0.05)
    args = parser.parse_args()
    baseline = json.loads(Path(args.baseline).read_text())["metrics"]
    candidate = json.loads(Path(args.candidate).read_text())["metrics"]
    failures = []
    if candidate["ttft_s"] > baseline["ttft_s"] * (1 + args.max_ttft_regression):
        failures.append(f"TTFT regressed: {baseline['ttft_s']}s -> {candidate['ttft_s']}s")
    if candidate["prefill_tps_wall"] < baseline["prefill_tps_wall"] * (1 - args.max_throughput_regression):
        failures.append("prefill throughput regressed: "
                        f"{baseline['prefill_tps_wall']} -> {candidate['prefill_tps_wall']} tok/s")
    if failures:
        print("Benchmark regression detected:\n- " + "\n- ".join(failures), file=sys.stderr)
        return 1
    print("benchmark gate passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
