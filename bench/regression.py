"""Fail a benchmark comparison when a candidate regresses beyond a threshold.

Usage: python bench/regression.py baseline.json candidate.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


FINGERPRINT_FIELDS = (
    "model",
    "governor",
    "kv_bits",
    "prompt_tokens",
    "max_tokens",
    "machine",
    "hw",
    "macos",
    "cache_mode",
)


def comparable(baseline: dict, candidate: dict) -> list[str]:
    """Return configuration mismatches that make a speed comparison invalid."""
    base_config = baseline.get("config", {})
    candidate_config = candidate.get("config", {})
    mismatches = []
    for field in FINGERPRINT_FIELDS:
        if base_config.get(field) != candidate_config.get(field):
            mismatches.append(
                f"{field}: {base_config.get(field)!r} != {candidate_config.get(field)!r}"
            )
    for field in ("mlx", "mlx_lm"):
        base_value = baseline.get("software", {}).get(field)
        candidate_value = candidate.get("software", {}).get(field)
        if base_value != candidate_value:
            mismatches.append(f"{field}: {base_value!r} != {candidate_value!r}")
    return mismatches


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline")
    parser.add_argument("candidate")
    parser.add_argument("--max-ttft-regression", type=float, default=0.10)
    parser.add_argument("--max-throughput-regression", type=float, default=0.05)
    parser.add_argument("--allow-config-mismatch", action="store_true",
                        help="compare unlike artifacts only when explicitly justified")
    args = parser.parse_args()
    baseline_artifact = json.loads(Path(args.baseline).read_text())
    candidate_artifact = json.loads(Path(args.candidate).read_text())
    mismatches = comparable(baseline_artifact, candidate_artifact)
    if mismatches and not args.allow_config_mismatch:
        print("Benchmark artifacts are not comparable:\n- " + "\n- ".join(mismatches), file=sys.stderr)
        return 2
    baseline = baseline_artifact["metrics"]
    candidate = candidate_artifact["metrics"]
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
