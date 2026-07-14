#!/usr/bin/env python3
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
    """Return configuration mismatches that make a speed comparison invalid.

    Artifacts without any config/software fingerprint (smoke flat output,
    thermal rounds) can't be checked — the caller decides whether unlike
    shapes may still be compared.
    """
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


def _has_fingerprint(artifact: dict) -> bool:
    return bool(artifact.get("config")) or bool(artifact.get("software"))


def _load_metrics(data: dict, path: Path) -> dict:
    """Extract the metrics dict from bench, smoke, or thermal output."""
    # bench.py / smoke.py structured output: {"metrics": {...}}
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
    parser.add_argument("--max-throughput-regression", type=float, default=0.05)
    parser.add_argument("--allow-config-mismatch", action="store_true",
                        help="compare unlike artifacts only when explicitly justified")
    args = parser.parse_args()

    baseline_artifact = json.loads(Path(args.baseline).read_text())
    candidate_artifact = json.loads(Path(args.candidate).read_text())

    # Fingerprint gate: only enforceable when at least one artifact carries a
    # config/software section. Legacy flat smoke and thermal-rounds artifacts
    # have none, and the CI smoke gate compares those routinely.
    if _has_fingerprint(baseline_artifact) or _has_fingerprint(candidate_artifact):
        mismatches = comparable(baseline_artifact, candidate_artifact)
        if mismatches and not args.allow_config_mismatch:
            print(
                "Benchmark artifacts are not comparable:\n- " + "\n- ".join(mismatches),
                file=sys.stderr,
            )
            return 2

    baseline = _load_metrics(baseline_artifact, Path(args.baseline))
    candidate = _load_metrics(candidate_artifact, Path(args.candidate))

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
