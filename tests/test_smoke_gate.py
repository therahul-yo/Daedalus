"""The smoke artifact must be a real, comparable input to the regression gate.

FIX 3 makes bench/smoke.py write a fingerprinted config/software/metrics
document. These tests pin the contract the CI gate relies on: the schema flows
through regression.py's loader, matching fingerprints compare, a genuine
regression is caught (exit 1), and an incomparable pair is refused (exit 2)
rather than silently passing.
"""

import importlib.util
import json
from pathlib import Path


_SPEC = importlib.util.spec_from_file_location(
    "bench_regression", Path(__file__).parents[1] / "bench" / "regression.py"
)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)


def smoke_artifact(**metric_overrides):
    """A document shaped exactly like bench/smoke.py's output."""
    metrics = {
        "ttft_s": 0.5,
        "prefill_tps_wall": 1000.0,
        "generation_tokens": 40,
        "generation_tps": 80.0,
        "peak_memory_gb": 1.2,
        "thermal_end": "NOMINAL",
    }
    metrics.update(metric_overrides)
    return {
        "schema_version": 2,
        "config": {
            "model": "mlx-community/Qwen3-0.6B-4bit",
            "governor": "default",
            "kv_bits": None,
            "prompt_tokens": 4000,
            "max_tokens": 40,
            "machine": "arm64",
            "macos": "15.5",
            "python": "3.12.0",
            "cache_mode": "cold",
            "hw": "Apple M4",
        },
        "software": {"mlx": "0.29.0", "mlx_lm": "0.31.0"},
        "metrics": metrics,
    }


def _run_gate(tmp_path, baseline, candidate, *extra):
    base = tmp_path / "baseline.json"
    cand = tmp_path / "candidate.json"
    base.write_text(json.dumps(baseline))
    cand.write_text(json.dumps(candidate))
    import sys

    argv = sys.argv
    sys.argv = ["regression.py", str(base), str(cand), *extra]
    try:
        return _MODULE.main()
    finally:
        sys.argv = argv


def test_smoke_schema_metrics_are_extracted():
    """_load_metrics reads the smoke schema's metrics block."""
    metrics = _MODULE._load_metrics(smoke_artifact(), Path("smoke.json"))
    assert metrics["ttft_s"] == 0.5
    assert metrics["prefill_tps_wall"] == 1000.0


def test_matching_smoke_fingerprints_pass(tmp_path):
    assert _run_gate(tmp_path, smoke_artifact(), smoke_artifact()) == 0


def test_throughput_regression_is_caught(tmp_path):
    slower = smoke_artifact(prefill_tps_wall=500.0)
    assert _run_gate(
        tmp_path, smoke_artifact(), slower, "--max-throughput-regression", "0.10"
    ) == 1


def test_ttft_regression_is_caught(tmp_path):
    slower = smoke_artifact(ttft_s=2.0)
    assert _run_gate(
        tmp_path, smoke_artifact(), slower, "--max-ttft-regression", "0.10"
    ) == 1


def test_fingerprint_mismatch_is_refused_not_passed(tmp_path):
    candidate = smoke_artifact()
    candidate["software"]["mlx"] = "0.30.0"
    # Exit 2 = incomparable; the CI gate turns this into a visible SKIP, never
    # a silent pass.
    assert _run_gate(tmp_path, smoke_artifact(), candidate) == 2
