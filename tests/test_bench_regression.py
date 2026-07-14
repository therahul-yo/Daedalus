"""Benchmark artifacts must match before they are used as a speed gate."""

import importlib.util
from pathlib import Path


_SPEC = importlib.util.spec_from_file_location(
    "bench_regression", Path(__file__).parents[1] / "bench" / "regression.py"
)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)
comparable = _MODULE.comparable


def artifact(**overrides):
    config = {
        "model": "model", "governor": "on", "kv_bits": 8,
        "prompt_tokens": 8000, "max_tokens": 64, "machine": "arm64",
        "hw": "Apple", "macos": "15.0", "cache_mode": "cold",
    }
    config.update(overrides)
    return {"config": config, "software": {"mlx": "1", "mlx_lm": "1"}}


def test_benchmark_artifacts_require_matching_fingerprint():
    assert comparable(artifact(), artifact()) == []
    assert "kv_bits" in comparable(artifact(), artifact(kv_bits=4))[0]
    changed = artifact()
    changed["software"]["mlx"] = "2"
    assert "mlx" in comparable(artifact(), changed)[0]
