"""Ops-polish slice of the codex audit plan: version, /health exposure split,
latency histograms, and the `daedalus status` command."""

import sys
from pathlib import Path

from fastapi.testclient import TestClient

import daedalus
from daedalus.cli import render_status
from daedalus.metrics import Histogram, ServerMetrics
from daedalus.server import create_app

sys.path.insert(0, str(Path(__file__).parent))
from test_server import FakeEngine, FakeStore  # noqa: E402


# ------------------------------------------------------------------ version


def test_version_matches_pyproject():
    import tomllib

    pyproject = Path(__file__).parent.parent / "pyproject.toml"
    with pyproject.open("rb") as f:
        assert daedalus.__version__ == tomllib.load(f)["project"]["version"]


# ------------------------------------------------------- /health exposure


def test_health_hides_diagnostics_on_keyed_server():
    """Key-protected (LAN-exposed) servers keep /health probe-friendly but
    reserve model identity and memory numbers for authorized callers."""
    app = create_app(FakeEngine(), FakeStore(), model_id="test-model", api_key="secret")
    client = TestClient(app)

    anonymous = client.get("/health")
    assert anonymous.status_code == 200  # liveness always answers
    assert anonymous.json() == {"status": "ok"}

    authorized = client.get("/health", headers={"Authorization": "Bearer secret"})
    assert authorized.json()["model"] == "test-model"
    assert "active_memory_bytes" in authorized.json()


def test_health_full_body_on_local_unkeyed_server():
    app = create_app(FakeEngine(), FakeStore(), model_id="test-model")
    r = TestClient(app).get("/health")
    assert r.json()["model"] == "test-model"
    assert r.json()["thermal"] == "NOMINAL"


# ---------------------------------------------------------------- histograms


def test_histogram_buckets_are_cumulative_with_inf():
    h = Histogram((1.0, 5.0))
    for v in (0.5, 0.7, 3.0, 100.0):
        h.observe(v)
    lines = h.render("x_seconds", "test")
    assert 'x_seconds_bucket{le="1.0"} 2' in lines
    assert 'x_seconds_bucket{le="5.0"} 3' in lines
    assert 'x_seconds_bucket{le="+Inf"} 4' in lines
    assert "x_seconds_count 4" in lines


def test_request_populates_latency_histograms():
    app = create_app(FakeEngine(), FakeStore(), model_id="test-model")
    client = TestClient(app)
    r = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 200
    text = client.get("/metrics").text
    assert "daedalus_ttft_seconds_count 1" in text
    assert "daedalus_queue_wait_seconds_count 1" in text
    assert 'daedalus_ttft_seconds_bucket{le="+Inf"} 1' in text


def test_metrics_render_valid_without_observations():
    m = ServerMetrics()
    out = m.render(active=0, limit=8, cache={}, thermal="NOMINAL")
    assert "daedalus_ttft_seconds_count 0" in out
    assert "daedalus_decode_tokens_per_second_count 0" in out


# ------------------------------------------------------------- status command


def test_render_status_full_diagnostics():
    out = render_status(
        {"status": "ok", "model": "m", "thermal": "NOMINAL",
         "active_memory_bytes": 5_000_000_000},
        {"status": "ready", "pending_requests": 1, "max_pending_requests": 8,
         "queue_depth": 0},
        {"entries": 4, "resident_entries": 2, "resident_bytes": 700_000_000,
         "hits": 3, "misses": 1},
    )
    assert "ok · m" in out
    assert "ready · 1/8 requests" in out
    assert "NOMINAL" in out
    assert "5.00 GB active" in out
    assert "4 entries (2 in RAM, 0.70 GB) · 75% hit rate" in out


def test_render_status_minimal_keyed_body():
    """Against a keyed server without a key: liveness only, no crashes."""
    out = render_status({"status": "ok"}, {}, {})
    assert "ok" in out
    assert "GB active" not in out
