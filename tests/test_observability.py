"""Tests for observability setup (guarded structlog, guarded OTel)."""

from __future__ import annotations

import logging

import pytest


def test_setup_logging_fallback_when_no_structlog():
    """When structlog is unavailable, setup_logging still configures stdlib
    logging at the expected level."""
    try:
        import structlog  # noqa: F401
        pytest.skip("structlog is installed — can't test fallback path")
    except ImportError:
        pass

    from daedalus.observability import setup_logging

    # Force the module to believe structlog is gone.
    import daedalus.observability as mod
    saved = mod.__dict__.get("_structlog")
    mod._structlog = None

    try:
        setup_logging(level="WARNING")

        root = logging.getLogger()
        assert root.level == logging.WARNING
        assert any(
            isinstance(h, logging.StreamHandler) and h.level == logging.WARNING
            for h in root.handlers
        )
    finally:
        for h in list(root.handlers):
            root.removeHandler(h)
        root.setLevel(logging.NOTSET)
        if saved is not None:
            mod._structlog = saved


def test_maybe_init_otel_returns_false_without_packages():
    """When opentelemetry is absent, maybe_init_otel returns False."""
    from daedalus.observability import maybe_init_otel

    result = maybe_init_otel(otlp_endpoint="http://localhost:4318/v1/traces")
    # In the test environment with no OTel installed, the import inside the
    # function will fail and return False.  If OTel *is* installed it needs
    # OTEL_EXPORTER_OTLP_ENDPOINT to be set, which it won't be in CI.
    assert result is False


def test_maybe_init_otel_noop_without_endpoint():
    """maybe_init_otel returns False when no endpoint is configured."""
    from daedalus.observability import maybe_init_otel

    result = maybe_init_otel()
    assert result is False


def test_get_tracer_never_none():
    """get_tracer() always returns a usable object, never None."""
    from daedalus.observability import get_tracer

    tracer = get_tracer()
    assert tracer is not None
    # It should not crash when instrumented.
    with tracer.start_as_current_span("test"):
        pass
