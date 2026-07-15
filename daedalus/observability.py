"""Observability setup with guarded optional dependencies.

``setup_logging()`` attempts to configure structlog for structured JSON
logging.  If structlog is not installed it falls back to standard library
``logging`` with a basic console handler.

``maybe_init_otel()`` starts OpenTelemetry SDK + OTLP export when the
``opentelemetry-distro`` package is installed and the ``OTEL_EXPORTER_OTLP_*``
environment variables are set.  Otherwise it's a no-op so the server starts
without any OTel dependency.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from typing import Any, Optional

# App logs rotate at 100 MiB; on a 16GB Air the log volume must never
# compete with model weights for the SSD.
_LOG_ROTATE_BYTES = 100 * 1024**2


def setup_logging(
    *,
    level: str = "INFO",
    json_output: bool = False,
    log_file: Optional[str] = None,
) -> None:
    """Configure the root logger.

    When ``structlog`` is available it is preferred for structured output.
    Falls back to stdlib ``logging`` with a simple formatter.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Attempt structlog ---------------------------------------------------
    _structlog: Any = None
    try:
        import structlog as _structlog  # type: ignore[import-untyped]
    except ImportError:
        _structlog = None

    if _structlog is not None and json_output:
        _setup_structlog(numeric_level, log_file)
        return

    # Fallback: stdlib logging --------------------------------------------
    root = logging.getLogger()
    root.setLevel(numeric_level)

    # Remove existing handlers to avoid duplicates on re-config.
    for h in list(root.handlers):
        root.removeHandler(h)

    handler: logging.Handler
    if log_file:
        handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=_LOG_ROTATE_BYTES, backupCount=3
        )
    else:
        handler = logging.StreamHandler(sys.stderr)

    handler.setLevel(numeric_level)

    if _structlog is not None:
        # Use structlog's formatter even for console output (nicer).
        handler.setFormatter(
            _structlog.stdlib.ProcessorFormatter(
                processor=_structlog.dev.ConsoleRenderer()
            )
        )
    else:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S",
            )
        )
    root.addHandler(handler)


def _setup_structlog(numeric_level: int, log_file: Optional[str] = None) -> None:
    """Internal: configure structlog with JSON rendering."""
    import structlog

    shared_processors: list[Any] = [
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if log_file:
        handler: logging.Handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=_LOG_ROTATE_BYTES, backupCount=3
        )
    else:
        handler = logging.StreamHandler(sys.stderr)

    handler.setLevel(numeric_level)
    handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            processor=structlog.processors.JSONRenderer(),
            foreign_pre_chain=shared_processors,
        )
    )

    root = logging.getLogger()
    root.setLevel(numeric_level)
    for h in list(root.handlers):
        root.removeHandler(h)
    root.addHandler(handler)

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


# ---------------------------------------------------------------------------
# OpenTelemetry (guarded)
# ---------------------------------------------------------------------------

def maybe_init_otel(
    service_name: str = "daedalus",
    *,
    otlp_endpoint: Optional[str] = None,
) -> bool:
    """Start the OpenTelemetry SDK if the optional packages are installed.

    Returns ``True`` if OTel was successfully initialised, ``False`` if the
    packages are unavailable or configuration is missing.

    ``otlp_endpoint`` overrides the ``OTEL_EXPORTER_OTLP_ENDPOINT`` env var
    when provided.  The standard ``OTEL_SERVICE_NAME`` env var is also
    respected as a fallback for ``service_name``.
    """
    try:
        from opentelemetry import trace  # type: ignore[import-untyped]
        from opentelemetry.sdk.resources import Resource  # type: ignore[import-untyped]
        from opentelemetry.sdk.trace import TracerProvider  # type: ignore[import-untyped]
        from opentelemetry.sdk.trace.export import (  # type: ignore[import-untyped]
            BatchSpanProcessor,
        )
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (  # type: ignore[import-untyped]
            OTLPSpanExporter,
        )
    except ImportError:
        return False

    endpoint = otlp_endpoint or os.environ.get(
        "OTEL_EXPORTER_OTLP_ENDPOINT", ""
    )
    if not endpoint:
        return False  # no exporter configured — no-op

    service = os.environ.get("OTEL_SERVICE_NAME", service_name)
    resource = Resource.create({"service.name": service})
    provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(endpoint=endpoint)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    return True


# Global tracer that code can ``from daedalus.observability import tracer``
_tracer: Any = None
try:
    from opentelemetry import trace as _otel_trace  # type: ignore[import-untyped]
    _tracer = _otel_trace.get_tracer(__name__)
except ImportError:
    _tracer = None


def get_tracer() -> Any:
    """Return a no-op tracer when OTel is not installed, the real one otherwise."""
    if _tracer is not None:
        return _tracer
    # Fallback: create a minimal no-op tracer so callers never get None.
    try:
        from opentelemetry.trace import _NoOpTracer  # type: ignore[import-untyped]
        return _NoOpTracer()
    except ImportError:
        # Last resort: a duck-typed no-op.
        class _NopSpan:
            def set_attribute(self, *a: Any, **kw: Any) -> None:
                return None

        class _NopTracer:
            def start_as_current_span(self, *a: Any, **kw: Any) -> Any:
                from contextlib import nullcontext
                return nullcontext(_NopSpan())
        return _NopTracer()
