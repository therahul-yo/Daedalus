"""Structured audit logger with rotating file output.

The ``audit_logger`` singleton emits structured JSON events to a captive
``daedalus.audit`` logger backed by a rotating file handler.  Import
guarded so the dependency (stdlib ``logging.handlers`` only — no extras)
is always available.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import time
from pathlib import Path
from typing import Any, Optional


_AUDIT_LOGGER_NAME = "daedalus.audit"

# Lazy-held singleton reference so ``close()`` can remove the handler.
_handler: Optional[logging.handlers.RotatingFileHandler] = None


def setup_audit_log(path: Path, *, max_bytes: int = 10 * 1024**2, backup_count: int = 5) -> logging.Logger:
    """Configure the ``daedalus.audit`` logger with a rotating file handler.

    Each event is written as a single JSON line (newline-delimited JSON / NDJSON)
    with at least the fields ``ts``, ``event``, and any event-specific payload.

    Returns the logger so callers can emit events with ``audit_logger.info(...)``
    or use the convenience helpers below.
    """
    global _handler
    log = logging.getLogger(_AUDIT_LOGGER_NAME)
    log.setLevel(logging.INFO)
    log.propagate = False  # don't double-log to the root handler

    # Avoid duplicate handlers if called more than once.
    if _handler is not None:
        log.removeHandler(_handler)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    _handler = logging.handlers.RotatingFileHandler(
        str(path),
        maxBytes=max_bytes,
        backupCount=backup_count,
    )
    _handler.setLevel(logging.INFO)
    log.addHandler(_handler)
    return log


def _audit_log() -> logging.Logger:
    """Return the daedalus.audit logger (may be a null logger if not configured)."""
    return logging.getLogger(_AUDIT_LOGGER_NAME)


def _emit(event: str, **payload: Any) -> None:
    """Emit a structured NDJSON audit event."""
    record = {"ts": time.time(), "event": event, **payload}
    _audit_log().info(json.dumps(record, default=str))


# ---------------------------------------------------------------------------
# Convenience emit helpers  (the real emit sites used by server.py)
# ---------------------------------------------------------------------------

def auth_failure(client_ip: str, reason: str = "invalid_api_key", **extra: Any) -> None:
    """Called when a request is rejected because the API key is missing/wrong.

    ``reason`` distinguishes:
    - ``missing_api_key`` — no Authorization header + server has a key configured
    - ``invalid_api_key`` — Authorization header present but doesn't match
    """
    _emit("auth_failure", client_ip=client_ip, reason=reason, **extra)


def rate_limit_hit(client_ip: str, policy: str, limit: int, **extra: Any) -> None:
    """Called when a per-client rate-limit bucket overflows.

    ``policy`` is e.g. ``requests_per_minute``.
    ``limit`` is the bound that was exceeded.
    """
    _emit("rate_limit_hit", client_ip=client_ip, policy=policy, limit=limit, **extra)


def request_rejected(client_ip: str, reason: str, **extra: Any) -> None:
    """Called when the server rejects a request before processing.

    Reasons include: ``queue_full``, ``memory_pressure``, ``request_too_large``.
    """
    _emit("request_rejected", client_ip=client_ip, reason=reason, **extra)


def cache_admin(action: str, client_ip: str, **extra: Any) -> None:
    """Called on cache-administration operations (clear, prune, etc.)."""
    _emit("cache_admin", action=action, client_ip=client_ip, **extra)


def model_swap(from_model: str, to_model: str, **extra: Any) -> None:
    """Called when the resident model is hot-swapped (multi-model mode)."""
    _emit("model_swap", from_model=from_model, to_model=to_model, **extra)


def close() -> None:
    """Flush and remove the rotating handler so the file can be rotated."""
    global _handler
    if _handler is not None:
        _handler.flush()
        _handler.close()
        log = logging.getLogger(_AUDIT_LOGGER_NAME)
        log.removeHandler(_handler)
        _handler = None
