"""HTTP-layer helpers: request-id propagation, rate limiters, body reading.

Small, FastAPI/Starlette-facing utilities shared by the app and streaming
paths — middleware for X-Request-ID, token-bucket rate limiters, a
byte-capped JSON body reader, trusted-proxy client-IP extraction, and the SSE
chunk/frame formatters.
"""

from __future__ import annotations

import contextvars
import json
import threading
import time
import uuid
from typing import Optional

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as StarletteRequest
from starlette.responses import Response as StarletteResponse

# Request ID context variable for propagation through all log lines
request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Middleware to extract and propagate X-Request-ID header through all log lines."""

    async def dispatch(self, request: StarletteRequest, call_next):
        request_id = request.headers.get("x-request-id") or f"req-{uuid.uuid4().hex[:12]}"
        token = request_id_var.set(request_id)
        try:
            response: StarletteResponse = await call_next(request)
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            request_id_var.reset(token)


class GlobalRateLimiter:
    """Global token bucket rate limiter for total RPS across all clients."""

    def __init__(self, max_rps: float, burst: int = 0):
        self.max_rps = max_rps
        self.burst = max(1, burst if burst > 0 else max(1, int(max_rps * 2)))
        self._tokens = float(self.burst)
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self.burst, self._tokens + elapsed * self.max_rps)
        self._last_refill = now

    def try_acquire(self) -> bool:
        with self._lock:
            self._refill()
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return True
            return False


class ClientRateLimiter(GlobalRateLimiter):
    """A per-client token bucket with a bounded, pruneable lifetime."""

    def __init__(self, requests_per_minute: int):
        super().__init__(requests_per_minute / 60.0, burst=requests_per_minute)
        self.last_seen = time.monotonic()

    def try_acquire(self) -> bool:
        allowed = super().try_acquire()
        self.last_seen = time.monotonic()
        return allowed


class RequestBodyTooLarge(ValueError):
    """Raised when a streamed request body exceeds the configured limit."""


async def read_json_body(request: Request, max_bytes: int) -> dict:
    """Read JSON with a hard byte cap even for chunked requests.

    Starlette's ``request.json()`` buffers the entire body.  Checking only
    Content-Length therefore leaves a memory-exhaustion path for chunked or
    deliberately headerless clients.
    """
    chunks: list[bytes] = []
    total = 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > max_bytes:
            raise RequestBodyTooLarge
        chunks.append(chunk)
    try:
        body = json.loads(b"".join(chunks))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("request body must be valid JSON") from exc
    if not isinstance(body, dict):
        raise ValueError("request body must be a JSON object")
    return body


def request_client_ip(request: Request, trusted_proxy_hosts: frozenset[str]) -> str:
    """Use forwarded client IPs only from an explicitly trusted proxy."""
    peer = request.client.host if request.client else "local"
    if peer not in trusted_proxy_hosts:
        return peer
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        candidate = forwarded.split(",", 1)[0].strip()
        if candidate:
            return candidate
    return peer


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


def _chunk(
    request_id: str,
    model: str,
    created: int,
    delta: dict,
    finish_reason: Optional[str] = None,
    usage: Optional[dict] = None,
) -> dict:
    chunk: dict = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {"index": 0, "delta": delta, "finish_reason": finish_reason}
        ],
    }
    if usage is not None:
        chunk["usage"] = usage
    return chunk
