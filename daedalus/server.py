"""OpenAI-compatible server for daedalus.

Endpoints: /v1/chat/completions (SSE + non-stream), /v1/models, /health,
/v1/cache/stats.

Client-compatibility rules learned from OpenCode/pi/Hermes research:
- SSE keepalive comments flow during prefill (pi's idle timeout resets on
  them; they also carry progress for humans watching).
- OpenCode's 300s deadline is a WHOLE-REQUEST timeout that streaming does
  not extend — the real fix is the prefix cache + checkpoint resume, which
  this server wires in for every request.
- Never emit an empty ``tool_calls: []`` array in a streamed chunk
  (OpenCode hangs forever). Tool-call deltas carry explicit ``index``.
- Stateless clients resend the whole conversation each turn: after prefill
  (before decode — hybrid caches can't be trimmed later) the KV state is
  snapshotted back into the store keyed by the prompt tokens.

Multi-model server: per-model engine/store/router, shared thermal governor,
request routing by model field in chat completions.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextvars
import hmac
import json
import logging
import logging.handlers
import os
import sys
import threading
import time
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Dict, List, Optional

from fastapi import FastAPI, Header, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from starlette.background import BackgroundTask
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as StarletteRequest
from starlette.responses import Response as StarletteResponse
from starlette.websockets import WebSocket as StarletteWebSocket

from daedalus.cache.store import PrefixCacheStore
from daedalus.engine import Engine, PrefillAborted
from daedalus.governor import ThermalGovernor
from daedalus.metrics import ServerMetrics
from daedalus.reasoning import ThinkStreamFilter
from daedalus.scheduler import PriorityLock
from daedalus.sensors import ThermalMonitor
from daedalus.tools import make_stream_filter

logger = logging.getLogger(__name__)

# Request ID context variable for propagation through all log lines
request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")

# Audit logger for structured security/audit logging
audit_logger: Optional[logging.Logger] = None

KEEPALIVE_INTERVAL_S = 1.0
CHECKPOINT_EVERY_TOKENS = 4096
CHECKPOINT_MIN_JOB_TOKENS = 8192
CHECKPOINT_MIN_INTERVAL_S = 8.0
SHORT_PROMPT_THRESHOLD = 2048  # tokens
DEFAULT_SHUTDOWN_TIMEOUT = 30.0  # seconds


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Middleware to extract and propagate X-Request-ID header through all log lines."""

    async def dispatch(self, request: StarletteRequest, call_next):
        # Extract or generate request ID
        request_id = request.headers.get("x-request-id") or f"req-{uuid.uuid4().hex[:12]}"
        
        # Set in context variable for log propagation
        token = request_id_var.set(request_id)
        
        try:
            # Add request ID to response headers
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
    
    def acquire_wait(self, timeout: float = 30.0) -> bool:
        """Wait for a token to become available."""
        start = time.monotonic()
        while time.monotonic() - start < timeout:
            if self.try_acquire():
                return True
            time.sleep(0.01)
        return False


@dataclass
class ServerState:
    engine: Engine
    store: PrefixCacheStore
    model_id: str
    lock: PriorityLock  # Priority queue for short prompts
    max_pending_requests: int
    api_key: Optional[str]
    metrics: ServerMetrics
    admission_lock: threading.Lock
    admitted_requests: int = 0
    accepting: bool = True
    token_cache: "PromptTokenCache" = None
    head_cache: "SharedHeadIndex" = None
    max_active_memory_bytes: Optional[int] = None
    max_prompt_tokens: int = 65536
    max_completion_tokens: int = 4096
    requests_per_minute: int = 0  # Per-client rate limit
    rate_windows: dict[str, tuple[float, int]] = None
    max_request_bytes: int = 2 * 1024 * 1024
    global_rate_limiter: Optional[GlobalRateLimiter] = None
    shutdown_timeout: float = DEFAULT_SHUTDOWN_TIMEOUT
    cors_origins: List[str] = field(default_factory=list)
    cors_allow_credentials: bool = False
    audit_log_path: Optional[str] = None
    in_flight_requests: int = 0
    in_flight_lock: threading.Lock = field(default_factory=threading.Lock)
    _multi_state: Optional["MultiModelServerState"] = field(default=None, repr=False)

    def try_admit(self) -> bool:
        with self.admission_lock:
            if not self.accepting or self.admitted_requests >= self.max_pending_requests:
                return False
            self.admitted_requests += 1
            return True

    def release(self) -> None:
        with self.admission_lock:
            self.admitted_requests = max(0, self.admitted_requests - 1)

    def memory_available(self) -> bool:
        if self.max_active_memory_bytes is None:
            return True
        active = getattr(self.engine, "active_memory_bytes", lambda: 0)()
        if active < self.max_active_memory_bytes:
            return True
        trim = getattr(self.store, "trim_ram", None)
        if trim:
            trim(0)
        return getattr(self.engine, "active_memory_bytes", lambda: 0)() < self.max_active_memory_bytes

    def allow_client(self, key: str) -> bool:
        """Per-client rate limiting."""
        if self.requests_per_minute <= 0:
            return True
        now = time.monotonic()
        with self.admission_lock:
            if len(self.rate_windows) > 1024:
                self.rate_windows = {
                    client: window for client, window in self.rate_windows.items()
                    if now - window[0] < 60
                }
            started, count = self.rate_windows.get(key, (now, 0))
            if now - started >= 60:
                started, count = now, 0
            if count >= self.requests_per_minute:
                self.rate_windows[key] = (started, count)
                return False
            self.rate_windows[key] = (started, count + 1)
            return True

    def allow_global_rate(self) -> bool:
        """Global rate limiting check (fast path)."""
        if self.global_rate_limiter is None:
            return True
        return self.global_rate_limiter.try_acquire()

    def acquire_global_rate(self, timeout: float = 30.0) -> bool:
        """Wait for global rate limiter token."""
        if self.global_rate_limiter is None:
            return True
        return self.global_rate_limiter.acquire_wait(timeout)

    def start_request(self) -> None:
        """Track in-flight request for graceful shutdown."""
        with self.in_flight_lock:
            self.in_flight_requests += 1

    def finish_request(self) -> None:
        """Track completed request for graceful shutdown."""
        with self.in_flight_lock:
            self.in_flight_requests = max(0, self.in_flight_requests - 1)

    def wait_for_drain(self, timeout: float) -> bool:
        """Wait for in-flight requests to complete."""
        start = time.monotonic()
        while time.monotonic() - start < timeout:
            with self.in_flight_lock:
                if self.in_flight_requests == 0:
                    return True
            time.sleep(0.1)
        return False


@dataclass
class MultiModelServerState:
    """Holds state for multiple models with shared resources."""
    models: Dict[str, ServerState]  # model_id -> ServerState
    governor: Optional[ThermalGovernor]
    monitor: Optional[ThermalMonitor]
    max_active_memory_bytes: Optional[int] = None
    global_rate_limiter: Optional[GlobalRateLimiter] = None
    accepting: bool = True
    in_flight_requests: int = 0
    in_flight_lock: threading.Lock = field(default_factory=threading.Lock)

    def get_model(self, model_id: str) -> Optional[ServerState]:
        return self.models.get(model_id)

    def list_models(self) -> List[dict]:
        """Return model info for /v1/models endpoint."""
        result = []
        for model_id, state in self.models.items():
            result.append({
                "id": model_id,
                "object": "model",
                "owned_by": "daedalus",
                "permission": [{"id": f"modelperm-{model_id}", "object": "model_permission", "created": int(time.time()), "allow_create_engine": False, "allow_sampling": True, "allow_logprobs": True, "allow_search_indices": False, "allow_view": True, "allow_fine_tuning": False, "organization": "*", "group": None, "is_blocking": False}],
                "root": model_id,
                "parent": None,
            })
        return result

    def total_active_memory(self) -> int:
        """Sum of active memory across all models."""
        total = 0
        for state in self.models.values():
            total += getattr(state.engine, "active_memory_bytes", lambda: 0)()
        return total

    def memory_available(self) -> bool:
        """Check if total active memory across all models is within limit."""
        if self.max_active_memory_bytes is None:
            return True
        return self.total_active_memory() < self.max_active_memory_bytes

    def start_request(self) -> None:
        with self.in_flight_lock:
            self.in_flight_requests += 1

    def finish_request(self) -> None:
        with self.in_flight_lock:
            self.in_flight_requests = max(0, self.in_flight_requests - 1)

    def wait_for_drain(self, timeout: float) -> bool:
        start = time.monotonic()
        while time.monotonic() - start < timeout:
            with self.in_flight_lock:
                if self.in_flight_requests == 0:
                    return True
            time.sleep(0.1)
        return False


class PromptTokenCache:
    """Tiny LRU for repeated stateless-agent chat-template rendering."""

    def __init__(self, max_entries: int = 256) -> None:
        self.max_entries = max_entries
        self.max_tokens = 200_000
        self._entries: OrderedDict[str, tuple[int, ...]] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        self.tokens = 0

    def get_or_build(self, messages: List[dict], tools: Optional[List[dict]], build) -> List[int]:
        key = json.dumps({"messages": messages, "tools": tools}, sort_keys=True, separators=(",", ":"))
        with self._lock:
            cached = self._entries.get(key)
            if cached is not None:
                self._entries.move_to_end(key)
                self.hits += 1
                return list(cached)
            self.misses += 1
        tokens = list(build())
        with self._lock:
            previous = self._entries.get(key)
            if previous is not None:
                self.tokens -= len(previous)
            self._entries[key] = tuple(tokens)
            self.tokens += len(tokens)
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries or self.tokens > self.max_tokens:
                _, evicted = self._entries.popitem(last=False)
                self.tokens -= len(evicted)
        return tokens

    def stats(self) -> dict:
        with self._lock:
            return {"entries": len(self._entries), "tokens": self.tokens, "hits": self.hits, "misses": self.misses}


class SharedHeadIndex:
    """Memoize stable system/tool prefix boundaries across agent sessions."""

    def __init__(self, max_entries: int = 512) -> None:
        self._entries: OrderedDict[str, int] = OrderedDict()
        self._lock = threading.Lock()
        self.max_entries = max_entries
        self.hits = 0

    def key(self, messages: List[dict], tools: Optional[List[dict]]) -> str:
        head = [m for m in messages if m.get("role") in ("system", "developer")] or messages[:1]
        return json.dumps({"head": normalize_messages(head), "tools": tools}, sort_keys=True, separators=(",", ":"))

    def get(self, key: str) -> Optional[int]:
        with self._lock:
            value = self._entries.get(key)
            if value is not None:
                self.hits += 1
                self._entries.move_to_end(key)
            return value

    def put(self, key: str, boundary: int) -> None:
        with self._lock:
            self._entries[key] = boundary
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)

    def stats(self) -> dict:
        with self._lock:
            return {"entries": len(self._entries), "hits": self.hits}


_TEMPLATE_ROLES = {"system", "user", "assistant", "tool"}


def normalize_messages(messages: List[dict]) -> List[dict]:
    """Map OpenAI wire-format quirks onto what HF chat templates accept.

    - role "developer" (newer OpenAI convention, sent by pi) -> "system";
      other unknown roles -> "user" (templates raise on unknown roles)
    - content parts [{type: "text", text: ...}] -> flattened string
    - assistant tool_calls function.arguments JSON string -> dict (Qwen-style
      templates iterate arguments as a mapping)
    - content: null -> ""
    """
    out = []
    for msg in messages:
        m = dict(msg)
        role = m.get("role")
        if role not in _TEMPLATE_ROLES:
            m["role"] = "system" if role == "developer" else "user"
            if role != "developer":
                logger.warning("unknown message role %r -> user", role)
        content = m.get("content")
        if isinstance(content, list):
            m["content"] = "".join(
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            )
        elif content is None:
            m["content"] = ""
        if m.get("tool_calls"):
            calls = []
            for call in m["tool_calls"]:
                call = json.loads(json.dumps(call))  # deep copy
                fn = call.get("function", {})
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        fn["arguments"] = json.loads(args) if args.strip() else {}
                    except json.JSONDecodeError:
                        fn["arguments"] = {"_raw": args}
                calls.append(call)
            m["tool_calls"] = calls
        out.append(m)
    return out


def build_prompt_tokens(
    state: ServerState, messages: List[dict], tools: Optional[List[dict]] = None
) -> List[int]:
    kwargs = {"add_generation_prompt": True}
    if tools:
        kwargs["tools"] = tools
    normalized = normalize_messages(messages)
    return state.token_cache.get_or_build(
        normalized, tools,
        lambda: state.engine.tokenizer.apply_chat_template(normalized, **kwargs),
    )


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


def create_app(
    engines: Dict[str, Engine],
    stores: Dict[str, PrefixCacheStore],
    model_ids: List[str],
    *,
    max_pending_requests: int = 8,
    api_key: Optional[str] = None,
    token_cache_entries: int = 1024,
    max_active_memory_bytes: Optional[int] = None,
    max_prompt_tokens: int = 65536,
    max_completion_tokens: int = 4096,
    requests_per_minute: int = 0,
    max_request_bytes: int = 2 * 1024 * 1024,
    global_rps: float = 0.0,
    global_burst: int = 0,
    shutdown_timeout: float = DEFAULT_SHUTDOWN_TIMEOUT,
    cors_origins: Optional[List[str]] = None,
    cors_allow_credentials: bool = False,
    audit_log_path: Optional[str] = None,
    governor: Optional[ThermalGovernor] = None,
    monitor: Optional[ThermalMonitor] = None,
    stream_interval: int = 1,  # NEW: tokens per SSE yield
) -> FastAPI:
    if max_pending_requests < 1:
        raise ValueError("max_pending_requests must be at least 1")
    if token_cache_entries < 1:
        raise ValueError("token_cache_entries must be at least 1")
    if max_prompt_tokens < 1 or max_completion_tokens < 1:
        raise ValueError("token limits must be positive")
    if requests_per_minute < 0:
        raise ValueError("requests_per_minute cannot be negative")
    if max_request_bytes < 1:
        raise ValueError("max_request_bytes must be positive")
    if global_rps < 0:
        raise ValueError("global_rps cannot be negative")
    if shutdown_timeout <= 0:
        raise ValueError("shutdown_timeout must be positive")

    # Set up audit logger
    global audit_logger
    if audit_log_path:
        audit_logger = logging.getLogger("daedalus.audit")
        audit_logger.setLevel(logging.INFO)
        audit_logger.propagate = False
        if audit_log_path == "stderr":
            handler = logging.StreamHandler(sys.stderr)
        else:
            handler = logging.handlers.RotatingFileHandler(
                audit_log_path, maxBytes=10 * 1024 * 1024, backupCount=5
            )
        handler.setFormatter(logging.Formatter("%(message)s"))
        audit_logger.addHandler(handler)

    # Create shared multi-model state first
    multi_state = MultiModelServerState(
        models={},
        governor=governor,
        monitor=monitor,
        max_active_memory_bytes=max_active_memory_bytes,
        global_rate_limiter=GlobalRateLimiter(global_rps, global_burst) if global_rps > 0 else None,
    )

    # Create per-model ServerState instances
    model_states = {}
    for model_id in model_ids:
        engine = engines[model_id]
        store = stores[model_id]
        model_states[model_id] = ServerState(
            engine=engine,
            store=store,
            model_id=model_id,
            lock=PriorityLock(SHORT_PROMPT_THRESHOLD),
            max_pending_requests=max_pending_requests,
            api_key=api_key,
            metrics=ServerMetrics(),
            admission_lock=threading.Lock(),
            token_cache=PromptTokenCache(token_cache_entries),
            head_cache=SharedHeadIndex(),
            max_active_memory_bytes=max_active_memory_bytes,
            max_prompt_tokens=max_prompt_tokens,
            max_completion_tokens=max_completion_tokens,
            requests_per_minute=requests_per_minute,
            rate_windows={},
            max_request_bytes=max_request_bytes,
            global_rate_limiter=GlobalRateLimiter(global_rps, global_burst) if global_rps > 0 else None,
            shutdown_timeout=shutdown_timeout,
            cors_origins=cors_origins or [],
            cors_allow_credentials=cors_allow_credentials,
            audit_log_path=audit_log_path,
        )
        model_states[model_id]._multi_state = multi_state  # Set back-reference for graceful shutdown

    # Update multi_state with the models
    multi_state.models = model_states

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        logger.info("daedalus multi-model server starting up")
        yield
        logger.info("daedalus server shutting down: stopping acceptance of new requests")
        with multi_state.in_flight_lock:
            multi_state.accepting = False
        for state in model_states.values():
            with state.admission_lock:
                state.accepting = False

        # Wait for in-flight requests to complete with timeout
        logger.info(f"waiting for in-flight requests to drain (timeout: {shutdown_timeout}s)")
        if not multi_state.wait_for_drain(shutdown_timeout):
            logger.warning(f"shutdown timeout ({shutdown_timeout}s) reached, forcing exit with {multi_state.in_flight_requests} requests still in flight")
        else:
            logger.info("all in-flight requests completed, shutdown complete")

        for state in model_states.values():
            close = getattr(state.store, "close", None)
            if close:
                close()

    app = FastAPI(title="daedalus", lifespan=lifespan)
    app.state.daedalus = multi_state

    # Add CORS middleware if origins are configured
    if cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_credentials=cors_allow_credentials,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # Add Request ID middleware for header propagation
    app.add_middleware(RequestIdMiddleware)

    def error(message: str, status_code: int, kind: str = "invalid_request_error"):
        return JSONResponse({"error": {"message": message, "type": kind}}, status_code=status_code)

    def extract_api_key(request: Request, authorization: Optional[str] = None) -> Optional[str]:
        """Extract API key from Authorization header, cookie, or websocket headers."""
        # Check Authorization header first (Bearer token)
        if authorization:
            return authorization.removeprefix("Bearer ").strip()
        
        # Check cookie for daedalus_api_key
        cookie_key = request.cookies.get("daedalus_api_key")
        if cookie_key:
            return cookie_key
        
        # Check X-API-Key header (for websocket handshake)
        header_key = request.headers.get("x-api-key")
        if header_key:
            return header_key
        
        return None

    def authorized(request: Request, authorization: Optional[str] = None) -> bool:
        if api_key is None:
            return True
        api_key_extracted = extract_api_key(request, authorization)
        return api_key_extracted is not None and hmac.compare_digest(api_key_extracted, api_key)

    def audit_log(event: str, **fields):
        """Structured audit logging for security events."""
        if audit_logger is None:
            return
        log_entry = {
            "timestamp": time.time(),
            "event": event,
            "request_id": request_id_var.get(),
            **fields,
        }
        audit_logger.info(json.dumps(log_entry))

    @app.get("/health")
    def health():
        # Return health for all models
        model_health = {}
        for model_id, state in model_states.items():
            model_health[model_id] = {
                "thermal": state.engine.governor.effective_level.name,
                "active_memory_bytes": getattr(state.engine, "active_memory_bytes", lambda: 0)(),
                "accepting": state.accepting,
                "in_flight": state.in_flight_requests,
            }
        return {
            "status": "ok",
            "models": model_health,
            "thermal": governor.effective_level.name if governor else "unknown",
            "accepting": multi_state.accepting,
            "in_flight": multi_state.in_flight_requests,
        }

    @app.get("/readyz")
    def readyz():
        ready = True
        pending_total = 0
        for state in model_states.values():
            with state.admission_lock:
                if not state.accepting or state.admitted_requests >= state.max_pending_requests:
                    ready = False
                pending_total += state.admitted_requests
        status = 200 if ready else 503
        return JSONResponse({
            "status": "ready" if ready else "busy",
            "pending_requests": pending_total,
            "models": {mid: {"pending": s.admitted_requests, "max_pending": s.max_pending_requests} for mid, s in model_states.items()},
            "accepting": multi_state.accepting,
            "in_flight": multi_state.in_flight_requests,
        }, status_code=status)

    @app.get("/metrics")
    def metrics():
        active = sum(s.admitted_requests for s in model_states.values())
        limit = max_pending_requests
        cache_stats = {
            "entries": sum(len(s.store.entries) for s in model_states.values()),
            "hits": sum(s.store.hits for s in model_states.values()),
            "misses": sum(s.store.misses for s in model_states.values()),
            "copy_seconds": sum(getattr(s.store, 'copy_seconds', 0) for s in model_states.values()),
            "lookup_seconds": sum(getattr(s.store, 'lookup_seconds', 0) for s in model_states.values()),
            "load_seconds": sum(getattr(s.store, 'load_seconds', 0) for s in model_states.values()),
            "tokenization": {
                "hits": sum(s.token_cache.hits for s in model_states.values()),
                "misses": sum(s.token_cache.misses for s in model_states.values()),
            },
            "shared_head": {
                "hits": sum(s.head_cache.hits for s in model_states.values()),
            },
        }
        thermal = governor.effective_level.name if governor else "unknown"
        # Use the first model state for rendering
        any_state = next(iter(model_states.values())) if model_states else None
        if any_state:
            return PlainTextResponse(any_state.metrics.render(
                active=active,
                limit=limit,
                cache=cache_stats,
                thermal=thermal,
            ), media_type="text/plain; version=0.0.4")
        return PlainTextResponse("", media_type="text/plain; version=0.0.4")

    @app.get("/v1/models")
    def models(request: Request, authorization: str = Header(default=None)):
        if not authorized(request, authorization):
            return error("invalid API key", 401, "authentication_error")
        return {
            "object": "list",
            "data": multi_state.list_models(),
        }

    @app.get("/v1/cache/stats")
    def cache_stats(request: Request, authorization: str = Header(default=None), model: Optional[str] = None):
        if not authorized(request, authorization):
            return error("invalid API key", 401, "authentication_error")
        if model:
            state = multi_state.get_model(model)
            if not state:
                return error(f"model {model} not found", 404, "not_found_error")
            return state.store.stats()
        # Return stats for all models
        return {mid: s.store.stats() for mid, s in model_states.items()}

    @app.delete("/v1/cache")
    def clear_cache(request: Request, authorization: str = Header(default=None), model: Optional[str] = None):
        if not authorized(request, authorization):
            return error("invalid API key", 401, "authentication_error")
        if model:
            state = multi_state.get_model(model)
            if not state:
                return error(f"model {model} not found", 404, "not_found_error")
            if state.admitted_requests:
                return error("cache cannot be cleared while requests are active", 409, "conflict_error")
            removed = state.store.clear()
            state.metrics.inc_cache_admin("clear")
            return {"removed_entries": removed}
        # Clear all caches
        total_removed = 0
        for state in model_states.values():
            if state.admitted_requests:
                return error("cache cannot be cleared while requests are active", 409, "conflict_error")
            removed = state.store.clear()
            state.metrics.inc_cache_admin("clear")
            total_removed += removed
        return {"removed_entries": total_removed}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request, authorization: str = Header(default=None)):
        # Extract or generate request ID from X-Request-ID header
        request_id = request.headers.get("X-Request-ID", f"chatcmpl-{uuid.uuid4().hex[:24]}")
        token = request_id_var.set(request_id)
        
        # Log request start with request ID
        logger.info(
            "request_start request_id=%s",
            request_id,
        )

        try:
            if not authorized(request, authorization):
                logger.warning("auth_failed request_id=%s", request_id)
                return error("invalid API key", 401, "authentication_error")
            
            client_key = authorization or (request.client.host if request.client else "local")
            
            # Check global rate limit first
            if not multi_state.global_rate_limiter or multi_state.global_rate_limiter.try_acquire():
                pass
            else:
                logger.warning("global_rate_limited request_id=%s", request_id)
                return error("global request rate limit exceeded", 429, "rate_limit_error")
            
            content_length = request.headers.get("content-length")
            if content_length:
                try:
                    if int(content_length) > max_request_bytes:
                        return error("request body exceeds server limit", 413)
                except ValueError:
                    return error("invalid Content-Length header", 400)
            
            try:
                body = await request.json()
            except Exception:
                return error("request body must be valid JSON", 400)
            
            if not isinstance(body, dict):
                return error("request body must be a JSON object", 400)
            
            messages = body.get("messages", [])
            if not isinstance(messages, list) or not messages:
                return error("messages must be a non-empty array", 400)
            
            stream = bool(body.get("stream", False))
            
            try:
                max_tokens = int(body.get("max_tokens") or body.get("max_completion_tokens") or 2048)
                temperature = float(body.get("temperature", 0.7))
                top_p = float(body.get("top_p", 1.0))
            except (TypeError, ValueError):
                return error("max_tokens, temperature, and top_p must be numeric", 400)
            
            if max_tokens < 1 or temperature < 0 or not 0 < top_p <= 1:
                return error("max_tokens must be positive, temperature non-negative, and top_p in (0, 1]", 400)
            
            # Extract model from request - route to appropriate model
            model_id = body.get("model")
            if not model_id:
                # Default to first model if not specified
                model_id = model_ids[0] if model_ids else None
            
            state = multi_state.get_model(model_id)
            if not state:
                return error(f"model '{model_id}' not found. Available models: {', '.join(model_ids)}", 404, "not_found_error")
            
            # Check per-model limits
            if max_tokens > state.max_completion_tokens:
                return error(f"max_tokens exceeds server limit ({state.max_completion_tokens})", 400)
            
            tools = body.get("tools") or None
            if body.get("tool_choice") == "none":
                tools = None

            try:
                tokens = build_prompt_tokens(state, messages, tools)
            except Exception as exc:
                logger.warning("prompt_build_failed request_id=%s error=%s", request_id, str(exc))
                return error("prompt could not be templated", 400)
            
            if len(tokens) > state.max_prompt_tokens:
                return error(f"prompt exceeds server limit ({state.max_prompt_tokens} tokens)", 413)
            
            # Check per-client rate limit
            if not state.allow_client(client_key):
                state.metrics.inc_request("rate_limited")
                logger.warning("rate_limited request_id=%s client=%s", request_id, client_key)
                return error("request rate limit exceeded", 429, "rate_limit_error")
            
            # Check global rate limit (wait if needed)
            if state.global_rate_limiter and not state.acquire_global_rate():
                state.metrics.inc_request("global_rate_limited")
                logger.warning("global_rate_limited request_id=%s", request_id)
                return error("global request rate limit exceeded", 429, "rate_limit_error")
            
            # Try to admit request
            if not state.try_admit():
                state.metrics.inc_request("rejected")
                return error("server queue is full; retry shortly", 429, "rate_limit_error")
            
            # Check memory availability across all models
            if not multi_state.memory_available():
                state.release()
                state.metrics.inc_request("memory_rejected")
                return error("insufficient free model memory; retry after active requests finish", 503, "server_overloaded_error")

            created = int(time.time())

            # Qwen3.5-style templates end the generation prompt inside an opened
            # think block; if so, the model's first output is reasoning.
            try:
                tail = state.engine.tokenizer.decode(tokens[-8:])
            except Exception:
                tail = ""
            prompt_in_think = tail.rstrip().endswith("think")

            # Shared-head boundary: the token count of the prompt's stable head
            # (system prompt + tool schemas), found by re-templating with a dummy
            # conversation and taking the longest common prefix. Snapshotting the
            # cache exactly there lets a NEW session/branch reuse the head even
            # on non-trimmable hybrid models, where any divergence otherwise
            # forces a full cold prefill.
            head_key = state.head_cache.key(messages, tools)
            head_boundary = state.head_cache.get(head_key)
            try:
                if head_boundary is None:
                    head_msgs = [
                        m for m in messages if m.get("role") in ("system", "developer")
                    ] or messages[:1]
                    probe = build_prompt_tokens(
                        state, head_msgs + [{"role": "user", "content": "†"}], tools
                    )
                    lcp = 0
                    for a, b in zip(tokens, probe):
                        if a != b:
                            break
                        lcp += 1
                    if 256 <= lcp < len(tokens) - 1:
                        head_boundary = lcp
                        state.head_cache.put(head_key, lcp)
            except Exception:
                pass

            # Log with request ID propagation
            rid = request_id[-6:]
            logger.info(
                "request_received",
                request_id=rid,
                model=model_id,
                prompt_tokens=len(tokens),
                tools=len(tools) if tools else 0,
                head_boundary=head_boundary,
                stream=stream,
            )

            gen = _Generation(
                state=state,
                tokens=tokens,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                tools=tools,
                prompt_in_think=prompt_in_think,
                head_boundary=head_boundary,
                request_id=request_id,
                created=created,
                stream_interval=stream_interval,
            )

            if stream:
                return StreamingResponse(
                    _stream_response(state, gen, request_id, created, request),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "X-Accel-Buffering": "no",
                        "X-Request-ID": request_id,
                    },
                    background=BackgroundTask(gen.release_slot),
                )

            try:
                result = await asyncio.to_thread(gen.run_to_completion)
                state.metrics.inc_request("completed")
            except Exception:
                state.metrics.inc_request("failed")
                raise
            finally:
                gen.release_slot()
            
            message: dict = {"role": "assistant", "content": result["text"] or None}
            if result["reasoning"]:
                message["reasoning_content"] = result["reasoning"]
            if result["tool_calls"]:
                # Response-format tool calls carry no "index" field.
                message["tool_calls"] = [
                    {k: v for k, v in c.items() if k != "index"}
                    for c in result["tool_calls"]
                ]
            
            return JSONResponse({
                "id": request_id,
                "object": "chat.completion",
                "created": created,
                "model": model_id,
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": result["finish_reason"],
                    }
                ],
                "usage": result["usage"],
            })
        finally:
            request_id_var.reset(token)


    return app


class _Generation:
    """Runs one request on the engine thread; the async side drains a queue.

    The engine work happens in a worker thread (MLX is happier off the event
    loop); events are handed to the async generator through a thread-safe
    queue so keepalives can be emitted while prefill is still running.
    """

    def __init__(
        self,
        state: ServerState,
        tokens: List[int],
        max_tokens: int,
        temperature: float,
        top_p: float,
        tools: Optional[List[dict]] = None,
        prompt_in_think: bool = False,
        head_boundary: Optional[int] = None,
        request_id: str = "",
        created: int = 0,
        stream_interval: int = 1,
    ):
        self.state = state
        self.tokens = tokens
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.tools = tools
        self.prompt_in_think = prompt_in_think
        self.head_boundary = head_boundary
        self.request_id = request_id
        self.created = created
        self.stream_interval = stream_interval
        self.cached_tokens = 0
        self.prefill_done = 0
        self.aborted = threading.Event()
        self.slot_taken = False
        self.rid = request_id[-6:]
        self.keepalive_line = f": keepalive {self.rid}\n\n"

    def take_slot(self) -> None:
        self.slot_taken = True

    def release_slot(self) -> None:
        if self.slot_taken:
            self.state.release()
            self.slot_taken = False

    def _run_engine(self):
        state = self.state
        already = 0
        last_checkpoint = 0
        last_checkpoint_time = time.monotonic()
        deferred_persist_tokens = None
        
        # Take the admission slot for this generation
        self.take_slot()
        
        # Track in-flight for graceful shutdown
        state.start_request()
        multi_state = getattr(state, '_multi_state', None)
        if multi_state:
            multi_state.start_request()
        try:
            with state.lock.acquire_with_priority(len(self.tokens)):
                # A client can disconnect while its FIFO ticket is waiting. It
                # still releases its turn, but never starts cache/model work.
                if self.aborted.is_set():
                    return
                hit = state.store.fetch(self.tokens)
                if hit is not None:
                    prompt_cache = hit.cache
                    already = hit.matched_tokens
                    self.cached_tokens = already
                    logger.info(
                        "  %s · cache hit %d/%d tok (%s) — prefilling %d",
                        self.rid,
                        already,
                        len(self.tokens),
                        hit.source,
                        len(self.tokens) - already,
                    )
                else:
                    prompt_cache = state.engine.make_cache()
                    already = 0
                    logger.info(
                        "  %s · cache miss — cold prefill %d tok",
                        self.rid,
                        len(self.tokens),
                    )

                last_checkpoint = already
                last_checkpoint_time = time.monotonic()
                deferred_persist_tokens: Optional[List[int]] = None
                last_progress_log = [time.monotonic()]

                def progress_cb(done: int, total: int) -> None:
                    self.prefill_done = done
                    now = time.monotonic()
                    if now - last_progress_log[0] >= 5.0 and done < total:
                        fresh = done - already
                        elapsed = now - t_start
                        rate = fresh / elapsed if elapsed > 0 else 0
                        logger.info(
                            "  %s · prefill %d/%d (%d%%) @ %.0f tok/s · thermal %s",
                            self.rid,
                            done,
                            total,
                            100 * done // max(total, 1),
                            rate,
                            state.engine.governor.effective_level.name,
                        )
                        last_progress_log[0] = now

                def checkpoint_cb(done: int, cache: List[Any]) -> None:
                    nonlocal last_checkpoint, last_checkpoint_time, deferred_persist_tokens
                    if done >= len(self.tokens) - 1:
                        # End-of-prefill snapshot keyed by the prompt: for
                        # non-trimmable (hybrid) caches this is the only state
                        # the next stateless-client turn can reuse.
                        state.store.put(self.tokens[:done], cache, persist=False)
                        deferred_persist_tokens = self.tokens[:done]
                    elif done == self.head_boundary and done > already:
                        # Shared-head snapshot: reused by NEW sessions/branches
                        # whose conversation diverges after the system prompt.
                        state.store.put(self.tokens[:done], cache)
                        logger.info(
                            "  %s · head snapshot at %d tok", self.rid, done
                        )
                    elif (
                        len(self.tokens) - already >= CHECKPOINT_MIN_JOB_TOKENS
                        and done - last_checkpoint >= CHECKPOINT_EVERY_TOKENS
                        and time.monotonic() - last_checkpoint_time >= CHECKPOINT_MIN_INTERVAL_S
                        and len(self.tokens) - done > CHECKPOINT_EVERY_TOKENS
                    ):
                        state.store.checkpoint(self.tokens, done, cache)
                        last_checkpoint = done
                        last_checkpoint_time = time.monotonic()

                think_filter = ThinkStreamFilter(
                    initially_thinking=self.prompt_in_think
                )
                tool_filter = make_stream_filter(state.engine.tokenizer, self.tools)
                call_index = 0
                emitted_calls = False
                finish_reason = "stop"
                n_generated = 0
                decode_tps = 0.0
                t_first_token = None

                def route(text):
                    """think-split, then tool-split the content half."""
                    nonlocal call_index, emitted_calls
                    reasoning, content = think_filter.feed(text)
                    events = []
                    if reasoning:
                        events.append({"type": "reasoning", "text": reasoning})
                    if content:
                        plain, calls = tool_filter.feed(content)
                        if plain:
                            events.append({"type": "delta", "text": plain})
                        if calls:
                            emitted_calls = True
                            events.append(
                                {
                                    "type": "tool_calls",
                                    "calls": [
                                        c.as_openai(call_index + i)
                                        for i, c in enumerate(calls)
                                    ],
                                }
                            )
                            call_index += len(calls)
                    return events

                t_start = time.monotonic()
                try:
                    tokens_since_yield = 0
                    for resp in state.engine.generate(
                        self.tokens,
                        max_tokens=self.max_tokens,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        prompt_cache=prompt_cache,
                        already_cached=already,
                        snap_points=(
                            [self.head_boundary] if self.head_boundary else None
                        ),
                        progress_cb=progress_cb,
                        checkpoint_cb=checkpoint_cb,
                        should_abort=self.aborted.is_set,
                    ):
                        if self.aborted.is_set():
                            finish_reason = "abort"
                            break
                        if t_first_token is None:
                            t_first_token = time.monotonic()
                        if resp.text:
                            output_events = route(resp.text)
                            # Yield tokens in batches based on stream_interval
                            for event in output_events:
                                yield event
                                if event.get("type") == "delta":
                                    tokens_since_yield += 1
                                    if tokens_since_yield >= self.stream_interval:
                                        # Force yield control to allow other coroutines
                                        tokens_since_yield = 0
                        # Generator yields above before continuing here, so the
                        # client can receive first content before disk I/O. MLX
                        # cache serialization must remain on this engine thread.
                        if deferred_persist_tokens is not None and resp.text and output_events:
                            persist = getattr(state.store, "persist", None)
                            if persist:
                                persist(deferred_persist_tokens)
                            deferred_persist_tokens = None
                        n_generated = resp.generation_tokens
                        decode_tps = resp.generation_tps
                        if resp.finish_reason is not None:
                            finish_reason = resp.finish_reason
                except PrefillAborted:
                    logger.info(
                        "  %s · prefill aborted at %d tok (client gone)",
                        self.rid,
                        self.prefill_done,
                    )
                    return

                reasoning, content = think_filter.finalize()
                if deferred_persist_tokens is not None:
                    persist = getattr(state.store, "persist", None)
                    if persist:
                        persist(deferred_persist_tokens)
                if reasoning:
                    yield {"type": "reasoning", "text": reasoning}
                if content:
                    plain, calls = tool_filter.feed(content)
                    content_tail, tail_calls = tool_filter.finalize()
                    if plain or content_tail:
                        yield {"type": "delta", "text": plain + content_tail}
                    calls = calls + tail_calls
                else:
                    content_tail, calls = tool_filter.finalize()
                    if content_tail:
                        yield {"type": "delta", "text": content_tail}
                if calls:
                    emitted_calls = True
                    yield {
                        "type": "tool_calls",
                        "calls": [
                            c.as_openai(call_index + i) for i, c in enumerate(calls)
                        ],
                    }

                if emitted_calls:
                    finish_reason = "tool_calls"

                total_s = time.monotonic() - t_start
                prefill_s = (t_first_token or time.monotonic()) - t_start
                fresh = len(self.tokens) - already
                logger.info(
                    "← %s · %.1fs · prefill %d tok %.1fs (%d cached) · "
                    "decode %d tok @ %.1f tok/s · thermal %s · finish=%s",
                    self.rid,
                    total_s,
                    fresh,
                    prefill_s,
                    already,
                    n_generated,
                    decode_tps,
                    state.engine.governor.effective_level.name,
                    finish_reason,
                )
                yield {
                    "type": "done",
                    "finish_reason": finish_reason or "stop",
                    "usage": {
                        "prompt_tokens": len(self.tokens),
                        "completion_tokens": n_generated,
                        "total_tokens": len(self.tokens) + n_generated,
                        "prompt_tokens_details": {
                            "cached_tokens": self.cached_tokens
                        },
                    },
                }
        finally:
            state.finish_request()
            if multi_state:
                multi_state.finish_request()

    def run_to_completion(self) -> dict:
        """Run generation to completion (non-streaming)."""
        last_event = None
        for event in self._run_engine():
            if event["type"] == "done":
                last_event = event
        return last_event


async def _stream_response(
    state: ServerState,
    gen: _Generation,
    request_id: str,
    created: int,
    request: Request,
) -> AsyncGenerator[str, None]:
    model = state.model_id
    queue: asyncio.Queue = asyncio.Queue(maxsize=64)
    loop = asyncio.get_running_loop()
    closed = threading.Event()

    def enqueue(event: dict) -> bool:
        """Backpressure worker output instead of accumulating slow-client RAM."""
        future = asyncio.run_coroutine_threadsafe(queue.put(event), loop)
        while True:
            try:
                future.result(timeout=0.1)
                return not closed.is_set()
            except concurrent.futures.TimeoutError:
                if closed.is_set():
                    future.cancel()
                    return False

    def worker():
        completed = False
        try:
            for event in gen._run_engine():
                completed = completed or event["type"] == "done"
                if not enqueue(event):
                    break
        except Exception as exc:  # surface engine errors to the client
            logger.exception("engine error")
            enqueue({"type": "error", "message": str(exc)})
        finally:
            state.metrics.inc_request("completed" if completed else "cancelled")
            # Idempotent slot release (leak fix from PR #4) + backpressured
            # eof (bounded queue from PR #3): put_nowait could raise on a
            # full queue, so eof goes through enqueue like every other event.
            gen.release_slot()
            if not closed.is_set():
                enqueue({"type": "eof"})

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    # Initial role delta.
    yield _sse(_chunk(request_id, model, created, {"role": "assistant"}))

    try:
        while True:
            try:
                event = await asyncio.wait_for(
                    queue.get(), timeout=KEEPALIVE_INTERVAL_S
                )
            except asyncio.TimeoutError:
                if await request.is_disconnected():
                    gen.aborted.set()
                    return
                # SSE comment: resets idle timeouts, invisible to JSON parsers.
                yield gen.keepalive_line()
                continue

            if event["type"] == "delta":
                yield _sse(
                    _chunk(request_id, model, created, {"content": event["text"]})
                )
            elif event["type"] == "reasoning":
                # DeepSeek/OpenAI-style reasoning delta: reasoning-aware
                # clients render it dimmed; others ignore the field.
                yield _sse(
                    _chunk(
                        request_id,
                        model,
                        created,
                        {"reasoning_content": event["text"]},
                    )
                )
            elif event["type"] == "tool_calls" and event["calls"]:
                # Guarded non-empty: an empty tool_calls array hangs OpenCode.
                yield _sse(
                    _chunk(
                        request_id, model, created, {"tool_calls": event["calls"]}
                    )
                )
            elif event["type"] == "done":
                yield _sse(
                    _chunk(
                        request_id,
                        model,
                        created,
                        {},
                        finish_reason=event["finish_reason"],
                        usage=event["usage"],
                    )
                )
            elif event["type"] == "error":
                yield _sse(
                    {
                        "error": {
                            "message": event["message"],
                            "type": "server_error",
                        }
                    }
                )
                break
            elif event["type"] == "eof":
                break
    finally:
        closed.set()
        gen.aborted.set()
    yield "data: [DONE]\n\n"