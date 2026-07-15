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

Single-user engine: one request at a time; a queue lock serializes access.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextvars
import hmac
import json
import logging
import math
import threading
import time
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncGenerator, Callable, List, Optional

from fastapi import FastAPI, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from starlette.background import BackgroundTask
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as StarletteRequest
from starlette.responses import Response as StarletteResponse

from daedalus.cache.store import PrefixCacheStore
from daedalus.engine import Engine, PrefillAborted
from daedalus.metrics import ServerMetrics
from daedalus.reasoning import ThinkStreamFilter
from daedalus.scheduler import FifoLock
from daedalus.tools import make_stream_filter

from daedalus import audit as audit_logger

logger = logging.getLogger(__name__)

# Request ID context variable for propagation through all log lines
request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")

KEEPALIVE_INTERVAL_S = 1.0
CHECKPOINT_EVERY_TOKENS = 4096
CHECKPOINT_MIN_JOB_TOKENS = 8192
CHECKPOINT_MIN_INTERVAL_S = 8.0


# ──────────────────────────────────────────────────────────────────────────
# Multi-model: memory profiles + admission (16GB M4 Air, swap-only)
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class ModelProfile:
    """Memory profile for a single model.

    Sizes are derived from the checkpoint on disk (exact, zero maintenance);
    see ``derive_model_profile``. The built-ins below are fallbacks / known
    hybrid-architecture overrides (where KV only grows on the GQA layers).
    """
    model_id: str
    weights_gb: float
    kv_gb_per_8k: float
    kv_gb_per_32k: float
    context_growing_layers: int = 0
    total_layers: int = 0
    hidden_size: int = 0
    num_attention_heads: int = 0

    def kv_gb(self, context_tokens: int) -> float:
        """KV cache size at k8v8 for a given context length."""
        if context_tokens <= 8192:
            return self.kv_gb_per_8k * (context_tokens / 8192)
        return self.kv_gb_per_8k * (context_tokens / 8192)

    def total_gb(self, context_tokens: int) -> float:
        """Total RAM footprint: weights + KV + 0.5GB Metal cache high-water."""
        return self.weights_gb + self.kv_gb(context_tokens) + 0.5


# 16GB M4 Air: 3.5 (macOS/system) + 0.8 (process) + 1.0 (safety) = 5.3 reserved
# Usable ceiling for (weights + KV) of the active model: 10.7 GB.
MODEL_MEMORY_CEILING_GB = 10.7
# Extra headroom kept free during a swap so both engines are never resident.
SWAP_SAFETY_GB = 1.0

# Known profiles (overrides for hybrid archs where only some layers grow).
MODEL_PROFILES: dict[str, ModelProfile] = {
    "qwen3.5-9b": ModelProfile(
        model_id="qwen3.5-9b", weights_gb=5.2, kv_gb_per_8k=0.65, kv_gb_per_32k=2.6,
        context_growing_layers=8, total_layers=32, hidden_size=3584, num_attention_heads=28,
    ),
    "qwen-7b": ModelProfile(
        model_id="qwen-7b", weights_gb=4.7, kv_gb_per_8k=0.58, kv_gb_per_32k=2.3,
        context_growing_layers=28, total_layers=28, hidden_size=3584, num_attention_heads=28,
    ),
    "qwen-3b": ModelProfile(
        model_id="qwen-3b", weights_gb=1.9, kv_gb_per_8k=0.24, kv_gb_per_32k=0.94,
        context_growing_layers=24, total_layers=24, hidden_size=2048, num_attention_heads=16,
    ),
}


def derive_model_profile(model_id: str, model_path: str) -> ModelProfile:
    """Derive a model's memory profile from the checkpoint on disk.

    Exact weights come from the safetensors index header; KV is estimated
    from ``num_hidden_layers`` (conservative for hybrid models — overcounts
    the constant-state layers, which is the safe direction). Built-in hybrid
    profiles (e.g. qwen3.5-9b) take precedence and are exact.
    """
    if model_id in MODEL_PROFILES:
        return MODEL_PROFILES[model_id]
    try:
        cfg = Path(model_path) / "config.json"
        if cfg.exists():
            config = json.loads(cfg.read_text())
            n_layers = int(config.get("num_hidden_layers", 0))
            hidden = int(config.get("hidden_size", 0))
            n_heads = max(int(config.get("num_attention_heads", 1)), 1)
            import glob
            total = 0
            for sf in glob.glob(str(Path(model_path) / "*.safetensors")):
                with open(sf, "rb") as f:
                    n = int.from_bytes(f.read(8), "little")
                    header = json.loads(f.read(n))
                for k, v in header.items():
                    if k == "__metadata__":
                        continue
                    total += v["data_offsets"][1] - v["data_offsets"][0]
            weights_gb = total / 1e9
            kv_per_8k = (2 * n_layers * (hidden // n_heads) * 8192) / 1e9
            return ModelProfile(
                model_id, weights_gb, kv_per_8k, kv_per_8k * 4,
                n_layers, n_layers, hidden, n_heads,
            )
    except Exception:
        pass
    return ModelProfile(model_id, 5.0, 0.6, 2.4)


def model_fits(
    candidate: ModelProfile,
    active: Optional[ModelProfile],
    max_prompt_tokens: int,
) -> "tuple[bool, float, float]":
    """Whether ``candidate`` can be admitted given the ``active`` model.

    Returns (fits, available_gb, required_gb).
    """
    required = candidate.total_gb(max_prompt_tokens)
    if active is not None:
        available = MODEL_MEMORY_CEILING_GB - active.total_gb(max_prompt_tokens)
    else:
        available = MODEL_MEMORY_CEILING_GB
    # Need SWAP_SAFETY_GB free so both engines are never resident at once.
    return available >= required + SWAP_SAFETY_GB, available, required


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


@dataclass
class ServerState:
    engine: Engine
    store: PrefixCacheStore
    model_id: str
    lock: FifoLock
    max_pending_requests: int
    api_key: Optional[str]
    metrics: ServerMetrics
    admission_lock: threading.Lock
    # Serializes cache maintenance with admission.  A clear must never race a
    # request that has passed the idle check but has not started using a cache.
    maintenance_lock: threading.Lock = field(default_factory=threading.Lock)
    admitted_requests: int = 0
    accepting: bool = True
    token_cache: "PromptTokenCache" = None
    head_cache: "SharedHeadIndex" = None
    token_cache_entries: int = 256
    max_active_memory_bytes: Optional[int] = None
    max_prompt_tokens: int = 65536
    max_completion_tokens: int = 4096
    requests_per_minute: int = 0
    client_rate_limiters: dict[str, ClientRateLimiter] = None
    max_request_bytes: int = 2 * 1024 * 1024
    global_rate_limiter: Optional[GlobalRateLimiter] = None
    shutdown_timeout: float = 30.0
    cors_origins: List[str] = field(default_factory=list)
    cors_allow_credentials: bool = False
    trusted_proxy_hosts: frozenset[str] = field(default_factory=frozenset)
    in_flight_requests: int = 0
    in_flight_lock: threading.Lock = field(default_factory=threading.Lock)
    # ── multi-model swap-only state ───────────────────────────────────────
    # Contains the resident model only.  Inactive models are paths plus a
    # loader, never live MLX objects: two 9B models cannot coexist on 16GB.
    models: dict[str, "tuple[Engine, PrefixCacheStore, PromptTokenCache, SharedHeadIndex, ModelProfile]"] = field(default_factory=dict)
    # Models the server will hot-swap to (beyond the default). Unknown -> 404.
    served_models: set[str] = field(default_factory=set)
    model_paths: dict[str, str] = field(default_factory=dict)
    model_loader: Optional[Callable[[str], "tuple[Engine, PrefixCacheStore]"]] = None
    model_context_override: Optional[int] = None
    model_context_tokens: Optional[int] = None
    swap_cooldown_seconds: float = 30.0
    _last_swap_time: float = 0.0
    swap_lock: threading.Lock = field(default_factory=threading.Lock)
    engine_tasks: int = 0
    engine_tasks_lock: threading.Condition = field(
        default_factory=lambda: threading.Condition(threading.Lock())
    )

    def register_model(self, model_id: str, model_path: str) -> None:
        """Mark ``model_id`` as a swap-eligible served model (CLI --model)."""
        self.served_models.add(model_id)
        if model_path:
            self.model_paths[model_id] = model_path

    def swap_model(self, model_id: str) -> "tuple[bool, str]":
        """Hot-swap to ``model_id`` if admitted.

        Returns (success, message). On success, ``engine``/``store``/caches on
        this state object are repointed at the swapped-in model; in-flight
        requests (which already captured ``state`` and the old engine) finish
        on the old engine, then acquire ``lock`` and see the new one.
        """
        if model_id == self.model_id:
            return True, "already active"
        if model_id not in self.served_models:
            return False, f"model {model_id!r} is not served (register with --model)"
        with self.swap_lock:
            with self.admission_lock:
                if not self.accepting:
                    return False, "server is shutting down"
            # All admission checks live under this lock: checking the cooldown
            # before it permits two callers to swap back-to-back.
            if model_id == self.model_id:
                return True, "already active"
            now = time.monotonic()
            if now - self._last_swap_time < self.swap_cooldown_seconds:
                wait = self.swap_cooldown_seconds - (now - self._last_swap_time)
                return False, f"swap cooldown active: retry after {wait:.0f}s"
            profile = derive_model_profile(model_id, model_path=self.model_paths.get(model_id, model_id))
            # The old model is released before loading the target, so admission
            # is for one resident model, not an impossible two-model total.
            fits, available, required = model_fits(profile, None, self.max_prompt_tokens)
            if not fits:
                return False, (
                    f"model {model_id!r} needs {required:.1f} GB (weights+KV), "
                    f"only {available:.1f} GB available for one resident model"
                )
            if self.model_loader is None:
                return False, "model swapping is not configured"
            # Block new engine admits, then drain work which is still persisting
            # pinned snapshots after it has released the FIFO lock.
            if not self.lock.acquire_for_swap():
                return False, "engine busy, retry"
            try:
                if not self.wait_for_engine_drain(timeout=10.0):
                    return False, "engine is still finishing, retry"
                old_id = self.model_id
                old_store, old_engine = self.store, self.engine
                close = getattr(old_store, "close", None)
                if close:
                    close()
                close = getattr(old_engine, "close", None)
                if close:
                    close()
                shutdown = getattr(old_engine, "shutdown", None)
                if shutdown:
                    shutdown()
                self.models.clear()
                # Drop all strong references before asking MLX to free Metal.
                import gc
                import mlx.core as mx
                self.engine = None
                self.store = None
                del old_store, old_engine
                gc.collect()
                mx.clear_cache()
                try:
                    eng, store = self.model_loader(model_id)
                except Exception as exc:
                    # A failed target load must not strand the service without
                    # its prior model; restore it before reporting the error.
                    try:
                        eng, store = self.model_loader(old_id)
                    except Exception:
                        raise RuntimeError(f"could not load {model_id!r}: {exc}") from exc
                    self.engine, self.store = eng, store
                    self.token_cache = PromptTokenCache(self.token_cache_entries)
                    self.head_cache = SharedHeadIndex()
                    self.models[old_id] = (eng, store, self.token_cache, self.head_cache,
                                           derive_model_profile(old_id, self.model_paths.get(old_id, old_id)))
                    return False, f"could not load {model_id!r}: {exc}"
                self.engine, self.store = eng, store
                self.token_cache = PromptTokenCache(self.token_cache_entries)
                self.head_cache = SharedHeadIndex()
                self.model_id = model_id
                self.model_context_tokens = (
                    self.model_context_override
                    or model_context_limit(getattr(eng, "model", None))
                )
                self.models[model_id] = (eng, store, self.token_cache, self.head_cache, profile)
            finally:
                self.lock.release_after_swap()
            self._last_swap_time = now
            audit_logger.model_swap(old_id, model_id)
            return True, "swapped"

    def try_admit(self) -> bool:
        with self.maintenance_lock:
            with self.admission_lock:
                if not self.accepting or self.admitted_requests >= self.max_pending_requests:
                    return False
                self.admitted_requests += 1
                return True

    def release(self) -> None:
        with self.admission_lock:
            self.admitted_requests = max(0, self.admitted_requests - 1)

    def memory_available(self, reserve_bytes: int = 0) -> bool:
        if self.max_active_memory_bytes is None:
            return True
        try:
            import psutil
            if psutil.virtual_memory().available < 1024 * 1024 * 512:
                trim = getattr(self.store, "trim_ram", None)
                if trim:
                    trim(0)
                if psutil.virtual_memory().available < 1024 * 1024 * 512:
                    return False
        except ImportError:
            pass

        active = getattr(self.engine, "active_memory_bytes", lambda: 0)()
        if active + reserve_bytes < self.max_active_memory_bytes:
            return True
        trim = getattr(self.store, "trim_ram", None)
        if trim:
            trim(0)
        return (
            getattr(self.engine, "active_memory_bytes", lambda: 0)()
            + reserve_bytes
            < self.max_active_memory_bytes
        )

    def allow_client(self, key: str) -> bool:
        if self.requests_per_minute <= 0:
            return True
        now = time.monotonic()
        with self.admission_lock:
            if len(self.client_rate_limiters) > 1024:
                self.client_rate_limiters = {
                    client: limiter
                    for client, limiter in self.client_rate_limiters.items()
                    if now - limiter.last_seen < 120
                }
            limiter = self.client_rate_limiters.get(key)
            if limiter is None:
                limiter = ClientRateLimiter(self.requests_per_minute)
                self.client_rate_limiters[key] = limiter
            return limiter.try_acquire()

    def allow_global_rate(self) -> bool:
        if self.global_rate_limiter is None:
            return True
        return self.global_rate_limiter.try_acquire()

    def start_request(self) -> None:
        with self.in_flight_lock:
            self.in_flight_requests += 1

    def finish_request(self) -> None:
        with self.in_flight_lock:
            self.in_flight_requests = max(0, self.in_flight_requests - 1)

    def wait_for_drain(self, timeout: float) -> bool:
        """Wait for in-flight requests to complete during shutdown."""
        start = time.monotonic()
        while time.monotonic() - start < timeout:
            with self.in_flight_lock:
                if self.in_flight_requests == 0:
                    return True
            time.sleep(0.1)
        return False

    def start_engine_task(self) -> None:
        with self.engine_tasks_lock:
            self.engine_tasks += 1

    def finish_engine_task(self) -> None:
        with self.engine_tasks_lock:
            self.engine_tasks = max(0, self.engine_tasks - 1)
            self.engine_tasks_lock.notify_all()

    def wait_for_engine_drain(self, timeout: float) -> bool:
        """Wait for post-generation cache persistence to finish during a swap."""
        deadline = time.monotonic() + timeout
        with self.engine_tasks_lock:
            while self.engine_tasks:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self.engine_tasks_lock.wait(remaining)
        return True

    def runtime_snapshot(self) -> "tuple[Engine, PrefixCacheStore, str, PromptTokenCache, SharedHeadIndex]":
        """Read a coherent active runtime while a swap may be loading weights."""
        with self.swap_lock:
            return self.engine, self.store, self.model_id, self.token_cache, self.head_cache


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

    def __init__(self, max_entries: int = 128) -> None:
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


def validate_tools(tools: Any) -> Optional[str]:
    """Return a public validation error for malformed OpenAI tool schemas."""
    if tools is None:
        return None
    if not isinstance(tools, list):
        return "tools must be an array"
    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            return "each tool must be a function definition"
        function = tool.get("function")
        if not isinstance(function, dict) or not isinstance(function.get("name"), str) or not function["name"].strip():
            return "each tool function must have a non-empty name"
        parameters = function.get("parameters")
        if parameters is not None and not isinstance(parameters, dict):
            return "tool function parameters must be an object"
    return None


def _model_config_owners(model: Any) -> List[Any]:
    """Config-bearing objects for a model, outermost first.

    mlx-lm models expose ``model.args`` (a ModelArgs dataclass), not
    ``.config`` — reading only ``.config`` made every downstream helper a
    silent no-op on real models. Vision-wrapped checkpoints (Qwen3.5) nest
    the text fields another level down in ``text_config``, which may be a
    dict or a dataclass depending on where in mlx-lm it was materialized.
    """
    owners: List[Any] = []
    for attr in ("config", "args"):
        cfg = getattr(model, attr, None)
        if cfg is None:
            continue
        owners.append(cfg)
        text = cfg.get("text_config") if isinstance(cfg, dict) else getattr(cfg, "text_config", None)
        if text is not None:
            owners.append(text)
    return owners


def _cfg_get(owner: Any, *names: str) -> Optional[int]:
    for name in names:
        value = owner.get(name) if isinstance(owner, dict) else getattr(owner, name, None)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return None


def model_context_limit(model: Any) -> Optional[int]:
    """Best-effort context-window discovery across common MLX model configs."""
    for owner in _model_config_owners(model):
        value = _cfg_get(owner, "max_position_embeddings", "max_seq_len", "max_sequence_length")
        if value is not None:
            return value
    return None


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


def estimate_kv_cache_bytes(model: Any, tokens: int, kv_bits: Optional[int]) -> Optional[int]:
    """Conservatively estimate one sequence's target-model KV-cache footprint.

    Returning ``None`` means the architecture does not expose enough standard
    config fields; callers retain the existing reactive memory guard instead
    of making up an unsafe number.
    """
    if tokens < 1:
        return None
    for config in _model_config_owners(model):
        def get(*names: str) -> Optional[int]:
            return _cfg_get(config, *names)

        layers = get("num_hidden_layers", "n_layer", "num_layers")
        heads = get("num_key_value_heads", "num_attention_heads", "n_head")
        head_dim = get("head_dim")
        if head_dim is None:
            hidden = get("hidden_size", "n_embd", "dim")
            attention_heads = get("num_attention_heads", "n_head")
            if hidden is not None and attention_heads is not None and hidden % attention_heads == 0:
                head_dim = hidden // attention_heads
        if layers is None or heads is None or head_dim is None:
            continue
        # Hybrid architectures grow per-token KV only in their full-attention
        # layers; the recurrent (Gated-DeltaNet/SSM) layers hold small
        # constant state. Counting every layer would over-reserve ~4x on
        # Qwen3.5 and spuriously reject long prompts. Two config spellings:
        # an explicit per-layer type list (HF) or an interval (mlx-lm).
        layer_types = (
            config.get("layer_types") if isinstance(config, dict)
            else getattr(config, "layer_types", None)
        )
        if isinstance(layer_types, (list, tuple)) and layer_types:
            kv_layers = sum(
                1 for t in layer_types if isinstance(t, str) and "full" in t
            )
            if kv_layers == 0:
                return None  # pure-recurrent: constant state, nothing to reserve
            layers = kv_layers
        else:
            interval = get("full_attention_interval")
            if interval is not None and interval > 1:
                layers = max(1, layers // interval)
        # keys + values.  Quantized KV has scale/zero-point overhead, so
        # reserve 20% above the ideal packed size rather than relying on an
        # optimistic bit count.  Unquantized MLX cache entries are float16.
        bytes_per_value = 2.0 if kv_bits is None else (kv_bits / 8.0) * 1.2
        return int(layers * 2 * heads * head_dim * tokens * bytes_per_value)
    return None


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
    engine: Engine,
    store: PrefixCacheStore,
    model_id: str,
    *,
    max_pending_requests: int = 8,
    api_key: Optional[str] = None,
    token_cache_entries: int = 256,
    max_active_memory_bytes: Optional[int] = None,
    max_prompt_tokens: int = 65536,
    max_completion_tokens: int = 4096,
    requests_per_minute: int = 0,
    max_request_bytes: int = 2 * 1024 * 1024,
    shutdown_drain_seconds: float = 10.0,
    audit_log_path: Optional[str] = None,
    global_rps: float = 0.0,
    global_burst: int = 0,
    shutdown_timeout: Optional[float] = None,
    cors_origins: Optional[List[str]] = None,
    cors_allow_credentials: bool = False,
    model_context_tokens: Optional[int] = None,
    trusted_proxy_hosts: Optional[List[str]] = None,
    model_paths: Optional[dict[str, str]] = None,
    model_loader: Optional[Callable[[str], "tuple[Engine, PrefixCacheStore]"]] = None,
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
    drain_timeout = shutdown_timeout if shutdown_timeout is not None else shutdown_drain_seconds
    if drain_timeout <= 0:
        raise ValueError("shutdown_timeout must be positive")

    # Audit log ########################################################
    if audit_log_path:
        audit_logger.setup_audit_log(Path(audit_log_path))
        logger.info("audit log enabled at %s", audit_log_path)

    state = ServerState(
        engine=engine, store=store, model_id=model_id, lock=FifoLock(),
        max_pending_requests=max_pending_requests, api_key=api_key,
        metrics=ServerMetrics(), admission_lock=threading.Lock(),
        token_cache=PromptTokenCache(token_cache_entries),
        head_cache=SharedHeadIndex(),
        token_cache_entries=token_cache_entries,
        max_active_memory_bytes=max_active_memory_bytes,
        max_prompt_tokens=max_prompt_tokens,
        max_completion_tokens=max_completion_tokens,
        requests_per_minute=requests_per_minute,
        client_rate_limiters={},
        max_request_bytes=max_request_bytes,
        global_rate_limiter=GlobalRateLimiter(global_rps, global_burst) if global_rps > 0 else None,
        shutdown_timeout=drain_timeout,
        cors_origins=cors_origins or [],
        cors_allow_credentials=cors_allow_credentials,
        trusted_proxy_hosts=frozenset(trusted_proxy_hosts or []),
        model_loader=model_loader,
        model_context_override=model_context_tokens,
    )
    state.model_context_tokens = (
        model_context_tokens or model_context_limit(getattr(engine, "model", None))
    )
    # Register the default model so single-model mode is a subset of the
    # swap path: admission math and /v1/models just work.
    state.models[model_id] = (engine, store, state.token_cache, state.head_cache,
                              derive_model_profile(model_id, model_path=model_id))
    state.served_models.add(model_id)
    state.model_paths[model_id] = model_id
    # Swap candidates are deliberately paths only.  Loading happens after the
    # active model has been torn down, keeping residency bounded to one model.
    for spec_id, spec_path in (model_paths or {}).items():
        state.served_models.add(spec_id)
        state.model_paths[spec_id] = spec_path

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        with state.admission_lock:
            state.accepting = False
        # Drain in-flight generations before releasing the cache flock:
        # engine worker threads may still be writing snapshots to disk.
        deadline = time.monotonic() + state.shutdown_timeout
        while time.monotonic() < deadline:
            # Use in_flight_requests in addition to admitted_requests
            in_flight = state.in_flight_requests
            with state.admission_lock:
                admitted = state.admitted_requests
            if admitted == 0 and in_flight == 0:
                break
            await asyncio.sleep(0.1)
        else:
            logger.warning(
                "shutdown: %d request(s) still active after drain deadline",
                state.admitted_requests,
            )
        close = getattr(state.store, "close", None)
        if close:
            close()
        engine_close = getattr(state.engine, "close", None)
        if engine_close:
            engine_close()
        engine_shutdown = getattr(state.engine, "shutdown", None)
        if engine_shutdown:
            engine_shutdown()
        audit_logger.close()

    app = FastAPI(title="daedalus", lifespan=lifespan)
    app.state.daedalus = state

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
        state.metrics.inc_error(kind)
        return JSONResponse({"error": {"message": message, "type": kind}}, status_code=status_code)

    def authorized(authorization: Optional[str]) -> bool:
        return state.api_key is None or hmac.compare_digest(
            authorization or "", f"Bearer {state.api_key}"
        )

    @app.get("/health")
    def health(authorization: Optional[str] = Header(default=None)):
        # Liveness must stay probe-friendly (launchd/uptime checks can't
        # attach headers), but on a key-protected (LAN-exposed) server the
        # model identity and memory numbers are diagnostics, not liveness —
        # they're only included for authorized callers. Local unkeyed
        # servers keep the full body.
        body: dict = {"status": "ok"}
        if authorized(authorization):
            active_engine, _, active_model, _, _ = state.runtime_snapshot()
            body.update(
                model=active_model,
                thermal=active_engine.governor.effective_level.name,
                active_memory_bytes=getattr(active_engine, "active_memory_bytes", lambda: 0)(),
            )
        return body

    @app.get("/readyz")
    def readyz():
        with state.admission_lock:
            ready = state.accepting and state.admitted_requests < state.max_pending_requests
            pending = state.admitted_requests
        status = 200 if ready else 503
        return JSONResponse({"status": "ready" if ready else "busy", "pending_requests": pending,
                             "queue_depth": state.lock.queued,
                             "max_pending_requests": state.max_pending_requests}, status_code=status)

    def client_ip(request: Request) -> str:
        return request_client_ip(request, state.trusted_proxy_hosts)

    @app.get("/metrics")
    def metrics(request: Request, authorization: Optional[str] = Header(default=None)):
        # Usage/cache telemetry is operational data: when the server is
        # key-protected (i.e. exposed beyond localhost), require the key.
        # /health and /readyz stay open — they leak nothing and probes
        # (launchd, uptime checks) can't attach headers.
        if state.api_key is not None and not authorized(authorization):
            audit_logger.auth_failure(client_ip(request), reason="missing_api_key")
            return error("invalid API key", 401, "authentication_error")
        with state.admission_lock:
            active = state.admitted_requests
        active_engine, active_store, _, token_cache, head_cache = state.runtime_snapshot()
        return PlainTextResponse(state.metrics.render(active=active, limit=state.max_pending_requests,
            cache={**active_store.stats(), "tokenization": token_cache.stats(), "shared_head": head_cache.stats()}, thermal=active_engine.governor.effective_level.name), media_type="text/plain; version=0.0.4")

    @app.get("/v1/models")
    def models(request: Request, authorization: Optional[str] = Header(default=None)):
        if not authorized(authorization):
            audit_logger.auth_failure(client_ip(request), reason="missing_api_key")
            return error("invalid API key", 401, "authentication_error")
        _, _, active_model, _, _ = state.runtime_snapshot()
        return {
            "object": "list",
            "data": [
                {
                    "id": candidate,
                    "object": "model",
                    "owned_by": "daedalus",
                    "active": candidate == active_model,
                }
                for candidate in sorted(state.served_models)
            ],
        }

    @app.get("/v1/cache/stats")
    def cache_stats(request: Request, authorization: Optional[str] = Header(default=None)):
        if not authorized(authorization):
            audit_logger.auth_failure(client_ip(request), reason="missing_api_key")
            return error("invalid API key", 401, "authentication_error")
        _, active_store, _, _, _ = state.runtime_snapshot()
        return active_store.stats()

    @app.delete("/v1/cache")
    def clear_cache(request: Request, authorization: Optional[str] = Header(default=None)):
        if not authorized(authorization):
            audit_logger.auth_failure(client_ip(request), reason="missing_api_key")
            return error("invalid API key", 401, "authentication_error")
        # Admission also takes this lock, so a request cannot slip in after
        # the idle check but before the cache contents are removed.
        with state.maintenance_lock:
            with state.admission_lock:
                if state.admitted_requests:
                    return error("cache cannot be cleared while requests are active", 409, "conflict_error")
            removed = state.store.clear()
        state.metrics.inc_cache_admin("clear")
        audit_logger.cache_admin("clear", client_ip=client_ip(request))
        return {"removed_entries": removed}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request, authorization: str | None = Header(default=None)):
        if not authorized(authorization):
            source_ip = client_ip(request)
            audit_logger.auth_failure(source_ip)
            return error("invalid API key", 401, "authentication_error")
        # Check global rate limit first
        if not state.allow_global_rate():
            state.metrics.inc_request("rate_limited")
            return error("global request rate limit exceeded", 429, "rate_limit_error")
        # Bucket by client IP, not Authorization: with a shared bearer
        # token every LAN client would otherwise share a single bucket.
        client_key = client_ip(request)
        if not state.allow_client(client_key):
            state.metrics.inc_request("rate_limited")
            audit_logger.rate_limit_hit(
                client_key, policy="requests_per_minute",
                limit=state.requests_per_minute,
            )
            return error("request rate limit exceeded", 429, "rate_limit_error")
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                if int(content_length) > state.max_request_bytes:
                    return error("request body exceeds server limit", 413)
            except ValueError:
                return error("invalid Content-Length header", 400)
        try:
            body = await read_json_body(request, state.max_request_bytes)
        except RequestBodyTooLarge:
            audit_logger.request_rejected(client_key, reason="request_too_large")
            return error("request body exceeds server limit", 413)
        except ValueError as exc:
            return error(str(exc), 400)
        requested_model = body.get("model")
        if requested_model is not None:
            if not isinstance(requested_model, str):
                return error("model must be a string", 400)
        messages = body.get("messages", [])
        if not isinstance(messages, list) or not messages:
            return error("messages must be a non-empty array", 400)
        stream = bool(body.get("stream", False))
        raw_max_tokens = body.get(
            "max_tokens", body.get("max_completion_tokens", 2048)
        )
        if isinstance(raw_max_tokens, bool) or not isinstance(raw_max_tokens, int):
            return error("max_tokens must be an integer", 400)
        max_tokens = raw_max_tokens
        try:
            temperature = float(body.get("temperature", 0.7))
            top_p = float(body.get("top_p", 1.0))
            freq_p = float(body.get("frequency_penalty", 0.0))
            pres_p = float(body.get("presence_penalty", 0.0))
        except (TypeError, ValueError):
            return error("max_tokens, temperature, top_p, frequency_penalty, presence_penalty must be numeric", 400)
        if (max_tokens < 1 or not math.isfinite(temperature) or temperature < 0
                or not math.isfinite(top_p) or not 0 < top_p <= 1
                or not math.isfinite(freq_p) or not -2.0 <= freq_p <= 2.0
                or not math.isfinite(pres_p) or not -2.0 <= pres_p <= 2.0):
            return error("max_tokens must be positive, temperature non-negative, and top_p in (0, 1]", 400)
        if max_tokens > state.max_completion_tokens:
            return error(f"max_tokens exceeds server limit ({state.max_completion_tokens})", 400)
        tools = body.get("tools") if "tools" in body else None
        tools_error = validate_tools(tools)
        if tools_error:
            return error(tools_error, 400)
        # Keep downstream template handling simple while retaining validation
        # above (an explicitly supplied empty array is valid and means none).
        tools = tools or None
        if body.get("tool_choice") == "none":
            tools = None
            
        stop = body.get("stop")
        if isinstance(stop, str):
            stop = [stop]
        elif stop is None:
            stop = []
        elif not isinstance(stop, list) or not all(
            isinstance(value, str) and value for value in stop
        ):
            return error("stop must be a string or an array of non-empty strings", 400)

        # Validate cheap request fields before a potentially multi-second
        # model swap. This prevents malformed traffic from churning weights.
        if requested_model is not None and requested_model != state.model_id:
            if requested_model not in state.served_models:
                return error(f"model {requested_model!r} is not served", 404, "model_not_found")
            ok, msg = state.swap_model(requested_model)
            if not ok:
                return error(msg, 409, "model_swap_conflict")
        served_model_id = state.model_id

        try:
            tokens = build_prompt_tokens(state, messages, tools)
        except Exception as exc:
            # Surface template/format problems as a proper 400 instead of an
            # opaque 500 the client silently retries.
            logger.warning("prompt build failed: %s", exc)
            return error("prompt could not be templated", 400)
        if len(tokens) > state.max_prompt_tokens:
            return error(f"prompt exceeds server limit ({state.max_prompt_tokens} tokens)", 413)
        context_limit = state.model_context_tokens
        if context_limit is not None and len(tokens) + max_tokens > context_limit:
            return error(
                f"prompt plus completion exceeds model context limit ({context_limit} tokens)",
                400,
            )
        if not state.try_admit():
            state.metrics.inc_request("rejected")
            audit_logger.request_rejected(client_key, reason="queue_full",
                                          limit=state.max_pending_requests)
            return error("server queue is full; retry shortly", 429, "rate_limit_error")
        reserve_bytes = estimate_kv_cache_bytes(
            getattr(state.engine, "model", None),
            len(tokens) + max_tokens,
            getattr(getattr(state.engine, "config", None), "kv_bits", None),
        ) or 0
        if not state.memory_available(reserve_bytes):
            state.release()
            state.metrics.inc_request("memory_rejected")
            audit_logger.request_rejected(client_key, reason="memory_pressure")
            return error("insufficient free model memory; retry after active requests finish", 503, "server_overloaded_error")
        state.start_request()
        request_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())

        # Qwen3.5-style templates end the generation prompt inside an opened
        # <think> block; if so, the model's first output is reasoning.
        try:
            tail = state.engine.tokenizer.decode(tokens[-8:])
        except Exception:
            tail = ""
        prompt_in_think = tail.rstrip().endswith("<think>")

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

        logger.info(
            "→ %s · %d tok%s%s%s",
            request_id[-6:],
            len(tokens),
            f" · {len(tools)} tools" if tools else "",
            f" · head {head_boundary}" if head_boundary else "",
            " · stream" if stream else "",
        )

        gen = _Generation(
            state=state,
            tokens=tokens,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            tools=tools,
            prompt_in_think=prompt_in_think,
            rid=request_id[-6:],
            head_boundary=head_boundary,
            stop=stop,
            frequency_penalty=freq_p,
            presence_penalty=pres_p,
        )

        if stream:
            stream_options = body.get("stream_options")
            include_usage = isinstance(stream_options, dict) and bool(
                stream_options.get("include_usage")
            )

            def _stream_cleanup():
                gen.release_slot()
                state.finish_request()

            return StreamingResponse(
                _stream_response(
                    state, gen, request_id, created, request, include_usage, served_model_id
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
                # Runs after the response cycle ends on every path, including
                # a disconnect before the generator is first iterated.
                background=BackgroundTask(_stream_cleanup),
            )

        try:
            # Run the engine in a thread, but keep watching the connection:
            # a non-streaming client that gives up (OpenCode's 300s retry)
            # must abort the burn instead of wasting a full prefill+decode.
            engine_task = asyncio.create_task(
                asyncio.to_thread(gen.run_to_completion)
            )
            while True:
                done_set, _ = await asyncio.wait({engine_task}, timeout=1.0)
                if done_set:
                    break
                if await request.is_disconnected():
                    gen.aborted.set()
                    await engine_task  # engine exits at the next chunk/abort poll
                    state.metrics.inc_request("cancelled")
                    return JSONResponse(
                        {"error": {"message": "client disconnected", "type": "cancelled"}},
                        status_code=499,
                    )
            result = engine_task.result()
            state.metrics.inc_request("completed")
        except Exception:
            state.metrics.inc_request("failed")
            raise
        finally:
            state.finish_request()
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
        return JSONResponse(
            {
                "id": request_id,
                "object": "chat.completion",
                "created": created,
                "model": served_model_id,
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": result["finish_reason"],
                    }
                ],
                "usage": result["usage"],
            }
        )

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
        tokens,
        max_tokens,
        temperature,
        top_p,
        tools=None,
        prompt_in_think=False,
        rid="",
        head_boundary=None,
        stop=None,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    ):
        self.state = state
        self.tokens = tokens
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.tools = tools
        self.prompt_in_think = prompt_in_think
        self.rid = rid
        self.head_boundary = head_boundary
        self.stop = stop or []
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.events: "asyncio.Queue[dict]" = None  # set in stream()
        self.aborted = threading.Event()
        self.prefill_done = 0
        self.prefill_total = len(tokens)
        self.prefill_started = time.monotonic()
        self.cached_tokens = 0
        self._release_lock = threading.Lock()
        self._released = False

    def release_slot(self) -> None:
        """Idempotently release this request's admission slot.

        Called from the engine worker's ``finally``, the non-streaming
        handler, AND the StreamingResponse background task — the last one is
        the guarantee: if a client disconnects before the response generator
        is ever iterated, the worker never starts, and without this the slot
        would leak until the server rejected everything.
        """
        with self._release_lock:
            if self._released:
                return
            self._released = True
        self.state.release()

    def keepalive_line(self) -> str:
        done, total = self.prefill_done, self.prefill_total
        thermal = self.state.engine.governor.effective_level.name.lower()
        elapsed = time.monotonic() - self.prefill_started
        fresh = done - self.cached_tokens
        if fresh > 0 and elapsed > 0 and done < total:
            eta = (total - done) / (fresh / elapsed)
            return f": prefill {done}/{total} thermal={thermal} eta={eta:.0f}s\n\n"
        return f": prefill {done}/{total} thermal={thermal}\n\n"

    # ------------------------------------------------------------- sync path

    def run_to_completion(self) -> dict:
        text_parts: List[str] = []
        reasoning_parts: List[str] = []
        tool_calls: List[dict] = []
        finish_reason = "stop"
        usage = {}
        for event in self._run_engine():
            if event["type"] == "delta":
                text_parts.append(event["text"])
            elif event["type"] == "reasoning":
                reasoning_parts.append(event["text"])
            elif event["type"] == "tool_calls":
                tool_calls.extend(event["calls"])
            elif event["type"] == "done":
                finish_reason = event["finish_reason"]
                usage = event["usage"]
        return {
            "text": "".join(text_parts),
            "reasoning": "".join(reasoning_parts),
            "tool_calls": tool_calls,
            "finish_reason": "tool_calls" if tool_calls else finish_reason,
            "usage": usage,
        }

    # ----------------------------------------------------------- engine loop

    def _run_engine(self):
        """Sync generator of events: prefill progress, text deltas, done."""
        state = self.state
        t_queue_start = time.monotonic()
        if not state.lock.acquire(self.aborted):
            return
        state.metrics.observe_queue_wait(time.monotonic() - t_queue_start)
        # This counter includes the post-lock tail which persists pinned
        # snapshots.  A swap waits for it before closing the old cache store.
        state.start_engine_task()
        try:
            yield from self._run_engine_locked()
        finally:
            state.finish_engine_task()

    def _run_engine_locked(self):
        """Run a request after the FIFO engine slot has been acquired."""
        state = self.state
        t_start = time.monotonic()
        engine = state.engine
        store = state.store
        try:
            hit = store.fetch(self.tokens)
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
                prompt_cache = engine.make_cache()
                already = 0
                logger.info(
                    "  %s · cache miss — cold prefill %d tok",
                    self.rid,
                    len(self.tokens),
                )

            last_checkpoint = already
            last_checkpoint_time = time.monotonic()
            deferred_persist_tokens: Optional[List[int]] = None
            deferred_head_tokens: Optional[List[int]] = None
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
                        engine.governor.effective_level.name,
                    )
                    last_progress_log[0] = now

            def checkpoint_cb(done: int, cache: List[Any]) -> None:
                nonlocal last_checkpoint, last_checkpoint_time
                nonlocal deferred_persist_tokens, deferred_head_tokens
                if done >= len(self.tokens) - 1:
                    # End-of-prefill snapshot keyed by the prompt: for
                    # non-trimmable (hybrid) caches this is the only state
                    # the next stateless-client turn can reuse.
                    store.put(self.tokens[:done], cache, persist=False)
                    deferred_persist_tokens = self.tokens[:done]
                elif done == self.head_boundary and done > already:
                    # Shared-head snapshot: reused by NEW sessions/branches
                    # whose conversation diverges after the system prompt.
                    # RAM-only here (pinned); disk write is deferred past the
                    # response so it never sits inside TTFT.
                    store.put(self.tokens[:done], cache, persist=False)
                    deferred_head_tokens = self.tokens[:done]
                    logger.info(
                        "  %s · head snapshot at %d tok", self.rid, done
                    )
                elif (
                    len(self.tokens) - already >= CHECKPOINT_MIN_JOB_TOKENS
                    and done - last_checkpoint >= CHECKPOINT_EVERY_TOKENS
                    and time.monotonic() - last_checkpoint_time >= CHECKPOINT_MIN_INTERVAL_S
                    and len(self.tokens) - done > CHECKPOINT_EVERY_TOKENS
                ):
                    store.checkpoint(self.tokens, done, cache)
                    last_checkpoint = done
                    last_checkpoint_time = time.monotonic()

            think_filter = ThinkStreamFilter(
                initially_thinking=self.prompt_in_think
            )
            tool_filter = make_stream_filter(engine.tokenizer, self.tools)
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

            # Keep only the suffix that can still form a stop marker.  The
            # prior implementation retained every decoded character and
            # rescanned it per token, which turns long generations into an
            # avoidable O(n²) hot path.
            stop_suffix = ""
            max_stop_len = max((len(s) for s in self.stop), default=0)
            stop_matched = False
            try:
                # Only engage penalty processors when a client asks for them:
                # the default path stays byte-identical to plain sampling.
                penalty_kwargs = {}
                if self.frequency_penalty or self.presence_penalty:
                    from mlx_lm.sample_utils import make_logits_processors

                    penalty_kwargs["logits_processors"] = make_logits_processors(
                        frequency_penalty=self.frequency_penalty,
                        presence_penalty=self.presence_penalty,
                    )

                for resp in engine.generate(
                    self.tokens,
                    **penalty_kwargs,
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
                        if self.stop:
                            stop_suffix += resp.text
                            matches = [
                                stop_suffix.find(marker)
                                for marker in self.stop
                                if marker in stop_suffix
                            ]
                            if matches:
                                text_to_emit = stop_suffix[:min(matches)]
                                if text_to_emit:
                                    yield from route(text_to_emit)
                                stop_matched = True
                                finish_reason = "stop"
                                break
                            safe = max(0, len(stop_suffix) - max_stop_len + 1)
                            if safe:
                                yield from route(stop_suffix[:safe])
                                stop_suffix = stop_suffix[safe:]
                        else:
                            yield from route(resp.text)
                    n_generated = resp.generation_tokens
                    decode_tps = resp.generation_tps
                    if resp.finish_reason is not None:
                        finish_reason = resp.finish_reason
                if stop_suffix and not stop_matched and not self.aborted.is_set():
                    yield from route(stop_suffix)
            except PrefillAborted:
                logger.info(
                    "  %s · prefill aborted at %d tok (client gone)",
                    self.rid,
                    self.prefill_done,
                )
                # A head snapshot taken before the abort is still valuable —
                # persist it (rare path; lock hold acceptable) so its pin
                # is released and the work isn't lost.
                if deferred_head_tokens is not None:
                    persist = getattr(store, "persist", None)
                    if persist:
                        persist(deferred_head_tokens)
                return
        finally:
            state.lock.release()

        # ---- FifoLock released: engine/GPU work is done. Filter finalize,
        # the done event, and deferred disk writes must not make the next
        # queued request wait behind this one's tail work.
        reasoning, content = think_filter.finalize()
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
        if t_first_token is not None:
            state.metrics.observe_ttft(prefill_s)
        state.metrics.observe_decode_tps(decode_tps)
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
            engine.governor.effective_level.name,
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

        # Deferred snapshot persists land after the client already has its
        # final chunk: outside the FifoLock, off the TTFT path, still on
        # this engine worker thread (MLX serialization is stream-bound).
        persist = getattr(store, "persist", None)
        if persist:
            for toks in (deferred_head_tokens, deferred_persist_tokens):
                if toks is not None:
                    persist(toks)


async def _stream_response(
    state: ServerState,
    gen: _Generation,
    request_id: str,
    created: int,
    request: Request,
    include_usage: bool = False,
    model: Optional[str] = None,
) -> AsyncGenerator[str, None]:
    model = model or state.model_id
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
                # OpenAI spec: with stream_options.include_usage the usage
                # rides a trailing chunk whose choices array is empty. That
                # empty array breaks clients that index choices[0] unguarded
                # (OpenCode among them), so the legacy default keeps usage on
                # the finish_reason chunk and never emits empty choices.
                yield _sse(
                    _chunk(
                        request_id,
                        model,
                        created,
                        {},
                        finish_reason=event["finish_reason"],
                        usage=None if include_usage else event["usage"],
                    )
                )
                if include_usage:
                    yield _sse(
                        {
                            "id": request_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [],
                            "usage": event["usage"],
                        }
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
