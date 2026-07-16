"""Server runtime state: admission, rate limiting, and the model hot-swap.

``ServerState`` holds the resident engine/store/caches plus every lock and
counter that serializes admission, memory guarding, drain, and the swap-only
multi-model lifecycle. ``PromptTokenCache`` and ``SharedHeadIndex`` are the
small LRUs that make stateless-agent turns cheap.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import psutil

from daedalus.cache.store import PrefixCacheStore
from daedalus.engine import Engine
from daedalus.metrics import ServerMetrics
from daedalus.scheduler import FifoLock

from daedalus import audit as audit_logger

from daedalus.server.http_utils import ClientRateLimiter, GlobalRateLimiter
from daedalus.server.profiles import (
    ModelProfile,
    derive_model_profile,
    model_context_limit,
    model_fits,
)
from daedalus.server.prompts import normalize_messages

logger = logging.getLogger(__name__)


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
    swap_cooldown_seconds: float = 30.0
    _last_swap_time: float = 0.0
    swap_lock: threading.Lock = field(default_factory=threading.Lock)
    # Monotonic counter bumped under swap_lock on every successful swap.  A
    # request captures it at admission (tokenization); if it changed by the
    # time the request reaches the engine, the tokens were built with the old
    # tokenizer and must be rebuilt against the resident model.
    swap_epoch: int = 0
    # Admission context window.  Recomputed on every swap so validation tracks
    # the resident model, not the one create_app() started with.  The override
    # (an explicit model_context_tokens argument) always wins when set.
    context_limit: Optional[int] = None
    context_limit_override: Optional[int] = None
    # Degraded mode: set when a swap fails to load the target AND fails to
    # restore the previous model, leaving no resident engine.
    degraded: bool = False
    degraded_reason: Optional[str] = None
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
                    # Full detail (with traceback) goes to the server log only;
                    # the client-visible message carries the class name at most.
                    logger.exception("model swap: loading %r failed", model_id)
                    # A failed target load must not strand the service without
                    # its prior model; restore it before reporting the error.
                    try:
                        eng, store = self.model_loader(old_id)
                    except Exception:
                        # Double failure: neither the target nor the previous
                        # model loads — there is no resident engine. Enter
                        # degraded mode so probes/clients get an honest 503
                        # instead of misleading 400s against a null engine.
                        logger.critical(
                            "model swap: reload of %r also failed after %r failed; "
                            "entering degraded mode", old_id, model_id,
                        )
                        self.degraded = True
                        self.degraded_reason = f"model load failed: {type(exc).__name__}"
                        with self.admission_lock:
                            self.accepting = False
                        audit_logger._emit(
                            "model_swap_degraded", from_model=old_id,
                            to_model=model_id, reason=type(exc).__name__,
                        )
                        return False, f"could not load {model_id!r}: {type(exc).__name__}"
                    self.engine, self.store = eng, store
                    self.token_cache = PromptTokenCache(self.token_cache_entries)
                    self.head_cache = SharedHeadIndex()
                    self.context_limit = self.context_limit_override or model_context_limit(
                        getattr(eng, "model", None)
                    )
                    self.models[old_id] = (eng, store, self.token_cache, self.head_cache,
                                           derive_model_profile(old_id, self.model_paths.get(old_id, old_id)))
                    return False, f"could not load {model_id!r}: {type(exc).__name__}"
                self.engine, self.store = eng, store
                self.token_cache = PromptTokenCache(self.token_cache_entries)
                self.head_cache = SharedHeadIndex()
                self.model_id = model_id
                # Admission must validate against the NEW model's window.
                self.context_limit = self.context_limit_override or model_context_limit(
                    getattr(eng, "model", None)
                )
                self.models[model_id] = (eng, store, self.token_cache, self.head_cache, profile)
                # In-flight requests captured the pre-swap epoch; bump it so
                # they re-tokenize against this model's tokenizer.
                self.swap_epoch += 1
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
        if psutil.virtual_memory().available < 1024 * 1024 * 512:
            trim = getattr(self.store, "trim_ram", None)
            if trim:
                trim(0)
            if psutil.virtual_memory().available < 1024 * 1024 * 512:
                return False

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
