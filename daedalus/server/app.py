"""FastAPI application factory wiring the OpenAI-compatible endpoints.

``create_app`` builds the ``ServerState``, installs middleware, and registers
every endpoint (/v1/chat/completions, /v1/models, /health, /readyz, /metrics,
/v1/cache/*) — delegating memory/profile math, prompt building, and the
generation loop to the sibling modules.
"""

from __future__ import annotations

import asyncio
import hmac
import logging
import math
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable, List, Optional

from fastapi import FastAPI, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool

from daedalus.cache.store import PrefixCacheStore
from daedalus.engine import Engine
from daedalus.metrics import ServerMetrics
from daedalus.scheduler import FifoLock

from daedalus import audit as audit_logger

from daedalus.server.generation import _Generation, _stream_response
from daedalus.server.http_utils import (
    GlobalRateLimiter,
    RequestBodyTooLarge,
    RequestIdMiddleware,
    read_json_body,
    request_client_ip,
)
from daedalus.server.profiles import (
    derive_model_profile,
    estimate_kv_cache_bytes,
    model_context_limit,
)
from daedalus.server.prompts import build_prompt_tokens, validate_tools
from daedalus.server.state import PromptTokenCache, ServerState, SharedHeadIndex

logger = logging.getLogger(__name__)


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
    )
    # Admission context window lives on state (not a closure local) so a swap
    # can recompute it; the explicit override always wins when supplied.
    state.context_limit_override = model_context_tokens
    state.context_limit = model_context_tokens or model_context_limit(getattr(engine, "model", None))
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
        active_engine, _, active_model, _, _ = state.runtime_snapshot()
        degraded = state.degraded or active_engine is None
        body: dict = {"status": "degraded" if degraded else "ok"}
        if authorized(authorization):
            body["model"] = active_model
            # engine can be None in degraded mode; never dereference it blindly.
            if active_engine is not None:
                body["thermal"] = active_engine.governor.effective_level.name
                body["active_memory_bytes"] = getattr(active_engine, "active_memory_bytes", lambda: 0)()
            if degraded and state.degraded_reason:
                body["degraded_reason"] = state.degraded_reason
        if degraded:
            return JSONResponse(body, status_code=503)
        return body

    @app.get("/readyz")
    def readyz():
        with state.admission_lock:
            ready = state.accepting and state.admitted_requests < state.max_pending_requests
            pending = state.admitted_requests
        if state.degraded or state.engine is None:
            ready = False
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
        active_engine, active_store, _, token_cache, head_cache = state.runtime_snapshot()
        if state.degraded or active_engine is None or active_store is None:
            return error("no model resident", 503, "server_error")
        with state.admission_lock:
            active = state.admitted_requests
        return PlainTextResponse(state.metrics.render(active=active, limit=state.max_pending_requests,
            cache={**active_store.stats(), "tokenization": token_cache.stats(), "shared_head": head_cache.stats()}, thermal=active_engine.governor.effective_level.name), media_type="text/plain; version=0.0.4")

    @app.get("/v1/models")
    def models(request: Request, authorization: Optional[str] = Header(default=None)):
        if not authorized(authorization):
            audit_logger.auth_failure(client_ip(request), reason="missing_api_key")
            return error("invalid API key", 401, "authentication_error")
        # Every swap-eligible model, resident one first then the rest sorted.
        _, _, resident, _, _ = state.runtime_snapshot()
        ordered = [resident] + sorted(m for m in state.served_models if m != resident)
        return {
            "object": "list",
            "data": [
                {
                    "id": mid,
                    "object": "model",
                    "owned_by": "daedalus",
                    "resident": mid == resident,
                }
                for mid in ordered
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
        # Degraded mode (a swap left no resident engine): fail fast and honest
        # rather than dereferencing a null engine into a misleading 400.
        if state.degraded or state.engine is None:
            return error("no model resident", 503, "server_error")
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
        needs_swap = False
        if requested_model is not None:
            if not isinstance(requested_model, str):
                return error("model must be a string", 400)
            if requested_model != state.model_id:
                # Unknown model -> 404 (cheap, stays early). A registered but
                # non-resident model defers its hot-swap until AFTER structural
                # validation so an invalid request can neither trigger a swap
                # nor burn the 30s swap cooldown.
                if requested_model not in state.served_models:
                    return error(f"model {requested_model!r} is not served", 404, "model_not_found")
                needs_swap = True
        messages = body.get("messages", [])
        if not isinstance(messages, list) or not messages:
            return error("messages must be a non-empty array", 400)
        # OpenAI SDKs send explicit JSON null for unset optional fields; treat
        # a null exactly like an absent key (fall through to the default).
        def _or_default(value: Any, default: Any) -> Any:
            return default if value is None else value

        stream = bool(_or_default(body.get("stream"), False))
        raw_max_tokens = body.get("max_tokens")
        if raw_max_tokens is None:
            raw_max_tokens = body.get("max_completion_tokens")
        if raw_max_tokens is None:
            raw_max_tokens = 2048
        if isinstance(raw_max_tokens, bool) or not isinstance(raw_max_tokens, int):
            return error("max_tokens must be an integer", 400)
        max_tokens = raw_max_tokens
        try:
            temperature = float(_or_default(body.get("temperature"), 0.7))
            top_p = float(_or_default(body.get("top_p"), 1.0))
            freq_p = float(_or_default(body.get("frequency_penalty"), 0.0))
            pres_p = float(_or_default(body.get("presence_penalty"), 0.0))
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

        # Structural validation has passed: only now may we hot-swap. The swap
        # is a 30-60s blocking model load, so it runs off the asyncio event
        # loop (otherwise it freezes every concurrent SSE stream/keepalive).
        if needs_swap:
            ok, msg = await run_in_threadpool(state.swap_model, requested_model)
            if not ok:
                if state.degraded:
                    return error("no model resident (model load failed)", 503, "server_error")
                # Over budget / cooldown -> 409 with detail (not 404).
                return error(msg, 409, "model_swap_conflict")
            # On success state.model_id is now requested_model; continue with
            # the target model's tokenizer and context window below.

        # The model actually serving this request (after any swap, or the
        # resident one when the swap was skipped). Used for the response
        # "model" field so it never reports a model a concurrent swap changed.
        served_model_id = state.model_id

        # Capture the swap epoch alongside tokenization: if a concurrent swap
        # bumps it before this request reaches the engine, the tokens below
        # were built with the wrong tokenizer and get rebuilt (see _Generation).
        admission_epoch = state.swap_epoch
        try:
            tokens = build_prompt_tokens(state, messages, tools)
        except Exception as exc:
            # Surface template/format problems as a proper 400 instead of an
            # opaque 500 the client silently retries.
            logger.warning("prompt build failed: %s", exc)
            return error("prompt could not be templated", 400)
        if len(tokens) > state.max_prompt_tokens:
            return error(f"prompt exceeds server limit ({state.max_prompt_tokens} tokens)", 413)
        if state.context_limit is not None and len(tokens) + max_tokens > state.context_limit:
            return error(
                f"prompt plus completion exceeds model context limit ({state.context_limit} tokens)",
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
            messages=messages,
            captured_epoch=admission_epoch,
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
        except _Generation._SwapConflict as exc:
            # A swap landed between admission and the engine slot and the
            # re-tokenized prompt no longer fits: retryable client error.
            state.metrics.inc_request("failed")
            return error(str(exc), 409, "model_swap_conflict")
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
