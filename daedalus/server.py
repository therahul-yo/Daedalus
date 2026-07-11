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
import json
import logging
import threading
import time
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncGenerator, List, Optional

from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

from daedalus.cache.store import PrefixCacheStore
from daedalus.engine import Engine, PrefillAborted
from daedalus.metrics import ServerMetrics
from daedalus.reasoning import ThinkStreamFilter
from daedalus.tools import make_stream_filter

logger = logging.getLogger(__name__)

KEEPALIVE_INTERVAL_S = 1.0
CHECKPOINT_EVERY_TOKENS = 4096


@dataclass
class ServerState:
    engine: Engine
    store: PrefixCacheStore
    model_id: str
    lock: threading.Lock
    max_pending_requests: int
    api_key: Optional[str]
    metrics: ServerMetrics
    admission_lock: threading.Lock
    admitted_requests: int = 0
    accepting: bool = True
    token_cache: "PromptTokenCache" = None

    def try_admit(self) -> bool:
        with self.admission_lock:
            if not self.accepting or self.admitted_requests >= self.max_pending_requests:
                return False
            self.admitted_requests += 1
            return True

    def release(self) -> None:
        with self.admission_lock:
            self.admitted_requests = max(0, self.admitted_requests - 1)


class PromptTokenCache:
    """Tiny LRU for repeated stateless-agent chat-template rendering."""

    def __init__(self, max_entries: int = 256) -> None:
        self.max_entries = max_entries
        self._entries: OrderedDict[str, tuple[int, ...]] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

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
            self._entries[key] = tuple(tokens)
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)
        return tokens

    def stats(self) -> dict:
        with self._lock:
            return {"entries": len(self._entries), "hits": self.hits, "misses": self.misses}


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
    engine: Engine,
    store: PrefixCacheStore,
    model_id: str,
    *,
    max_pending_requests: int = 8,
    api_key: Optional[str] = None,
    token_cache_entries: int = 256,
) -> FastAPI:
    if max_pending_requests < 1:
        raise ValueError("max_pending_requests must be at least 1")
    if token_cache_entries < 1:
        raise ValueError("token_cache_entries must be at least 1")
    state = ServerState(
        engine=engine, store=store, model_id=model_id, lock=threading.Lock(),
        max_pending_requests=max_pending_requests, api_key=api_key,
        metrics=ServerMetrics(), admission_lock=threading.Lock(),
        token_cache=PromptTokenCache(token_cache_entries),
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        with state.admission_lock:
            state.accepting = False
        close = getattr(state.store, "close", None)
        if close:
            close()

    app = FastAPI(title="daedalus", lifespan=lifespan)
    app.state.daedalus = state

    def error(message: str, status_code: int, kind: str = "invalid_request_error"):
        state.metrics.inc_error(kind)
        return JSONResponse({"error": {"message": message, "type": kind}}, status_code=status_code)

    def authorized(authorization: Optional[str]) -> bool:
        return state.api_key is None or authorization == f"Bearer {state.api_key}"

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "model": state.model_id,
            "thermal": state.engine.governor.effective_level.name,
        }

    @app.get("/readyz")
    def readyz():
        with state.admission_lock:
            ready = state.accepting and state.admitted_requests < state.max_pending_requests
            pending = state.admitted_requests
        status = 200 if ready else 503
        return JSONResponse({"status": "ready" if ready else "busy", "pending_requests": pending,
                             "max_pending_requests": state.max_pending_requests}, status_code=status)

    @app.get("/metrics")
    def metrics():
        with state.admission_lock:
            active = state.admitted_requests
        return PlainTextResponse(state.metrics.render(active=active, limit=state.max_pending_requests,
            cache={**state.store.stats(), "tokenization": state.token_cache.stats()}, thermal=state.engine.governor.effective_level.name), media_type="text/plain; version=0.0.4")

    @app.get("/v1/models")
    def models(authorization: Optional[str] = Header(default=None)):
        if not authorized(authorization):
            return error("invalid API key", 401, "authentication_error")
        return {
            "object": "list",
            "data": [
                {
                    "id": state.model_id,
                    "object": "model",
                    "owned_by": "daedalus",
                }
            ],
        }

    @app.get("/v1/cache/stats")
    def cache_stats(authorization: Optional[str] = Header(default=None)):
        if not authorized(authorization):
            return error("invalid API key", 401, "authentication_error")
        return state.store.stats()

    @app.delete("/v1/cache")
    def clear_cache(authorization: Optional[str] = Header(default=None)):
        if not authorized(authorization):
            return error("invalid API key", 401, "authentication_error")
        if state.admitted_requests:
            return error("cache cannot be cleared while requests are active", 409, "conflict_error")
        removed = state.store.clear()
        state.metrics.inc_cache_admin("clear")
        return {"removed_entries": removed}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request, authorization: Optional[str] = Header(default=None)):
        if not authorized(authorization):
            return error("invalid API key", 401, "authentication_error")
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
        tools = body.get("tools") or None
        if body.get("tool_choice") == "none":
            tools = None

        try:
            tokens = build_prompt_tokens(state, messages, tools)
        except Exception as exc:
            # Surface template/format problems as a proper 400 instead of an
            # opaque 500 the client silently retries.
            logger.warning("prompt build failed: %s", exc)
            return error(f"prompt could not be templated: {exc}", 400)
        if not state.try_admit():
            state.metrics.inc_request("rejected")
            return error("server queue is full; retry shortly", 429, "rate_limit_error")
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
        head_boundary = None
        try:
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
        )

        if stream:
            return StreamingResponse(
                _stream_response(state, gen, request_id, created, request),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )

        try:
            result = await asyncio.to_thread(gen.run_to_completion)
            state.metrics.inc_request("completed")
        except Exception:
            state.metrics.inc_request("failed")
            raise
        finally:
            state.release()
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
                "model": state.model_id,
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
        self.events: "asyncio.Queue[dict]" = None  # set in stream()
        self.aborted = threading.Event()
        self.prefill_done = 0
        self.prefill_total = len(tokens)
        self.prefill_started = time.monotonic()
        self.cached_tokens = 0

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
        t_start = time.monotonic()
        with state.lock:
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
                nonlocal last_checkpoint, deferred_persist_tokens
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
                elif done - last_checkpoint >= CHECKPOINT_EVERY_TOKENS:
                    state.store.checkpoint(self.tokens, done, cache)
                    last_checkpoint = done

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

            try:
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
                        yield from output_events
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


async def _stream_response(
    state: ServerState,
    gen: _Generation,
    request_id: str,
    created: int,
    request: Request,
) -> AsyncGenerator[str, None]:
    model = state.model_id
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def worker():
        completed = False
        try:
            for event in gen._run_engine():
                completed = completed or event["type"] == "done"
                loop.call_soon_threadsafe(queue.put_nowait, event)
        except Exception as exc:  # surface engine errors to the client
            logger.exception("engine error")
            loop.call_soon_threadsafe(
                queue.put_nowait, {"type": "error", "message": str(exc)}
            )
        finally:
            state.metrics.inc_request("completed" if completed else "cancelled")
            state.release()
            loop.call_soon_threadsafe(queue.put_nowait, {"type": "eof"})

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
        gen.aborted.set()
    yield "data: [DONE]\n\n"
