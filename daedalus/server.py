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
from dataclasses import dataclass
from typing import Any, AsyncGenerator, List, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from daedalus.cache.store import PrefixCacheStore
from daedalus.engine import Engine, PrefillAborted
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
    return state.engine.tokenizer.apply_chat_template(
        normalize_messages(messages), **kwargs
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
    engine: Engine, store: PrefixCacheStore, model_id: str
) -> FastAPI:
    app = FastAPI(title="daedalus")
    state = ServerState(
        engine=engine, store=store, model_id=model_id, lock=threading.Lock()
    )
    app.state.daedalus = state

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "model": state.model_id,
            "thermal": state.engine.governor.effective_level.name,
        }

    @app.get("/v1/models")
    def models():
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
    def cache_stats():
        return state.store.stats()

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        body = await request.json()
        messages = body.get("messages", [])
        if not messages:
            return JSONResponse(
                {"error": {"message": "messages required", "type": "invalid_request_error"}},
                status_code=400,
            )
        stream = bool(body.get("stream", False))
        max_tokens = int(
            body.get("max_tokens") or body.get("max_completion_tokens") or 2048
        )
        temperature = float(body.get("temperature", 0.7))
        top_p = float(body.get("top_p", 1.0))
        tools = body.get("tools") or None
        if body.get("tool_choice") == "none":
            tools = None

        try:
            tokens = build_prompt_tokens(state, messages, tools)
        except Exception as exc:
            # Surface template/format problems as a proper 400 instead of an
            # opaque 500 the client silently retries.
            logger.warning("prompt build failed: %s", exc)
            return JSONResponse(
                {
                    "error": {
                        "message": f"prompt could not be templated: {exc}",
                        "type": "invalid_request_error",
                    }
                },
                status_code=400,
            )
        request_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())

        gen = _Generation(
            state=state,
            tokens=tokens,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            tools=tools,
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

        result = await asyncio.to_thread(gen.run_to_completion)
        message: dict = {"role": "assistant", "content": result["text"] or None}
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
        self, state: ServerState, tokens, max_tokens, temperature, top_p, tools=None
    ):
        self.state = state
        self.tokens = tokens
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.tools = tools
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
        tool_calls: List[dict] = []
        finish_reason = "stop"
        usage = {}
        for event in self._run_engine():
            if event["type"] == "delta":
                text_parts.append(event["text"])
            elif event["type"] == "tool_calls":
                tool_calls.extend(event["calls"])
            elif event["type"] == "done":
                finish_reason = event["finish_reason"]
                usage = event["usage"]
        return {
            "text": "".join(text_parts),
            "tool_calls": tool_calls,
            "finish_reason": "tool_calls" if tool_calls else finish_reason,
            "usage": usage,
        }

    # ----------------------------------------------------------- engine loop

    def _run_engine(self):
        """Sync generator of events: prefill progress, text deltas, done."""
        state = self.state
        with state.lock:
            hit = state.store.fetch(self.tokens)
            if hit is not None:
                prompt_cache = hit.cache
                already = hit.matched_tokens
                self.cached_tokens = already
                logger.info(
                    "cache hit: %d/%d tokens (%s)",
                    already,
                    len(self.tokens),
                    hit.source,
                )
            else:
                prompt_cache = state.engine.make_cache()
                already = 0

            last_checkpoint = already

            def checkpoint_cb(done: int, cache: List[Any]) -> None:
                nonlocal last_checkpoint
                self.prefill_done = done
                if done >= len(self.tokens) - 1:
                    # End-of-prefill snapshot keyed by the prompt: for
                    # non-trimmable (hybrid) caches this is the only state
                    # the next stateless-client turn can reuse.
                    state.store.put(self.tokens[:done], cache)
                elif done - last_checkpoint >= CHECKPOINT_EVERY_TOKENS:
                    state.store.checkpoint(self.tokens, done, cache)
                    last_checkpoint = done

            tool_filter = make_stream_filter(state.engine.tokenizer, self.tools)
            call_index = 0
            emitted_calls = False
            finish_reason = "stop"
            n_generated = 0
            try:
                for resp in state.engine.generate(
                    self.tokens,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    prompt_cache=prompt_cache,
                    already_cached=already,
                    checkpoint_cb=checkpoint_cb,
                    should_abort=self.aborted.is_set,
                ):
                    if self.aborted.is_set():
                        finish_reason = "abort"
                        break
                    if resp.text:
                        content, calls = tool_filter.feed(resp.text)
                        if content:
                            yield {"type": "delta", "text": content}
                        if calls:
                            emitted_calls = True
                            yield {
                                "type": "tool_calls",
                                "calls": [
                                    c.as_openai(call_index + i)
                                    for i, c in enumerate(calls)
                                ],
                            }
                            call_index += len(calls)
                    n_generated = resp.generation_tokens
                    if resp.finish_reason is not None:
                        finish_reason = resp.finish_reason
            except PrefillAborted:
                logger.info("prefill aborted at %d tokens", self.prefill_done)
                return

            content, calls = tool_filter.finalize()
            if content:
                yield {"type": "delta", "text": content}
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
        try:
            for event in gen._run_engine():
                loop.call_soon_threadsafe(queue.put_nowait, event)
        except Exception as exc:  # surface engine errors to the client
            logger.exception("engine error")
            loop.call_soon_threadsafe(
                queue.put_nowait, {"type": "error", "message": str(exc)}
            )
        finally:
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
