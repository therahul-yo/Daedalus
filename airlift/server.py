"""OpenAI-compatible server for airlift.

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

from airlift.cache.store import PrefixCacheStore
from airlift.engine import Engine, PrefillAborted

logger = logging.getLogger(__name__)

KEEPALIVE_INTERVAL_S = 1.0
CHECKPOINT_EVERY_TOKENS = 4096


@dataclass
class ServerState:
    engine: Engine
    store: PrefixCacheStore
    model_id: str
    lock: threading.Lock


def build_prompt_tokens(state: ServerState, messages: List[dict]) -> List[int]:
    return state.engine.tokenizer.apply_chat_template(
        messages, add_generation_prompt=True
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
    app = FastAPI(title="airlift")
    state = ServerState(
        engine=engine, store=store, model_id=model_id, lock=threading.Lock()
    )
    app.state.airlift = state

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
                    "owned_by": "airlift",
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

        tokens = build_prompt_tokens(state, messages)
        request_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())

        gen = _Generation(
            state=state,
            tokens=tokens,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
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
        return JSONResponse(
            {
                "id": request_id,
                "object": "chat.completion",
                "created": created,
                "model": state.model_id,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": result["text"]},
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

    def __init__(self, state: ServerState, tokens, max_tokens, temperature, top_p):
        self.state = state
        self.tokens = tokens
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.events: "asyncio.Queue[dict]" = None  # set in stream()
        self.aborted = threading.Event()
        self.prefill_done = 0
        self.prefill_total = len(tokens)
        self.cached_tokens = 0

    # ------------------------------------------------------------- sync path

    def run_to_completion(self) -> dict:
        text_parts: List[str] = []
        finish_reason = "stop"
        usage = {}
        for event in self._run_engine():
            if event["type"] == "delta":
                text_parts.append(event["text"])
            elif event["type"] == "done":
                finish_reason = event["finish_reason"]
                usage = event["usage"]
        return {
            "text": "".join(text_parts),
            "finish_reason": finish_reason,
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
                        yield {"type": "delta", "text": resp.text}
                    n_generated = resp.generation_tokens
                    if resp.finish_reason is not None:
                        finish_reason = resp.finish_reason
            except PrefillAborted:
                logger.info("prefill aborted at %d tokens", self.prefill_done)
                return

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
                yield f": prefill {gen.prefill_done}/{gen.prefill_total}\n\n"
                continue

            if event["type"] == "delta":
                yield _sse(
                    _chunk(request_id, model, created, {"content": event["text"]})
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
