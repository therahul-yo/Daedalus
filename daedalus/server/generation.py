"""Per-request generation: the engine worker and the SSE streaming pump.

``_Generation`` runs one request on a worker thread (MLX off the event loop),
handling cache fetch/checkpointing, cross-swap re-tokenization, think/tool
stream filtering, and stop sequences; ``_stream_response`` drains its events
onto an SSE response with keepalives and backpressure.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import threading
import time
from typing import TYPE_CHECKING, Any, AsyncGenerator, List, Optional

from fastapi import Request

from daedalus.engine import PrefillAborted
from daedalus.reasoning import ThinkStreamFilter
from daedalus.tools import make_stream_filter

from daedalus.server.http_utils import _chunk, _sse
from daedalus.server.prompts import build_prompt_tokens

if TYPE_CHECKING:
    from daedalus.server.state import ServerState

logger = logging.getLogger(__name__)

KEEPALIVE_INTERVAL_S = 1.0
CHECKPOINT_EVERY_TOKENS = 4096
CHECKPOINT_MIN_JOB_TOKENS = 8192
CHECKPOINT_MIN_INTERVAL_S = 8.0


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
        messages=None,
        captured_epoch=0,
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
        # Kept so the engine loop can re-tokenize if a swap lands between
        # admission and the engine slot (the captured epoch no longer matches).
        self.messages = messages
        self.captured_epoch = captured_epoch
        self.events: "asyncio.Queue[dict]" = None  # set in stream()
        self.aborted = threading.Event()
        self.prefill_done = 0
        self.prefill_total = len(tokens)
        self.prefill_started = time.monotonic()
        self.cached_tokens = 0
        self._release_lock = threading.Lock()
        self._released = False
        # Deferred (pinned) cache snapshots recorded during prefill; finalized
        # (persisted + unpinned) on every exit so pinned RAM never leaks.
        self._deferred_head_tokens: Optional[List[int]] = None
        self._deferred_persist_tokens: Optional[List[int]] = None
        self._pins_finalized = False
        self._pin_lock = threading.Lock()

    class _SwapConflict(Exception):
        """Re-tokenized prompt no longer fits after a mid-flight swap."""

    def finalize_deferred(self) -> None:
        """Persist (and thereby unpin) deferred snapshots exactly once.

        Runs on every exit path — success, engine exception, and client
        disconnect — so the RAM-only pinned entries created by
        ``checkpoint_cb`` can never grow un-evictably.
        """
        with self._pin_lock:
            if self._pins_finalized:
                return
            self._pins_finalized = True
            head, full = self._deferred_head_tokens, self._deferred_persist_tokens
        persist = getattr(self.state.store, "persist", None)
        if persist is None:
            return
        for toks in (head, full):
            if toks is not None:
                try:
                    persist(toks)
                except Exception:
                    logger.exception("deferred snapshot persist failed")

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
            # Persist/unpin any deferred snapshots before releasing the swap
            # drain barrier: still holding an engine task keeps the store this
            # request generated on from being torn down under us.
            self.finalize_deferred()
            state.finish_engine_task()

    def _run_engine_locked(self):
        """Run a request after the FIFO engine slot has been acquired."""
        state = self.state
        t_start = time.monotonic()
        engine = state.engine
        store = state.store
        try:
            # Cross-tokenizer guard: a swap between admission (where these tokens
            # were built) and now would run OLD-tokenizer ids on the NEW model —
            # silent garbage.  Re-tokenize with the resident tokenizer and re-check
            # the prompt/context limits; a clean 400-style error beats corruption.
            # NOTE: this must live INSIDE the try whose finally releases
            # state.lock — raising _SwapConflict before the try would leak the
            # FIFO lock and deadlock the engine.
            if state.swap_epoch != self.captured_epoch and self.messages is not None:
                try:
                    self.tokens = build_prompt_tokens(state, self.messages, self.tools)
                except Exception as exc:
                    logger.warning("re-tokenize after swap failed: %s", exc)
                    raise self._SwapConflict("prompt could not be re-templated after a model swap") from exc
                self.captured_epoch = state.swap_epoch
                self.prefill_total = len(self.tokens)
                if len(self.tokens) > state.max_prompt_tokens or (
                    state.context_limit is not None
                    and len(self.tokens) + self.max_tokens > state.context_limit
                ):
                    raise self._SwapConflict(
                        "prompt no longer fits the resident model's context after a model swap"
                    )
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
                if done >= len(self.tokens) - 1:
                    # End-of-prefill snapshot keyed by the prompt: for
                    # non-trimmable (hybrid) caches this is the only state
                    # the next stateless-client turn can reuse.
                    store.put(self.tokens[:done], cache, persist=False)
                    self._deferred_persist_tokens = self.tokens[:done]
                elif done == self.head_boundary and done > already:
                    # Shared-head snapshot: reused by NEW sessions/branches
                    # whose conversation diverges after the system prompt.
                    # RAM-only here (pinned); disk write is deferred past the
                    # response so it never sits inside TTFT.
                    store.put(self.tokens[:done], cache, persist=False)
                    self._deferred_head_tokens = self.tokens[:done]
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
                # A head snapshot taken before the abort is still valuable and
                # its pin is released by finalize_deferred (runs on every exit).
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
        # final chunk: outside the FifoLock, off the TTFT path, still on this
        # engine worker thread (MLX serialization is stream-bound). Routed
        # through finalize_deferred so success, exception, and disconnect all
        # take the same unpin path exactly once.
        self.finalize_deferred()


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
        engine_gen = gen._run_engine()
        try:
            for event in engine_gen:
                completed = completed or event["type"] == "done"
                if not enqueue(event):
                    break
        except _Generation._SwapConflict as exc:
            enqueue({"type": "error", "message": str(exc), "kind": "model_swap_conflict"})
        except Exception:  # surface engine errors to the client
            # Never leak raw exception text/stack to the client; the full
            # detail goes to the server log only.
            logger.exception("engine error")
            enqueue({"type": "error", "message": "internal error during generation"})
        finally:
            # Deterministically run the generator's finally (finalize_deferred +
            # finish_engine_task) even when the loop broke early on a slow /
            # disconnected client, so pinned snapshots never leak.
            engine_gen.close()
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
                            "type": event.get("kind", "server_error"),
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
