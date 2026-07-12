"""The daedalus engine: thermally-paced chunked prefill + streamed decode.

Built on mlx-lm's public primitives (``model(tokens, cache=...)``,
``make_prompt_cache``, ``maybe_quantize_kv_cache``, ``stream_generate``) —
deliberately NOT on ``BatchGenerator`` internals, which are private and have
broken downstream projects before.

Flow for one request:

1. ``paced_prefill`` computes the KV cache for all prompt tokens except the
   last, one governor-sized chunk at a time, idling between chunks per the
   thermal duty cycle. Hooks fire per chunk: progress (SSE keepalives),
   checkpoint (resumable prefill), abort.
2. ``stream_generate`` is handed the warm cache plus the final prompt token;
   its internal prefill loop sees a 1-token prompt and skips straight to
   decode. Decode is memory-bandwidth-bound and runs unpaced.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Generator, Iterable, List, Optional

import mlx.core as mx
from mlx_lm.generate import (
    GenerationResponse,
    generation_stream,
    maybe_quantize_kv_cache,
    stream_generate,
    wired_limit,
)
from mlx_lm.models import cache as cache_mod
from mlx_lm.sample_utils import make_sampler

from daedalus.governor import ThermalGovernor
from daedalus.sensors import ThermalLevel, ThermalMonitor

# Structured logging
try:
    import structlog
    HAS_STRUCTLOG = True
except ImportError:
    structlog = None
    HAS_STRUCTLOG = False

# OpenTelemetry
try:
    from opentelemetry import trace
    HAS_OTEL = True
except ImportError:
    trace = None
    HAS_OTEL = False

logger = structlog.get_logger(__name__) if HAS_STRUCTLOG else __import__('logging').getLogger(__name__)
tracer = trace.get_tracer("daedalus.engine") if HAS_OTEL and trace else None


class PrefillAborted(Exception):
    """Raised when a prefill is abandoned via the abort hook."""


@dataclass
class PrefillReport:
    total_tokens: int
    computed_tokens: int = 0
    chunks: int = 0
    burn_seconds: float = 0.0
    idle_seconds: float = 0.0
    max_level: int = 0
    model_seconds: float = 0.0
    quantize_seconds: float = 0.0
    eval_seconds: float = 0.0


@dataclass
class EngineConfig:
    kv_bits: Optional[int] = 8
    kv_group_size: int = 64
    quantized_kv_start: int = 4096
    # A measured best nominal prefill chunk. Thermal policies still take over
    # as soon as pressure rises, so tuning cannot disable thermal protection.
    prefill_chunk_tokens: Optional[int] = None
    # Clearing Metal's allocator after every chunk can force reallocation on
    # the next chunk. Keep allocations during a request by default; expose the
    # old behavior for tight-memory machines.
    clear_metal_cache_between_chunks: bool = False
    # Retained allocations must still be bounded: a 30k-token prefill can
    # otherwise accumulate buffer-cache into swap on a 16GB Air. When MLX's
    # cache exceeds this high-water mark, it is cleared at the next chunk
    # boundary regardless of the flag above.
    metal_cache_high_water_bytes: int = 1_536_000_000
    # Poll the abort hook at this interval while idling between chunks.
    idle_poll_seconds: float = 0.1
    # Optional MLX-LM speculative decoding; only enable after a RAM/throughput
    # benchmark confirms that the draft model helps this machine.
    num_draft_tokens: int = 0


class Engine:
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        governor: ThermalGovernor,
        config: Optional[EngineConfig] = None,
        draft_model: Optional[Any] = None,
        clock: Callable[[], float] = time.perf_counter,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.governor = governor
        self.config = config or EngineConfig()
        self.draft_model = draft_model
        self._clock = clock
        self._sleep = sleep

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        monitor: Optional[ThermalMonitor] = None,
        governor: Optional[ThermalGovernor] = None,
        config: Optional[EngineConfig] = None,
        draft_model_path: Optional[str] = None,
    ) -> "Engine":
        from mlx_lm import load

        model, tokenizer = load(model_path)
        draft_model = None
        if draft_model_path:
            if not cache_mod.can_trim_prompt_cache(cache_mod.make_prompt_cache(model)):
                raise ValueError(
                    "speculative decoding requires trimmable prompt caches; "
                    "this hybrid/sliding-window model is not compatible"
                )
            draft_model, draft_tokenizer = load(draft_model_path)
            if draft_tokenizer.vocab_size != tokenizer.vocab_size:
                raise ValueError("draft model tokenizer vocabulary does not match target model")
            if not cache_mod.can_trim_prompt_cache(cache_mod.make_prompt_cache(draft_model)):
                raise ValueError("draft model prompt cache is not trimmable")
        if governor is None:
            monitor = (monitor or ThermalMonitor()).start()
            governor = ThermalGovernor(monitor)
        return cls(model, tokenizer, governor, config, draft_model=draft_model)

    def make_cache(self) -> List[Any]:
        cache = cache_mod.make_prompt_cache(self.model)
        if self.draft_model is not None:
            cache += cache_mod.make_prompt_cache(self.draft_model)
        return cache

    def active_memory_bytes(self) -> int:
        """Current MLX active allocation, used for conservative admission."""
        return int(mx.get_active_memory())

    def paced_prefill(
        self,
        tokens: List[int],
        prompt_cache: List[Any],
        *,
        already_cached: int = 0,
        snap_points: Optional[List[int]] = None,
        progress_cb: Optional[Callable[[int, int], None]] = None,
        checkpoint_cb: Optional[Callable[[int, List[Any]], None]] = None,
        should_abort: Optional[Callable[[], bool]] = None,
    ) -> PrefillReport:
        """Prefill ``tokens[already_cached:-1]`` into ``prompt_cache``.

        ``already_cached`` counts tokens whose KV is already present (prefix
        cache hit or checkpoint resume). The final token is left uncomputed —
        the first decode step consumes it (mlx-lm convention).

        ``snap_points`` are token offsets a chunk must end exactly on (e.g.
        the shared system-prompt boundary), so ``checkpoint_cb`` observes a
        cache state that non-trimmable (hybrid) models can reuse as-is.
        """
        total = len(tokens)
        job_tokens = max(0, total - already_cached - 1)
        snaps = sorted(s for s in (snap_points or []) if 0 < s < total - 1)
        report = PrefillReport(total_tokens=total, computed_tokens=already_cached)
        if progress_cb:
            progress_cb(report.computed_tokens, total)

        # Start span for the entire prefill
        prefill_span = tracer.start_span("prefill") if tracer else None
        if prefill_span:
            prefill_span.set_attribute("total_tokens", total)
            prefill_span.set_attribute("already_cached", already_cached)
            prefill_span.set_attribute("job_tokens", job_tokens)

        thermal_chunk_tokens = self.governor.initial_chunk_tokens()
        chunk_tokens = min(
            self.config.prefill_chunk_tokens or thermal_chunk_tokens,
            thermal_chunk_tokens,
        )
        
        while total - report.computed_tokens > 1:
            if should_abort and should_abort():
                if prefill_span:
                    prefill_span.set_attribute("aborted", True)
                    prefill_span.end()
                raise PrefillAborted(
                    f"aborted at {report.computed_tokens}/{total} tokens"
                )

            n = min(chunk_tokens, total - report.computed_tokens - 1)
            for snap in snaps:
                if report.computed_tokens < snap < report.computed_tokens + n:
                    n = snap - report.computed_tokens
                    break
            piece = tokens[report.computed_tokens : report.computed_tokens + n]

            # Start span for this chunk
            chunk_span = tracer.start_span("prefill_chunk") if tracer else None
            if chunk_span:
                chunk_span.set_attribute("chunk_index", report.chunks)
                chunk_span.set_attribute("chunk_tokens", n)
                chunk_span.set_attribute("computed_tokens", report.computed_tokens)
            
            start = self._clock()
            with mx.stream(generation_stream):
                self.model(mx.array(piece)[None], cache=prompt_cache)
                model_end = self._clock()
                maybe_quantize_kv_cache(
                    prompt_cache,
                    quantized_kv_start=self.config.quantized_kv_start,
                    kv_group_size=self.config.kv_group_size,
                    kv_bits=self.config.kv_bits,
                )
                quantize_end = self._clock()
                mx.eval([c.state for c in prompt_cache])
                eval_end = self._clock()
            burn = self._clock() - start
            if (
                self.config.clear_metal_cache_between_chunks
                or mx.get_cache_memory() > self.config.metal_cache_high_water_bytes
            ):
                mx.clear_cache()

            report.computed_tokens += n
            report.chunks += 1
            report.burn_seconds += burn
            report.model_seconds += model_end - start
            report.quantize_seconds += quantize_end - model_end
            report.eval_seconds += eval_end - quantize_end

            if progress_cb:
                progress_cb(report.computed_tokens, total)
            if checkpoint_cb:
                checkpoint_cb(report.computed_tokens, prompt_cache)

            decision = self.governor.pace(chunk_seconds=burn, job_tokens=job_tokens)
            chunk_tokens = (
                self.config.prefill_chunk_tokens
                if self.config.prefill_chunk_tokens
                and decision.effective_level == ThermalLevel.NOMINAL
                else decision.next_chunk_tokens
            )
            report.max_level = max(report.max_level, int(decision.effective_level))
            if decision.sleep_seconds > 0 and total - report.computed_tokens > 1:
                report.idle_seconds += self._idle(
                    decision.sleep_seconds, should_abort
                )
            
            # Log structured info for this chunk
            logger.info(
                "prefill_chunk",
                chunks=report.chunks,
                computed_tokens=report.computed_tokens,
                total_tokens=total,
                chunk_tokens=n,
                burn_seconds=burn,
                thermal_level=decision.effective_level.name,
                next_chunk_tokens=decision.next_chunk_tokens,
                sleep_seconds=decision.sleep_seconds,
            )
            
            if chunk_span:
                chunk_span.set_attribute("thermal_level", decision.effective_level.name)
                chunk_span.set_attribute("next_chunk_tokens", decision.next_chunk_tokens)
                chunk_span.set_attribute("sleep_seconds", decision.sleep_seconds)
                chunk_span.end()
        
        if prefill_span:
            prefill_span.set_attribute("chunks", report.chunks)
            prefill_span.set_attribute("burn_seconds", report.burn_seconds)
            prefill_span.set_attribute("idle_seconds", report.idle_seconds)
            prefill_span.set_attribute("max_thermal_level", report.max_level)
            prefill_span.end()
        
        return report

    def _idle(
        self, seconds: float, should_abort: Optional[Callable[[], bool]]
    ) -> float:
        """Sleep in small increments so aborts cancel the idle promptly."""
        deadline = self._clock() + seconds
        slept = 0.0
        while True:
            remaining = deadline - self._clock()
            if remaining <= 0:
                return slept
            if should_abort and should_abort():
                raise PrefillAborted("aborted during thermal idle")
            step = min(self.config.idle_poll_seconds, remaining)
            self._sleep(step)
            slept += step

    def generate(
        self,
        tokens: List[int],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 1.0,
        prompt_cache: Optional[List[Any]] = None,
        already_cached: int = 0,
        snap_points: Optional[List[int]] = None,
        progress_cb: Optional[Callable[[int, int], None]] = None,
        checkpoint_cb: Optional[Callable[[int, List[Any]], None]] = None,
        should_abort: Optional[Callable[[], bool]] = None,
    ) -> Generator[GenerationResponse, None, None]:
        """Paced prefill, then streamed decode. Yields GenerationResponse."""
        if prompt_cache is None:
            prompt_cache = self.make_cache()

        with wired_limit(self.model, [generation_stream]):
            self.paced_prefill(
                tokens,
                prompt_cache,
                already_cached=already_cached,
                snap_points=snap_points,
                progress_cb=progress_cb,
                checkpoint_cb=checkpoint_cb,
                should_abort=should_abort,
            )

        # Decode phase span
        decode_span = tracer.start_span("decode") if tracer else None
        if decode_span:
            decode_span.set_attribute("max_tokens", max_tokens)
            decode_span.set_attribute("temperature", temperature)
            decode_span.set_attribute("top_p", top_p)
        
        sampler = make_sampler(temp=temperature, top_p=top_p)
        try:
            for resp in stream_generate(
                self.model,
                self.tokenizer,
                prompt=tokens[-1:],
                prompt_cache=prompt_cache,
                max_tokens=max_tokens,
                sampler=sampler,
                kv_bits=self.config.kv_bits,
                kv_group_size=self.config.kv_group_size,
                quantized_kv_start=self.config.quantized_kv_start,
                draft_model=self.draft_model,
                num_draft_tokens=self.config.num_draft_tokens,
            ):
                if decode_span:
                    decode_span.add_event("token_generated", {
                        "token": resp.text[:50] if resp.text else "",
                        "generation_tokens": resp.generation_tokens,
                        "generation_tps": resp.generation_tps,
                    })
                yield resp
        finally:
            if decode_span:
                decode_span.end()
