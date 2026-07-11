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
from daedalus.sensors import ThermalMonitor


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


@dataclass
class EngineConfig:
    kv_bits: Optional[int] = 8
    kv_group_size: int = 64
    quantized_kv_start: int = 4096
    # Poll the abort hook at this interval while idling between chunks.
    idle_poll_seconds: float = 0.1


class Engine:
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        governor: ThermalGovernor,
        config: Optional[EngineConfig] = None,
        clock: Callable[[], float] = time.perf_counter,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.governor = governor
        self.config = config or EngineConfig()
        self._clock = clock
        self._sleep = sleep

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        monitor: Optional[ThermalMonitor] = None,
        governor: Optional[ThermalGovernor] = None,
        config: Optional[EngineConfig] = None,
    ) -> "Engine":
        from mlx_lm import load

        model, tokenizer = load(model_path)
        if governor is None:
            monitor = (monitor or ThermalMonitor()).start()
            governor = ThermalGovernor(monitor)
        return cls(model, tokenizer, governor, config)

    def make_cache(self) -> List[Any]:
        return cache_mod.make_prompt_cache(self.model)

    def paced_prefill(
        self,
        tokens: List[int],
        prompt_cache: List[Any],
        *,
        already_cached: int = 0,
        progress_cb: Optional[Callable[[int, int], None]] = None,
        checkpoint_cb: Optional[Callable[[int, List[Any]], None]] = None,
        should_abort: Optional[Callable[[], bool]] = None,
    ) -> PrefillReport:
        """Prefill ``tokens[already_cached:-1]`` into ``prompt_cache``.

        ``already_cached`` counts tokens whose KV is already present (prefix
        cache hit or checkpoint resume). The final token is left uncomputed —
        the first decode step consumes it (mlx-lm convention).
        """
        total = len(tokens)
        report = PrefillReport(total_tokens=total, computed_tokens=already_cached)
        if progress_cb:
            progress_cb(report.computed_tokens, total)

        chunk_tokens = self.governor.initial_chunk_tokens()
        while total - report.computed_tokens > 1:
            if should_abort and should_abort():
                raise PrefillAborted(
                    f"aborted at {report.computed_tokens}/{total} tokens"
                )

            n = min(chunk_tokens, total - report.computed_tokens - 1)
            piece = tokens[report.computed_tokens : report.computed_tokens + n]

            start = self._clock()
            with mx.stream(generation_stream):
                self.model(mx.array(piece)[None], cache=prompt_cache)
                maybe_quantize_kv_cache(
                    prompt_cache,
                    quantized_kv_start=self.config.quantized_kv_start,
                    kv_group_size=self.config.kv_group_size,
                    kv_bits=self.config.kv_bits,
                )
                mx.eval([c.state for c in prompt_cache])
            burn = self._clock() - start
            mx.clear_cache()

            report.computed_tokens += n
            report.chunks += 1
            report.burn_seconds += burn

            if progress_cb:
                progress_cb(report.computed_tokens, total)
            if checkpoint_cb:
                checkpoint_cb(report.computed_tokens, prompt_cache)

            decision = self.governor.pace(chunk_seconds=burn)
            chunk_tokens = decision.next_chunk_tokens
            report.max_level = max(report.max_level, int(decision.effective_level))
            if decision.sleep_seconds > 0 and total - report.computed_tokens > 1:
                report.idle_seconds += self._idle(
                    decision.sleep_seconds, should_abort
                )
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
                progress_cb=progress_cb,
                checkpoint_cb=checkpoint_cb,
                should_abort=should_abort,
            )

        sampler = make_sampler(temp=temperature, top_p=top_p)
        yield from stream_generate(
            self.model,
            self.tokenizer,
            prompt=tokens[-1:],
            prompt_cache=prompt_cache,
            max_tokens=max_tokens,
            sampler=sampler,
            kv_bits=self.config.kv_bits,
            kv_group_size=self.config.kv_group_size,
            quantized_kv_start=self.config.quantized_kv_start,
        )
