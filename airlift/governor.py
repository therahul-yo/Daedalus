"""Thermal governor: turns thermal-pressure readings into prefill pacing.

The engine calls ``pace(chunk_seconds)`` after every prefill chunk and gets
back how many tokens the next chunk may process and how long to idle first.

Policy model — duty cycling: at effective level L with duty ``d``, after a
chunk that burned the GPU for ``t`` seconds, idle ``t * (1 - d) / d`` so the
GPU is busy at most fraction ``d`` of wall-clock. Sleeping is proportional to
the *measured* burn, so the policy self-adapts to hardware speed and chunk
size without knowing tokens/sec.

Escalation is instant; de-escalation is hysteretic: the effective level steps
down one level only after the observed level has stayed below it for
``step_down_seconds``. Fanless chassis cool slowly — rushing back to full
duty just re-throttles.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional

from airlift.sensors import ThermalLevel, ThermalMonitor


@dataclass(frozen=True)
class LevelPolicy:
    chunk_tokens: int
    duty: float  # fraction of wall-clock the GPU may burn, 0 < duty <= 1


DEFAULT_POLICIES: Mapping[ThermalLevel, LevelPolicy] = {
    ThermalLevel.NOMINAL: LevelPolicy(chunk_tokens=2048, duty=1.0),
    ThermalLevel.MODERATE: LevelPolicy(chunk_tokens=1024, duty=0.6),
    ThermalLevel.HEAVY: LevelPolicy(chunk_tokens=512, duty=0.25),
    ThermalLevel.TRAPPING: LevelPolicy(chunk_tokens=256, duty=0.10),
    ThermalLevel.SLEEPING: LevelPolicy(chunk_tokens=256, duty=0.05),
}


@dataclass(frozen=True)
class PaceDecision:
    next_chunk_tokens: int
    sleep_seconds: float
    effective_level: ThermalLevel


@dataclass
class GovernorConfig:
    policies: Mapping[ThermalLevel, LevelPolicy] = field(
        default_factory=lambda: dict(DEFAULT_POLICIES)
    )
    step_down_seconds: float = 20.0
    max_sleep_seconds: float = 10.0
    # Optional global duty ceiling (e.g. user "quiet mode" = 0.5 even when cool).
    max_duty: float = 1.0


class ThermalGovernor:
    def __init__(
        self,
        monitor: ThermalMonitor,
        config: Optional[GovernorConfig] = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._monitor = monitor
        self._config = config or GovernorConfig()
        self._clock = clock
        self._effective = monitor.level
        self._below_since: Optional[float] = None

    @property
    def effective_level(self) -> ThermalLevel:
        return self._effective

    def _update_effective(self, observed: ThermalLevel, now: float) -> None:
        if observed >= self._effective:
            # Escalate (or hold) immediately.
            self._effective = observed
            self._below_since = None
            return
        if self._below_since is None:
            self._below_since = now
        elif now - self._below_since >= self._config.step_down_seconds:
            self._effective = ThermalLevel(int(self._effective) - 1)
            # Restart the timer; stepping down multiple levels takes
            # step_down_seconds per level.
            self._below_since = now if observed < self._effective else None

    def pace(self, chunk_seconds: float) -> PaceDecision:
        """Decide the next chunk size and pre-chunk idle after a chunk that
        took ``chunk_seconds`` of GPU time."""
        now = self._clock()
        self._update_effective(self._monitor.level, now)
        policy = self._config.policies[self._effective]
        duty = min(policy.duty, self._config.max_duty)
        if duty >= 1.0 or chunk_seconds <= 0:
            sleep = 0.0
        else:
            sleep = min(
                self._config.max_sleep_seconds, chunk_seconds * (1.0 - duty) / duty
            )
        return PaceDecision(
            next_chunk_tokens=policy.chunk_tokens,
            sleep_seconds=sleep,
            effective_level=self._effective,
        )

    def initial_chunk_tokens(self) -> int:
        """Chunk size for the first chunk of a prefill (no burn measured yet)."""
        self._update_effective(self._monitor.level, self._clock())
        return self._config.policies[self._effective].chunk_tokens
