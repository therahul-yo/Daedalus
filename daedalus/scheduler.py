"""Fair, single-engine request scheduling."""

from __future__ import annotations

import threading
import heapq
import time


class FifoLock:
    """A context-manager lock that admits waiting callers in arrival order."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._next_ticket = 0
        self._serving = 0
        self._held = False

    @property
    def queued(self) -> int:
        with self._condition:
            return max(0, self._next_ticket - self._serving - (1 if self._held else 0))

    def __enter__(self) -> "FifoLock":
        with self._condition:
            ticket = self._next_ticket
            self._next_ticket += 1
            while ticket != self._serving:
                self._condition.wait()
            self._held = True
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        with self._condition:
            self._held = False
            self._serving += 1
            self._condition.notify_all()


class PriorityLock:
    """A context-manager lock that admits waiting callers by priority (short prompts first)."""

    def __init__(self, short_prompt_threshold: int = 2048) -> None:
        self._condition = threading.Condition()
        self._heap: list[tuple[int, float, int, threading.Event]] = []
        self._counter = 0
        self._held = False
        self._short_prompt_threshold = short_prompt_threshold

    @property
    def queued(self) -> int:
        with self._condition:
            return len(self._heap)

    def __enter__(self) -> "PriorityLock":
        # NOTE: acquire_with_priority() is called explicitly before entering the
        # with block (e.g. `with lock.acquire_with_priority(n):`).  That call
        # already acquires the lock, so __enter__ must NOT acquire a second time.
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        with self._condition:
            self._held = False
            self._condition.notify_all()

    def acquire_with_priority(self, prompt_tokens: int) -> "PriorityLock":
        """Acquire the lock with priority based on prompt length.

        Args:
            prompt_tokens: Number of tokens in the prompt. Short prompts (< threshold) get priority 0.
        """
        priority = 0 if prompt_tokens < self._short_prompt_threshold else 1
        with self._condition:
            ticket = self._counter
            self._counter += 1
            event = threading.Event()
            # heap entry: (priority, timestamp, ticket, event)
            # Lower priority value = higher priority (processed first)
            heapq.heappush(self._heap, (priority, time.monotonic(), ticket, event))
            while self._held or (self._heap and self._heap[0][2] != ticket):
                self._condition.wait()
            self._held = True
            heapq.heappop(self._heap)
        return self