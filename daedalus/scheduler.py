"""Fair, single-engine request scheduling."""

from __future__ import annotations

import threading


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
