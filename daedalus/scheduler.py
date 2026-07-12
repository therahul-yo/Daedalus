"""Fair, single-engine request scheduling."""

from __future__ import annotations

import threading


class FifoLock:
    """A lock that admits waiting callers in arrival order, supporting aborts."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._next_ticket = 0
        self._serving = 0
        self._held = False

    @property
    def queued(self) -> int:
        with self._condition:
            return max(0, self._next_ticket - self._serving - (1 if self._held else 0))

    def acquire(self, abort_event: threading.Event = None, timeout: float = 1.0) -> bool:
        with self._condition:
            ticket = self._next_ticket
            self._next_ticket += 1
            while ticket != self._serving:
                if abort_event and abort_event.is_set():
                    self._serving += 1
                    self._condition.notify_all()
                    return False
                self._condition.wait(timeout)
            self._held = True
            return True

    def release(self) -> None:
        with self._condition:
            self._held = False
            self._serving += 1
            self._condition.notify_all()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
