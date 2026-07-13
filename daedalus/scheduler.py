"""Fair, single-engine request scheduling."""

from __future__ import annotations

import threading


class FifoLock:
    """A lock that admits waiting callers in arrival order, supporting aborts.

    A swap (multi-model hot-swap) uses ``acquire_for_swap`` to block new
    admits and wait until the engine is idle, then repoint ``state.engine``/
    ``state.store``; requests already holding the lock finish on the old engine.
    """

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._next_ticket = 0
        self._serving = 0
        self._held = False
        self._abandoned: set[int] = set()
        # Held by a swap to block new ``acquire`` calls and wait for idle.
        self._swap_gate = threading.Lock()

    @property
    def queued(self) -> int:
        with self._condition:
            return max(
                0,
                self._next_ticket
                - self._serving
                - len(self._abandoned)
                - (1 if self._held else 0),
            )

    def acquire(self, abort_event: threading.Event = None, timeout: float = 1.0) -> bool:
        # A swap may be in progress. We must not let a new admit start while
        # the swap is waiting for idle, but we also must NOT serialize admits
        # against each other (that would break FIFO). So: grab a ticket under
        # the swap gate, then release the gate immediately and wait under the
        # condition alone.
        with self._swap_gate:
            with self._condition:
                ticket = self._next_ticket
                self._next_ticket += 1
        while ticket != self._serving:
            with self._condition:
                if abort_event and abort_event.is_set():
                    with self._condition:
                        self._abandoned.add(ticket)
                        self._condition.notify_all()
                    return False
                if ticket == self._serving:
                    break
                self._condition.wait(timeout)
        with self._condition:
            if abort_event and abort_event.is_set():
                self._advance()
                self._condition.notify_all()
                return False
            self._held = True
            return True

    def release(self) -> None:
        with self._condition:
            self._held = False
            self._advance()
            self._condition.notify_all()

    def acquire_for_swap(self, timeout: float = 5.0) -> bool:
        """Block new admits and wait until the engine is idle (no holder).

        Caller repoints ``state.engine``/``state.store`` while holding this,
        then calls ``release_after_swap``. Returns True if idle was reached.
        """
        self._swap_gate.acquire()
        with self._condition:
            waited = 0.0
            while self._held or self._next_ticket != self._serving:
                if waited >= timeout:
                    self._swap_gate.release()
                    return False
                self._condition.wait(min(0.1, timeout - waited))
                waited += 0.1
        return True

    def release_after_swap(self) -> None:
        """Release the swap gate, letting new admits proceed."""
        self._swap_gate.release()

    def _advance(self) -> None:
        # Skip past tickets whose waiters abandoned the queue.
        self._serving += 1
        while self._serving in self._abandoned:
            self._abandoned.discard(self._serving)
            self._serving += 1

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
