"""macOS thermal sensing without sudo.

Two signals, both readable from an unprivileged process:

1. The 5-level thermal pressure Darwin notification
   ``com.apple.system.thermalpressurelevel`` (libnotify). This is the signal
   that matters: level 2 ("Heavy") is where real clock throttling starts.
   NSProcessInfo collapses Moderate and Heavy into a single "fair" state,
   which is why we go to libnotify directly.

2. ``NSProcessInfo.thermalState`` (4 coarse states) as a fallback when the
   notify key is unavailable.

No pyobjc dependency — plain ctypes against libSystem / libobjc.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import enum
import threading
import time
from collections import deque
from typing import Callable, Optional


class ThermalLevel(enum.IntEnum):
    """Darwin thermal pressure levels (kOSThermalPressureLevel*)."""

    NOMINAL = 0
    MODERATE = 1
    HEAVY = 2
    TRAPPING = 3
    SLEEPING = 4


_NOTIFY_KEY = b"com.apple.system.thermalpressurelevel"

# NSProcessInfo.thermalState values (coarser than ThermalLevel).
_PROCESSINFO_TO_LEVEL = {
    0: ThermalLevel.NOMINAL,   # nominal
    1: ThermalLevel.MODERATE,  # fair (hides Moderate vs Heavy — assume Moderate)
    2: ThermalLevel.HEAVY,     # serious
    3: ThermalLevel.TRAPPING,  # critical
}


class _NotifyPressureReader:
    """Reads the 5-level pressure via libnotify. Registers once, reads cheaply."""

    def __init__(self) -> None:
        self._libc = ctypes.CDLL(
            ctypes.util.find_library("System") or "/usr/lib/libSystem.B.dylib"
        )
        self._token = ctypes.c_int()
        rc = self._libc.notify_register_check(
            ctypes.c_char_p(_NOTIFY_KEY), ctypes.byref(self._token)
        )
        if rc != 0:
            raise OSError(f"notify_register_check({_NOTIFY_KEY!r}) failed: rc={rc}")

    def read(self) -> ThermalLevel:
        state = ctypes.c_uint64()
        rc = self._libc.notify_get_state(self._token, ctypes.byref(state))
        if rc != 0:
            raise OSError(f"notify_get_state failed: rc={rc}")
        return ThermalLevel(min(int(state.value), int(ThermalLevel.SLEEPING)))

    def close(self) -> None:
        try:
            self._libc.notify_cancel(self._token)
        except Exception:
            pass


class _ProcessInfoReader:
    """Fallback: NSProcessInfo.thermalState via raw objc_msgSend."""

    def __init__(self) -> None:
        self._objc = ctypes.CDLL("/usr/lib/libobjc.A.dylib")
        # Foundation must be loaded for the NSProcessInfo class to exist.
        ctypes.CDLL("/System/Library/Frameworks/Foundation.framework/Foundation")
        self._objc.objc_getClass.restype = ctypes.c_void_p
        self._objc.sel_registerName.restype = ctypes.c_void_p
        send = ctypes.cast(
            self._objc.objc_msgSend,
            ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p),
        )
        cls = self._objc.objc_getClass(b"NSProcessInfo")
        self._process_info = send(cls, self._objc.sel_registerName(b"processInfo"))
        if not self._process_info:
            raise OSError("NSProcessInfo unavailable")
        self._send_long = ctypes.cast(
            self._objc.objc_msgSend,
            ctypes.CFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p),
        )
        self._sel_thermal = self._objc.sel_registerName(b"thermalState")

    def read(self) -> ThermalLevel:
        state = self._send_long(self._process_info, self._sel_thermal)
        return _PROCESSINFO_TO_LEVEL.get(int(state), ThermalLevel.HEAVY)

    def close(self) -> None:
        pass


def make_pressure_reader() -> Callable[[], ThermalLevel]:
    """Best available no-sudo thermal reader for this host."""
    try:
        reader = _NotifyPressureReader()
    except OSError:
        reader = _ProcessInfoReader()
    return reader.read


class ThermalMonitor:
    """Polls thermal pressure and keeps a short history for trend decisions.

    The engine reads ``level`` at every prefill-chunk boundary (seconds apart),
    so a poll interval of ~2s gives the governor fresher data than it can act
    on. ``on_change`` callbacks fire from the poll thread.
    """

    def __init__(
        self,
        reader: Optional[Callable[[], ThermalLevel]] = None,
        poll_interval: float = 2.0,
        history_seconds: float = 120.0,
    ) -> None:
        self._read = reader or make_pressure_reader()
        self._poll_interval = poll_interval
        self._history: deque[tuple[float, ThermalLevel]] = deque(
            maxlen=max(2, int(history_seconds / poll_interval))
        )
        self._level = self._read()
        self._history.append((time.monotonic(), self._level))
        self._callbacks: list[Callable[[ThermalLevel, ThermalLevel], None]] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @property
    def level(self) -> ThermalLevel:
        with self._lock:
            return self._level

    def refresh(self) -> ThermalLevel:
        """Synchronously re-read the sensor (also records history)."""
        level = self._read()
        with self._lock:
            previous, self._level = self._level, level
            self._history.append((time.monotonic(), level))
            callbacks = list(self._callbacks) if level != previous else []
        for cb in callbacks:
            cb(previous, level)
        return level

    def rising(self, window_seconds: float = 30.0) -> bool:
        """True if pressure increased within the recent window (hysteresis input)."""
        cutoff = time.monotonic() - window_seconds
        with self._lock:
            recent = [lvl for ts, lvl in self._history if ts >= cutoff]
        return len(recent) >= 2 and recent[-1] > recent[0]

    def on_change(self, cb: Callable[[ThermalLevel, ThermalLevel], None]) -> None:
        with self._lock:
            self._callbacks.append(cb)

    def start(self) -> "ThermalMonitor":
        if self._thread is None:
            self._thread = threading.Thread(
                target=self._run, name="airlift-thermal", daemon=True
            )
            self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self._poll_interval + 1)
            self._thread = None
        self._stop.clear()

    def _run(self) -> None:
        while not self._stop.wait(self._poll_interval):
            try:
                self.refresh()
            except OSError:
                # Sensor read failure: keep last known level, keep polling.
                pass
