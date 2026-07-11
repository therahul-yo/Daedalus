import sys
import time

import pytest

from daedalus.sensors import ThermalLevel, ThermalMonitor, make_pressure_reader

is_macos = sys.platform == "darwin"


@pytest.mark.skipif(not is_macos, reason="macOS-only sensor")
def test_real_reader_returns_valid_level():
    read = make_pressure_reader()
    level = read()
    assert isinstance(level, ThermalLevel)
    assert ThermalLevel.NOMINAL <= level <= ThermalLevel.SLEEPING


def test_monitor_with_injected_reader_tracks_changes():
    levels = iter(
        [ThermalLevel.NOMINAL, ThermalLevel.NOMINAL, ThermalLevel.MODERATE, ThermalLevel.HEAVY]
    )
    current = ThermalLevel.NOMINAL

    def fake_reader():
        nonlocal current
        current = next(levels, current)
        return current

    monitor = ThermalMonitor(reader=fake_reader, poll_interval=0.01)
    assert monitor.level == ThermalLevel.NOMINAL

    changes = []
    monitor.on_change(lambda old, new: changes.append((old, new)))

    monitor.refresh()  # NOMINAL -> NOMINAL: no callback
    assert changes == []
    monitor.refresh()  # -> MODERATE
    monitor.refresh()  # -> HEAVY
    assert changes == [
        (ThermalLevel.NOMINAL, ThermalLevel.MODERATE),
        (ThermalLevel.MODERATE, ThermalLevel.HEAVY),
    ]
    assert monitor.level == ThermalLevel.HEAVY
    assert monitor.rising(window_seconds=60)


def test_monitor_poll_thread_start_stop():
    monitor = ThermalMonitor(reader=lambda: ThermalLevel.NOMINAL, poll_interval=0.01)
    monitor.start()
    time.sleep(0.05)
    monitor.stop()
    assert monitor.level == ThermalLevel.NOMINAL


def test_monitor_survives_reader_failure():
    calls = {"n": 0}

    def flaky_reader():
        calls["n"] += 1
        if calls["n"] > 1:
            raise OSError("sensor gone")
        return ThermalLevel.MODERATE

    monitor = ThermalMonitor(reader=flaky_reader, poll_interval=0.01)
    monitor.start()
    time.sleep(0.05)
    monitor.stop()
    # Last known level is retained despite read failures.
    assert monitor.level == ThermalLevel.MODERATE
    assert calls["n"] > 2  # kept polling after failures
