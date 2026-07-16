"""`daedalus doctor` and its --json machine-readable report."""

import json

from daedalus.cli import main

_STATUSES = {"pass", "warn", "fail"}


def _run(monkeypatch, capsys, *argv):
    """Invoke the CLI entrypoint with a fabricated argv; return (rc, stdout)."""
    monkeypatch.setattr("sys.argv", ["daedalus", "doctor", *argv])
    rc = main()
    return rc, capsys.readouterr().out


def test_doctor_json_is_a_single_clean_object(monkeypatch, capsys):
    rc, out = _run(monkeypatch, capsys, "--json")
    # Exactly one JSON object on stdout — no ANSI, no log lines, no human report.
    assert "\x1b" not in out
    assert "doctor:" not in out
    report = json.loads(out)  # would raise if stdout carried anything else
    assert isinstance(report["ok"], bool)
    names = [c["name"] for c in report["checks"]]
    assert names == ["machine", "macos", "thermal_pressure", "mlx"]
    for c in report["checks"]:
        assert set(c) == {"name", "status", "detail"}
        assert c["status"] in _STATUSES
    # Exit code tracks ok exactly (same semantics as the human report).
    assert rc == (0 if report["ok"] else 1)


def test_doctor_json_reports_thermal_failure(monkeypatch, capsys):
    def boom():
        raise RuntimeError("no sensor")

    monkeypatch.setattr("daedalus.sensors.make_pressure_reader", boom)
    rc, out = _run(monkeypatch, capsys, "--json")
    report = json.loads(out)  # still a single JSON object, no partial human output
    assert report["ok"] is False
    assert rc == 1
    thermal = next(c for c in report["checks"] if c["name"] == "thermal_pressure")
    assert thermal["status"] == "fail"
    assert "no sensor" in thermal["detail"]


def test_doctor_json_warns_when_metal_unavailable(monkeypatch, capsys):
    monkeypatch.setattr("mlx.core.metal.is_available", lambda: False)
    rc, out = _run(monkeypatch, capsys, "--json")
    report = json.loads(out)
    mlx = next(c for c in report["checks"] if c["name"] == "mlx")
    assert mlx["status"] == "warn"
    # A warn is not a failure: exit code stays 0.
    assert report["ok"] is True
    assert rc == 0


def test_doctor_human_report_and_exit_code(monkeypatch, capsys):
    rc, out = _run(monkeypatch, capsys)
    assert "machine:" in out
    assert "thermal pressure" in out
    assert out.rstrip().endswith("doctor: all good")
    assert "{" not in out  # not JSON
    assert rc == 0


def test_doctor_human_failure_returns_one(monkeypatch, capsys):
    def boom():
        raise RuntimeError("no sensor")

    monkeypatch.setattr("daedalus.sensors.make_pressure_reader", boom)
    rc, out = _run(monkeypatch, capsys)
    assert "thermal pressure: FAILED (no sensor)" in out
    assert out.rstrip().endswith("doctor: problems found")
    assert rc == 1
