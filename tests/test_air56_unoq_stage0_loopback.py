import json

from tools.air56_unoq_stage0_loopback import run_loopback_selftest
from tools import air56_unoq_stage0_loopback


def test_stage0_loopback_selftest_passes_with_crc() -> None:
    report = run_loopback_selftest(packets=4, period_ms=10, timeout_ms=100, crc=True)

    assert report.passed
    assert report.telemetry_size == 20
    assert report.command_size == 9
    assert report.fallback_after_timeout
    assert [frame.index for frame in report.frames] == [0, 1, 2, 3]
    assert all(frame.enable_ai == 1 for frame in report.frames)
    assert all(frame.crc_ok for frame in report.frames)


def test_stage0_loopback_selftest_supports_crc_disabled() -> None:
    report = run_loopback_selftest(packets=2, crc=False)

    assert report.passed
    assert not report.crc_enabled
    assert all(frame.crc_ok for frame in report.frames)


def test_stage0_loopback_cli_writes_report(tmp_path, monkeypatch, capsys) -> None:
    out_json = tmp_path / "stage0.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "air56_unoq_stage0_loopback.py",
            "--packets",
            "2",
            "--out-json",
            str(out_json),
        ],
    )

    assert air56_unoq_stage0_loopback.main() == 0
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert payload["packets"] == 2
    assert "passed" in capsys.readouterr().out
