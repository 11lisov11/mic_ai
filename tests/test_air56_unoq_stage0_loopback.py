from tools.air56_unoq_stage0_loopback import run_loopback_selftest


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
