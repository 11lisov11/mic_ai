from __future__ import annotations

import json
from pathlib import Path

from tools import air56_unoq_hardware_acceptance
from tools.air56_unoq_hardware_acceptance import _load_report, build_acceptance_summary


def _passing_report() -> dict:
    return {
        "schema": "mic_theory.air56_unoq.hardware_acceptance.v1",
        "board_id": "unoq-air56-bench-001",
        "operator": "test",
        "stages": {
            "stage0": {
                "passed": True,
                "struct_sizes_ok": True,
                "crc_error_rejected": True,
                "loopback_duration_s": 600.0,
                "fallback_ms": 80.0,
                "telemetry_period_ms_max": 10.5,
            },
            "stage1": {
                "passed": True,
                "mock_adapter_enabled": False,
                "production_build_without_mock": True,
                "current_scaling_ok": True,
                "speed_scaling_ok": True,
                "vdc_scaling_ok": True,
                "p_in_estimate_ok": True,
                "fault_bits_ok": True,
                "safe_disable_ok": True,
            },
            "stage2": {
                "passed": True,
                "ai_enabled": False,
                "bridge_dry_run": True,
                "telemetry_period_ms_max": 10.8,
                "decoded_telemetry_mismatch_pct": 0.5,
            },
            "stage3": {
                "passed": True,
                "ai_enabled": True,
                "id_ref_limits_tight": True,
                "disable_on_fault": True,
                "fallback_ms": 90.0,
                "tracking_guard_regression": False,
            },
            "stage4": {
                "passed": True,
                "documented": True,
                "guard_fail_delta": 0.0,
                "tracking_guard_regression": False,
                "current_thermal_limit_ok": True,
                "fallback_oscillation": False,
                "power_saving_pct": 0.1,
            },
        },
    }


def test_hardware_acceptance_passes_only_with_all_physical_stages() -> None:
    summary = build_acceptance_summary(_passing_report())

    assert summary["hardware_ready"] is True
    assert all(row["passed"] for row in summary["checks"])


def test_hardware_acceptance_rejects_mock_or_missing_ab_evidence() -> None:
    report = _passing_report()
    report["stages"]["stage1"]["mock_adapter_enabled"] = True
    report["stages"]["stage4"]["documented"] = False

    summary = build_acceptance_summary(report)
    failed = {row["name"] for row in summary["checks"] if not row["passed"]}

    assert summary["hardware_ready"] is False
    assert "stage1.production_no_mock" in failed
    assert "stage4.documented" in failed


def test_hardware_acceptance_report_is_json_serializable() -> None:
    summary = build_acceptance_summary(_passing_report())

    assert '"hardware_ready": true' in json.dumps(summary)


def test_hardware_acceptance_cli_writes_summary(tmp_path: Path, monkeypatch) -> None:
    report = tmp_path / "report.json"
    out_json = tmp_path / "summary.json"
    report.write_text(json.dumps(_passing_report()), encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        [
            "air56_unoq_hardware_acceptance.py",
            "--report",
            str(report),
            "--out-json",
            str(out_json),
        ],
    )

    assert air56_unoq_hardware_acceptance.main() == 0
    assert json.loads(out_json.read_text(encoding="utf-8"))["hardware_ready"] is True


def test_hardware_acceptance_cli_returns_failure_for_template_like_report(tmp_path: Path, monkeypatch) -> None:
    report_data = _passing_report()
    report_data["stages"]["stage0"]["passed"] = False
    report = tmp_path / "report.json"
    report.write_text(json.dumps(report_data), encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        [
            "air56_unoq_hardware_acceptance.py",
            "--report",
            str(report),
        ],
    )

    assert air56_unoq_hardware_acceptance.main() == 1


def test_hardware_acceptance_rejects_non_object_report(tmp_path: Path) -> None:
    report = tmp_path / "bad.json"
    report.write_text("[]", encoding="utf-8")

    try:
        _load_report(report)
    except ValueError as exc:
        assert "must be a JSON object" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("non-object acceptance report was accepted")


def test_hardware_acceptance_handles_malformed_stage_values() -> None:
    summary = build_acceptance_summary(
        {
            "schema": "mic_theory.air56_unoq.hardware_acceptance.v1",
            "board_id": "board",
            "operator": "operator",
            "stages": [],
        }
    )
    assert summary["hardware_ready"] is False

    report = _passing_report()
    report["stages"]["stage0"]["fallback_ms"] = object()
    summary = build_acceptance_summary(report)
    failed = {row["name"] for row in summary["checks"] if not row["passed"]}
    assert "stage0.fallback" in failed
