from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import air56_unoq_build_hardware_report
from tools.air56_unoq_build_hardware_report import build_hardware_report, build_hardware_report_from_paths
from tools.air56_unoq_hardware_acceptance import build_acceptance_summary


def _stage0() -> dict:
    return {
        "passed": True,
        "telemetry_size": 20,
        "command_size": 9,
        "crc_error_rejected": True,
        "loopback_duration_s": 600,
        "fallback_ms": 80,
        "frames": [{"telemetry_t_ms": 0}, {"telemetry_t_ms": 10}, {"telemetry_t_ms": 20}],
    }


def _stage1() -> dict:
    return {
        "passed": True,
        "mock_adapter_enabled": False,
        "production_build_without_mock": True,
        "current_scaling_ok": True,
        "speed_scaling_ok": True,
        "vdc_scaling_ok": True,
        "p_in_estimate_ok": True,
        "fault_bits_ok": True,
        "safe_disable_ok": True,
    }


def _stage2() -> dict:
    return {"passed": True, "ai_enabled": False, "bridge_dry_run": True}


def _stage2_rows() -> list[dict[str, str]]:
    return [
        {
            "t_ms": "0",
            "omega_meas": "144.0",
            "stm_omega_meas": "144.0",
            "omega_ref": "144.5",
            "stm_omega_ref": "144.5",
            "id": "1.30",
            "stm_id": "1.30",
            "iq": "0.40",
            "stm_iq": "0.40",
            "vdc": "24.1",
            "stm_vdc": "24.1",
            "i_rms": "1.45",
            "stm_i_rms": "1.45",
            "p_in": "42.0",
            "stm_p_in": "42.0",
        },
        {
            "t_ms": "10",
            "omega_meas": "144.1",
            "stm_omega_meas": "144.1",
            "omega_ref": "144.5",
            "stm_omega_ref": "144.5",
            "id": "1.29",
            "stm_id": "1.29",
            "iq": "0.41",
            "stm_iq": "0.41",
            "vdc": "24.1",
            "stm_vdc": "24.1",
            "i_rms": "1.46",
            "stm_i_rms": "1.46",
            "p_in": "42.2",
            "stm_p_in": "42.2",
        },
        {
            "t_ms": "20",
            "omega_meas": "144.1",
            "stm_omega_meas": "144.1",
            "omega_ref": "144.5",
            "stm_omega_ref": "144.5",
            "id": "1.29",
            "stm_id": "1.29",
            "iq": "0.41",
            "stm_iq": "0.41",
            "vdc": "24.1",
            "stm_vdc": "24.1",
            "i_rms": "1.46",
            "stm_i_rms": "1.46",
            "p_in": "42.3",
            "stm_p_in": "42.3",
        },
    ]


def _stage3() -> dict:
    return {
        "passed": True,
        "ai_enabled": True,
        "id_ref_limits_tight": True,
        "disable_on_fault": True,
        "fallback_ms": 90,
        "tracking_guard_regression": False,
    }


def _stage4() -> dict:
    return {
        "passed": True,
        "documented": True,
        "guard_fail_delta": 0,
        "tracking_guard_regression": False,
        "current_thermal_limit_ok": True,
        "fallback_oscillation": False,
        "power_saving_pct": 0.2,
    }


def test_build_hardware_report_accepts_realistic_stage_logs() -> None:
    report = build_hardware_report(
        board_id="unoq-air56-bench-001",
        operator="test",
        stage0=_stage0(),
        stage1=_stage1(),
        stage2=_stage2(),
        stage2_rows=_stage2_rows(),
        stage3=_stage3(),
        stage4=_stage4(),
        notes="bench acceptance",
    )

    assert report["schema"] == air56_unoq_build_hardware_report.SCHEMA
    assert report["stages"]["stage0"]["telemetry_period_ms_max"] == 10.0
    assert report["stages"]["stage2"]["decoded_telemetry_mismatch_pct"] == 0.0
    assert build_acceptance_summary(report)["hardware_ready"] is True


def test_build_hardware_report_fails_safe_when_stage2_csv_does_not_match() -> None:
    rows = _stage2_rows()
    rows[0]["stm_p_in"] = "99.0"

    report = build_hardware_report(
        board_id="board",
        operator="operator",
        stage0=_stage0(),
        stage1=_stage1(),
        stage2=_stage2(),
        stage2_rows=rows,
        stage3=_stage3(),
        stage4=_stage4(),
    )

    assert report["stages"]["stage2"]["passed"] is False
    assert report["stages"]["stage2"]["decoded_telemetry_mismatch_pct"] > 2.0
    assert build_acceptance_summary(report)["hardware_ready"] is False


def test_build_hardware_report_accepts_explicit_stage2_summary_without_csv() -> None:
    stage2 = {
        "passed": "yes",
        "ai_enabled": "false",
        "telemetry_only": "true",
        "telemetry_period_ms_max": 10.5,
        "decoded_telemetry_mismatch_pct": 0.5,
    }

    report = build_hardware_report(
        board_id="board",
        operator="operator",
        stage0=_stage0(),
        stage1=_stage1(),
        stage2=stage2,
        stage2_rows=[],
        stage3=_stage3(),
        stage4=_stage4(),
    )

    assert report["stages"]["stage2"]["bridge_dry_run"] is True
    assert build_acceptance_summary(report)["hardware_ready"] is True


def test_build_hardware_report_from_paths_and_cli_write_outputs(tmp_path: Path, monkeypatch) -> None:
    paths = {}
    for name, payload in {
        "stage0": _stage0(),
        "stage1": _stage1(),
        "stage2": _stage2(),
        "stage3": _stage3(),
        "stage4": _stage4(),
    }.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        paths[name] = path

    csv_path = tmp_path / "stage2.csv"
    csv_path.write_text(
        "t_ms,omega_meas,stm_omega_meas,p_in,stm_p_in\n"
        "0,144.0,144.0,42.0,42.0\n"
        "10,144.1,144.1,42.1,42.1\n",
        encoding="utf-8",
    )
    report = build_hardware_report_from_paths(
        board_id="board",
        operator="operator",
        stage0_json=paths["stage0"],
        stage1_json=paths["stage1"],
        stage2_json=paths["stage2"],
        stage2_csv=csv_path,
        stage3_json=paths["stage3"],
        stage4_json=paths["stage4"],
    )
    assert build_acceptance_summary(report)["hardware_ready"] is True

    out_report = tmp_path / "report.json"
    out_summary = tmp_path / "summary.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "air56_unoq_build_hardware_report.py",
            "--board-id",
            "board",
            "--operator",
            "operator",
            "--stage0-json",
            str(paths["stage0"]),
            "--stage1-json",
            str(paths["stage1"]),
            "--stage2-json",
            str(paths["stage2"]),
            "--stage2-csv",
            str(csv_path),
            "--stage3-json",
            str(paths["stage3"]),
            "--stage4-json",
            str(paths["stage4"]),
            "--out-json",
            str(out_report),
            "--out-summary-json",
            str(out_summary),
        ],
    )

    assert air56_unoq_build_hardware_report.main() == 0
    assert json.loads(out_report.read_text(encoding="utf-8"))["board_id"] == "board"
    assert json.loads(out_summary.read_text(encoding="utf-8"))["hardware_ready"] is True


def test_build_hardware_report_rejects_non_object_json(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="log JSON must be an object"):
        air56_unoq_build_hardware_report._read_json(bad)


def test_build_hardware_report_helper_fail_safe_edges() -> None:
    helper = air56_unoq_build_hardware_report

    assert helper._read_csv_rows(None) == []
    assert helper._truthy(None, default=True) is True
    assert helper._truthy(0) is False
    assert helper._float({"bad": object()}, "bad", default=7.0) == 7.0
    assert helper._csv_float({"x": "not-a-number"}, "x") is None
    assert helper._max_delta_ms_from_values([10.0]) == float("inf")
    assert helper._max_period_ms_from_frames({"not": "a-list"}) == float("inf")
    assert helper._max_period_ms_from_frames([{"telemetry_t_ms": "bad"}, {"t_ms": 10}, {"t_ms": 22}]) == 12.0
    assert helper._decoded_mismatch_pct([{"t_ms": "0"}]) == 100.0

    report = build_hardware_report(
        board_id="board",
        operator="operator",
        stage0={
            "passed": True,
            "telemetry_size": 20,
            "command_size": 9,
            "crc_error_rejected": True,
            "loopback_duration_s": 600,
            "fallback_after_timeout": True,
            "timeout_ms": 90,
            "frames": [{"telemetry_t_ms": 0}, {"telemetry_t_ms": 10}],
        },
        stage1=_stage1(),
        stage2={
            "passed": True,
            "ai_enabled": False,
            "bridge_dry_run": True,
            "telemetry_period_ms_max": 10,
            "decoded_telemetry_mismatch_pct": 0,
        },
        stage2_rows=[],
        stage3=_stage3(),
        stage4=_stage4(),
    )

    assert report["stages"]["stage0"]["fallback_ms"] == 90.0
    assert build_acceptance_summary(report)["hardware_ready"] is True


def test_build_hardware_report_cli_returns_failure_for_templates(tmp_path: Path, monkeypatch) -> None:
    paths = {}
    for name, payload in {
        "stage0": {"passed": False},
        "stage1": {"passed": False},
        "stage2": {"passed": False},
        "stage3": {"passed": False},
        "stage4": {"passed": False},
    }.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        paths[name] = path

    out_report = tmp_path / "report.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "air56_unoq_build_hardware_report.py",
            "--board-id",
            "board",
            "--operator",
            "operator",
            "--stage0-json",
            str(paths["stage0"]),
            "--stage1-json",
            str(paths["stage1"]),
            "--stage2-json",
            str(paths["stage2"]),
            "--stage3-json",
            str(paths["stage3"]),
            "--stage4-json",
            str(paths["stage4"]),
            "--out-json",
            str(out_report),
        ],
    )

    assert air56_unoq_build_hardware_report.main() == 1
    assert json.loads(out_report.read_text(encoding="utf-8"))["stages"]["stage0"]["passed"] is False
