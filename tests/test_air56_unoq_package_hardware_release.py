from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from tools import air56_unoq_package_hardware_release
from tools import check_air56_unoq_coverage_gate
from tools.air56_unoq_package_hardware_release import build_hardware_release_package
from tools.air56_unoq_validate_hw_binding import REQUIRED_SYMBOLS, SCHEMA as BINDING_SCHEMA


def _adapter_source() -> str:
    return """
#include "air56_unoq_hw_port.h"
extern "C" float board_speed_feedback_rad_s();
extern "C" float board_speed_reference_rad_s();
extern "C" float board_current_d_amp();
extern "C" float board_current_q_amp();
extern "C" float board_dc_bus_volt();
extern "C" float board_current_rms_amp();
extern "C" float board_input_power_watt();
extern "C" uint16_t board_fault_bits();
extern "C" void board_set_flux_reference_amp(float value);
extern "C" float air56_foc_get_omega_meas_rad_s(void) { return board_speed_feedback_rad_s(); }
extern "C" float air56_foc_get_omega_ref_rad_s(void) { return board_speed_reference_rad_s(); }
extern "C" float air56_foc_get_id_amp(void) { return board_current_d_amp(); }
extern "C" float air56_foc_get_iq_amp(void) { return board_current_q_amp(); }
extern "C" float air56_foc_get_vdc_volt(void) { return board_dc_bus_volt(); }
extern "C" float air56_foc_get_irms_amp(void) { return board_current_rms_amp(); }
extern "C" float air56_foc_get_pin_watt(void) { return board_input_power_watt(); }
extern "C" uint16_t air56_foc_get_status_bits(void) { return board_fault_bits(); }
extern "C" void air56_foc_set_id_ref_amp(float id_ref_amp) { board_set_flux_reference_amp(id_ref_amp); }
"""


def _binding_manifest(source: Path) -> dict:
    return {
        "schema": BINDING_SCHEMA,
        "board_id": "unoq-air56-bench-001",
        "board_revision": "revA",
        "stm32": {"mcu": "STM32U585", "board_definition": "custom_unoq_stm32u585", "build_target": "air56_unoq_stm32u585_port"},
        "adapter": {"mock_adapter_enabled": False, "production_build_without_mock": True, "source_files": [source.as_posix()]},
        "serial": {"uart_instance": "USART1", "tx_pin": "PA9", "rx_pin": "PA10", "baud": 921600, "crc_enabled": True},
        "control_loop": {"telemetry_period_ms": 10, "command_timeout_ms": 100},
        "scaling": {
            "current": {"amp_per_adc_count": 0.0025, "offset_calibrated": True},
            "vdc": {"volt_per_adc_count": 0.015},
            "speed": {"source": "encoder_or_observer", "units": "rad_s"},
            "p_in": {"validated": True},
        },
        "faults": {"inverter_fault_pin": "PB2", "inverter_enable_pin": "PB1", "fault_bits_mapped": True, "safe_disable_verified": True},
        "symbol_map": {symbol: f"real::{symbol}" for symbol in REQUIRED_SYMBOLS},
    }


def _hardware_report() -> dict:
    return {
        "schema": "mic_theory.air56_unoq.hardware_acceptance.v1",
        "board_id": "unoq-air56-bench-001",
        "operator": "test",
        "stages": {
            "stage0": {"passed": True, "struct_sizes_ok": True, "crc_error_rejected": True, "loopback_duration_s": 600, "fallback_ms": 80, "telemetry_period_ms_max": 10.5},
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
            "stage2": {"passed": True, "ai_enabled": False, "bridge_dry_run": True, "telemetry_period_ms_max": 10.8, "decoded_telemetry_mismatch_pct": 0.5},
            "stage3": {"passed": True, "ai_enabled": True, "id_ref_limits_tight": True, "disable_on_fault": True, "fallback_ms": 90, "tracking_guard_regression": False},
            "stage4": {"passed": True, "documented": True, "guard_fail_delta": 0, "tracking_guard_regression": False, "current_thermal_limit_ok": True, "fallback_oscillation": False, "power_saving_pct": 0.1},
        },
    }


def _coverage_payload() -> dict:
    return {
        "totals": {"percent_covered": check_air56_unoq_coverage_gate.MIN_TOTAL},
        "files": {
            path: {"summary": {"percent_covered": required}}
            for path, required in check_air56_unoq_coverage_gate.MIN_BY_FILE.items()
        },
    }


def _write_evidence(tmp_path: Path) -> dict[str, Path]:
    source = tmp_path / "air56_real_adapter.cpp"
    source.write_text(_adapter_source(), encoding="utf-8")
    paths = {
        "binding": tmp_path / "binding.json",
        "hardware": tmp_path / "hardware.json",
        "smoke": tmp_path / "smoke.json",
        "coverage": tmp_path / "coverage.json",
    }
    paths["binding"].write_text(json.dumps(_binding_manifest(source)), encoding="utf-8")
    paths["hardware"].write_text(json.dumps(_hardware_report()), encoding="utf-8")
    paths["smoke"].write_text(json.dumps({"passed": True, "steps": []}), encoding="utf-8")
    paths["coverage"].write_text(json.dumps(_coverage_payload()), encoding="utf-8")
    return paths


def test_package_hardware_release_writes_manifest_and_hashes(tmp_path: Path) -> None:
    paths = _write_evidence(tmp_path)
    out_dir = tmp_path / "release_pkg"

    summary = build_hardware_release_package(
        package_tag="bench001",
        out_dir=out_dir,
        binding_manifest=paths["binding"],
        hardware_report=paths["hardware"],
        deploy_smoke_json=paths["smoke"],
        coverage_json=paths["coverage"],
        repo_root=tmp_path,
    )

    manifest = json.loads(Path(summary["manifest"]).read_text(encoding="utf-8"))
    assert summary["release_ready"] is True
    assert manifest["schema"] == air56_unoq_package_hardware_release.SCHEMA
    assert manifest["release_ready"] is True
    assert manifest["package_tag"] == "bench001"
    assert len(manifest["evidence"]) == 5
    assert {row["role"] for row in manifest["evidence"]} == {
        "hardware_binding_manifest",
        "hardware_acceptance_report",
        "deploy_smoke_report",
        "coverage_gate_json",
        "release_gate_summary",
    }
    for row in manifest["evidence"]:
        assert Path(row["path"]).is_file()
        assert row["sha256"] == air56_unoq_package_hardware_release._sha256(Path(row["path"]))


def test_package_hardware_release_cli_status_and_allow_not_ready(tmp_path: Path, monkeypatch) -> None:
    paths = _write_evidence(tmp_path)
    paths["smoke"].write_text(json.dumps({"passed": False}), encoding="utf-8")
    out_dir = tmp_path / "release_pkg"
    monkeypatch.setattr(
        "sys.argv",
        [
            "air56_unoq_package_hardware_release.py",
            "--package-tag",
            "bench001",
            "--out-dir",
            str(out_dir),
            "--binding-manifest",
            str(paths["binding"]),
            "--hardware-report",
            str(paths["hardware"]),
            "--deploy-smoke-json",
            str(paths["smoke"]),
            "--coverage-json",
            str(paths["coverage"]),
            "--repo-root",
            str(tmp_path),
        ],
    )

    assert air56_unoq_package_hardware_release.main() == 1
    manifest = json.loads((out_dir / "hardware_release_manifest.json").read_text(encoding="utf-8"))
    assert manifest["release_ready"] is False


def test_package_hardware_release_cli_allow_not_ready_returns_success(tmp_path: Path, monkeypatch) -> None:
    paths = _write_evidence(tmp_path)
    paths["smoke"].write_text(json.dumps({"passed": False}), encoding="utf-8")
    out_dir = tmp_path / "release_pkg"
    monkeypatch.setattr(
        "sys.argv",
        [
            "air56_unoq_package_hardware_release.py",
            "--package-tag",
            "bench001",
            "--out-dir",
            str(out_dir),
            "--binding-manifest",
            str(paths["binding"]),
            "--hardware-report",
            str(paths["hardware"]),
            "--deploy-smoke-json",
            str(paths["smoke"]),
            "--coverage-json",
            str(paths["coverage"]),
            "--repo-root",
            str(tmp_path),
            "--allow-not-ready",
        ],
    )

    assert air56_unoq_package_hardware_release.main() == 0
    manifest = json.loads((out_dir / "hardware_release_manifest.json").read_text(encoding="utf-8"))
    assert manifest["allow_not_ready"] is True
    assert manifest["release_ready"] is False


def test_package_hardware_release_fails_on_missing_evidence(tmp_path: Path) -> None:
    paths = _write_evidence(tmp_path)
    missing = tmp_path / "missing.json"

    with pytest.raises(FileNotFoundError):
        build_hardware_release_package(
            package_tag="bench001",
            out_dir=tmp_path / "release_pkg",
            binding_manifest=missing,
            hardware_report=paths["hardware"],
            deploy_smoke_json=paths["smoke"],
            coverage_json=paths["coverage"],
            repo_root=tmp_path,
        )


def test_package_hardware_release_git_head_success_and_fallback(tmp_path: Path, monkeypatch) -> None:
    def fake_run_success(cmd, cwd, check, capture_output, text):
        assert cmd == ["git", "rev-parse", "HEAD"]
        assert cwd == tmp_path
        assert check is True
        assert capture_output is True
        assert text is True
        return subprocess.CompletedProcess(cmd, 0, stdout="abc123\n")

    monkeypatch.setattr(air56_unoq_package_hardware_release.subprocess, "run", fake_run_success)
    assert air56_unoq_package_hardware_release._git_head(tmp_path) == "abc123"

    def fake_run_fail(*_args, **_kwargs):
        raise OSError("no git")

    monkeypatch.setattr(air56_unoq_package_hardware_release.subprocess, "run", fake_run_fail)
    assert air56_unoq_package_hardware_release._git_head(tmp_path) == "unknown"
