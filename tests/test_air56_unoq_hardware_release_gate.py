from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import air56_unoq_hardware_release_gate
from tools import check_air56_unoq_coverage_gate
from tools.air56_unoq_hardware_release_gate import build_release_gate_summary
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


def test_release_gate_passes_with_all_evidence(tmp_path: Path) -> None:
    paths = _write_evidence(tmp_path)

    summary = build_release_gate_summary(
        binding_manifest=paths["binding"],
        hardware_report=paths["hardware"],
        deploy_smoke_json=paths["smoke"],
        coverage_json=paths["coverage"],
        repo_root=tmp_path,
    )

    assert summary["release_ready"] is True
    assert all(row["passed"] for row in summary["checks"])
    assert summary["details"]["coverage_gate"]["passed"] is True


def test_release_gate_fails_safe_for_bad_evidence(tmp_path: Path) -> None:
    paths = _write_evidence(tmp_path)
    hardware = _hardware_report()
    hardware["stages"]["stage4"]["documented"] = False
    paths["hardware"].write_text(json.dumps(hardware), encoding="utf-8")
    paths["smoke"].write_text(json.dumps({"passed": False}), encoding="utf-8")
    coverage = _coverage_payload()
    coverage["files"]["tools/uno_q_protocol.py"]["summary"]["percent_covered"] = 1.0
    paths["coverage"].write_text(json.dumps(coverage), encoding="utf-8")

    summary = build_release_gate_summary(
        binding_manifest=paths["binding"],
        hardware_report=paths["hardware"],
        deploy_smoke_json=paths["smoke"],
        coverage_json=paths["coverage"],
        repo_root=tmp_path,
    )
    failed = {row["name"] for row in summary["checks"] if not row["passed"]}

    assert summary["release_ready"] is False
    assert failed == {"hardware_acceptance", "deploy_smoke", "coverage_gate"}
    assert summary["details"]["hardware_acceptance"]["hardware_ready"] is False
    assert summary["details"]["deploy_smoke"]["passed"] is False
    assert summary["details"]["coverage_gate"]["passed"] is False


def test_release_gate_records_loading_errors_as_failed_checks(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    summary = build_release_gate_summary(
        binding_manifest=missing,
        hardware_report=missing,
        deploy_smoke_json=missing,
        coverage_json=missing,
        repo_root=tmp_path,
    )

    assert summary["release_ready"] is False
    assert {row["name"] for row in summary["checks"] if not row["passed"]} == {
        "hardware_binding",
        "hardware_acceptance",
        "deploy_smoke",
        "coverage_gate",
    }


def test_release_gate_cli_writes_summary(tmp_path: Path, monkeypatch) -> None:
    paths = _write_evidence(tmp_path)
    out_json = tmp_path / "release_gate.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "air56_unoq_hardware_release_gate.py",
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
            "--out-json",
            str(out_json),
        ],
    )

    assert air56_unoq_hardware_release_gate.main() == 0
    assert json.loads(out_json.read_text(encoding="utf-8"))["release_ready"] is True

    paths["smoke"].write_text(json.dumps({"passed": False}), encoding="utf-8")
    assert air56_unoq_hardware_release_gate.main() == 1


def test_release_gate_rejects_non_object_auxiliary_json(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="deploy smoke report must be a JSON object"):
        air56_unoq_hardware_release_gate._load_json_object(bad, label="deploy smoke report")
