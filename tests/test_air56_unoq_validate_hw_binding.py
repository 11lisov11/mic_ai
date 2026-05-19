from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import air56_unoq_validate_hw_binding
from tools.air56_unoq_validate_hw_binding import build_binding_summary


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


def _manifest(source: Path) -> dict:
    return {
        "schema": air56_unoq_validate_hw_binding.SCHEMA,
        "board_id": "unoq-air56-bench-001",
        "board_revision": "revA",
        "stm32": {
            "mcu": "STM32U585",
            "board_definition": "custom_unoq_stm32u585",
            "build_target": "air56_unoq_stm32u585_port",
        },
        "adapter": {
            "mock_adapter_enabled": False,
            "production_build_without_mock": True,
            "source_files": [source.as_posix()],
        },
        "serial": {
            "uart_instance": "USART1",
            "tx_pin": "PA9",
            "rx_pin": "PA10",
            "baud": 921600,
            "crc_enabled": True,
        },
        "control_loop": {
            "telemetry_period_ms": 10,
            "command_timeout_ms": 100,
        },
        "scaling": {
            "current": {
                "amp_per_adc_count": 0.0025,
                "offset_calibrated": True,
            },
            "vdc": {
                "volt_per_adc_count": 0.015,
            },
            "speed": {
                "source": "encoder_or_observer",
                "units": "rad_s",
            },
            "p_in": {
                "validated": True,
            },
        },
        "faults": {
            "inverter_fault_pin": "PB2",
            "inverter_enable_pin": "PB1",
            "fault_bits_mapped": True,
            "safe_disable_verified": True,
        },
        "symbol_map": {
            symbol: f"real::{symbol}"
            for symbol in air56_unoq_validate_hw_binding.REQUIRED_SYMBOLS
        },
    }


def test_hw_binding_validator_accepts_real_adapter_manifest(tmp_path: Path) -> None:
    source = tmp_path / "air56_real_adapter.cpp"
    source.write_text(_adapter_source(), encoding="utf-8")
    manifest = _manifest(source)

    summary = build_binding_summary(manifest, repo_root=tmp_path)

    assert summary["hardware_binding_ready"] is True
    assert all(row["passed"] for row in summary["checks"])


def test_hw_binding_validator_rejects_template_and_mock_adapter(tmp_path: Path) -> None:
    source = tmp_path / "air56_template.cpp"
    source.write_text(
        """
#include "air56_unoq_hw_mock.h"
extern "C" float air56_foc_get_omega_meas_rad_s(void) { return 0.0f; }
extern "C" void air56_foc_set_id_ref_amp(float id_ref_amp) { (void)id_ref_amp; }
#error "not implemented"
""",
        encoding="utf-8",
    )
    manifest = _manifest(source)
    manifest["adapter"]["mock_adapter_enabled"] = True
    manifest["adapter"]["production_build_without_mock"] = False
    manifest["symbol_map"]["air56_foc_get_iq_amp"] = ""

    summary = build_binding_summary(manifest, repo_root=tmp_path)
    failed = {row["name"] for row in summary["checks"] if not row["passed"]}

    assert summary["hardware_binding_ready"] is False
    assert "adapter.no_mock" in failed
    assert "adapter.production_build" in failed
    assert "adapter.no_forbidden_stub_text" in failed
    assert "adapter.required_symbols" in failed
    assert "adapter.symbol_map" in failed


def test_hw_binding_validator_rejects_missing_manifest_fields_and_sources(tmp_path: Path) -> None:
    missing_source = tmp_path / "missing.cpp"
    manifest = _manifest(missing_source)
    manifest["schema"] = "wrong"
    manifest["board_id"] = ""
    manifest["stm32"]["mcu"] = "STM32F103"
    manifest["serial"]["baud"] = 115200
    manifest["serial"]["crc_enabled"] = False
    manifest["control_loop"]["telemetry_period_ms"] = 20
    manifest["control_loop"]["command_timeout_ms"] = 250
    manifest["scaling"]["current"]["amp_per_adc_count"] = 0
    manifest["scaling"]["current"]["offset_calibrated"] = False
    manifest["scaling"]["vdc"]["volt_per_adc_count"] = 0
    manifest["scaling"]["speed"]["units"] = "rpm"
    manifest["scaling"]["p_in"]["validated"] = False
    manifest["faults"]["fault_bits_mapped"] = False
    manifest["faults"]["safe_disable_verified"] = False
    manifest["faults"]["inverter_fault_pin"] = ""

    summary = build_binding_summary(manifest, repo_root=tmp_path)
    failed = {row["name"] for row in summary["checks"] if not row["passed"]}

    assert summary["hardware_binding_ready"] is False
    assert "schema" in failed
    assert "board_id" in failed
    assert "stm32.mcu" in failed
    assert "adapter.sources_exist" in failed
    assert "serial.baud" in failed
    assert "serial.crc" in failed
    assert "control_loop.period" in failed
    assert "control_loop.timeout" in failed
    assert "scaling.current" in failed
    assert "scaling.vdc" in failed
    assert "scaling.speed" in failed
    assert "scaling.pin" in failed
    assert "faults.fault_bits" in failed
    assert "faults.safe_disable" in failed
    assert "faults.lines" in failed


def test_hw_binding_validator_cli_writes_summary(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "air56_real_adapter.cpp"
    source.write_text(_adapter_source(), encoding="utf-8")
    manifest_path = tmp_path / "binding.json"
    manifest_path.write_text(json.dumps(_manifest(source)), encoding="utf-8")
    out_json = tmp_path / "summary.json"

    monkeypatch.setattr(
        "sys.argv",
        [
            "air56_unoq_validate_hw_binding.py",
            "--manifest",
            str(manifest_path),
            "--repo-root",
            str(tmp_path),
            "--out-json",
            str(out_json),
        ],
    )

    assert air56_unoq_validate_hw_binding.main() == 0
    assert json.loads(out_json.read_text(encoding="utf-8"))["hardware_binding_ready"] is True

    manifest = _manifest(source)
    manifest["adapter"]["source_files"] = []
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert air56_unoq_validate_hw_binding.main() == 1


def test_hw_binding_validator_rejects_non_object_manifest(tmp_path: Path) -> None:
    manifest_path = tmp_path / "bad.json"
    manifest_path.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="must be a JSON object"):
        air56_unoq_validate_hw_binding._load_manifest(manifest_path)


def test_hw_binding_validator_helper_edges(tmp_path: Path) -> None:
    helper = air56_unoq_validate_hw_binding

    assert helper._dict({"x": []}, "x") == {}
    assert helper._bool({"x": "yes"}, "x") is True
    assert helper._bool({"x": 0}, "x") is False
    assert helper._float({"x": object()}, "x", default=3.0) == 3.0
    assert helper._list({"x": "bad"}, "x") == []
    assert helper._source_forbidden_hits("return 0;") == ["adapter source must not return a constant zero stub"]
    assert helper._resolve_source_paths({"adapter": {"source_files": ["adapter.cpp", ""]}}, tmp_path) == [tmp_path / "adapter.cpp"]
