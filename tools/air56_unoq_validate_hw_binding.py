from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


SCHEMA = "mic_theory.air56_unoq.hw_binding.v1"
REQUIRED_SYMBOLS = (
    "air56_foc_get_omega_meas_rad_s",
    "air56_foc_get_omega_ref_rad_s",
    "air56_foc_get_id_amp",
    "air56_foc_get_iq_amp",
    "air56_foc_get_vdc_volt",
    "air56_foc_get_irms_amp",
    "air56_foc_get_pin_watt",
    "air56_foc_get_status_bits",
    "air56_foc_set_id_ref_amp",
)

FORBIDDEN_SOURCE_PATTERNS = (
    (re.compile(r"#\s*error\b"), "adapter source must not contain #error"),
    (re.compile(r"\bAIR56_UNOQ_USE_MOCK_HW\b"), "production adapter must not define/use mock hardware"),
    (re.compile(r"\bair56_unoq_hw_mock\b"), "production adapter must not include mock adapter"),
    (re.compile(r"\bTODO\b|\bFIXME\b|not\s+implemented|template\s+only", re.IGNORECASE), "adapter source must not contain TODO/stub text"),
    (re.compile(r"\(void\)\s*id_ref_amp"), "id_ref setter must use id_ref_amp"),
    (re.compile(r"return\s+0(?:\.0f?|u)?\s*;"), "adapter source must not return a constant zero stub"),
)


@dataclass(frozen=True)
class BindingCheck:
    name: str
    passed: bool
    detail: str


def _load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"hardware binding manifest must be a JSON object: {path}")
    return payload


def _dict(payload: dict[str, Any], name: str) -> dict[str, Any]:
    value = payload.get(name, {})
    return value if isinstance(value, dict) else {}


def _bool(payload: dict[str, Any], name: str, default: bool = False) -> bool:
    value = payload.get(name, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on", "ok", "pass", "passed"}


def _float(payload: dict[str, Any], name: str, default: float = float("inf")) -> float:
    try:
        return float(payload.get(name, default))
    except (TypeError, ValueError):
        return float(default)


def _nonempty(payload: dict[str, Any], name: str) -> bool:
    return bool(str(payload.get(name, "")).strip())


def _list(payload: dict[str, Any], name: str) -> list[Any]:
    value = payload.get(name, [])
    return value if isinstance(value, list) else []


def _resolve_source_paths(manifest: dict[str, Any], repo_root: Path) -> list[Path]:
    adapter = _dict(manifest, "adapter")
    paths: list[Path] = []
    for item in _list(adapter, "source_files"):
        raw = str(item).strip()
        if not raw:
            continue
        path = Path(raw)
        paths.append(path if path.is_absolute() else repo_root / path)
    return paths


def _read_sources(paths: list[Path]) -> tuple[str, list[Path]]:
    chunks: list[str] = []
    missing: list[Path] = []
    for path in paths:
        if not path.is_file():
            missing.append(path)
            continue
        chunks.append(path.read_text(encoding="utf-8", errors="replace"))
    return "\n".join(chunks), missing


def _has_symbol_implementation(source: str, symbol: str) -> bool:
    pattern = re.compile(rf"\b{re.escape(symbol)}\s*\([^)]*\)\s*(?:noexcept\s*)?\{{", re.MULTILINE)
    return bool(pattern.search(source))


def _source_forbidden_hits(source: str) -> list[str]:
    hits: list[str] = []
    for pattern, detail in FORBIDDEN_SOURCE_PATTERNS:
        if pattern.search(source):
            hits.append(detail)
    return hits


def _symbol_map_complete(manifest: dict[str, Any]) -> bool:
    symbol_map = _dict(manifest, "symbol_map")
    return all(_nonempty(symbol_map, symbol) for symbol in REQUIRED_SYMBOLS)


def evaluate_hw_binding(manifest: dict[str, Any], *, repo_root: Path) -> list[BindingCheck]:
    stm32 = _dict(manifest, "stm32")
    serial = _dict(manifest, "serial")
    control_loop = _dict(manifest, "control_loop")
    adapter = _dict(manifest, "adapter")
    scaling = _dict(manifest, "scaling")
    current = _dict(scaling, "current")
    vdc = _dict(scaling, "vdc")
    speed = _dict(scaling, "speed")
    p_in = _dict(scaling, "p_in")
    faults = _dict(manifest, "faults")

    source_paths = _resolve_source_paths(manifest, repo_root)
    source_text, missing_sources = _read_sources(source_paths)
    forbidden_hits = _source_forbidden_hits(source_text)
    missing_symbols = [symbol for symbol in REQUIRED_SYMBOLS if not _has_symbol_implementation(source_text, symbol)]

    checks = [
        BindingCheck("schema", str(manifest.get("schema", "")) == SCHEMA, f"schema must be {SCHEMA}"),
        BindingCheck("board_id", _nonempty(manifest, "board_id"), "board_id is required"),
        BindingCheck("board_revision", _nonempty(manifest, "board_revision"), "board_revision is required"),
        BindingCheck("stm32.mcu", "STM32U585" in str(stm32.get("mcu", "")), "MCU must be STM32U585"),
        BindingCheck("stm32.board_definition", _nonempty(stm32, "board_definition"), "board definition is required"),
        BindingCheck("stm32.build_target", _nonempty(stm32, "build_target"), "production build target is required"),
        BindingCheck("adapter.no_mock", not _bool(adapter, "mock_adapter_enabled", True), "mock adapter must be disabled"),
        BindingCheck("adapter.production_build", _bool(adapter, "production_build_without_mock"), "production build without mock must be verified"),
        BindingCheck("adapter.source_files", bool(source_paths), "adapter source file list is required"),
        BindingCheck("adapter.sources_exist", not missing_sources, f"missing adapter sources: {[str(path) for path in missing_sources]}"),
        BindingCheck("adapter.no_forbidden_stub_text", not forbidden_hits, "; ".join(forbidden_hits) if forbidden_hits else "no forbidden stub text"),
        BindingCheck("adapter.required_symbols", not missing_symbols, f"missing symbols: {missing_symbols}" if missing_symbols else "all required symbols implemented"),
        BindingCheck("adapter.symbol_map", _symbol_map_complete(manifest), "symbol_map must document every air56_foc_* mapping"),
        BindingCheck("serial.uart_instance", _nonempty(serial, "uart_instance"), "UART instance is required"),
        BindingCheck("serial.tx_pin", _nonempty(serial, "tx_pin"), "TX pin is required"),
        BindingCheck("serial.rx_pin", _nonempty(serial, "rx_pin"), "RX pin is required"),
        BindingCheck("serial.baud", _float(serial, "baud", 0.0) >= 921600.0, "baud must be >= 921600"),
        BindingCheck("serial.crc", _bool(serial, "crc_enabled"), "CRC must be enabled"),
        BindingCheck("control_loop.period", _float(control_loop, "telemetry_period_ms", 999.0) <= 10.0, "telemetry target must be <= 10 ms"),
        BindingCheck("control_loop.timeout", _float(control_loop, "command_timeout_ms", 999.0) <= 100.0, "command timeout must be <= 100 ms"),
        BindingCheck("scaling.current", _float(current, "amp_per_adc_count", 0.0) > 0.0 and _bool(current, "offset_calibrated"), "current scaling and offset calibration are required"),
        BindingCheck("scaling.vdc", _float(vdc, "volt_per_adc_count", 0.0) > 0.0, "Vdc scaling is required"),
        BindingCheck("scaling.speed", str(speed.get("units", "")).strip() == "rad_s" and _nonempty(speed, "source"), "speed source must be rad_s"),
        BindingCheck("scaling.pin", _bool(p_in, "validated"), "P_in estimator must be validated"),
        BindingCheck("faults.fault_bits", _bool(faults, "fault_bits_mapped"), "fault bits must be mapped"),
        BindingCheck("faults.safe_disable", _bool(faults, "safe_disable_verified"), "safe disable must be verified"),
        BindingCheck("faults.lines", _nonempty(faults, "inverter_fault_pin") and _nonempty(faults, "inverter_enable_pin"), "fault and enable pins are required"),
    ]
    return checks


def build_binding_summary(manifest: dict[str, Any], *, repo_root: Path) -> dict[str, Any]:
    checks = evaluate_hw_binding(manifest, repo_root=repo_root)
    return {
        "hardware_binding_ready": all(check.passed for check in checks),
        "checks": [asdict(check) for check in checks],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate AIR56 UNO Q STM32U585 production hardware binding manifest and adapter source.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--out-json", default="")
    args = parser.parse_args()

    repo_root = Path(str(args.repo_root)).resolve()
    manifest = _load_manifest(Path(str(args.manifest)).resolve())
    summary = build_binding_summary(manifest, repo_root=repo_root)
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    if str(args.out_json).strip():
        out_path = Path(str(args.out_json)).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if bool(summary["hardware_binding_ready"]) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
