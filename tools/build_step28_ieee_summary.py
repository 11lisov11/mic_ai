from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.common_utils import json_load as _json_load_shared
from tools.common_utils import read_csv as _read_csv_shared
from tools.common_utils import write_csv as _write_csv_shared
from tools.step27_artifacts import find_acceptance_json


def _read_csv(path: Path) -> List[Dict[str, str]]:
    return _read_csv_shared(path)


def _read_json(path: Path) -> Dict[str, object]:
    return _json_load_shared(path)


def _to_float(row: Dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "0"))
    except Exception:
        return 0.0


def _fmt(v: float) -> str:
    return f"{float(v):.3f}"


def _collect_mode(mode_name: str, mode_dir: Path) -> Dict[str, object]:
    global_rows = _read_csv(mode_dir / "step27_final_pi_vs_foc_vs_mic.csv")
    motor_rows = _read_csv(mode_dir / "step27_stats_motor_controller.csv")
    acceptance = _read_json(find_acceptance_json(mode_dir))
    reproducibility = _read_json(mode_dir / "step27_reproducibility.json")

    idx_global = {str(r["controller"]): r for r in global_rows}
    idx_motor = {(str(r["motor"]), str(r["controller"])): r for r in motor_rows}

    return {
        "name": mode_name,
        "dir": str(mode_dir),
        "global_rows": idx_global,
        "motor_rows": idx_motor,
        "acceptance": acceptance,
        "reproducibility": reproducibility,
    }


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    _write_csv_shared(path, rows)


def _build_md(mode1: Dict[str, object], mode2: Dict[str, object]) -> str:
    lines: List[str] = []
    lines.append("# Step28 IEEE Summary")
    lines.append("")

    for mode in (mode1, mode2):
        name = str(mode["name"])
        acc = dict(mode["acceptance"])
        repro = dict(mode["reproducibility"])
        glob = dict(mode["global_rows"])
        mic = dict(glob.get("MIC", {}))
        foc = dict(glob.get("FOC", {}))
        lines.append(f"## {name}")
        lines.append("")
        lines.append(f"- out_dir: `{mode['dir']}`")
        lines.append(
            "- AIR56 acceptance: mean=`{}` worst_case=`{}`".format(
                bool(acc.get("mean_pass", False)),
                bool(acc.get("worst_case_pass", False)),
            )
        )
        lines.append(
            "- reproducibility: stable_vs_previous=`{}` sha=`{}`".format(
                repro.get("stable_vs_previous"),
                repro.get("table_sha256", ""),
            )
        )
        lines.append(
            "- MIC global: power(mean/std/min)=`{}/{}/{}` eta(mean/std/min)=`{}/{}/{}` err(mean/max)=`{}/{}` start_stop(mean/min)=`{}/{}`".format(
                _fmt(_to_float(mic, "avg_power_saving_pct_mean")),
                _fmt(_to_float(mic, "avg_power_saving_pct_std")),
                _fmt(_to_float(mic, "avg_power_saving_pct_min")),
                _fmt(_to_float(mic, "avg_eta_gain_pct_mean")),
                _fmt(_to_float(mic, "avg_eta_gain_pct_std")),
                _fmt(_to_float(mic, "avg_eta_gain_pct_min")),
                _fmt(_to_float(mic, "err_failures_mean")),
                _fmt(_to_float(mic, "err_failures_max")),
                _fmt(_to_float(mic, "start_stop_power_saving_pct_mean")),
                _fmt(_to_float(mic, "start_stop_power_saving_pct_min")),
            )
        )
        lines.append(
            "- FOC global: power(mean/std/min)=`{}/{}/{}` eta(mean/std/min)=`{}/{}/{}` err(mean/max)=`{}/{}` start_stop(mean/min)=`{}/{}`".format(
                _fmt(_to_float(foc, "avg_power_saving_pct_mean")),
                _fmt(_to_float(foc, "avg_power_saving_pct_std")),
                _fmt(_to_float(foc, "avg_power_saving_pct_min")),
                _fmt(_to_float(foc, "avg_eta_gain_pct_mean")),
                _fmt(_to_float(foc, "avg_eta_gain_pct_std")),
                _fmt(_to_float(foc, "avg_eta_gain_pct_min")),
                _fmt(_to_float(foc, "err_failures_mean")),
                _fmt(_to_float(foc, "err_failures_max")),
                _fmt(_to_float(foc, "start_stop_power_saving_pct_mean")),
                _fmt(_to_float(foc, "start_stop_power_saving_pct_min")),
            )
        )
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build IEEE mode1/mode2 combined summary for step28.")
    parser.add_argument("--mode1-dir", required=True)
    parser.add_argument("--mode2-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    mode1 = _collect_mode("mode1_foc_encoder_vs_mic_sensorless", Path(args.mode1_dir).resolve())
    mode2 = _collect_mode("mode2_foc_sensorless_vs_mic_sensorless", Path(args.mode2_dir).resolve())

    rows: List[Dict[str, object]] = []
    for mode in (mode1, mode2):
        name = str(mode["name"])
        acc = dict(mode["acceptance"])
        repro = dict(mode["reproducibility"])
        glob = dict(mode["global_rows"])
        for controller in ("PI", "FOC", "MIC"):
            row = dict(glob.get(controller, {}))
            if not row:
                continue
            rows.append(
                {
                    "mode": name,
                    "controller": controller,
                    "avg_power_saving_pct_mean": _to_float(row, "avg_power_saving_pct_mean"),
                    "avg_power_saving_pct_std": _to_float(row, "avg_power_saving_pct_std"),
                    "avg_power_saving_pct_min": _to_float(row, "avg_power_saving_pct_min"),
                    "avg_eta_gain_pct_mean": _to_float(row, "avg_eta_gain_pct_mean"),
                    "avg_eta_gain_pct_std": _to_float(row, "avg_eta_gain_pct_std"),
                    "avg_eta_gain_pct_min": _to_float(row, "avg_eta_gain_pct_min"),
                    "err_failures_mean": _to_float(row, "err_failures_mean"),
                    "err_failures_max": _to_float(row, "err_failures_max"),
                    "start_stop_power_saving_pct_mean": _to_float(row, "start_stop_power_saving_pct_mean"),
                    "start_stop_power_saving_pct_min": _to_float(row, "start_stop_power_saving_pct_min"),
                    "air56_mean_pass": bool(acc.get("mean_pass", False)),
                    "air56_worst_case_pass": bool(acc.get("worst_case_pass", False)),
                    "stable_vs_previous": repro.get("stable_vs_previous"),
                    "table_sha256": str(repro.get("table_sha256", "")),
                }
            )

    _write_csv(out_dir / "step28_ieee_summary.csv", rows)
    (out_dir / "step28_ieee_summary.md").write_text(_build_md(mode1, mode2), encoding="utf-8")


if __name__ == "__main__":
    main()
