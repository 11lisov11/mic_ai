from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def test_check_motor_acceptance_envelopes_smoke(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_json(
        run_dir / "runs" / "air56" / "seed_101" / "mic_summary_rows.json",
        [
            {
                "scenario": "speed_step",
                "err_ok": True,
                "power_saving_pct": 0.8,
                "eta_gain_pct": 0.1,
                "current_peak_ratio": 1.05,
                "current_mean_ratio": 1.01,
            },
            {
                "scenario": "start_stop",
                "err_ok": True,
                "power_saving_pct": -0.3,
                "eta_gain_pct": -1.0,
                "current_peak_ratio": 1.12,
                "current_mean_ratio": 1.05,
            },
        ],
    )

    env_json = tmp_path / "envelopes.json"
    _write_json(
        env_json,
        {
            "common": {
                "speed_step": {
                    "power_saving_pct_min": -0.5,
                    "eta_gain_pct_min": -0.5,
                    "current_peak_ratio_max": 1.2,
                    "current_mean_ratio_max": 1.1,
                    "err_ok_required": True,
                },
                "start_stop": {
                    "power_saving_pct_min": -0.5,
                    "eta_gain_pct_min": -5.0,
                    "current_peak_ratio_max": 1.3,
                    "current_mean_ratio_max": 1.2,
                    "err_ok_required": True,
                },
            },
            "motors": {"air56": {}},
        },
    )

    out_dir = tmp_path / "out"
    cmd = [
        sys.executable,
        "tools/check_motor_acceptance_envelopes.py",
        "--run-dir",
        str(run_dir),
        "--envelopes",
        str(env_json),
        "--motors",
        "air56",
        "--out-dir",
        str(out_dir),
    ]
    subprocess.run(cmd, check=True)

    assert (out_dir / "acceptance_envelope_checks.csv").exists()
    assert (out_dir / "acceptance_envelope_scenarios.csv").exists()
    assert (out_dir / "acceptance_envelope_summary.csv").exists()
    summary = json.loads((out_dir / "acceptance_envelope_summary.json").read_text(encoding="utf-8"))
    assert summary["rows_scenarios"] == 2
    assert summary["rows_summary"] == 2
    assert summary["all_rows_pass"] is True

