from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from tools import build_against_passport_table as passport


def test_build_against_passport_table_smoke(monkeypatch, tmp_path: Path) -> None:
    def _fake_build_rows_for_motor(**kwargs):
        motor = str(kwargs["motor_key"])
        return [
            {
                "motor": motor,
                "policy": "FOC",
                "load_factor": 1.0,
                "p2_kw_model": 0.25,
                "p2_kw_nameplate": 0.25,
                "p2_kw_delta_pct": 0.0,
                "i1_a_model": 0.58,
                "i1_a_nameplate": 0.60,
                "i1_a_delta_pct": -3.33,
                "n2_rpm_model": 1378.0,
                "n2_rpm_nameplate": 1380.0,
                "n2_rpm_delta_pct": -0.14,
                "eta_pct_model": 76.0,
                "eta_pct_nameplate": 74.0,
                "eta_pct_delta_abs": 2.0,
                "cos_phi_model": 0.79,
                "cos_phi_nameplate": 0.78,
                "cos_phi_delta_abs": 0.01,
                "checkpoint_used": "",
                "omega_ref_pu_used": 1.0,
                "load_csv": "dummy.csv",
                "nominal_proxy_used": False,
                "nominal_proxy_abs_err": 0.0,
            }
        ]

    monkeypatch.setattr(passport, "_build_rows_for_motor", _fake_build_rows_for_motor)

    out_root = tmp_path / "passport"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_against_passport_table.py",
            "--motors",
            "air56,al31,ao2",
            "--out-root",
            str(out_root),
            "--tag",
            "t01",
        ],
    )
    passport.main()

    out_dir = out_root / "t01"
    csv_path = out_dir / "passport_compare_3motors.csv"
    md_path = out_dir / "passport_compare_3motors.md"
    json_path = out_dir / "passport_compare_3motors.json"
    assert csv_path.exists()
    assert md_path.exists()
    assert json_path.exists()

    df = pd.read_csv(csv_path)
    assert set(df["motor"].astype(str)) == {"air56", "al31", "ao2"}
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload.get("failures") == []
