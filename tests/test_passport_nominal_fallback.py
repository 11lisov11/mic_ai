from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import pandas as pd
import pytest

from tools import build_against_passport_table as passport


def _write_load_csv(
    path: Path,
    *,
    foc_p2: float,
    foc_i: float,
    foc_n: float,
    foc_eta: float,
    foc_cos: float,
    mic_p2: float,
    mic_i: float,
    mic_n: float,
    mic_eta: float,
    mic_cos: float,
) -> None:
    rows = [
        {"policy": "FOC", "load_factor": 1.0, "p2_kw": foc_p2, "i_rms": foc_i, "n2_rpm": foc_n, "eta_pct": foc_eta, "cos_phi": foc_cos},
        {"policy": "MIC_AI", "load_factor": 1.0, "p2_kw": mic_p2, "i_rms": mic_i, "n2_rpm": mic_n, "eta_pct": mic_eta, "cos_phi": mic_cos},
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_passport_fallback_to_next_omega_when_nominal_row_invalid(monkeypatch, tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    calls: list[float] = []

    def _fake_run_drive_characteristics(**kwargs):
        omega = float(kwargs["omega_ref_pu"])
        calls.append(omega)
        csv_path = Path(kwargs["out_dir"]) / "load_characteristics.csv"
        if abs(omega - 1.0) < 1e-12:
            _write_load_csv(
                csv_path,
                foc_p2=0.0,
                foc_i=float("nan"),
                foc_n=0.0,
                foc_eta=0.0,
                foc_cos=float("nan"),
                mic_p2=0.0,
                mic_i=10.0,
                mic_n=0.0,
                mic_eta=0.0,
                mic_cos=0.0,
            )
        else:
            _write_load_csv(
                csv_path,
                foc_p2=0.45,
                foc_i=1.8,
                foc_n=1100.0,
                foc_eta=70.0,
                foc_cos=0.82,
                mic_p2=0.44,
                mic_i=1.7,
                mic_n=1098.0,
                mic_eta=71.0,
                mic_cos=0.83,
            )

    monkeypatch.setattr(passport, "_run_drive_characteristics", _fake_run_drive_characteristics)
    monkeypatch.setattr(passport, "_resolve_checkpoint", lambda *a, **k: None)
    monkeypatch.setattr(
        passport,
        "_load_module",
        lambda _p: SimpleNamespace(NAMEPLATE_TEST={"P_n": 600.0, "I_n": 1.7, "n_rated": 1390.0, "eta_n": 0.78, "cos_phi_n": 0.74}),
    )

    rows = passport._build_rows_for_motor(
        motor_key="al31",
        config_path="dummy.py",
        raw_dir=raw_dir,
        checkpoint_registry="dummy.json",
        load_factors="0.2,0.4,0.6,0.8,1.0",
        t_end=1.0,
        window_frac=0.3,
    )

    assert len(rows) == 2
    assert calls[:2] == [1.0, 0.9]
    assert all(abs(float(r["omega_ref_pu_used"]) - 0.9) < 1e-12 for r in rows)
    assert all(float(r["p2_kw_model"]) > 0.0 for r in rows)


def test_passport_raises_when_all_omega_invalid(monkeypatch, tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"

    def _fake_run_drive_characteristics(**kwargs):
        csv_path = Path(kwargs["out_dir"]) / "load_characteristics.csv"
        _write_load_csv(
            csv_path,
            foc_p2=0.0,
            foc_i=float("nan"),
            foc_n=0.0,
            foc_eta=0.0,
            foc_cos=float("nan"),
            mic_p2=0.0,
            mic_i=0.0,
            mic_n=0.0,
            mic_eta=0.0,
            mic_cos=0.0,
        )

    monkeypatch.setattr(passport, "_run_drive_characteristics", _fake_run_drive_characteristics)
    monkeypatch.setattr(passport, "_resolve_checkpoint", lambda *a, **k: None)
    monkeypatch.setattr(
        passport,
        "_load_module",
        lambda _p: SimpleNamespace(NAMEPLATE_TEST={"P_n": 3000.0, "I_n": 7.2, "n_rated": 1430.0, "eta_n": 0.84, "cos_phi_n": 0.82}),
    )

    with pytest.raises(RuntimeError):
        passport._build_rows_for_motor(
            motor_key="ao2",
            config_path="dummy.py",
            raw_dir=raw_dir,
            checkpoint_registry="dummy.json",
            load_factors="0.2,0.4,0.6,0.8,1.0",
            t_end=1.0,
            window_frac=0.3,
        )


def test_passport_accepts_foc_only_when_mic_unavailable(monkeypatch, tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"

    def _fake_run_drive_characteristics(**kwargs):
        csv_path = Path(kwargs["out_dir"]) / "load_characteristics.csv"
        _write_load_csv(
            csv_path,
            foc_p2=0.40,
            foc_i=1.6,
            foc_n=1080.0,
            foc_eta=68.0,
            foc_cos=0.80,
            mic_p2=0.0,
            mic_i=0.0,
            mic_n=0.0,
            mic_eta=0.0,
            mic_cos=0.0,
        )

    monkeypatch.setattr(passport, "_run_drive_characteristics", _fake_run_drive_characteristics)
    monkeypatch.setattr(passport, "_resolve_checkpoint", lambda *a, **k: None)
    monkeypatch.setattr(
        passport,
        "_load_module",
        lambda _p: SimpleNamespace(NAMEPLATE_TEST={"P_n": 600.0, "I_n": 1.7, "n_rated": 1390.0, "eta_n": 0.78, "cos_phi_n": 0.74}),
    )

    rows = passport._build_rows_for_motor(
        motor_key="al31",
        config_path="dummy.py",
        raw_dir=raw_dir,
        checkpoint_registry="dummy.json",
        load_factors="0.2,0.4,0.6,0.8,1.0",
        t_end=1.0,
        window_frac=0.3,
    )
    assert len(rows) == 1
    assert str(rows[0]["policy"]) == "FOC"
