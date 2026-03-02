from __future__ import annotations

from pathlib import Path

import pandas as pd

from tools.validate_theory_working_characteristics import run_validation


def _write_csv(path: Path, rows: list[dict]) -> Path:
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return path


def test_theory_validator_passes_on_plausible_curves(tmp_path: Path) -> None:
    rows: list[dict] = []
    for policy in ("FOC", "MIC"):
        p2s = [0.05, 0.10, 0.15, 0.20, 0.23]
        m2s = [0.3, 0.6, 0.9, 1.2, 1.4]
        i1s = [0.22, 0.27, 0.33, 0.41, 0.48]
        n2s = [1384, 1383, 1382, 1381, 1380]
        etas = [62.0, 70.0, 75.0, 77.0, 76.5]
        coss = [0.32, 0.52, 0.68, 0.79, 0.80]
        for p2, m2, i1, n2, eta, cos in zip(p2s, m2s, i1s, n2s, etas, coss):
            rows.append(
                {
                    "policy": policy,
                    "p2_kw": p2,
                    "m2": m2,
                    "i_rms": i1,
                    "n2_rpm": n2,
                    "eta_pct": eta,
                    "cos_phi": cos,
                    "p_el_pos": p2 * 1000.0 / max(eta / 100.0, 1e-6),
                }
            )
    path = _write_csv(tmp_path / "ok.csv", rows)
    report = run_validation(path)
    assert bool(report["passed"]) is True
    assert int(report["hard_fail_count"]) == 0


def test_theory_validator_fails_on_out_of_bounds_cosphi(tmp_path: Path) -> None:
    rows = [
        {"policy": "FOC", "p2_kw": 0.05, "m2": 0.3, "i_rms": 0.2, "n2_rpm": 1384, "eta_pct": 60.0, "cos_phi": 0.3},
        {"policy": "FOC", "p2_kw": 0.10, "m2": 0.6, "i_rms": 0.3, "n2_rpm": 1383, "eta_pct": 70.0, "cos_phi": 1.2},
        {"policy": "FOC", "p2_kw": 0.15, "m2": 0.9, "i_rms": 0.4, "n2_rpm": 1382, "eta_pct": 75.0, "cos_phi": 0.8},
        {"policy": "FOC", "p2_kw": 0.20, "m2": 1.2, "i_rms": 0.5, "n2_rpm": 1381, "eta_pct": 74.0, "cos_phi": 0.82},
    ]
    path = _write_csv(tmp_path / "bad.csv", rows)
    report = run_validation(path)
    assert bool(report["passed"]) is False
    assert int(report["hard_fail_count"]) >= 1

