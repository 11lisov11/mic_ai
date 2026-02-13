import csv
import json
from pathlib import Path

import numpy as np

from mic_ai.tools import calibrate_losses


def _write_csv(path: Path, rows: list[dict]) -> None:
    headers = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_calibrate_losses_cli(tmp_path, monkeypatch) -> None:
    r_inv = 0.5
    b_core = 2.0
    omega = np.linspace(1.0, 5.0, 20)
    i_rms = np.linspace(0.5, 2.0, 20)
    p_mech = np.ones_like(omega) * 10.0
    p_el = p_mech + 3.0 * r_inv * i_rms**2 + b_core * np.abs(omega)

    rows = []
    for o, i, pe, pm in zip(omega, i_rms, p_el, p_mech):
        rows.append({"omega": float(o), "i_rms": float(i), "p_el": float(pe), "p_mech": float(pm)})

    csv_path = tmp_path / "data.csv"
    _write_csv(csv_path, rows)
    snippet = tmp_path / "snippet.txt"

    argv = [
        "calibrate_losses",
        "--csv",
        str(csv_path),
        "--omega-col",
        "omega",
        "--i-rms-col",
        "i_rms",
        "--p-el-col",
        "p_el",
        "--p-mech-col",
        "p_mech",
        "--omega-exp",
        "1.0",
        "--psi-exp",
        "0.0",
        "--write-snippet",
        str(snippet),
    ]
    monkeypatch.setattr("sys.argv", argv)
    calibrate_losses.main()

    content = snippet.read_text(encoding="utf-8")
    assert "loss_inv_r" in content


def test_calibrate_losses_grid_search(tmp_path, monkeypatch) -> None:
    a = 1.2
    b = 0.7
    omega_exp = 1.5
    psi_exp = 2.0
    omega = np.linspace(1.0, 3.0, 15)
    i_rms = np.linspace(0.5, 1.5, 15)
    psi = np.linspace(0.8, 1.2, 15)
    p_mech = np.ones_like(omega) * 5.0
    p_el = p_mech + a * i_rms**2 + b * (np.abs(omega) ** omega_exp) * (np.abs(psi) ** psi_exp)

    rows = []
    for o, i, ps, pe, pm in zip(omega, i_rms, psi, p_el, p_mech):
        rows.append({"omega": float(o), "i_rms": float(i), "psi": float(ps), "p_el": float(pe), "p_mech": float(pm)})

    csv_path = tmp_path / "grid.csv"
    report_path = tmp_path / "report.json"
    _write_csv(csv_path, rows)

    argv = [
        "calibrate_losses",
        "--csv",
        str(csv_path),
        "--omega-col",
        "omega",
        "--i-rms-col",
        "i_rms",
        "--psi-col",
        "psi",
        "--p-el-col",
        "p_el",
        "--p-mech-col",
        "p_mech",
        "--omega-exp-range",
        "1.0,2.0",
        "--psi-exp-range",
        "1.0,3.0",
        "--omega-exp-grid",
        "3",
        "--psi-exp-grid",
        "3",
        "--write-report",
        str(report_path),
    ]
    monkeypatch.setattr("sys.argv", argv)
    calibrate_losses.main()

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["grid_used"] is True
    assert abs(report["loss_core_omega_exp"] - omega_exp) < 1e-6
    assert abs(report["loss_core_psi_exp"] - psi_exp) < 1e-6
