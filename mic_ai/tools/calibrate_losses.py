from __future__ import annotations

"""Calibrate loss coefficients from time-series CSV logs.

Expected columns (default names):
- omega
- i_rms
- p_el (or p_in_total)
- p_mech
Optional:
- i_d (for psi estimate)
- psi_s (if already computed)

The model fits:
  p_loss ~= a * i_rms^2 + b * |omega|^omega_exp * |psi|^psi_exp
and reports:
  loss_inv_r = a / 3
  loss_core_k = b
"""

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

from mic_ai.core.env import make_env_from_config


def _parse_range(text: str | None) -> Tuple[float, float] | None:
    if not text:
        return None
    raw = str(text).strip().replace(":", ",")
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if len(parts) != 2:
        return None
    try:
        lo = float(parts[0])
        hi = float(parts[1])
    except ValueError:
        return None
    if hi < lo:
        lo, hi = hi, lo
    return lo, hi


def _load_csv(path: Path) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            out: Dict[str, float] = {}
            for k, v in row.items():
                if v is None or v == "":
                    continue
                try:
                    out[k] = float(v)
                except ValueError:
                    continue
            if out:
                rows.append(out)
    return rows


def _collect_files(csv_paths: Iterable[str] | None, directory: str | None, pattern: str) -> List[Path]:
    files: List[Path] = []
    if csv_paths:
        for item in csv_paths:
            path = Path(item).expanduser().resolve()
            if path.is_file():
                files.append(path)
    if directory:
        root = Path(directory).expanduser().resolve()
        if root.is_dir():
            files.extend(sorted(root.glob(pattern)))
    # Deduplicate
    unique = []
    seen = set()
    for path in files:
        if path in seen:
            continue
        seen.add(path)
        unique.append(path)
    return unique


def _fit_coeffs(x1: np.ndarray, x2: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    X = np.stack([x1, x2], axis=1)
    coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
    a = float(max(coeffs[0], 0.0))
    b = float(max(coeffs[1], 0.0))
    y_hat = X @ np.array([a, b], dtype=float)
    rmse = float(np.sqrt(np.mean((y - y_hat) ** 2))) if y.size else 0.0
    return a, b, rmse


def _grid_values(lo: float, hi: float, count: int) -> List[float]:
    if count <= 1:
        return [float(lo)]
    return [float(x) for x in np.linspace(lo, hi, count)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate loss coefficients from CSV logs.")
    parser.add_argument("--csv", nargs="+", help="CSV file(s) with time-series data.")
    parser.add_argument("--dir", default=None, help="Directory with CSV files.")
    parser.add_argument("--pattern", default="*.csv", help="Glob pattern for CSV files.")
    parser.add_argument("--config", default=None, help="Env config (.py) to pull Rs/Rr/B/Lm defaults.")
    parser.add_argument("--omega-col", default="omega")
    parser.add_argument("--i-rms-col", default="i_rms")
    parser.add_argument("--p-in-col", default="p_in_total")
    parser.add_argument("--p-el-col", default="p_el")
    parser.add_argument("--p-mech-col", default="p_mech")
    parser.add_argument("--psi-col", default=None)
    parser.add_argument("--id-col", default=None, help="Column with i_d for psi estimate (psi = Lm * i_d).")
    parser.add_argument("--i-dr-col", default=None, help="Column with rotor i_d (optional).")
    parser.add_argument("--i-qr-col", default=None, help="Column with rotor i_q (optional).")
    parser.add_argument("--lm", type=float, default=None, help="Lm (H) for psi estimate when --id-col is used.")
    parser.add_argument("--rs", type=float, default=None, help="Stator resistance for copper loss removal.")
    parser.add_argument("--rr", type=float, default=None, help="Rotor resistance for copper loss removal.")
    parser.add_argument("--b", type=float, default=None, help="Viscous friction (B) for mech loss removal.")
    parser.add_argument("--subtract-copper", action="store_true", help="Subtract Rs/Rr copper losses before fitting.")
    parser.add_argument("--subtract-mech", action="store_true", help="Subtract B*omega^2 before fitting.")
    parser.add_argument("--omega-exp", type=float, default=1.0)
    parser.add_argument("--psi-exp", type=float, default=2.0)
    parser.add_argument("--omega-exp-range", type=str, default=None, help="Grid search range for omega exponent (min,max).")
    parser.add_argument("--psi-exp-range", type=str, default=None, help="Grid search range for psi exponent (min,max).")
    parser.add_argument("--omega-exp-grid", type=int, default=1, help="Grid points for omega exponent search.")
    parser.add_argument("--psi-exp-grid", type=int, default=1, help="Grid points for psi exponent search.")
    parser.add_argument("--clip-negative", action="store_true", help="Clamp p_loss to >= 0.")
    parser.add_argument("--write-snippet", default=None, help="Write python snippet with calibrated params.")
    parser.add_argument("--write-report", default=None, help="Write JSON report with fitted params.")
    args = parser.parse_args()

    files = _collect_files(args.csv, args.dir, args.pattern)
    if not files:
        raise SystemExit("No CSV files found. Use --csv or --dir.")

    omega_col = args.omega_col
    i_rms_col = args.i_rms_col
    p_mech_col = args.p_mech_col
    p_in_col = args.p_in_col
    p_el_col = args.p_el_col

    rs = args.rs
    rr = args.rr
    b = args.b
    lm = args.lm
    if args.config:
        env_cfg = make_env_from_config(args.config).env_config
        if rs is None:
            rs = float(getattr(env_cfg.motor, "Rs", 0.0))
        if rr is None:
            rr = float(getattr(env_cfg.motor, "Rr", 0.0))
        if b is None:
            b = float(getattr(env_cfg.motor, "B", 0.0))
        if lm is None:
            lm = float(getattr(env_cfg.motor, "Lm", 0.0))

    x1_vals: List[float] = []
    omega_vals: List[float] = []
    psi_vals: List[float] = []
    y_vals: List[float] = []

    for path in files:
        rows = _load_csv(path)
        for row in rows:
            if omega_col not in row or i_rms_col not in row or p_mech_col not in row:
                continue
            omega = float(row[omega_col])
            i_rms = float(row[i_rms_col])
            p_mech = float(row[p_mech_col])

            if p_in_col in row:
                p_in = float(row[p_in_col])
            elif p_el_col in row:
                p_in = float(row[p_el_col])
            else:
                continue

            if not np.isfinite(omega) or not np.isfinite(i_rms) or not np.isfinite(p_in) or not np.isfinite(p_mech):
                continue

            p_in_pos = max(0.0, p_in)
            p_loss = p_in_pos - p_mech
            if args.clip_negative:
                p_loss = max(0.0, p_loss)
            if not np.isfinite(p_loss):
                continue

            psi_val = 1.0
            if args.psi_col and args.psi_col in row:
                psi_val = float(row[args.psi_col])
            elif args.id_col and args.id_col in row and lm is not None:
                psi_val = float(lm) * float(row[args.id_col])

            if args.subtract_copper and rs is not None and rs > 0.0:
                p_loss -= 3.0 * float(rs) * (i_rms ** 2)
            if args.subtract_copper and rr is not None and rr > 0.0 and args.i_dr_col and args.i_qr_col:
                if args.i_dr_col in row and args.i_qr_col in row:
                    i_dr = float(row[args.i_dr_col])
                    i_qr = float(row[args.i_qr_col])
                    i_r_rms = math.sqrt(i_dr * i_dr + i_qr * i_qr)
                    p_loss -= 3.0 * float(rr) * (i_r_rms ** 2)
            if args.subtract_mech and b is not None and b > 0.0:
                p_loss -= float(b) * (omega ** 2)

            x1_vals.append(i_rms * i_rms)
            omega_vals.append(abs(omega))
            psi_vals.append(abs(psi_val))
            y_vals.append(p_loss)

    if not y_vals:
        raise SystemExit("No usable rows found. Check column names.")

    x1 = np.asarray(x1_vals, dtype=float)
    omega_abs = np.asarray(omega_vals, dtype=float)
    psi_abs = np.asarray(psi_vals, dtype=float)
    y = np.asarray(y_vals, dtype=float)
    omega_range = _parse_range(args.omega_exp_range)
    psi_range = _parse_range(args.psi_exp_range)

    omega_exp = float(args.omega_exp)
    psi_exp = float(args.psi_exp)
    best = None
    if omega_range or psi_range:
        omega_lo, omega_hi = omega_range if omega_range else (omega_exp, omega_exp)
        psi_lo, psi_hi = psi_range if psi_range else (psi_exp, psi_exp)
        omega_grid = _grid_values(omega_lo, omega_hi, int(max(args.omega_exp_grid, 1)))
        psi_grid = _grid_values(psi_lo, psi_hi, int(max(args.psi_exp_grid, 1)))
        for omega_exp_val in omega_grid:
            for psi_exp_val in psi_grid:
                x2 = (omega_abs ** float(omega_exp_val)) * (psi_abs ** float(psi_exp_val))
                a, b, rmse = _fit_coeffs(x1, x2, y)
                if best is None or rmse < best["rmse"]:
                    best = {
                        "a": a,
                        "b": b,
                        "rmse": rmse,
                        "omega_exp": float(omega_exp_val),
                        "psi_exp": float(psi_exp_val),
                    }
        assert best is not None
        omega_exp = float(best["omega_exp"])
        psi_exp = float(best["psi_exp"])
        a = float(best["a"])
        b = float(best["b"])
        rmse = float(best["rmse"])
    else:
        x2 = (omega_abs ** float(omega_exp)) * (psi_abs ** float(psi_exp))
        a, b, rmse = _fit_coeffs(x1, x2, y)

    loss_inv_r = a / 3.0
    loss_core_k = b

    print("Calibrated loss parameters:")
    print(f"  loss_inv_r      = {loss_inv_r:.6g}")
    print(f"  loss_core_k     = {loss_core_k:.6g}")
    print(f"  loss_core_omega_exp = {float(omega_exp):.3g}")
    print(f"  loss_core_psi_exp   = {float(psi_exp):.3g}")
    print(f"  rmse (W)        = {rmse:.6g}")

    if args.write_snippet:
        snippet = (
            f"loss_inv_r = {loss_inv_r:.6g}\n"
            f"loss_core_k = {loss_core_k:.6g}\n"
            f"loss_core_omega_exp = {float(omega_exp):.6g}\n"
            f"loss_core_psi_exp = {float(psi_exp):.6g}\n"
        )
        Path(args.write_snippet).write_text(snippet, encoding="utf-8")
        print(f"Wrote snippet to {args.write_snippet}")

    if args.write_report:
        report = {
            "loss_inv_r": loss_inv_r,
            "loss_core_k": loss_core_k,
            "loss_core_omega_exp": float(omega_exp),
            "loss_core_psi_exp": float(psi_exp),
            "rmse": rmse,
            "samples": int(len(y_vals)),
            "grid_used": bool(omega_range or psi_range),
            "omega_exp_range": omega_range,
            "psi_exp_range": psi_range,
            "omega_exp_grid": int(max(args.omega_exp_grid, 1)),
            "psi_exp_grid": int(max(args.psi_exp_grid, 1)),
        }
        Path(args.write_report).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote report to {args.write_report}")


if __name__ == "__main__":
    main()
