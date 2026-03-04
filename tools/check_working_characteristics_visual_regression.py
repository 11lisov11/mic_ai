from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


REQUIRED_COLUMNS: Tuple[str, ...] = (
    "policy",
    "P2_kW",
    "M2_Nm",
    "I1_A",
    "n2_rpm",
    "eta_pct",
    "cosphi",
)


def _score_monotonic(
    x: np.ndarray,
    y: np.ndarray,
    *,
    direction: str,
    eps: float = 1e-9,
) -> Dict[str, float]:
    if x.size < 3:
        return {"ok": 1.0, "violation_ratio": 0.0}
    dy = np.diff(y)
    if direction == "inc":
        bad = int(np.sum(dy < -abs(eps)))
    elif direction == "dec":
        bad = int(np.sum(dy > abs(eps)))
    else:
        raise ValueError(direction)
    total = max(int(dy.size), 1)
    ratio = float(bad / total)
    return {"ok": 1.0 if ratio <= 0.25 else 0.0, "violation_ratio": ratio}


def _score_hump(y: np.ndarray, eps: float = 1e-6, max_sign_changes: int = 3) -> Dict[str, float]:
    if y.size < 4:
        return {"ok": 1.0, "sign_changes": 0.0}
    dy = np.diff(y)
    sign = np.zeros_like(dy, dtype=int)
    sign[dy > eps] = 1
    sign[dy < -eps] = -1
    compact: List[int] = []
    for s in sign.tolist():
        if s == 0:
            continue
        if not compact or compact[-1] != s:
            compact.append(s)
    # Hump-like with tolerance to small local oscillations from sampled traces.
    sign_changes = int(max(0, len(compact) - 1))
    ok = 1.0 if sign_changes <= int(max_sign_changes) else 0.0
    return {"ok": ok, "sign_changes": float(sign_changes)}


def _interp_grid(x: np.ndarray, y: np.ndarray, grid: np.ndarray) -> np.ndarray:
    return np.interp(grid, x, y)


def _prepare_policy(df: pd.DataFrame, policy: str) -> pd.DataFrame:
    sub = df[df["policy"].astype(str) == policy].copy()
    sub = sub.sort_values("P2_kW")
    sub = sub.drop_duplicates("P2_kW", keep="last")
    return sub


def _load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    for c in REQUIRED_COLUMNS:
        if c == "policy":
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=[c for c in REQUIRED_COLUMNS if c != "policy"])
    return df


def run_check(
    *,
    csv_path: Path,
    foc_policy: str,
    mic_policy: str,
    x_min_tol_kw: float,
    x_max_tol_kw: float,
    out_json: Path | None = None,
    out_md: Path | None = None,
) -> Dict[str, object]:
    df = _load_csv(csv_path)
    foc = _prepare_policy(df, foc_policy)
    mic = _prepare_policy(df, mic_policy)
    if foc.empty or mic.empty:
        raise ValueError(f"Missing policy data: foc={foc_policy} rows={len(foc)}, mic={mic_policy} rows={len(mic)}")

    x_f = foc["P2_kW"].to_numpy(dtype=float)
    x_m = mic["P2_kW"].to_numpy(dtype=float)

    axis_min_diff = abs(float(np.nanmin(x_f)) - float(np.nanmin(x_m)))
    axis_max_diff = abs(float(np.nanmax(x_f)) - float(np.nanmax(x_m)))
    axis_ok = bool(axis_min_diff <= float(x_min_tol_kw) and axis_max_diff <= float(x_max_tol_kw))

    shape_rows: List[Dict[str, object]] = []
    for name, grp in ((foc_policy, foc), (mic_policy, mic)):
        x = grp["P2_kW"].to_numpy(dtype=float)
        m2 = grp["M2_Nm"].to_numpy(dtype=float)
        i1 = grp["I1_A"].to_numpy(dtype=float)
        n2 = grp["n2_rpm"].to_numpy(dtype=float)
        eta = grp["eta_pct"].to_numpy(dtype=float)
        cosphi = grp["cosphi"].to_numpy(dtype=float)

        s_m2 = _score_monotonic(x, m2, direction="inc")
        s_i1 = _score_monotonic(x, i1, direction="inc")
        s_n2 = _score_monotonic(x, n2, direction="dec")
        s_eta = _score_hump(eta)
        s_cosphi = _score_hump(cosphi)
        eta_range_ok = bool(np.nanmin(eta) >= -1e-6 and np.nanmax(eta) <= 100.0 + 1e-6)
        cosphi_range_ok = bool(np.nanmin(cosphi) >= -1e-6 and np.nanmax(cosphi) <= 1.0 + 1e-6)
        shape_ok = bool(
            s_m2["ok"] and s_i1["ok"] and s_n2["ok"] and s_eta["ok"] and s_cosphi["ok"] and eta_range_ok and cosphi_range_ok
        )
        shape_rows.append(
            {
                "policy": name,
                "shape_ok": shape_ok,
                "M2_violation_ratio": float(s_m2["violation_ratio"]),
                "I1_violation_ratio": float(s_i1["violation_ratio"]),
                "n2_violation_ratio": float(s_n2["violation_ratio"]),
                "eta_sign_changes": float(s_eta["sign_changes"]),
                "cosphi_sign_changes": float(s_cosphi["sign_changes"]),
                "eta_range_ok": eta_range_ok,
                "cosphi_range_ok": cosphi_range_ok,
            }
        )

    # Cross-policy drift on overlap (weak visual regression signal).
    lo = max(float(np.nanmin(x_f)), float(np.nanmin(x_m)))
    hi = min(float(np.nanmax(x_f)), float(np.nanmax(x_m)))
    overlap_ok = bool(hi > lo)
    drift: Dict[str, float] = {}
    if overlap_ok:
        grid = np.linspace(lo, hi, 64)
        for src, col in (
            ("M2_Nm", "M2"),
            ("I1_A", "I1"),
            ("n2_rpm", "n2"),
            ("eta_pct", "eta"),
            ("cosphi", "cosphi"),
        ):
            y_f = _interp_grid(x_f, foc[src].to_numpy(dtype=float), grid)
            y_m = _interp_grid(x_m, mic[src].to_numpy(dtype=float), grid)
            denom = max(float(np.nanmax(np.abs(y_f))), 1e-9)
            drift[f"{col}_nrmse"] = float(np.sqrt(np.mean((y_m - y_f) ** 2)) / denom)

    all_shape_ok = bool(all(bool(r["shape_ok"]) for r in shape_rows))
    passed = bool(axis_ok and all_shape_ok and overlap_ok)
    payload: Dict[str, object] = {
        "csv_path": str(csv_path),
        "foc_policy": foc_policy,
        "mic_policy": mic_policy,
        "axis_consistency": {
            "ok": axis_ok,
            "min_diff_kw": axis_min_diff,
            "max_diff_kw": axis_max_diff,
            "x_min_tol_kw": float(x_min_tol_kw),
            "x_max_tol_kw": float(x_max_tol_kw),
        },
        "shape_checks": shape_rows,
        "cross_policy_drift": drift,
        "overlap_ok": overlap_ok,
        "passed": passed,
    }

    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if out_md is not None:
        out_md.parent.mkdir(parents=True, exist_ok=True)
        lines: List[str] = []
        lines.append("# Visual Regression Report")
        lines.append("")
        lines.append(f"- csv_path: `{csv_path}`")
        lines.append(f"- passed: `{passed}`")
        axis = payload["axis_consistency"]
        lines.append(f"- axis_ok: `{axis['ok']}` (min_diff={axis['min_diff_kw']:.6f}, max_diff={axis['max_diff_kw']:.6f})")
        lines.append("")
        lines.append("## Shape Checks")
        for row in shape_rows:
            lines.append(f"- `{row['policy']}`: shape_ok=`{row['shape_ok']}`")
            lines.append(
                f"  M2_violation={row['M2_violation_ratio']:.3f}, "
                f"I1_violation={row['I1_violation_ratio']:.3f}, "
                f"n2_violation={row['n2_violation_ratio']:.3f}, "
                f"eta_changes={row['eta_sign_changes']:.0f}, cosphi_changes={row['cosphi_sign_changes']:.0f}"
            )
        out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Check visual regression constraints for AIR56 FOC/MIC working-characteristics table.")
    parser.add_argument("--csv", default="paper/pgups_2026/fig/working_characteristics_air56_foc_mic_table.csv")
    parser.add_argument("--foc-policy", default="FOC")
    parser.add_argument("--mic-policy", default="MIC_AI")
    parser.add_argument("--x-min-tol-kw", type=float, default=0.005)
    parser.add_argument("--x-max-tol-kw", type=float, default=0.03)
    parser.add_argument("--out-json", default="")
    parser.add_argument("--out-md", default="")
    args = parser.parse_args()

    csv_path = Path(str(args.csv)).resolve()
    out_json = Path(str(args.out_json)).resolve() if str(args.out_json).strip() else csv_path.with_name(
        csv_path.stem + "_visual_regression.json"
    )
    out_md = Path(str(args.out_md)).resolve() if str(args.out_md).strip() else csv_path.with_name(
        csv_path.stem + "_visual_regression.md"
    )
    payload = run_check(
        csv_path=csv_path,
        foc_policy=str(args.foc_policy),
        mic_policy=str(args.mic_policy),
        x_min_tol_kw=float(args.x_min_tol_kw),
        x_max_tol_kw=float(args.x_max_tol_kw),
        out_json=out_json,
        out_md=out_md,
    )
    print(f"saved: {out_json}")
    print(f"saved: {out_md}")
    print(f"passed: {bool(payload.get('passed', False))}")
    if not bool(payload.get("passed", False)):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
