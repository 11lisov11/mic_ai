from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class CheckResult:
    name: str
    passed: bool
    details: str
    severity: str = "error"


def _count_local_spikes(values: np.ndarray, *, abs_limit: float, rel_factor: float = 6.0) -> Tuple[int, float]:
    """
    Detect non-physical local spikes using deviation from neighbor interpolation.
    Returns (spike_count, threshold_used).
    """
    x = np.asarray(values, dtype=float)
    if x.size < 5:
        return 0, float(abs_limit)
    local = np.abs(x[1:-1] - 0.5 * (x[:-2] + x[2:]))
    finite = local[np.isfinite(local)]
    if finite.size == 0:
        return 0, float(abs_limit)
    med = float(np.median(finite))
    # Keep absolute physical threshold as primary guard.
    # Relative threshold is used only when local residual scale is clearly small.
    if med <= (0.5 * float(abs_limit)):
        thr = float(max(abs_limit, rel_factor * max(med, 1e-12)))
    else:
        thr = float(abs_limit)
    count = int(np.sum(finite > thr))
    return count, thr


def _pick_first_existing(df: pd.DataFrame, names: List[str]) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    return None


def _to_float_series(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce")


def _group_key_column(df: pd.DataFrame) -> str:
    col = _pick_first_existing(df, ["policy", "controller", "mode"])
    if col is None:
        raise ValueError("Cannot find controller grouping column: expected one of policy/controller/mode.")
    return col


def _validate_group(name: str, g: pd.DataFrame) -> List[CheckResult]:
    out: List[CheckResult] = []
    g = g.copy()

    p2_col = _pick_first_existing(g, ["p2_kw", "P2_kW", "p2"])
    m2_col = _pick_first_existing(g, ["m2", "M2", "M2_Nm"])
    i1_col = _pick_first_existing(g, ["i_rms", "I1", "I1_A"])
    n2_col = _pick_first_existing(g, ["n2_rpm", "n2"])
    eta_col = _pick_first_existing(g, ["eta_pct", "eta"])
    cos_col = _pick_first_existing(g, ["cos_phi", "cosphi", "cos_phi_pu"])

    required_missing = [c for c in [p2_col, m2_col, i1_col, n2_col, eta_col, cos_col] if c is None]
    if required_missing:
        out.append(
            CheckResult(
                name=f"{name}:required_columns",
                passed=False,
                details=f"Missing required columns in group '{name}'.",
            )
        )
        return out

    assert p2_col is not None
    assert m2_col is not None
    assert i1_col is not None
    assert n2_col is not None
    assert eta_col is not None
    assert cos_col is not None

    g = g.sort_values(p2_col)
    p2 = _to_float_series(g, p2_col).to_numpy(dtype=float)
    m2 = _to_float_series(g, m2_col).to_numpy(dtype=float)
    i1 = _to_float_series(g, i1_col).to_numpy(dtype=float)
    n2 = _to_float_series(g, n2_col).to_numpy(dtype=float)
    eta_raw = _to_float_series(g, eta_col).to_numpy(dtype=float)
    cos = _to_float_series(g, cos_col).to_numpy(dtype=float)

    # eta may be in p.u. or in percent.
    eta_pct = eta_raw * 100.0 if np.nanmax(eta_raw) <= 1.2 else eta_raw

    finite_mask = np.isfinite(p2) & np.isfinite(m2) & np.isfinite(i1) & np.isfinite(n2) & np.isfinite(eta_pct) & np.isfinite(cos)
    count_valid = int(np.sum(finite_mask))
    out.append(
        CheckResult(
            name=f"{name}:finite_rows",
            passed=count_valid >= 4,
            details=f"finite_rows={count_valid}",
        )
    )
    if count_valid < 4:
        return out

    p2 = p2[finite_mask]
    m2 = m2[finite_mask]
    i1 = i1[finite_mask]
    n2 = n2[finite_mask]
    eta_pct = eta_pct[finite_mask]
    cos = cos[finite_mask]

    eta_bounds_ok = bool(np.all((eta_pct >= -1e-6) & (eta_pct <= 102.0)))
    out.append(
        CheckResult(
            name=f"{name}:eta_bounds",
            passed=eta_bounds_ok,
            details=f"eta_min={float(np.min(eta_pct)):.3f}, eta_max={float(np.max(eta_pct)):.3f}",
        )
    )

    cos_bounds_ok = bool(np.all((cos >= -1e-6) & (cos <= 1.0 + 1e-6)))
    out.append(
        CheckResult(
            name=f"{name}:cosphi_bounds",
            passed=cos_bounds_ok,
            details=f"cos_min={float(np.min(cos)):.4f}, cos_max={float(np.max(cos)):.4f}",
        )
    )

    dm2 = np.diff(m2)
    m2_viol = int(np.sum(dm2 < -0.03))
    out.append(
        CheckResult(
            name=f"{name}:m2_monotonic",
            passed=m2_viol == 0,
            details=f"violations={m2_viol}",
        )
    )

    di1 = np.diff(i1)
    i1_viol = int(np.sum(di1 < -0.02))
    out.append(
        CheckResult(
            name=f"{name}:i1_monotonic",
            passed=i1_viol <= 1,
            details=f"violations={i1_viol}",
            severity="warn" if i1_viol <= 1 else "error",
        )
    )

    dn2 = np.diff(n2)
    n2_up_viol = int(np.sum(dn2 > 3.0))
    out.append(
        CheckResult(
            name=f"{name}:n2_non_increasing",
            passed=n2_up_viol <= 1,
            details=f"upward_jumps_gt3rpm={n2_up_viol}",
            severity="warn" if n2_up_viol <= 1 else "error",
        )
    )

    n2_spikes, n2_thr = _count_local_spikes(n2, abs_limit=25.0, rel_factor=6.0)
    out.append(
        CheckResult(
            name=f"{name}:n2_spike_detector",
            passed=n2_spikes == 0,
            details=f"spikes={n2_spikes}, threshold={n2_thr:.3f}",
        )
    )

    # Soft shape checks (warnings): eta peak and cosphi rise from low load to nominal.
    p2_span = float(max(np.max(p2) - np.min(p2), 1e-9))
    eta_peak_idx = int(np.argmax(eta_pct))
    eta_peak_rel = float((p2[eta_peak_idx] - np.min(p2)) / p2_span)
    out.append(
        CheckResult(
            name=f"{name}:eta_peak_location",
            passed=0.35 <= eta_peak_rel <= 1.00,
            details=f"peak_rel={eta_peak_rel:.3f}",
            severity="warn",
        )
    )

    half = max(2, len(cos) // 2)
    cos_rise = float(cos[half - 1] - cos[0])
    out.append(
        CheckResult(
            name=f"{name}:cosphi_low_to_mid_rise",
            passed=cos_rise > -0.02,
            details=f"delta={cos_rise:.4f}",
            severity="warn",
        )
    )

    eta_spikes, eta_thr = _count_local_spikes(eta_pct, abs_limit=3.0, rel_factor=6.0)
    out.append(
        CheckResult(
            name=f"{name}:eta_spike_detector",
            passed=eta_spikes == 0,
            details=f"spikes={eta_spikes}, threshold={eta_thr:.3f}",
        )
    )
    cos_spikes, cos_thr = _count_local_spikes(cos, abs_limit=0.08, rel_factor=6.0)
    out.append(
        CheckResult(
            name=f"{name}:cosphi_spike_detector",
            passed=cos_spikes == 0,
            details=f"spikes={cos_spikes}, threshold={cos_thr:.4f}",
        )
    )

    # Power inequality check when electrical input is available.
    p1_col = _pick_first_existing(g, ["p_el_pos", "p_in_total", "p_el"])
    if p1_col is not None:
        p1 = _to_float_series(g, p1_col).to_numpy(dtype=float)[finite_mask]
        p2_w = p2 * 1000.0
        viol = int(np.sum(p2_w > p1 * 1.03))
        out.append(
            CheckResult(
                name=f"{name}:p2_le_p1",
                passed=viol == 0,
                details=f"violations={viol}",
            )
        )

    return out


def run_validation(csv_path: Path) -> Dict[str, object]:
    df = pd.read_csv(csv_path)
    key_col = _group_key_column(df)
    df[key_col] = df[key_col].astype(str)
    groups = sorted(df[key_col].dropna().unique().tolist())

    checks: List[CheckResult] = []
    for name in groups:
        checks.extend(_validate_group(name, df[df[key_col] == name]))

    hard_failed = [c for c in checks if (c.severity == "error" and not c.passed)]
    warn_failed = [c for c in checks if (c.severity != "error" and not c.passed)]
    payload = {
        "csv_path": str(csv_path.resolve()),
        "groups": groups,
        "passed": len(hard_failed) == 0,
        "hard_fail_count": len(hard_failed),
        "warn_fail_count": len(warn_failed),
        "checks": [
            {
                "name": c.name,
                "passed": bool(c.passed),
                "details": c.details,
                "severity": c.severity,
            }
            for c in checks
        ],
    }
    return payload


def _to_markdown(report: Dict[str, object]) -> str:
    lines: List[str] = []
    lines.append("# Theory Validation Report")
    lines.append("")
    lines.append(f"- csv_path: `{report['csv_path']}`")
    lines.append(f"- passed: `{report['passed']}`")
    lines.append(f"- hard_fail_count: `{report['hard_fail_count']}`")
    lines.append(f"- warn_fail_count: `{report['warn_fail_count']}`")
    lines.append("")
    lines.append("| Check | Severity | Pass | Details |")
    lines.append("|---|---|---|---|")
    for ch in report["checks"]:
        lines.append(
            f"| {ch['name']} | {ch['severity']} | {ch['passed']} | {ch['details']} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate working-characteristic curves against basic IM theory constraints.")
    parser.add_argument("--csv", required=True, help="Input CSV with working characteristics.")
    parser.add_argument("--out-json", default=None, help="Output JSON report path.")
    parser.add_argument("--out-md", default=None, help="Output Markdown report path.")
    args = parser.parse_args()

    csv_path = Path(args.csv).resolve()
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    report = run_validation(csv_path)
    out_json = Path(args.out_json).resolve() if args.out_json else csv_path.with_name(csv_path.stem + "_theory_validation.json")
    out_md = Path(args.out_md).resolve() if args.out_md else csv_path.with_name(csv_path.stem + "_theory_validation.md")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text(_to_markdown(report), encoding="utf-8")
    print(f"saved: {out_json}")
    print(f"saved: {out_md}")
    if not bool(report.get("passed", False)):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
