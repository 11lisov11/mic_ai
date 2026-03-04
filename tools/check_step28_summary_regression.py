from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


def _to_float(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _load_row(df: pd.DataFrame, mode: str, controller: str) -> pd.Series | None:
    part = df[(df["mode"].astype(str) == str(mode)) & (df["controller"].astype(str).str.upper() == str(controller).upper())]
    if part.empty:
        return None
    return part.iloc[0]


def _check_metric(
    *,
    actual: float,
    expected: float,
    abs_tol: float,
    rel_tol: float,
) -> Tuple[bool, float, float]:
    diff_abs = abs(actual - expected)
    scale = max(abs(expected), 1e-12)
    diff_rel = diff_abs / scale
    ok = bool(diff_abs <= abs_tol or diff_rel <= rel_tol)
    return ok, diff_abs, diff_rel


def _render_md(payload: Dict[str, object]) -> str:
    lines: List[str] = []
    lines.append("# Step28 Regression Guard Report")
    lines.append("")
    lines.append(f"- generated_utc: `{payload.get('generated_utc', '')}`")
    lines.append(f"- summary_csv: `{payload.get('summary_csv', '')}`")
    lines.append(f"- baseline_json: `{payload.get('baseline_json', '')}`")
    lines.append(f"- checks_total: `{payload.get('checks_total', 0)}`")
    lines.append(f"- checks_failed: `{payload.get('checks_failed', 0)}`")
    lines.append(f"- ok: `{payload.get('ok', False)}`")
    lines.append("")
    failures = list(payload.get("failures", []))
    if failures:
        lines.append("## Failures")
        for f in failures:
            if not isinstance(f, dict):
                continue
            lines.append(
                "- mode={mode}, controller={controller}, metric={metric}, actual={actual}, expected={expected}, abs_diff={abs_diff}, rel_diff={rel_diff}, abs_tol={abs_tol}, rel_tol={rel_tol}".format(
                    mode=f.get("mode", ""),
                    controller=f.get("controller", ""),
                    metric=f.get("metric", ""),
                    actual=f.get("actual", ""),
                    expected=f.get("expected", ""),
                    abs_diff=f.get("abs_diff", ""),
                    rel_diff=f.get("rel_diff", ""),
                    abs_tol=f.get("abs_tol", ""),
                    rel_tol=f.get("rel_tol", ""),
                )
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Regression guard for frozen step28 summary metrics against baseline.")
    parser.add_argument("--summary-csv", required=True)
    parser.add_argument("--baseline-json", required=True)
    parser.add_argument("--out-json", default="")
    parser.add_argument("--out-md", default="")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    summary_csv = Path(str(args.summary_csv)).expanduser().resolve()
    baseline_json = Path(str(args.baseline_json)).expanduser().resolve()
    if not summary_csv.exists():
        raise FileNotFoundError(summary_csv)
    if not baseline_json.exists():
        raise FileNotFoundError(baseline_json)

    out_json = (
        Path(str(args.out_json)).expanduser().resolve()
        if str(args.out_json).strip()
        else (summary_csv.parent / "STEP28_REGRESSION_GUARD.json")
    )
    out_md = (
        Path(str(args.out_md)).expanduser().resolve()
        if str(args.out_md).strip()
        else (summary_csv.parent / "STEP28_REGRESSION_GUARD.md")
    )

    df = pd.read_csv(summary_csv)
    baseline = dict(json.loads(baseline_json.read_text(encoding="utf-8")))
    rows = baseline.get("rows", [])
    if not isinstance(rows, list):
        raise ValueError("baseline rows must be a list")

    checks_total = 0
    failures: List[Dict[str, object]] = []

    for item in rows:
        if not isinstance(item, dict):
            continue
        mode = str(item.get("mode", "")).strip()
        controller = str(item.get("controller", "")).strip()
        metrics = item.get("metrics", {})
        tolerances = item.get("tolerance", {})
        abs_map = tolerances.get("abs", {}) if isinstance(tolerances, dict) else {}
        rel_map = tolerances.get("rel", {}) if isinstance(tolerances, dict) else {}
        row = _load_row(df, mode=mode, controller=controller)
        if row is None:
            failures.append(
                {
                    "mode": mode,
                    "controller": controller,
                    "metric": "__row__",
                    "actual": None,
                    "expected": "present",
                    "abs_diff": None,
                    "rel_diff": None,
                    "abs_tol": None,
                    "rel_tol": None,
                }
            )
            continue
        if not isinstance(metrics, dict):
            continue
        for metric, expected_raw in metrics.items():
            checks_total += 1
            expected = _to_float(expected_raw)
            actual = _to_float(row.get(metric))
            abs_tol = _to_float(abs_map.get(metric, 1e-6 if math.isfinite(expected) else 0.0))
            rel_tol = _to_float(rel_map.get(metric, 1e-4 if math.isfinite(expected) else 0.0))
            ok, diff_abs, diff_rel = _check_metric(actual=actual, expected=expected, abs_tol=abs_tol, rel_tol=rel_tol)
            if not ok:
                failures.append(
                    {
                        "mode": mode,
                        "controller": controller,
                        "metric": metric,
                        "actual": actual,
                        "expected": expected,
                        "abs_diff": diff_abs,
                        "rel_diff": diff_rel,
                        "abs_tol": abs_tol,
                        "rel_tol": rel_tol,
                    }
                )

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "summary_csv": str(summary_csv),
        "baseline_json": str(baseline_json),
        "checks_total": int(checks_total),
        "checks_failed": int(len(failures)),
        "ok": bool(len(failures) == 0),
        "failures": failures,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(_render_md(payload), encoding="utf-8")
    print(f"saved: {out_json}")
    print(f"saved: {out_md}")
    print(f"ok: {bool(payload.get('ok', False))}")

    if bool(args.strict) and not bool(payload.get("ok", False)):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
