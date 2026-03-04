from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.validate_theory_working_characteristics import run_validation


def _to_markdown(summary: Dict[str, object], rows: List[Dict[str, object]]) -> str:
    lines: List[str] = []
    lines.append("# Theory Validation (3 Motors)")
    lines.append("")
    lines.append(f"- generated_utc: `{summary['generated_utc']}`")
    lines.append(f"- tag: `{summary['tag']}`")
    lines.append(f"- all_passed: `{summary['all_passed']}`")
    lines.append("")
    lines.append("| motor | csv | passed | hard_fail_count | warn_fail_count |")
    lines.append("|---|---|---|---:|---:|")
    for row in rows:
        lines.append(
            "| {motor} | {csv_path} | {passed} | {hard_fail_count} | {warn_fail_count} |".format(
                **row
            )
        )
    lines.append("")
    return "\n".join(lines)


def _find_input_csv(passport_raw_dir: Path, motor: str) -> Path:
    candidates = [
        passport_raw_dir / motor / "working_characteristics_filtered.csv",
        passport_raw_dir / motor / "working_characteristics.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"No working_characteristics CSV for motor={motor} under {passport_raw_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build per-motor theory validation reports from passport working-characteristics CSV files.")
    parser.add_argument("--tag", required=True, help="Passport tag directory name, e.g. 20260304_al31_robust_rand009_nodrift_v3")
    parser.add_argument("--passport-root", default="paper/ieee_2026/data/passport")
    parser.add_argument("--out-root", default="paper/ieee_2026/data/theory_validation")
    parser.add_argument("--motors", default="air56,al31,ao2")
    args = parser.parse_args()

    motors = [m.strip().lower() for m in str(args.motors).split(",") if m.strip()]
    passport_dir = (Path(args.passport_root).expanduser().resolve() / str(args.tag))
    passport_raw_dir = passport_dir / "raw"
    if not passport_raw_dir.exists():
        raise FileNotFoundError(passport_raw_dir)

    out_dir = (Path(args.out_root).expanduser().resolve() / str(args.tag))
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, object]] = []
    for motor in motors:
        in_csv = _find_input_csv(passport_raw_dir, motor)
        report = run_validation(in_csv)
        motor_dir = out_dir / motor
        motor_dir.mkdir(parents=True, exist_ok=True)
        out_json = motor_dir / "theory_validation_report.json"
        out_md = motor_dir / "theory_validation_report.md"
        out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        lines = [
            f"# Theory Validation Report: {motor.upper()}",
            "",
            f"- csv_path: `{report['csv_path']}`",
            f"- passed: `{report['passed']}`",
            f"- hard_fail_count: `{report['hard_fail_count']}`",
            f"- warn_fail_count: `{report['warn_fail_count']}`",
            "",
            "| Check | Severity | Pass | Details |",
            "|---|---|---|---|",
        ]
        for ch in report["checks"]:
            lines.append(f"| {ch['name']} | {ch['severity']} | {ch['passed']} | {ch['details']} |")
        lines.append("")
        out_md.write_text("\n".join(lines), encoding="utf-8")

        summary_rows.append(
            {
                "motor": motor,
                "csv_path": str(in_csv),
                "passed": bool(report.get("passed", False)),
                "hard_fail_count": int(report.get("hard_fail_count", 0)),
                "warn_fail_count": int(report.get("warn_fail_count", 0)),
                "report_json": str(out_json),
                "report_md": str(out_md),
            }
        )
        print(f"saved: {out_json}")
        print(f"saved: {out_md}")

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "tag": str(args.tag),
        "all_passed": all(bool(r["passed"]) for r in summary_rows),
        "rows": summary_rows,
    }

    summary_csv = out_dir / "theory_validation_summary.csv"
    summary_json = out_dir / "theory_validation_summary.json"
    summary_md = out_dir / "theory_validation_summary.md"
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_md.write_text(_to_markdown(summary, summary_rows), encoding="utf-8")
    print(f"saved: {summary_csv}")
    print(f"saved: {summary_json}")
    print(f"saved: {summary_md}")


if __name__ == "__main__":
    main()
