from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: List[str], *, cwd: Path, dry_run: bool) -> Dict[str, object]:
    started = time.time()
    rec: Dict[str, object] = {
        "cmd": cmd,
        "cwd": str(cwd),
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    if dry_run:
        rec["dry_run"] = True
        rec["returncode"] = 0
        rec["elapsed_sec"] = 0.0
        return rec
    proc = subprocess.run(cmd, cwd=cwd, check=False)
    rec["dry_run"] = False
    rec["returncode"] = int(proc.returncode)
    rec["elapsed_sec"] = round(float(time.time() - started), 3)
    return rec


def _md_report(rows: List[Dict[str, object]], payload: Dict[str, object]) -> str:
    lines: List[str] = []
    lines.append("# Integration Pipeline Report")
    lines.append("")
    lines.append(f"- created_utc: `{payload.get('created_utc', '')}`")
    lines.append(f"- dry_run: `{bool(payload.get('dry_run', False))}`")
    lines.append(f"- all_ok: `{bool(payload.get('all_ok', False))}`")
    lines.append("")
    lines.append("## Steps")
    for i, row in enumerate(rows, start=1):
        cmd = " ".join(str(x) for x in row.get("cmd", []))
        lines.append(f"{i}. rc=`{row.get('returncode', '')}` elapsed=`{row.get('elapsed_sec', '')}`s")
        lines.append(f"   - `{cmd}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run integration contour for step27 -> robust_motor_hardening -> reproduce_ieee_step28 "
            "with one command and save report."
        )
    )
    parser.add_argument("--out-root", default="outputs/integration_pipeline")
    parser.add_argument("--motors", default="air56")
    parser.add_argument("--seeds", default="101")
    parser.add_argument("--scenarios", default="speed_step")
    parser.add_argument("--package-tag", default="integration_smoke")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-step27", action="store_true")
    parser.add_argument("--skip-robust", action="store_true")
    parser.add_argument("--skip-step28", action="store_true")
    args = parser.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.out_root).resolve() / ts
    out_root.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    if not bool(args.skip_step27):
        step27_out = out_root / "step27"
        cmd = [
            sys.executable,
            "tools/step27_pipeline.py",
            "--motors",
            str(args.motors),
            "--seeds",
            str(args.seeds),
            "--scenarios",
            str(args.scenarios),
            "--mic-mode",
            "rule",
            "--skip-air56-tune",
            "--out-dir",
            str(step27_out),
        ]
        rows.append(_run(cmd, cwd=ROOT, dry_run=bool(args.dry_run)))

    if not bool(args.skip_robust):
        robust_out = out_root / "robust"
        cmd = [
            sys.executable,
            "tools/robust_motor_hardening.py",
            "--motors",
            "al31",
            "--out-dir",
            str(robust_out),
            "--dry-run",
        ]
        rows.append(_run(cmd, cwd=ROOT, dry_run=bool(args.dry_run)))

    if not bool(args.skip_step28):
        step28_out = out_root / "step28"
        pkg_root = out_root / "pkg"
        bundle = out_root / "bundle"
        cmd = [
            sys.executable,
            "tools/reproduce_ieee_step28.py",
            "--out-root",
            str(step28_out),
            "--package-root",
            str(pkg_root),
            "--package-tag",
            str(args.package_tag),
            "--motors",
            str(args.motors),
            "--seeds",
            str(args.seeds),
            "--scenarios",
            str(args.scenarios),
            "--mic-mode",
            "rule",
            "--skip-air56-tune",
            "--submission-bundle-out-dir",
            str(bundle),
        ]
        rows.append(_run(cmd, cwd=ROOT, dry_run=bool(args.dry_run)))

    all_ok = bool(all(int(r.get("returncode", 1)) == 0 for r in rows))
    payload: Dict[str, object] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dry_run": bool(args.dry_run),
        "out_root": str(out_root),
        "all_ok": all_ok,
        "steps": rows,
    }
    report_json = out_root / "integration_pipeline_report.json"
    report_md = out_root / "integration_pipeline_report.md"
    report_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    report_md.write_text(_md_report(rows, payload), encoding="utf-8")

    print(f"[integration] report_json={report_json}")
    print(f"[integration] report_md={report_md}")
    if not all_ok:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
