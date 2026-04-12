from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.step27_artifacts import (
    STEP27_AIR56_ACCEPTANCE_JSON_LEGACY,
    STEP27_MOTOR_ACCEPTANCE_JSON,
    existing_acceptance_jsons,
)


ROOT_FILES = (
    "step28_ieee_summary.csv",
    "step28_ieee_summary.md",
)

MODE_FILES = (
    "step27_per_seed_metrics.csv",
    "step27_stats_motor_controller.csv",
    "step27_final_pi_vs_foc_vs_mic.csv",
    "step27_reproducibility.json",
    "step27_report.md",
    "step27_seed_perturbations.csv",
)

MODE_DIRS = (
    "mode1_foc_encoder_vs_mic_sensorless",
    "mode2_foc_sensorless_vs_mic_sensorless",
)


def _copy_file(src: Path, dst: Path, copied: List[str], missing: List[str]) -> None:
    if not src.exists():
        missing.append(str(src))
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    copied.append(str(dst))


def _copy_acceptance_files(mode_src: Path, mode_dst: Path, copied: List[str], missing: List[str]) -> None:
    acceptance_paths = existing_acceptance_jsons(mode_src)
    if not acceptance_paths:
        missing.append(str(mode_src / STEP27_MOTOR_ACCEPTANCE_JSON))
        return

    primary_payload = acceptance_paths[0].read_text(encoding="utf-8")
    target_primary = mode_dst / STEP27_MOTOR_ACCEPTANCE_JSON
    target_primary.parent.mkdir(parents=True, exist_ok=True)
    target_primary.write_text(primary_payload, encoding="utf-8")
    copied.append(str(target_primary))

    target_legacy = mode_dst / STEP27_AIR56_ACCEPTANCE_JSON_LEGACY
    if not target_legacy.exists():
        target_legacy.write_text(primary_payload, encoding="utf-8")
        copied.append(str(target_legacy))

    for src_path in acceptance_paths:
        dst_path = mode_dst / src_path.name
        if dst_path.exists():
            continue
        shutil.copy2(src_path, dst_path)
        copied.append(str(dst_path))


def main() -> None:
    parser = argparse.ArgumentParser(description="Package step28 artifacts into paper/ieee_2026 tree.")
    parser.add_argument("--step28-out", required=True, help="Path to output root produced by run_step28_ieee_protocol.*")
    parser.add_argument("--dest-root", default="paper/ieee_2026/data/step28", help="Destination root for packaged artifacts")
    parser.add_argument("--tag", default="", help="Package tag. Default: UTC timestamp")
    parser.add_argument("--include-runs", action="store_true", help="Include heavy mode*/runs raw traces")
    parser.add_argument("--strict", action="store_true", help="Fail if any expected file is missing")
    parser.add_argument("--theory-csv", default="", help="Optional CSV for theory validator.")
    parser.add_argument("--passport-dir", default="", help="Optional directory with passport_compare_3motors.* artifacts.")
    args = parser.parse_args()

    src_root = Path(args.step28_out).expanduser()
    if not src_root.is_absolute():
        src_root = (Path.cwd() / src_root).resolve()
    if not src_root.exists():
        raise FileNotFoundError(f"Step28 output root not found: {src_root}")

    tag = str(args.tag).strip()
    if not tag:
        tag = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    dst_root = Path(args.dest_root).expanduser()
    if not dst_root.is_absolute():
        dst_root = (Path.cwd() / dst_root).resolve()
    package_dir = dst_root / tag
    package_dir.mkdir(parents=True, exist_ok=True)

    copied: List[str] = []
    missing: List[str] = []

    for rel in ROOT_FILES:
        _copy_file(src_root / rel, package_dir / rel, copied, missing)

    for mode in MODE_DIRS:
        mode_src = src_root / mode
        mode_dst = package_dir / mode
        for rel in MODE_FILES:
            _copy_file(mode_src / rel, mode_dst / rel, copied, missing)
        _copy_acceptance_files(mode_src, mode_dst, copied, missing)
        if args.include_runs:
            runs_src = mode_src / "runs"
            runs_dst = mode_dst / "runs"
            if runs_src.exists():
                shutil.copytree(runs_src, runs_dst, dirs_exist_ok=True)
                copied.append(str(runs_dst))
            else:
                missing.append(str(runs_src))

    theory_csv = str(args.theory_csv).strip()
    if theory_csv:
        theory_out_dir = package_dir / "theory_validation"
        theory_out_dir.mkdir(parents=True, exist_ok=True)
        out_json = theory_out_dir / "report.json"
        out_md = theory_out_dir / "report.md"
        cmd = [
            sys.executable,
            "tools/validate_theory_working_characteristics.py",
            "--csv",
            str(theory_csv),
            "--out-json",
            str(out_json),
            "--out-md",
            str(out_md),
        ]
        subprocess.run(cmd, check=True)
        copied.append(str(out_json))
        copied.append(str(out_md))

    passport_dir_raw = str(args.passport_dir).strip()
    if passport_dir_raw:
        passport_dir = Path(passport_dir_raw).expanduser()
        if not passport_dir.is_absolute():
            passport_dir = (Path.cwd() / passport_dir).resolve()
        for name in ("passport_compare_3motors.csv", "passport_compare_3motors.md", "passport_compare_3motors.json"):
            _copy_file(
                passport_dir / name,
                package_dir / "passport" / name,
                copied,
                missing,
            )

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_step28_out": str(src_root),
        "package_dir": str(package_dir),
        "copied_count": len(copied),
        "missing_count": len(missing),
        "copied": copied,
        "missing": missing,
        "include_runs": bool(args.include_runs),
    }
    (package_dir / "package_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[package-step28] package_dir={package_dir}")
    print(f"[package-step28] copied={len(copied)} missing={len(missing)}")

    if args.strict and missing:
        raise FileNotFoundError(f"Missing required artifacts: {len(missing)}. See {package_dir / 'package_manifest.json'}")


if __name__ == "__main__":
    main()
