from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.step27_artifacts import STEP27_MOTOR_ACCEPTANCE_JSON, existing_acceptance_jsons


MODE_DIRS = (
    "mode1_foc_encoder_vs_mic_sensorless",
    "mode2_foc_sensorless_vs_mic_sensorless",
)

STEP28_ROOT_REQUIRED = (
    "step28_ieee_summary.csv",
    "step28_ieee_summary.md",
    "package_manifest.json",
)

MODE_REQUIRED = (
    "step27_per_seed_metrics.csv",
    "step27_stats_motor_controller.csv",
    "step27_final_pi_vs_foc_vs_mic.csv",
    STEP27_MOTOR_ACCEPTANCE_JSON,
    "step27_reproducibility.json",
    "step27_report.md",
)

DERIVED_REQUIRED = (
    "ieee_pi_foc_mic_stats.csv",
    "ieee_pi_foc_mic_stats.md",
    "fig_ieee_pi_foc_mic_power.png",
    "fig_ieee_pi_foc_mic_power.pdf",
    "fig_ieee_pi_foc_mic_power.svg",
    "motor_tuning_acceptance_summary.csv",
    "motor_tuning_acceptance_summary.json",
)

PASSPORT_REQUIRED = (
    "passport_compare_3motors.csv",
    "passport_compare_3motors.md",
    "passport_compare_3motors.json",
)

PUBLICATION_OPTIONAL = (
    "manuscript.md",
    "fig/fig1_mic_methodology.png",
    "fig/fig2_pi_foc_mic_power.pdf",
    "fig/fig3_air56_working_characteristics.pdf",
    "fig/fig4_cross_motor_robustness.pdf",
    "fig/fig5_training_to_foc.pdf",
)

RELEASE_OPTIONAL = (
    "promotion_manifest.json",
    "release_snapshot.json",
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_motors_from_summary(path: Path) -> Set[str]:
    motors: Set[str] = set()
    if not path.exists():
        return motors
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            motor = str(row.get("motor", "")).strip().lower()
            if motor:
                motors.add(motor)
    return motors


def _render_path(path: Path, *, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except Exception:
        return str(path.resolve())


def _add_path(
    *,
    path: Path,
    group: str,
    required: bool,
    files: List[Dict[str, object]],
    missing_required: List[str],
    missing_optional: List[str],
    repo_root: Path,
) -> None:
    rendered = _render_path(path, root=repo_root)
    if not path.exists():
        if required:
            missing_required.append(rendered)
        else:
            missing_optional.append(rendered)
        return
    files.append(
        {
            "path": rendered,
            "group": group,
            "required": bool(required),
            "size_bytes": int(path.stat().st_size),
            "sha256": _sha256(path),
        }
    )


def _motor_report_files(motors: Iterable[str]) -> Tuple[str, ...]:
    rows: List[str] = []
    for motor in sorted({str(x).strip().lower() for x in motors if str(x).strip()}):
        rows.append(f"motor_{motor}_tuning_report.md")
    return tuple(rows)


def build_lock(
    *,
    step28_dir: Path,
    ieee_root: Path | None,
    release_tag: str,
    require_publication_assets: bool,
    require_release_assets: bool,
) -> Dict[str, object]:
    repo_root = Path(__file__).resolve().parents[1]
    step28_dir = step28_dir.resolve()
    release_dir = None if ieee_root is None else (ieee_root.resolve() / "data" / "release" / release_tag)

    files: List[Dict[str, object]] = []
    missing_required: List[str] = []
    missing_optional: List[str] = []

    # Step28 root.
    for rel in STEP28_ROOT_REQUIRED:
        _add_path(
            path=step28_dir / rel,
            group="step28_root",
            required=True,
            files=files,
            missing_required=missing_required,
            missing_optional=missing_optional,
            repo_root=repo_root,
        )

    # Step28 mode subfolders.
    for mode in MODE_DIRS:
        mode_dir = step28_dir / mode
        for rel in MODE_REQUIRED:
            if rel == STEP27_MOTOR_ACCEPTANCE_JSON:
                acceptance_paths = existing_acceptance_jsons(mode_dir)
                if acceptance_paths:
                    for path in acceptance_paths:
                        _add_path(
                            path=path,
                            group=f"step28_mode:{mode}",
                            required=True,
                            files=files,
                            missing_required=missing_required,
                            missing_optional=missing_optional,
                            repo_root=repo_root,
                        )
                else:
                    _add_path(
                        path=mode_dir / rel,
                        group=f"step28_mode:{mode}",
                        required=True,
                        files=files,
                        missing_required=missing_required,
                        missing_optional=missing_optional,
                        repo_root=repo_root,
                    )
                continue
            _add_path(
                path=mode_dir / rel,
                group=f"step28_mode:{mode}",
                required=True,
                files=files,
                missing_required=missing_required,
                missing_optional=missing_optional,
                repo_root=repo_root,
            )

    # Step28 derived.
    derived_dir = step28_dir / "derived_ieee"
    for rel in DERIVED_REQUIRED:
        _add_path(
            path=derived_dir / rel,
            group="derived_ieee",
            required=True,
            files=files,
            missing_required=missing_required,
            missing_optional=missing_optional,
            repo_root=repo_root,
        )
    motors = _read_motors_from_summary(step28_dir / "step28_ieee_summary.csv")
    for rel in _motor_report_files(motors):
        _add_path(
            path=derived_dir / rel,
            group="derived_ieee",
            required=True,
            files=files,
            missing_required=missing_required,
            missing_optional=missing_optional,
            repo_root=repo_root,
        )

    # Passport files become required only when passport folder exists.
    passport_dir = step28_dir / "passport"
    if passport_dir.exists():
        for rel in PASSPORT_REQUIRED:
            _add_path(
                path=passport_dir / rel,
                group="passport",
                required=True,
                files=files,
                missing_required=missing_required,
                missing_optional=missing_optional,
                repo_root=repo_root,
            )

    # Publication assets from ieee root.
    if ieee_root is not None:
        for rel in PUBLICATION_OPTIONAL:
            _add_path(
                path=ieee_root / rel,
                group="publication_assets",
                required=bool(require_publication_assets),
                files=files,
                missing_required=missing_required,
                missing_optional=missing_optional,
                repo_root=repo_root,
            )

    # Release snapshot assets.
    if release_dir is not None:
        for rel in RELEASE_OPTIONAL:
            _add_path(
                path=release_dir / rel,
                group="release_assets",
                required=bool(require_release_assets),
                files=files,
                missing_required=missing_required,
                missing_optional=missing_optional,
                repo_root=repo_root,
            )

    files_sorted = sorted(files, key=lambda x: str(x["path"]))
    aggregate = hashlib.sha256()
    for row in files_sorted:
        aggregate.update(str(row["path"]).encode("utf-8"))
        aggregate.update(str(row["sha256"]).encode("utf-8"))
        aggregate.update(str(int(row["size_bytes"])).encode("utf-8"))

    lock_ok = len(missing_required) == 0
    payload: Dict[str, object] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "step28_dir": _render_path(step28_dir, root=repo_root),
        "ieee_root": "" if ieee_root is None else _render_path(ieee_root.resolve(), root=repo_root),
        "release_tag": release_tag,
        "release_dir": "" if release_dir is None else _render_path(release_dir, root=repo_root),
        "require_publication_assets": bool(require_publication_assets),
        "require_release_assets": bool(require_release_assets),
        "required_files_missing": sorted(missing_required),
        "optional_files_missing": sorted(missing_optional),
        "hashed_files_count": len(files_sorted),
        "aggregate_sha256": aggregate.hexdigest(),
        "lock_ok": bool(lock_ok),
        "files": files_sorted,
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze IEEE submission candidate with SHA256 lockfile for step28 package."
    )
    parser.add_argument("--step28-dir", required=True, help="Frozen step28 package directory.")
    parser.add_argument("--ieee-root", default="", help="IEEE root with manuscript/fig/release assets.")
    parser.add_argument(
        "--release-tag",
        default="",
        help="Release tag under ieee_root/data/release. Default: step28_dir name.",
    )
    parser.add_argument(
        "--out-json",
        default="",
        help="Output lock json path. Default: <step28-dir>/submission_candidate_lock.json",
    )
    parser.add_argument("--require-publication-assets", action="store_true")
    parser.add_argument("--require-release-assets", action="store_true")
    parser.add_argument("--strict", action="store_true", help="Return non-zero when lock_ok=false.")
    args = parser.parse_args()

    step28_dir = Path(str(args.step28_dir)).expanduser().resolve()
    if not step28_dir.exists():
        raise FileNotFoundError(step28_dir)

    ieee_root = None
    if str(args.ieee_root).strip():
        ieee_root = Path(str(args.ieee_root)).expanduser().resolve()
        if not ieee_root.exists():
            raise FileNotFoundError(ieee_root)

    release_tag = str(args.release_tag).strip() or step28_dir.name
    out_json = (
        Path(str(args.out_json)).expanduser().resolve()
        if str(args.out_json).strip()
        else (step28_dir / "submission_candidate_lock.json")
    )

    payload = build_lock(
        step28_dir=step28_dir,
        ieee_root=ieee_root,
        release_tag=release_tag,
        require_publication_assets=bool(args.require_publication_assets),
        require_release_assets=bool(args.require_release_assets),
    )
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"saved: {out_json}")
    print(f"lock_ok: {bool(payload.get('lock_ok', False))}")
    print(f"hashed_files_count: {int(payload.get('hashed_files_count', 0))}")

    if bool(args.strict) and not bool(payload.get("lock_ok", False)):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
