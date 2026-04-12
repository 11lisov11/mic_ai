from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List


def _run(cmd: List[str], *, cwd: Path, dry_run: bool, executed: List[List[str]]) -> None:
    executed.append(list(cmd))
    print("[reproduce-step28] run:", " ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True, cwd=cwd)


@dataclass(frozen=True)
class ReproducePaths:
    root: Path
    out_root: Path
    mode1_dir: Path
    mode2_dir: Path
    package_root: Path
    package_dir: Path


def _resolve_paths(*, root: Path, out_root_arg: str, package_root_arg: str, package_tag: str) -> ReproducePaths:
    out_root = (root / str(out_root_arg)).resolve()
    package_root = (root / str(package_root_arg)).resolve()
    return ReproducePaths(
        root=root,
        out_root=out_root,
        mode1_dir=out_root / "mode1_foc_encoder_vs_mic_sensorless",
        mode2_dir=out_root / "mode2_foc_sensorless_vs_mic_sensorless",
        package_root=package_root,
        package_dir=package_root / package_tag,
    )


def _python_cmd(script: str, *args: str) -> List[str]:
    return [sys.executable, script, *args]


def _build_step27_base_cmd(args: argparse.Namespace) -> List[str]:
    cmd = _python_cmd(
        "tools/step27_pipeline.py",
        "--motors",
        str(args.motors),
        "--seeds",
        str(args.seeds),
        "--scenarios",
        str(args.scenarios),
        "--mic-mode",
        str(args.mic_mode),
        "--ai-control-mode",
        str(args.ai_control_mode),
        "--checkpoint-registry",
        str(args.checkpoint_registry),
        "--seed-perturbation",
        "--seed-perturb-level",
        str(float(args.seed_perturb_level)),
    )
    if bool(args.skip_air56_tune):
        cmd.append("--skip-air56-tune")
    return cmd


def _build_step27_mode_cmd(
    base_cmd: List[str],
    *,
    out_dir: Path,
    foc_feedback_mode: str,
    mic_feedback_mode: str,
) -> List[str]:
    return list(base_cmd) + [
        "--out-dir",
        str(out_dir),
        "--foc-feedback-mode",
        str(foc_feedback_mode),
        "--mic-feedback-mode",
        str(mic_feedback_mode),
    ]


def _build_passport_cmd(args: argparse.Namespace, *, passport_out_root: Path, passport_tag: str) -> List[str]:
    return _python_cmd(
        "tools/build_against_passport_table.py",
        "--motors",
        str(args.motors),
        "--checkpoint-registry",
        str(args.checkpoint_registry),
        "--out-root",
        str(passport_out_root),
        "--tag",
        str(passport_tag),
    )


def _build_package_cmd(
    args: argparse.Namespace,
    *,
    root: Path,
    step28_out: Path,
    package_root: Path,
    package_tag: str,
    passport_dir: Path | None,
) -> List[str]:
    cmd = _python_cmd(
        "scripts/package_ieee_step28.py",
        "--step28-out",
        str(step28_out),
        "--dest-root",
        str(package_root),
        "--tag",
        str(package_tag),
    )
    if bool(args.strict_package):
        cmd.append("--strict")
    theory_csv = str(args.theory_csv).strip()
    if theory_csv:
        cmd.extend(["--theory-csv", str((root / theory_csv).resolve())])
    if passport_dir is not None:
        cmd.extend(["--passport-dir", str(passport_dir)])
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="One-command reproducibility pipeline for IEEE step28 artifacts."
    )
    parser.add_argument("--out-root", default="outputs/progress_step28_ieee_repro")
    parser.add_argument("--motors", default="air56,al31,ao2")
    parser.add_argument("--seeds", default="101,202,303,404,505")
    parser.add_argument("--scenarios", default="speed_step,ramp,load_step,start_stop")
    parser.add_argument("--seed-perturb-level", type=float, default=0.2)
    parser.add_argument("--mic-mode", choices=["ai", "rule"], default="rule")
    parser.add_argument(
        "--ai-control-mode",
        choices=["ai_id_ref", "ai_id_ref_hybrid", "ai_current", "ai_voltage", "foc_assist", "ai_speed"],
        default="ai_id_ref",
    )
    parser.add_argument("--checkpoint-registry", default="config/checkpoint_registry.json")
    parser.add_argument("--skip-air56-tune", dest="skip_air56_tune", action="store_true")
    parser.add_argument("--no-skip-air56-tune", dest="skip_air56_tune", action="store_false")
    parser.set_defaults(skip_air56_tune=True)

    parser.add_argument("--package-root", default="paper/ieee_2026/data/step28")
    parser.add_argument("--package-tag", default="")
    parser.add_argument("--strict-package", dest="strict_package", action="store_true")
    parser.add_argument("--no-strict-package", dest="strict_package", action="store_false")
    parser.set_defaults(strict_package=True)
    parser.add_argument("--build-figures-tables", dest="build_figures_tables", action="store_true")
    parser.add_argument("--no-build-figures-tables", dest="build_figures_tables", action="store_false")
    parser.set_defaults(build_figures_tables=True)
    parser.add_argument("--promote-release", dest="promote_release", action="store_true")
    parser.add_argument("--no-promote-release", dest="promote_release", action="store_false")
    parser.set_defaults(promote_release=False)
    parser.add_argument("--ieee-root", default="paper/ieee_2026")
    parser.add_argument("--pgups-fig-dir", default="paper/pgups_2026/fig")
    parser.add_argument("--manuscript", default="paper/ieee_2026/manuscript.md")
    parser.add_argument("--guardrails-policy", default="paper/ieee_2026/guardrails_policy.json")
    parser.add_argument("--freeze-submission-candidate", dest="freeze_submission_candidate", action="store_true")
    parser.add_argument("--no-freeze-submission-candidate", dest="freeze_submission_candidate", action="store_false")
    parser.set_defaults(freeze_submission_candidate=True)
    parser.add_argument(
        "--freeze-require-publication-assets",
        dest="freeze_require_publication_assets",
        action="store_true",
    )
    parser.add_argument(
        "--freeze-require-release-assets",
        dest="freeze_require_release_assets",
        action="store_true",
    )

    parser.add_argument("--theory-csv", default="")
    parser.add_argument("--build-passport", dest="build_passport", action="store_true")
    parser.add_argument("--no-build-passport", dest="build_passport", action="store_false")
    parser.add_argument("--passport-out-root", default="paper/ieee_2026/data/passport")
    parser.add_argument("--passport-tag", default="")
    parser.add_argument("--passport-dir", default="")
    parser.set_defaults(build_passport=True)
    parser.add_argument("--build-submission-bundle", dest="build_submission_bundle", action="store_true")
    parser.add_argument("--no-build-submission-bundle", dest="build_submission_bundle", action="store_false")
    parser.set_defaults(build_submission_bundle=True)
    parser.add_argument("--strict-verify", dest="strict_verify", action="store_true")
    parser.add_argument("--no-strict-verify", dest="strict_verify", action="store_false")
    parser.set_defaults(strict_verify=False)
    parser.add_argument("--submission-bundle-out-dir", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    package_tag = str(args.package_tag).strip() or timestamp
    paths = _resolve_paths(
        root=root,
        out_root_arg=str(args.out_root),
        package_root_arg=str(args.package_root),
        package_tag=package_tag,
    )
    paths.out_root.mkdir(parents=True, exist_ok=True)

    executed: List[List[str]] = []

    base_cmd = _build_step27_base_cmd(args)
    _run(
        _build_step27_mode_cmd(
            base_cmd,
            out_dir=paths.mode1_dir,
            foc_feedback_mode="encoder",
            mic_feedback_mode="sensorless",
        ),
        cwd=root,
        dry_run=bool(args.dry_run),
        executed=executed,
    )
    _run(
        _build_step27_mode_cmd(
            base_cmd,
            out_dir=paths.mode2_dir,
            foc_feedback_mode="sensorless",
            mic_feedback_mode="sensorless",
        ),
        cwd=root,
        dry_run=bool(args.dry_run),
        executed=executed,
    )

    _run(
        _python_cmd(
            "tools/build_step28_ieee_summary.py",
            "--mode1-dir",
            str(paths.mode1_dir),
            "--mode2-dir",
            str(paths.mode2_dir),
            "--out-dir",
            str(paths.out_root),
        ),
        cwd=root,
        dry_run=bool(args.dry_run),
        executed=executed,
    )

    passport_dir: Path | None = None
    passport_dir_arg = str(args.passport_dir).strip()
    if passport_dir_arg:
        passport_dir = (root / passport_dir_arg).resolve()
    elif bool(args.build_passport):
        passport_tag = str(args.passport_tag).strip() or package_tag
        passport_out_root = (root / str(args.passport_out_root)).resolve()
        _run(
            _build_passport_cmd(args, passport_out_root=passport_out_root, passport_tag=passport_tag),
            cwd=root,
            dry_run=bool(args.dry_run),
            executed=executed,
        )
        passport_dir = passport_out_root / passport_tag

    _run(
        _build_package_cmd(
            args,
            root=root,
            step28_out=paths.out_root,
            package_root=paths.package_root,
            package_tag=package_tag,
            passport_dir=passport_dir,
        ),
        cwd=root,
        dry_run=bool(args.dry_run),
        executed=executed,
    )

    package_dir = paths.package_dir
    if bool(args.build_figures_tables):
        figs_cmd = [
            sys.executable,
            "tools/build_ieee_figures_tables.py",
            "--step28-dir",
            str(package_dir),
            "--out-dir",
            str(package_dir / "derived_ieee"),
        ]
        _run(figs_cmd, cwd=root, dry_run=bool(args.dry_run), executed=executed)

    motor_tuning_cmd = [
        sys.executable,
        "tools/build_motor_tuning_reports_from_step28.py",
        "--step28-dir",
        str(package_dir),
        "--out-dir",
        str(package_dir / "derived_ieee"),
    ]
    _run(motor_tuning_cmd, cwd=root, dry_run=bool(args.dry_run), executed=executed)

    if bool(args.promote_release):
        promote_cmd = [
            sys.executable,
            "tools/promote_ieee_release.py",
            "--step28-dir",
            str(package_dir),
            "--ieee-root",
            str((root / str(args.ieee_root)).resolve()),
            "--pgups-fig-dir",
            str((root / str(args.pgups_fig_dir)).resolve()),
            "--tag",
            package_tag,
        ]
        _run(promote_cmd, cwd=root, dry_run=bool(args.dry_run), executed=executed)

    if bool(args.freeze_submission_candidate):
        freeze_cmd = [
            sys.executable,
            "tools/freeze_ieee_submission_candidate.py",
            "--step28-dir",
            str(package_dir),
            "--out-json",
            str(package_dir / "submission_candidate_lock.json"),
            "--ieee-root",
            str((root / str(args.ieee_root)).resolve()),
            "--release-tag",
            package_tag,
        ]
        if bool(args.freeze_require_publication_assets):
            freeze_cmd.append("--require-publication-assets")
        if bool(args.promote_release) and bool(args.freeze_require_release_assets):
            freeze_cmd.append("--require-release-assets")
        _run(freeze_cmd, cwd=root, dry_run=bool(args.dry_run), executed=executed)

    checklist_cmd = [
        sys.executable,
        "tools/build_ieee_final_checklist.py",
        "--step28-dir",
        str(package_dir),
        "--out-md",
        str(package_dir / "FINAL_CHECKLIST_AUTO.md"),
        "--guardrails-policy",
        str((root / str(args.guardrails_policy)).resolve()),
    ]
    if bool(args.freeze_submission_candidate):
        checklist_cmd.append("--require-lock")
    _run(checklist_cmd, cwd=root, dry_run=bool(args.dry_run), executed=executed)

    candidate_note_cmd = [
        sys.executable,
        "tools/build_submission_candidate_note.py",
        "--step28-dir",
        str(package_dir),
        "--ieee-root",
        str((root / str(args.ieee_root)).resolve()),
        "--tag",
        package_tag,
        "--checklist-md",
        str(package_dir / "FINAL_CHECKLIST_AUTO.md"),
        "--out-md",
        str(package_dir / "SUBMISSION_CANDIDATE.md"),
        "--out-json",
        str(package_dir / "SUBMISSION_CANDIDATE.json"),
    ]
    _run(candidate_note_cmd, cwd=root, dry_run=bool(args.dry_run), executed=executed)

    release_manifest_cmd = [
        sys.executable,
        "tools/build_ieee_release_commit_manifest.py",
        "--step28-dir",
        str(package_dir),
        "--ieee-root",
        str((root / str(args.ieee_root)).resolve()),
        "--tag",
        package_tag,
        "--out-json",
        str(package_dir / "RELEASE_COMMIT_MANIFEST.json"),
        "--out-md",
        str(package_dir / "RELEASE_COMMIT_MANIFEST.md"),
        "--allow-dirty",
    ]
    _run(release_manifest_cmd, cwd=root, dry_run=bool(args.dry_run), executed=executed)

    dossier_cmd = [
        sys.executable,
        "tools/build_ieee_submission_dossier.py",
        "--step28-dir",
        str(package_dir),
        "--ieee-root",
        str((root / str(args.ieee_root)).resolve()),
        "--tag",
        package_tag,
        "--out-json",
        str(package_dir / "IEEE_SUBMISSION_DOSSIER.json"),
        "--out-md",
        str(package_dir / "IEEE_SUBMISSION_DOSSIER.md"),
    ]
    _run(dossier_cmd, cwd=root, dry_run=bool(args.dry_run), executed=executed)

    manuscript = (root / str(args.manuscript)).resolve()
    manuscript_consistency_cmd = [
        sys.executable,
        "tools/check_ieee_manuscript_consistency.py",
        "--manuscript",
        str(manuscript),
        "--out-json",
        str(package_dir / "MANUSCRIPT_CONSISTENCY_REPORT.json"),
        "--out-md",
        str(package_dir / "MANUSCRIPT_CONSISTENCY_REPORT.md"),
        "--strict",
    ]
    _run(manuscript_consistency_cmd, cwd=root, dry_run=bool(args.dry_run), executed=executed)

    manuscript_template_cmd = [
        sys.executable,
        "tools/check_ieee_manuscript_template.py",
        "--manuscript",
        str(manuscript),
        "--out-json",
        str(package_dir / "MANUSCRIPT_TEMPLATE_REPORT.json"),
        "--out-md",
        str(package_dir / "MANUSCRIPT_TEMPLATE_REPORT.md"),
        "--strict",
    ]
    _run(manuscript_template_cmd, cwd=root, dry_run=bool(args.dry_run), executed=executed)

    verify_cmd = [
        sys.executable,
        "tools/verify_ieee_submission_candidate.py",
        "--step28-dir",
        str(package_dir),
        "--ieee-root",
        str((root / str(args.ieee_root)).resolve()),
        "--guardrails-policy",
        str((root / str(args.guardrails_policy)).resolve()),
        "--manuscript",
        str(manuscript),
        "--allow-dirty",
    ]
    if bool(args.strict_verify):
        verify_cmd.append("--strict")
    _run(verify_cmd, cwd=root, dry_run=bool(args.dry_run), executed=executed)

    if bool(args.build_submission_bundle):
        bundle_cmd = [
            sys.executable,
            "tools/build_ieee_submission_bundle.py",
            "--step28-dir",
            str(package_dir),
            "--ieee-root",
            str((root / str(args.ieee_root)).resolve()),
            "--tag",
            package_tag,
            "--strict",
        ]
        bundle_out_dir = str(args.submission_bundle_out_dir).strip()
        if bundle_out_dir:
            bundle_cmd.extend(["--out-dir", str((root / bundle_out_dir).resolve())])
        _run(bundle_cmd, cwd=root, dry_run=bool(args.dry_run), executed=executed)

    final_dossier_cmd = [
        sys.executable,
        "tools/build_ieee_submission_dossier.py",
        "--step28-dir",
        str(package_dir),
        "--ieee-root",
        str((root / str(args.ieee_root)).resolve()),
        "--tag",
        package_tag,
        "--out-json",
        str(package_dir / "IEEE_SUBMISSION_DOSSIER.json"),
        "--out-md",
        str(package_dir / "IEEE_SUBMISSION_DOSSIER.md"),
    ]
    if bool(args.strict_verify):
        final_dossier_cmd.append("--strict")
    _run(final_dossier_cmd, cwd=root, dry_run=bool(args.dry_run), executed=executed)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "out_root": str(paths.out_root),
        "package_dir": str(package_dir),
        "package_tag": package_tag,
        "passport_dir": str(passport_dir) if passport_dir is not None else "",
        "dry_run": bool(args.dry_run),
        "executed_commands": executed,
    }
    manifest_path = paths.out_root / "step28_reproduce_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[reproduce-step28] manifest: {manifest_path}", flush=True)
    if not bool(args.dry_run):
        print(f"[reproduce-step28] package:  {package_dir}", flush=True)


if __name__ == "__main__":
    main()
