from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


def _read_json(path: Path) -> Dict[str, object]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _safe_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "ok"}


def _to_float(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _fmt_pct(value: float) -> str:
    return "nan" if not math.isfinite(value) else f"{value:+.3f}%"


def _check_ready_from_md(text: str) -> bool:
    return "ready_for_submission: `true`" in str(text).lower()


def build_dossier(step28_dir: Path, ieee_root: Path, tag: str) -> Dict[str, object]:
    checklist_md = step28_dir / "FINAL_CHECKLIST_AUTO.md"
    lock_json = step28_dir / "submission_candidate_lock.json"
    candidate_json = step28_dir / "SUBMISSION_CANDIDATE.json"
    release_manifest_json = step28_dir / "RELEASE_COMMIT_MANIFEST.json"
    verify_json = step28_dir / "VERIFY_SUBMISSION_CANDIDATE.json"
    template_json = step28_dir / "MANUSCRIPT_TEMPLATE_REPORT.json"
    motor_summary_json = step28_dir / "derived_ieee" / "motor_tuning_acceptance_summary.json"
    passport_json = step28_dir / "passport" / "passport_compare_3motors.json"
    bundle_manifest_json = ieee_root / "submission_bundle" / tag / "submission_bundle_manifest.json"

    checklist_ready = checklist_md.exists() and _check_ready_from_md(checklist_md.read_text(encoding="utf-8"))
    lock_ok = False
    lock_aggregate = ""
    if lock_json.exists():
        lock = _read_json(lock_json)
        lock_ok = _safe_bool(lock.get("lock_ok", False))
        lock_aggregate = str(lock.get("aggregate_sha256", ""))

    candidate_ready = False
    if candidate_json.exists():
        candidate = _read_json(candidate_json)
        candidate_ready = _safe_bool(candidate.get("ready_for_submission", False))

    release_manifest_ok = False
    release_aggregate = ""
    git_commit = ""
    git_branch = ""
    git_dirty = None
    if release_manifest_json.exists():
        release = _read_json(release_manifest_json)
        release_manifest_ok = _safe_bool(release.get("manifest_ok", False))
        release_aggregate = str(release.get("aggregate_sha256", ""))
        git = release.get("git", {})
        if isinstance(git, dict):
            git_commit = str(git.get("commit", ""))
            git_branch = str(git.get("branch", ""))
            raw_dirty = git.get("dirty", None)
            git_dirty = None if raw_dirty is None else bool(raw_dirty)

    verify_ok = False
    if verify_json.exists():
        verify = _read_json(verify_json)
        verify_ok = _safe_bool(verify.get("verification_ok", False))

    template_ok = False
    if template_json.exists():
        template = _read_json(template_json)
        template_ok = _safe_bool(template.get("ok", False))

    bundle_ok = False
    bundle_present = bundle_manifest_json.exists()
    if bundle_present:
        bundle = _read_json(bundle_manifest_json)
        bundle_ok = _safe_bool(bundle.get("bundle_ok", False))

    motor_rows: List[Dict[str, object]] = []
    if motor_summary_json.exists():
        payload = _read_json(motor_summary_json)
        rows = payload.get("rows", [])
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, dict):
                    motor_rows.append(dict(row))
    motor_rows = sorted(motor_rows, key=lambda x: str(x.get("motor", "")))

    passport_failures = 0
    passport_warnings = 0
    if passport_json.exists():
        p = _read_json(passport_json)
        failures = p.get("failures", [])
        warnings = p.get("warnings", [])
        passport_failures = len(failures) if isinstance(failures, list) else 0
        passport_warnings = len(warnings) if isinstance(warnings, list) else 0

    dossier_ok = bool(checklist_ready and lock_ok and candidate_ready and release_manifest_ok and template_ok)
    if verify_json.exists():
        dossier_ok = bool(dossier_ok and verify_ok)
    if bundle_present:
        dossier_ok = bool(dossier_ok and bundle_ok)

    payload: Dict[str, object] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "tag": tag,
        "step28_dir": str(step28_dir),
        "ieee_root": str(ieee_root),
        "status": {
            "checklist_ready_for_submission": checklist_ready,
            "lock_ok": lock_ok,
            "candidate_ready_for_submission": candidate_ready,
            "release_commit_manifest_ok": release_manifest_ok,
            "verify_ok": verify_ok,
            "manuscript_template_ok": template_ok,
            "bundle_present": bundle_present,
            "bundle_ok": bundle_ok,
            "passport_failures": passport_failures,
            "passport_warnings": passport_warnings,
            "dossier_ok": dossier_ok,
        },
        "hashes": {
            "submission_lock_aggregate_sha256": lock_aggregate,
            "release_manifest_aggregate_sha256": release_aggregate,
        },
        "git": {
            "commit": git_commit,
            "branch": git_branch,
            "dirty": git_dirty,
        },
        "motor_summary_rows": motor_rows,
        "artifacts": {
            "checklist_md": str(checklist_md),
            "lock_json": str(lock_json),
            "candidate_json": str(candidate_json),
            "release_manifest_json": str(release_manifest_json),
            "verify_json": str(verify_json),
            "template_json": str(template_json),
            "bundle_manifest_json": str(bundle_manifest_json),
            "motor_summary_json": str(motor_summary_json),
            "passport_json": str(passport_json),
        },
    }
    return payload


def _render_md(payload: Dict[str, object]) -> str:
    status = dict(payload.get("status", {}))
    hashes = dict(payload.get("hashes", {}))
    git = dict(payload.get("git", {}))
    rows = payload.get("motor_summary_rows", [])
    rows_list = rows if isinstance(rows, list) else []

    lines: List[str] = []
    lines.append("# IEEE Submission Dossier")
    lines.append("")
    lines.append(f"- generated_utc: `{payload.get('generated_utc', '')}`")
    lines.append(f"- tag: `{payload.get('tag', '')}`")
    lines.append(f"- step28_dir: `{payload.get('step28_dir', '')}`")
    lines.append(f"- dossier_ok: `{status.get('dossier_ok', False)}`")
    lines.append("")
    lines.append("## Status")
    lines.append(f"- checklist_ready_for_submission: `{status.get('checklist_ready_for_submission', False)}`")
    lines.append(f"- lock_ok: `{status.get('lock_ok', False)}`")
    lines.append(f"- candidate_ready_for_submission: `{status.get('candidate_ready_for_submission', False)}`")
    lines.append(f"- release_commit_manifest_ok: `{status.get('release_commit_manifest_ok', False)}`")
    lines.append(f"- verify_ok: `{status.get('verify_ok', False)}`")
    lines.append(f"- manuscript_template_ok: `{status.get('manuscript_template_ok', False)}`")
    lines.append(f"- bundle_present: `{status.get('bundle_present', False)}`")
    lines.append(f"- bundle_ok: `{status.get('bundle_ok', False)}`")
    lines.append(f"- passport_failures: `{status.get('passport_failures', 0)}`")
    lines.append(f"- passport_warnings: `{status.get('passport_warnings', 0)}`")
    lines.append("")
    lines.append("## Hashes")
    lines.append(f"- submission_lock_aggregate_sha256: `{hashes.get('submission_lock_aggregate_sha256', '')}`")
    lines.append(f"- release_manifest_aggregate_sha256: `{hashes.get('release_manifest_aggregate_sha256', '')}`")
    lines.append("")
    lines.append("## Git")
    lines.append(f"- commit: `{git.get('commit', '')}`")
    lines.append(f"- branch: `{git.get('branch', '')}`")
    lines.append(f"- dirty: `{git.get('dirty', '')}`")
    lines.append("")
    lines.append("## Motor Summary")
    if not rows_list:
        lines.append("- no motor summary rows")
    else:
        for row in rows_list:
            if not isinstance(row, dict):
                continue
            motor = str(row.get("motor", ""))
            acc = _safe_bool(row.get("acceptance_pass", False))
            p_mean = _fmt_pct(_to_float(row.get("avg_power_saving_pct_mean")))
            p_min = _fmt_pct(_to_float(row.get("avg_power_saving_pct_min")))
            eta_mean = _fmt_pct(_to_float(row.get("avg_eta_gain_pct_mean")))
            eta_min = _fmt_pct(_to_float(row.get("avg_eta_gain_pct_min")))
            err_max = _to_float(row.get("err_failures_max"))
            err_max_s = "nan" if not math.isfinite(err_max) else f"{err_max:.2f}"
            lines.append(
                f"- motor={motor}, acceptance_pass={acc}, power_mean={p_mean}, power_min={p_min}, eta_mean={eta_mean}, eta_min={eta_min}, err_max={err_max_s}"
            )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build IEEE submission dossier from frozen package artifacts.")
    parser.add_argument("--step28-dir", required=True)
    parser.add_argument("--ieee-root", default="paper/ieee_2026")
    parser.add_argument("--tag", default="", help="Default: step28 directory name")
    parser.add_argument("--out-json", default="", help="Default: <step28-dir>/IEEE_SUBMISSION_DOSSIER.json")
    parser.add_argument("--out-md", default="", help="Default: <step28-dir>/IEEE_SUBMISSION_DOSSIER.md")
    parser.add_argument("--strict", action="store_true", help="Return non-zero when dossier_ok=false.")
    args = parser.parse_args()

    step28_dir = Path(str(args.step28_dir)).expanduser().resolve()
    if not step28_dir.exists():
        raise FileNotFoundError(step28_dir)
    ieee_root = Path(str(args.ieee_root)).expanduser().resolve()
    if not ieee_root.exists():
        raise FileNotFoundError(ieee_root)

    tag = str(args.tag).strip() or step28_dir.name
    out_json = (
        Path(str(args.out_json)).expanduser().resolve()
        if str(args.out_json).strip()
        else (step28_dir / "IEEE_SUBMISSION_DOSSIER.json")
    )
    out_md = (
        Path(str(args.out_md)).expanduser().resolve()
        if str(args.out_md).strip()
        else (step28_dir / "IEEE_SUBMISSION_DOSSIER.md")
    )

    payload = build_dossier(step28_dir=step28_dir, ieee_root=ieee_root, tag=tag)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(_render_md(payload), encoding="utf-8")
    print(f"saved: {out_json}")
    print(f"saved: {out_md}")
    print(f"dossier_ok: {bool(dict(payload.get('status', {})).get('dossier_ok', False))}")

    if bool(args.strict) and not bool(dict(payload.get("status", {})).get("dossier_ok", False)):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
