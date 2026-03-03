from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


def _run(cmd: List[str], *, cwd: Path) -> None:
    print("[verify-ieee] run:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=cwd)


def _read_json(path: Path) -> Dict[str, object]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _check_ready(checklist_text: str) -> bool:
    return "ready_for_submission: `true`" in str(checklist_text).lower()


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify existing IEEE submission-candidate package.")
    parser.add_argument("--step28-dir", required=True)
    parser.add_argument("--ieee-root", default="paper/ieee_2026")
    parser.add_argument("--tag", default="", help="Release tag. Default: step28 directory name.")
    parser.add_argument("--guardrails-policy", default="paper/ieee_2026/guardrails_policy.json")
    parser.add_argument("--manuscript", default="paper/ieee_2026/manuscript.md")
    parser.add_argument("--allow-dirty", action="store_true", help="Allow dirty git worktree in release manifest stage.")
    parser.add_argument("--out-json", default="", help="Default: <step28-dir>/VERIFY_SUBMISSION_CANDIDATE.json")
    parser.add_argument("--strict", action="store_true", help="Return non-zero when verification fails.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    step28_dir = Path(str(args.step28_dir)).expanduser().resolve()
    if not step28_dir.exists():
        raise FileNotFoundError(step28_dir)
    ieee_root = Path(str(args.ieee_root)).expanduser().resolve()
    if not ieee_root.exists():
        raise FileNotFoundError(ieee_root)
    policy = Path(str(args.guardrails_policy)).expanduser().resolve()
    if not policy.exists():
        raise FileNotFoundError(policy)
    tag = str(args.tag).strip() or step28_dir.name
    out_json = (
        Path(str(args.out_json)).expanduser().resolve()
        if str(args.out_json).strip()
        else (step28_dir / "VERIFY_SUBMISSION_CANDIDATE.json")
    )

    checklist_path = step28_dir / "FINAL_CHECKLIST_AUTO.md"
    checklist_cmd = [
        sys.executable,
        "tools/build_ieee_final_checklist.py",
        "--step28-dir",
        str(step28_dir),
        "--out-md",
        str(checklist_path),
        "--ieee-root",
        str(ieee_root),
        "--guardrails-policy",
        str(policy),
        "--require-lock",
    ]
    if bool(args.strict):
        checklist_cmd.append("--strict")
    _run(checklist_cmd, cwd=root)

    manuscript = Path(str(args.manuscript)).expanduser().resolve()
    manuscript_json = step28_dir / "MANUSCRIPT_CONSISTENCY_REPORT.json"
    manuscript_md = step28_dir / "MANUSCRIPT_CONSISTENCY_REPORT.md"
    manuscript_consistency_cmd = [
        sys.executable,
        "tools/check_ieee_manuscript_consistency.py",
        "--manuscript",
        str(manuscript),
        "--out-json",
        str(manuscript_json),
        "--out-md",
        str(manuscript_md),
    ]
    if bool(args.strict):
        manuscript_consistency_cmd.append("--strict")
    _run(manuscript_consistency_cmd, cwd=root)

    template_json = step28_dir / "MANUSCRIPT_TEMPLATE_REPORT.json"
    template_md = step28_dir / "MANUSCRIPT_TEMPLATE_REPORT.md"
    manuscript_template_cmd = [
        sys.executable,
        "tools/check_ieee_manuscript_template.py",
        "--manuscript",
        str(manuscript),
        "--out-json",
        str(template_json),
        "--out-md",
        str(template_md),
    ]
    if bool(args.strict):
        manuscript_template_cmd.append("--strict")
    _run(manuscript_template_cmd, cwd=root)

    candidate_md = step28_dir / "SUBMISSION_CANDIDATE.md"
    candidate_json = step28_dir / "SUBMISSION_CANDIDATE.json"
    _run(
        [
            sys.executable,
            "tools/build_submission_candidate_note.py",
            "--step28-dir",
            str(step28_dir),
            "--ieee-root",
            str(ieee_root),
            "--tag",
            tag,
            "--checklist-md",
            str(checklist_path),
            "--out-md",
            str(candidate_md),
            "--out-json",
            str(candidate_json),
        ],
        cwd=root,
    )

    release_md = step28_dir / "RELEASE_COMMIT_MANIFEST.md"
    release_json = step28_dir / "RELEASE_COMMIT_MANIFEST.json"
    rel_cmd = [
        sys.executable,
        "tools/build_ieee_release_commit_manifest.py",
        "--step28-dir",
        str(step28_dir),
        "--ieee-root",
        str(ieee_root),
        "--tag",
        tag,
        "--out-json",
        str(release_json),
        "--out-md",
        str(release_md),
    ]
    if bool(args.strict):
        rel_cmd.append("--strict")
    if bool(args.allow_dirty):
        rel_cmd.append("--allow-dirty")
    _run(rel_cmd, cwd=root)

    dossier_md = step28_dir / "IEEE_SUBMISSION_DOSSIER.md"
    dossier_json = step28_dir / "IEEE_SUBMISSION_DOSSIER.json"
    dossier_cmd = [
        sys.executable,
        "tools/build_ieee_submission_dossier.py",
        "--step28-dir",
        str(step28_dir),
        "--ieee-root",
        str(ieee_root),
        "--tag",
        tag,
        "--out-json",
        str(dossier_json),
        "--out-md",
        str(dossier_md),
    ]
    if bool(args.strict):
        dossier_cmd.append("--strict")
    _run(dossier_cmd, cwd=root)

    checklist_ready = _check_ready(checklist_path.read_text(encoding="utf-8"))
    release_payload = _read_json(release_json)
    manifest_ok = bool(release_payload.get("manifest_ok", False))
    template_payload = _read_json(template_json)
    template_ok = bool(template_payload.get("ok", False))
    verification_ok = bool(checklist_ready and manifest_ok and template_ok)

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "step28_dir": str(step28_dir),
        "ieee_root": str(ieee_root),
        "tag": tag,
        "guardrails_policy": str(policy),
        "allow_dirty": bool(args.allow_dirty),
        "checklist_ready_for_submission": checklist_ready,
        "release_manifest_ok": manifest_ok,
        "manuscript_template_ok": template_ok,
        "verification_ok": verification_ok,
        "artifacts": {
            "checklist_md": str(checklist_path),
            "candidate_md": str(candidate_md),
            "candidate_json": str(candidate_json),
            "release_manifest_md": str(release_md),
            "release_manifest_json": str(release_json),
            "dossier_md": str(dossier_md),
            "dossier_json": str(dossier_json),
            "manuscript_consistency_md": str(manuscript_md),
            "manuscript_consistency_json": str(manuscript_json),
            "manuscript_template_md": str(template_md),
            "manuscript_template_json": str(template_json),
        },
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"saved: {out_json}")
    print(f"verification_ok: {verification_ok}")

    if bool(args.strict) and not verification_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
