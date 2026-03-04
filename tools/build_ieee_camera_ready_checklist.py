from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple


def _read_json(path: Path) -> Dict[str, object]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "ok"}


def _check_ready_from_md(path: Path) -> bool:
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8")
    return "ready_for_submission: `true`" in text.lower()


def _item(name: str, path: Path, ok: bool) -> Dict[str, object]:
    return {"name": name, "path": str(path), "exists": bool(path.exists()), "ok": bool(ok)}


def _render_md(payload: Dict[str, object]) -> str:
    lines: List[str] = []
    lines.append("# IEEE Camera-Ready Checklist")
    lines.append("")
    lines.append(f"- generated_utc: `{payload.get('generated_utc', '')}`")
    lines.append(f"- tag: `{payload.get('tag', '')}`")
    lines.append(f"- camera_ready_ok: `{payload.get('camera_ready_ok', False)}`")
    lines.append("")
    lines.append("## Checks")
    for row in list(payload.get("checks", [])):
        if not isinstance(row, dict):
            continue
        marker = "[x]" if bool(row.get("ok", False)) else "[ ]"
        lines.append(
            "- {mark} {name}: exists={exists}, ok={ok}, path=`{path}`".format(
                mark=marker,
                name=row.get("name", ""),
                exists=bool(row.get("exists", False)),
                ok=bool(row.get("ok", False)),
                path=row.get("path", ""),
            )
        )
    lines.append("")
    return "\n".join(lines)


def build_payload(step28_dir: Path, ieee_root: Path, tag: str) -> Dict[str, object]:
    bundle_manifest = ieee_root / "submission_bundle" / tag / "submission_bundle_manifest.json"

    final_checklist_md = step28_dir / "FINAL_CHECKLIST_AUTO.md"
    verify_json = step28_dir / "VERIFY_SUBMISSION_CANDIDATE.json"
    template_json = step28_dir / "MANUSCRIPT_TEMPLATE_REPORT.json"
    consistency_json = step28_dir / "MANUSCRIPT_CONSISTENCY_REPORT.json"
    dossier_json = step28_dir / "IEEE_SUBMISSION_DOSSIER.json"
    handoff_json = step28_dir / "IEEE_SUBMISSION_HANDOFF.json"
    release_notes_json = step28_dir / "IEEE_RELEASE_NOTES.json"
    regression_guard_json = step28_dir / "STEP28_REGRESSION_GUARD.json"

    verify_ok = False
    if verify_json.exists():
        verify_ok = _as_bool(_read_json(verify_json).get("verification_ok", False))
    template_ok = False
    if template_json.exists():
        template_ok = _as_bool(_read_json(template_json).get("ok", False))
    consistency_ok = False
    if consistency_json.exists():
        consistency_ok = _as_bool(_read_json(consistency_json).get("ok", False))
    dossier_ok = False
    if dossier_json.exists():
        dossier_ok = _as_bool(dict(_read_json(dossier_json).get("status", {})).get("dossier_ok", False))
    handoff_ok = False
    if handoff_json.exists():
        handoff_ok = _as_bool(_read_json(handoff_json).get("handoff_ready", False))
    release_notes_ok = False
    if release_notes_json.exists():
        release_payload = _read_json(release_notes_json)
        release_notes_ok = _as_bool(release_payload.get("strict_ready", False)) or _as_bool(
            release_payload.get("release_note_ready", False)
        )
    regression_guard_ok = False
    if regression_guard_json.exists():
        regression_guard_ok = _as_bool(_read_json(regression_guard_json).get("ok", False))
    bundle_ok = False
    if bundle_manifest.exists():
        bundle_ok = _as_bool(_read_json(bundle_manifest).get("bundle_ok", False))

    checks = [
        _item("final_checklist_ready", final_checklist_md, _check_ready_from_md(final_checklist_md)),
        _item("verify_submission_candidate", verify_json, verify_ok),
        _item("manuscript_template", template_json, template_ok),
        _item("manuscript_consistency", consistency_json, consistency_ok),
        _item("submission_dossier", dossier_json, dossier_ok),
        _item("submission_handoff", handoff_json, handoff_ok),
        _item("release_notes", release_notes_json, release_notes_ok),
        _item("step28_regression_guard", regression_guard_json, regression_guard_ok),
        _item("submission_bundle_manifest", bundle_manifest, bundle_ok),
    ]

    camera_ready_ok = all(bool(row.get("exists", False)) and bool(row.get("ok", False)) for row in checks)
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "tag": tag,
        "step28_dir": str(step28_dir),
        "ieee_root": str(ieee_root),
        "camera_ready_ok": bool(camera_ready_ok),
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build camera-ready checklist from frozen step28 artifacts.")
    parser.add_argument("--step28-dir", required=True)
    parser.add_argument("--ieee-root", default="paper/ieee_2026")
    parser.add_argument("--tag", default="", help="Default: step28 directory name")
    parser.add_argument("--out-json", default="", help="Default: <step28-dir>/CAMERA_READY_CHECKLIST.json")
    parser.add_argument("--out-md", default="", help="Default: <step28-dir>/CAMERA_READY_CHECKLIST.md")
    parser.add_argument("--strict", action="store_true")
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
        else (step28_dir / "CAMERA_READY_CHECKLIST.json")
    )
    out_md = (
        Path(str(args.out_md)).expanduser().resolve()
        if str(args.out_md).strip()
        else (step28_dir / "CAMERA_READY_CHECKLIST.md")
    )

    payload = build_payload(step28_dir=step28_dir, ieee_root=ieee_root, tag=tag)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(_render_md(payload), encoding="utf-8")
    print(f"saved: {out_json}")
    print(f"saved: {out_md}")
    print(f"camera_ready_ok: {bool(payload.get('camera_ready_ok', False))}")

    if bool(args.strict) and not bool(payload.get("camera_ready_ok", False)):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
