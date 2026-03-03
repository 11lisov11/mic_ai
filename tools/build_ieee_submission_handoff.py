from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


def _read_json(path: Path) -> Dict[str, object]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "ok"}


def _render_md(payload: Dict[str, object]) -> str:
    lines: List[str] = []
    lines.append("# IEEE Submission Handoff")
    lines.append("")
    lines.append(f"- generated_utc: `{payload.get('generated_utc', '')}`")
    lines.append(f"- tag: `{payload.get('tag', '')}`")
    lines.append(f"- handoff_ready: `{payload.get('handoff_ready', False)}`")
    lines.append(f"- verify_ok: `{payload.get('verify_ok', False)}`")
    lines.append(f"- dossier_ok: `{payload.get('dossier_ok', False)}`")
    lines.append(f"- bundle_ok: `{payload.get('bundle_ok', False)}`")
    lines.append("")
    lines.append("## Bundle Artifacts")
    bundle = dict(payload.get("bundle", {}))
    lines.append(f"- manifest_json: `{bundle.get('manifest_json', '')}`")
    lines.append(f"- manifest_md: `{bundle.get('manifest_md', '')}`")
    lines.append(f"- zip: `{bundle.get('zip', '')}`")
    lines.append(f"- tar_gz: `{bundle.get('tar_gz', '')}`")
    lines.append("")
    lines.append("## Upload Checklist")
    lines.append("1. Upload ZIP archive to IEEE submission portal.")
    lines.append("2. Attach manuscript source and generated PDF according to template.")
    lines.append("3. Attach reproducibility manifests from step28 package.")
    lines.append("4. Copy key metrics from `IEEE_SUBMISSION_DOSSIER.json` to submission notes.")
    lines.append("5. Verify hashes in `submission_bundle_manifest.json` after upload.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build IEEE submission handoff note from frozen package.")
    parser.add_argument("--step28-dir", required=True)
    parser.add_argument("--ieee-root", default="paper/ieee_2026")
    parser.add_argument("--bundle-dir", default="", help="Default: <ieee-root>/submission_bundle/<tag>")
    parser.add_argument("--tag", default="", help="Default: step28 directory name")
    parser.add_argument("--out-json", default="", help="Default: <step28-dir>/IEEE_SUBMISSION_HANDOFF.json")
    parser.add_argument("--out-md", default="", help="Default: <step28-dir>/IEEE_SUBMISSION_HANDOFF.md")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    step28_dir = Path(str(args.step28_dir)).expanduser().resolve()
    if not step28_dir.exists():
        raise FileNotFoundError(step28_dir)
    ieee_root = Path(str(args.ieee_root)).expanduser().resolve()
    if not ieee_root.exists():
        raise FileNotFoundError(ieee_root)
    tag = str(args.tag).strip() or step28_dir.name

    bundle_dir = (
        Path(str(args.bundle_dir)).expanduser().resolve()
        if str(args.bundle_dir).strip()
        else (ieee_root / "submission_bundle" / tag)
    )
    if not bundle_dir.exists():
        raise FileNotFoundError(bundle_dir)

    verify_json = step28_dir / "VERIFY_SUBMISSION_CANDIDATE.json"
    dossier_json = step28_dir / "IEEE_SUBMISSION_DOSSIER.json"
    bundle_manifest = bundle_dir / "submission_bundle_manifest.json"
    if not verify_json.exists():
        raise FileNotFoundError(verify_json)
    if not dossier_json.exists():
        raise FileNotFoundError(dossier_json)
    if not bundle_manifest.exists():
        raise FileNotFoundError(bundle_manifest)

    verify = _read_json(verify_json)
    dossier = _read_json(dossier_json)
    bundle = _read_json(bundle_manifest)

    verify_ok = _bool(verify.get("verification_ok", False))
    dossier_ok = _bool(dict(dossier.get("status", {})).get("dossier_ok", False))
    bundle_ok = _bool(bundle.get("bundle_ok", False))
    handoff_ready = bool(verify_ok and dossier_ok and bundle_ok)

    out_json = (
        Path(str(args.out_json)).expanduser().resolve()
        if str(args.out_json).strip()
        else (step28_dir / "IEEE_SUBMISSION_HANDOFF.json")
    )
    out_md = (
        Path(str(args.out_md)).expanduser().resolve()
        if str(args.out_md).strip()
        else (step28_dir / "IEEE_SUBMISSION_HANDOFF.md")
    )

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "tag": tag,
        "step28_dir": str(step28_dir),
        "ieee_root": str(ieee_root),
        "bundle_dir": str(bundle_dir),
        "verify_ok": verify_ok,
        "dossier_ok": dossier_ok,
        "bundle_ok": bundle_ok,
        "handoff_ready": handoff_ready,
        "bundle": {
            "manifest_json": str(bundle_manifest),
            "manifest_md": str(bundle_dir / "submission_bundle_manifest.md"),
            "zip": str(dict(bundle.get("archives", {})).get("zip", "")),
            "tar_gz": str(dict(bundle.get("archives", {})).get("tar_gz", "")),
        },
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(_render_md(payload), encoding="utf-8")
    print(f"saved: {out_json}")
    print(f"saved: {out_md}")
    print(f"handoff_ready: {handoff_ready}")

    if bool(args.strict) and not handoff_ready:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
