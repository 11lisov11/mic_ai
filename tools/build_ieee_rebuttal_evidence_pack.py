from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _to_rel(path: Path, roots: List[Path]) -> str:
    resolved = path.resolve()
    for root in roots:
        try:
            return str(resolved.relative_to(root.resolve())).replace("\\", "/")
        except Exception:
            continue
    raw = str(resolved).replace("\\", "/")
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    return f"external/{digest}_{path.name}"


def _collect_paths(step28_dir: Path, ieee_root: Path, tag: str, include_archives: bool) -> Tuple[List[Path], List[Path]]:
    bundle_dir = ieee_root / "submission_bundle" / tag
    release_dir = ieee_root / "data" / "release" / tag
    mode1 = step28_dir / "mode1_foc_encoder_vs_mic_sensorless"
    mode2 = step28_dir / "mode2_foc_sensorless_vs_mic_sensorless"

    required = [
        step28_dir / "step28_ieee_summary.csv",
        step28_dir / "step28_ieee_summary.md",
        step28_dir / "VERIFY_SUBMISSION_CANDIDATE.json",
        step28_dir / "FINAL_CHECKLIST_AUTO.md",
        step28_dir / "MANUSCRIPT_CONSISTENCY_REPORT.json",
        step28_dir / "MANUSCRIPT_TEMPLATE_REPORT.json",
        step28_dir / "IEEE_SUBMISSION_DOSSIER.json",
        step28_dir / "IEEE_SUBMISSION_HANDOFF.json",
        step28_dir / "RELEASE_COMMIT_MANIFEST.json",
        step28_dir / "IEEE_RELEASE_NOTES.json",
        step28_dir / "STEP28_REGRESSION_GUARD.json",
        step28_dir / "CAMERA_READY_CHECKLIST.json",
        mode1 / "step27_per_seed_metrics.csv",
        mode1 / "step27_stats_motor_controller.csv",
        mode1 / "step27_final_pi_vs_foc_vs_mic.csv",
        mode2 / "step27_per_seed_metrics.csv",
        mode2 / "step27_stats_motor_controller.csv",
        mode2 / "step27_final_pi_vs_foc_vs_mic.csv",
        step28_dir / "derived_ieee" / "ieee_pi_foc_mic_stats.csv",
        step28_dir / "derived_ieee" / "fig_ieee_pi_foc_mic_power.pdf",
        step28_dir / "passport" / "passport_compare_3motors.csv",
        step28_dir / "passport" / "passport_compare_3motors.json",
        bundle_dir / "submission_bundle_manifest.json",
        ieee_root / "manuscript.md",
        ieee_root / "guardrails_policy.json",
    ]

    optional = [
        step28_dir / "MANUSCRIPT_CONSISTENCY_REPORT.md",
        step28_dir / "MANUSCRIPT_TEMPLATE_REPORT.md",
        step28_dir / "IEEE_SUBMISSION_DOSSIER.md",
        step28_dir / "IEEE_SUBMISSION_HANDOFF.md",
        step28_dir / "RELEASE_COMMIT_MANIFEST.md",
        step28_dir / "IEEE_RELEASE_NOTES.md",
        step28_dir / "STEP28_REGRESSION_GUARD.md",
        step28_dir / "CAMERA_READY_CHECKLIST.md",
        step28_dir / "submission_candidate_lock.json",
        step28_dir / "SUBMISSION_CANDIDATE.json",
        step28_dir / "SUBMISSION_CANDIDATE.md",
        step28_dir / "derived_ieee" / "fig_ieee_pi_foc_mic_power.png",
        step28_dir / "derived_ieee" / "fig_ieee_pi_foc_mic_power.svg",
        release_dir / "promotion_manifest.json",
        release_dir / "release_snapshot.json",
        bundle_dir / "submission_bundle_manifest.md",
    ]
    if include_archives:
        optional.extend(
            [
                bundle_dir / f"ieee_submission_{tag}.zip",
                bundle_dir / f"ieee_submission_{tag}.tar.gz",
            ]
        )

    return required, optional


def _copy_with_manifest(paths: List[Path], *, rel_roots: List[Path], out_content: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for src in paths:
        src_rel = _to_rel(src, rel_roots)
        dst = out_content / src_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
        rows.append(
            {
                "src_rel": src_rel,
                "dst_rel": _to_rel(dst, [out_content]),
                "size_bytes": int(dst.stat().st_size),
                "sha256": _sha256(dst),
            }
        )
    return rows


def _render_md(payload: Dict[str, object]) -> str:
    lines: List[str] = []
    lines.append("# IEEE Rebuttal Evidence Pack")
    lines.append("")
    lines.append(f"- generated_utc: `{payload.get('generated_utc', '')}`")
    lines.append(f"- tag: `{payload.get('tag', '')}`")
    lines.append(f"- strict_ready: `{payload.get('strict_ready', False)}`")
    lines.append(f"- required_missing_count: `{payload.get('required_missing_count', 0)}`")
    lines.append(f"- optional_missing_count: `{payload.get('optional_missing_count', 0)}`")
    lines.append(f"- copied_files_count: `{payload.get('copied_files_count', 0)}`")
    lines.append("")

    missing_required = list(payload.get("required_missing", []))
    if missing_required:
        lines.append("## Missing Required")
        for item in missing_required:
            lines.append(f"- {item}")
        lines.append("")

    lines.append("## Copied Files")
    for row in list(payload.get("copied_files", [])):
        if not isinstance(row, dict):
            continue
        lines.append(f"- `{row.get('src_rel', '')}` sha256=`{row.get('sha256', '')}`")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build IEEE rebuttal evidence pack (tables/figures/hashes/logs).")
    parser.add_argument("--step28-dir", required=True)
    parser.add_argument("--ieee-root", default="paper/ieee_2026")
    parser.add_argument("--tag", default="", help="Default: step28 directory name")
    parser.add_argument("--out-dir", default="", help="Default: <ieee-root>/data/rebuttal/<tag>")
    parser.add_argument("--include-archives", action="store_true", help="Copy zip/tar.gz bundle archives too.")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    step28_dir = Path(str(args.step28_dir)).expanduser().resolve()
    if not step28_dir.exists():
        raise FileNotFoundError(step28_dir)
    ieee_root = Path(str(args.ieee_root)).expanduser().resolve()
    if not ieee_root.exists():
        raise FileNotFoundError(ieee_root)
    tag = str(args.tag).strip() or step28_dir.name

    out_dir = (
        Path(str(args.out_dir)).expanduser().resolve()
        if str(args.out_dir).strip()
        else (ieee_root / "data" / "rebuttal" / tag)
    )
    out_content = out_dir / "content"
    out_content.mkdir(parents=True, exist_ok=True)

    required, optional = _collect_paths(
        step28_dir=step28_dir,
        ieee_root=ieee_root,
        tag=tag,
        include_archives=bool(args.include_archives),
    )
    required_present = [p for p in required if p.exists()]
    rel_roots = [step28_dir, ieee_root, repo_root]
    required_missing = [_to_rel(p, rel_roots) for p in required if not p.exists()]
    optional_present = [p for p in optional if p.exists()]
    optional_missing = [_to_rel(p, rel_roots) for p in optional if not p.exists()]

    copied = _copy_with_manifest([*required_present, *optional_present], rel_roots=rel_roots, out_content=out_content)

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "tag": tag,
        "step28_dir": str(step28_dir),
        "ieee_root": str(ieee_root),
        "out_dir": str(out_dir),
        "strict_ready": bool(len(required_missing) == 0),
        "required_missing_count": int(len(required_missing)),
        "optional_missing_count": int(len(optional_missing)),
        "required_missing": required_missing,
        "optional_missing": optional_missing,
        "copied_files_count": int(len(copied)),
        "copied_files": copied,
    }

    out_json = out_dir / "REBUTTAL_EVIDENCE_PACK.json"
    out_md = out_dir / "REBUTTAL_EVIDENCE_PACK.md"
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    out_md.write_text(_render_md(payload), encoding="utf-8")
    print(f"saved: {out_json}")
    print(f"saved: {out_md}")
    print(f"strict_ready: {bool(payload.get('strict_ready', False))}")

    if bool(args.strict) and not bool(payload.get("strict_ready", False)):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
