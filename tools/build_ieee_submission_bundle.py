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
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except Exception:
        raw = str(path.resolve()).replace("\\", "/")
        short = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
        return f"_external/{short}_{path.name}"


def _add_unique(paths: List[Path], path: Path) -> None:
    if path not in paths:
        paths.append(path)


def _collect_paths(step28_dir: Path, ieee_root: Path, tag: str) -> Tuple[List[Path], List[Path]]:
    required: List[Path] = [
        step28_dir / "step28_ieee_summary.csv",
        step28_dir / "step28_ieee_summary.md",
        step28_dir / "package_manifest.json",
        step28_dir / "FINAL_CHECKLIST_AUTO.md",
        step28_dir / "submission_candidate_lock.json",
        step28_dir / "SUBMISSION_CANDIDATE.md",
        step28_dir / "SUBMISSION_CANDIDATE.json",
        step28_dir / "RELEASE_COMMIT_MANIFEST.md",
        step28_dir / "RELEASE_COMMIT_MANIFEST.json",
        step28_dir / "IEEE_SUBMISSION_DOSSIER.md",
        step28_dir / "IEEE_SUBMISSION_DOSSIER.json",
        step28_dir / "VERIFY_SUBMISSION_CANDIDATE.json",
        step28_dir / "MANUSCRIPT_CONSISTENCY_REPORT.md",
        step28_dir / "MANUSCRIPT_CONSISTENCY_REPORT.json",
        step28_dir / "MANUSCRIPT_TEMPLATE_REPORT.md",
        step28_dir / "MANUSCRIPT_TEMPLATE_REPORT.json",
        step28_dir / "derived_ieee" / "ieee_pi_foc_mic_stats.csv",
        step28_dir / "derived_ieee" / "motor_tuning_acceptance_summary.csv",
        step28_dir / "derived_ieee" / "motor_tuning_acceptance_summary.json",
        ieee_root / "manuscript.md",
        ieee_root / "FINAL_CHECKLIST.md",
        ieee_root / "guardrails_policy.json",
        ieee_root / "fig" / "README.md",
        ieee_root / "fig" / "fig1_mic_methodology.png",
        ieee_root / "fig" / "fig2_pi_foc_mic_power.pdf",
        ieee_root / "fig" / "fig3_air56_working_characteristics.pdf",
        ieee_root / "fig" / "fig4_cross_motor_robustness.pdf",
        ieee_root / "fig" / "fig5_training_to_foc.pdf",
    ]

    optional: List[Path] = [
        ieee_root / "FINAL_CHECKLIST_AUTO.md",
        ieee_root / "SUBMISSION_CANDIDATE.md",
        ieee_root / "SUBMISSION_CANDIDATE.json",
        ieee_root / "MANUSCRIPT_CONSISTENCY_REPORT.md",
        ieee_root / "MANUSCRIPT_CONSISTENCY_REPORT.json",
        ieee_root / "MANUSCRIPT_TEMPLATE_REPORT.md",
        ieee_root / "MANUSCRIPT_TEMPLATE_REPORT.json",
        ieee_root / "fig" / "fig2_pi_foc_mic_power.png",
        ieee_root / "fig" / "fig2_pi_foc_mic_power.svg",
        ieee_root / "fig" / "fig3_air56_working_characteristics.png",
        ieee_root / "fig" / "fig3_air56_working_characteristics.svg",
        ieee_root / "fig" / "fig4_cross_motor_robustness.png",
        ieee_root / "fig" / "fig4_cross_motor_robustness.svg",
        ieee_root / "fig" / "fig5_training_to_foc.png",
        ieee_root / "fig" / "fig5_training_to_foc.svg",
        step28_dir / "passport" / "passport_compare_3motors.csv",
        step28_dir / "passport" / "passport_compare_3motors.md",
        step28_dir / "passport" / "passport_compare_3motors.json",
        ieee_root / "data" / "release" / tag / "promotion_manifest.json",
        ieee_root / "data" / "release" / tag / "release_snapshot.json",
    ]

    release_tables = ieee_root / "data" / "release" / tag / "tables"
    if release_tables.exists():
        for p in sorted(release_tables.glob("*.csv")):
            _add_unique(optional, p)
    return required, optional


def _copy_files(
    paths: List[Path],
    *,
    repo_root: Path,
    payload_root: Path,
    required: bool,
) -> Tuple[List[Dict[str, object]], List[str]]:
    rows: List[Dict[str, object]] = []
    missing: List[str] = []
    for src in paths:
        rel = _rel(src, repo_root)
        if not src.exists():
            missing.append(rel)
            continue
        dst = payload_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        rows.append(
            {
                "required": bool(required),
                "path": rel,
                "size_bytes": int(src.stat().st_size),
                "sha256": _sha256(src),
            }
        )
    return rows, missing


def _aggregate_sha(rows: List[Dict[str, object]]) -> str:
    agg = hashlib.sha256()
    for row in sorted(rows, key=lambda x: str(x["path"])):
        agg.update(str(row["path"]).encode("utf-8"))
        agg.update(str(row["sha256"]).encode("utf-8"))
        agg.update(str(int(row["size_bytes"])).encode("utf-8"))
    return agg.hexdigest()


def _render_md(payload: Dict[str, object]) -> str:
    lines: List[str] = []
    lines.append("# IEEE Submission Bundle")
    lines.append("")
    lines.append(f"- generated_utc: `{payload.get('generated_utc', '')}`")
    lines.append(f"- tag: `{payload.get('tag', '')}`")
    lines.append(f"- bundle_dir: `{payload.get('bundle_dir', '')}`")
    lines.append(f"- bundle_ok: `{payload.get('bundle_ok', False)}`")
    lines.append(f"- files_copied: `{payload.get('files_copied', 0)}`")
    lines.append(f"- aggregate_sha256: `{payload.get('aggregate_sha256', '')}`")
    lines.append("")
    archives = dict(payload.get("archives", {}))
    lines.append("## Archives")
    lines.append(f"- zip: `{archives.get('zip', '')}`")
    lines.append(f"- tar_gz: `{archives.get('tar_gz', '')}`")
    lines.append("")
    req_missing = list(payload.get("required_missing", []))
    if req_missing:
        lines.append("## Missing Required")
        for item in req_missing:
            lines.append(f"- {item}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build IEEE submission bundle (content + zip/tar.gz + hash manifest).")
    parser.add_argument("--step28-dir", required=True)
    parser.add_argument("--ieee-root", default="paper/ieee_2026")
    parser.add_argument("--tag", default="", help="Default: step28 directory name")
    parser.add_argument("--out-dir", default="", help="Default: <ieee-root>/submission_bundle/<tag>")
    parser.add_argument("--strict", action="store_true", help="Return non-zero if required files are missing.")
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
        else (ieee_root / "submission_bundle" / tag)
    )
    payload_root = out_dir / "content"
    if payload_root.exists():
        shutil.rmtree(payload_root)
    payload_root.mkdir(parents=True, exist_ok=True)

    required_paths, optional_paths = _collect_paths(step28_dir=step28_dir, ieee_root=ieee_root, tag=tag)
    req_rows, req_missing = _copy_files(required_paths, repo_root=repo_root, payload_root=payload_root, required=True)
    opt_rows, _opt_missing = _copy_files(optional_paths, repo_root=repo_root, payload_root=payload_root, required=False)
    rows = sorted([*req_rows, *opt_rows], key=lambda x: str(x["path"]))

    archive_base = out_dir / f"ieee_submission_{tag}"
    zip_path = Path(shutil.make_archive(str(archive_base), "zip", root_dir=str(payload_root)))
    tar_path = Path(shutil.make_archive(str(archive_base), "gztar", root_dir=str(payload_root)))

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "tag": tag,
        "repo_root": str(repo_root),
        "step28_dir": str(step28_dir),
        "ieee_root": str(ieee_root),
        "bundle_dir": str(out_dir),
        "payload_root": str(payload_root),
        "bundle_ok": len(req_missing) == 0,
        "files_copied": len(rows),
        "required_missing_count": len(req_missing),
        "required_missing": req_missing,
        "aggregate_sha256": _aggregate_sha(rows),
        "files": rows,
        "archives": {
            "zip": str(zip_path),
            "tar_gz": str(tar_path),
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "submission_bundle_manifest.json"
    out_md = out_dir / "submission_bundle_manifest.md"
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    out_md.write_text(_render_md(payload), encoding="utf-8")

    print(f"saved: {out_json}")
    print(f"saved: {out_md}")
    print(f"saved: {zip_path}")
    print(f"saved: {tar_path}")
    print(f"bundle_ok: {bool(payload.get('bundle_ok', False))}")

    if bool(args.strict) and not bool(payload.get("bundle_ok", False)):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
