from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _run_git(args: List[str], *, cwd: Path) -> str:
    proc = subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)
    return proc.stdout.strip()


def _safe_git(args: List[str], *, cwd: Path) -> str:
    try:
        return _run_git(args, cwd=cwd)
    except Exception:
        return ""


def _render_rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except Exception:
        return str(path.resolve())


def _collect_paths(step28_dir: Path, ieee_root: Path, tag: str) -> Tuple[List[Tuple[str, Path]], List[Tuple[str, Path]]]:
    required = [
        ("step28_root", step28_dir / "step28_ieee_summary.csv"),
        ("step28_root", step28_dir / "step28_ieee_summary.md"),
        ("step28_root", step28_dir / "package_manifest.json"),
        ("step28_quality", step28_dir / "FINAL_CHECKLIST_AUTO.md"),
        ("step28_quality", step28_dir / "submission_candidate_lock.json"),
        ("step28_quality", step28_dir / "SUBMISSION_CANDIDATE.md"),
        ("step28_quality", step28_dir / "SUBMISSION_CANDIDATE.json"),
    ]
    optional = [
        ("step28_derived", step28_dir / "derived_ieee" / "ieee_pi_foc_mic_stats.csv"),
        ("step28_derived", step28_dir / "derived_ieee" / "motor_tuning_acceptance_summary.json"),
        ("step28_passport", step28_dir / "passport" / "passport_compare_3motors.json"),
        ("ieee_root", ieee_root / "FINAL_CHECKLIST_AUTO.md"),
        ("ieee_root", ieee_root / "SUBMISSION_CANDIDATE.md"),
        ("ieee_root", ieee_root / "SUBMISSION_CANDIDATE.json"),
        ("ieee_root", ieee_root / "manuscript.md"),
        ("ieee_root", ieee_root / "guardrails_policy.json"),
        ("release", ieee_root / "data" / "release" / tag / "promotion_manifest.json"),
        ("release", ieee_root / "data" / "release" / tag / "release_snapshot.json"),
    ]
    return required, optional


def main() -> None:
    parser = argparse.ArgumentParser(description="Build immutable release commit manifest for IEEE submission package.")
    parser.add_argument("--step28-dir", required=True)
    parser.add_argument("--ieee-root", default="paper/ieee_2026")
    parser.add_argument("--tag", default="", help="Release tag. Default: step28 directory name.")
    parser.add_argument("--out-json", default="", help="Default: <step28-dir>/RELEASE_COMMIT_MANIFEST.json")
    parser.add_argument("--out-md", default="", help="Default: <step28-dir>/RELEASE_COMMIT_MANIFEST.md")
    parser.add_argument("--allow-dirty", action="store_true", help="Do not fail when git worktree is dirty.")
    parser.add_argument("--strict", action="store_true", help="Fail when required artifacts are missing.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
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
        else (step28_dir / "RELEASE_COMMIT_MANIFEST.json")
    )
    out_md = (
        Path(str(args.out_md)).expanduser().resolve()
        if str(args.out_md).strip()
        else (step28_dir / "RELEASE_COMMIT_MANIFEST.md")
    )

    required_paths, optional_paths = _collect_paths(step28_dir, ieee_root, tag)
    files: List[Dict[str, object]] = []
    missing_required: List[str] = []
    missing_optional: List[str] = []

    for group, path in required_paths:
        if not path.exists():
            missing_required.append(_render_rel(path, repo_root))
            continue
        files.append(
            {
                "group": group,
                "required": True,
                "path": _render_rel(path, repo_root),
                "size_bytes": int(path.stat().st_size),
                "sha256": _sha256(path),
            }
        )

    for group, path in optional_paths:
        if not path.exists():
            missing_optional.append(_render_rel(path, repo_root))
            continue
        files.append(
            {
                "group": group,
                "required": False,
                "path": _render_rel(path, repo_root),
                "size_bytes": int(path.stat().st_size),
                "sha256": _sha256(path),
            }
        )

    files = sorted(files, key=lambda x: str(x["path"]))
    agg = hashlib.sha256()
    for row in files:
        agg.update(str(row["path"]).encode("utf-8"))
        agg.update(str(row["sha256"]).encode("utf-8"))
        agg.update(str(int(row["size_bytes"])).encode("utf-8"))
    aggregate_sha = agg.hexdigest()

    git_commit = _safe_git(["rev-parse", "HEAD"], cwd=repo_root)
    git_branch = _safe_git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_root)
    git_status = _safe_git(["status", "--porcelain"], cwd=repo_root)
    dirty_lines = [ln for ln in git_status.splitlines() if str(ln).strip()]
    git_dirty = len(dirty_lines) > 0

    required_ok = len(missing_required) == 0
    dirty_ok = (not git_dirty) or bool(args.allow_dirty)
    manifest_ok = bool(required_ok and dirty_ok)

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "tag": tag,
        "repo_root": str(repo_root),
        "step28_dir": str(step28_dir),
        "ieee_root": str(ieee_root),
        "git": {
            "commit": git_commit,
            "branch": git_branch,
            "dirty": git_dirty,
            "dirty_lines_count": len(dirty_lines),
        },
        "allow_dirty": bool(args.allow_dirty),
        "required_ok": required_ok,
        "manifest_ok": manifest_ok,
        "required_missing": missing_required,
        "optional_missing": missing_optional,
        "files_count": len(files),
        "aggregate_sha256": aggregate_sha,
        "files": files,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines: List[str] = []
    lines.append("# Release Commit Manifest")
    lines.append("")
    lines.append(f"- generated_utc: `{payload['generated_utc']}`")
    lines.append(f"- tag: `{tag}`")
    lines.append(f"- git_commit: `{git_commit}`")
    lines.append(f"- git_branch: `{git_branch}`")
    lines.append(f"- git_dirty: `{git_dirty}`")
    lines.append(f"- dirty_lines_count: `{len(dirty_lines)}`")
    lines.append(f"- required_ok: `{required_ok}`")
    lines.append(f"- manifest_ok: `{manifest_ok}`")
    lines.append(f"- files_count: `{len(files)}`")
    lines.append(f"- aggregate_sha256: `{aggregate_sha}`")
    lines.append("")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"saved: {out_json}")
    print(f"saved: {out_md}")
    if bool(args.strict) and not manifest_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
