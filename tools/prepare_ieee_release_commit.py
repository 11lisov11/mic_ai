from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple


def _run_git(args: List[str], *, cwd: Path) -> str:
    proc = subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)
    return proc.stdout.strip()


def _safe_git(args: List[str], *, cwd: Path) -> str:
    try:
        return _run_git(args, cwd=cwd)
    except Exception:
        return ""


def _read_json(path: Path) -> Dict[str, object]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _to_rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except Exception:
        return str(path.resolve()).replace("\\", "/")


def _collect_required(step28_dir: Path, bundle_dir: Path) -> List[Path]:
    return [
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
        bundle_dir / "submission_bundle_manifest.md",
        bundle_dir / "submission_bundle_manifest.json",
        bundle_dir / f"ieee_submission_{step28_dir.name}.zip",
        bundle_dir / f"ieee_submission_{step28_dir.name}.tar.gz",
    ]


def _collect_optional(ieee_root: Path) -> List[Path]:
    return [
        ieee_root / "FINAL_CHECKLIST_AUTO.md",
        ieee_root / "SUBMISSION_CANDIDATE.md",
        ieee_root / "SUBMISSION_CANDIDATE.json",
        ieee_root / "MANUSCRIPT_CONSISTENCY_REPORT.md",
        ieee_root / "MANUSCRIPT_CONSISTENCY_REPORT.json",
        ieee_root / "MANUSCRIPT_TEMPLATE_REPORT.md",
        ieee_root / "MANUSCRIPT_TEMPLATE_REPORT.json",
    ]


def _build_plan(
    *,
    repo_root: Path,
    step28_dir: Path,
    ieee_root: Path,
    bundle_dir: Path,
    tag: str,
    commit_message: str,
    tag_name: str,
) -> Dict[str, object]:
    verify = _read_json(step28_dir / "VERIFY_SUBMISSION_CANDIDATE.json")
    dossier = _read_json(step28_dir / "IEEE_SUBMISSION_DOSSIER.json")
    bundle = _read_json(bundle_dir / "submission_bundle_manifest.json")

    verify_ok = bool(verify.get("verification_ok", False))
    dossier_ok = bool(dict(dossier.get("status", {})).get("dossier_ok", False))
    bundle_ok = bool(bundle.get("bundle_ok", False))

    required = _collect_required(step28_dir, bundle_dir)
    optional = _collect_optional(ieee_root)

    missing_required: List[str] = []
    existing_required: List[str] = []
    existing_optional: List[str] = []
    for p in required:
        rel = _to_rel(p, repo_root)
        if p.exists():
            existing_required.append(rel)
        else:
            missing_required.append(rel)
    for p in optional:
        if p.exists():
            existing_optional.append(_to_rel(p, repo_root))

    git_add_paths = sorted(set([*existing_required, *existing_optional]))
    release_ready = bool(verify_ok and dossier_ok and bundle_ok and len(missing_required) == 0)
    plan = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(repo_root),
        "step28_dir": str(step28_dir),
        "ieee_root": str(ieee_root),
        "bundle_dir": str(bundle_dir),
        "tag": tag,
        "commit_message": commit_message,
        "tag_name": tag_name,
        "verify_ok": verify_ok,
        "dossier_ok": dossier_ok,
        "bundle_ok": bundle_ok,
        "missing_required_count": len(missing_required),
        "missing_required": missing_required,
        "git_add_paths_count": len(git_add_paths),
        "git_add_paths": git_add_paths,
        "release_ready": release_ready,
        "git": {
            "branch": _safe_git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_root),
            "commit": _safe_git(["rev-parse", "HEAD"], cwd=repo_root),
            "status_porcelain": _safe_git(["status", "--porcelain"], cwd=repo_root).splitlines(),
        },
    }
    return plan


def _render_md(payload: Dict[str, object]) -> str:
    lines: List[str] = []
    lines.append("# IEEE Release Git Plan")
    lines.append("")
    lines.append(f"- generated_utc: `{payload.get('generated_utc', '')}`")
    lines.append(f"- tag: `{payload.get('tag', '')}`")
    lines.append(f"- release_ready: `{payload.get('release_ready', False)}`")
    lines.append(f"- verify_ok: `{payload.get('verify_ok', False)}`")
    lines.append(f"- dossier_ok: `{payload.get('dossier_ok', False)}`")
    lines.append(f"- bundle_ok: `{payload.get('bundle_ok', False)}`")
    lines.append(f"- missing_required_count: `{payload.get('missing_required_count', 0)}`")
    lines.append(f"- git_add_paths_count: `{payload.get('git_add_paths_count', 0)}`")
    lines.append(f"- commit_message: `{payload.get('commit_message', '')}`")
    lines.append(f"- tag_name: `{payload.get('tag_name', '')}`")
    lines.append("")
    missing = list(payload.get("missing_required", []))
    if missing:
        lines.append("## Missing Required")
        for item in missing:
            lines.append(f"- {item}")
        lines.append("")
    lines.append("## Git Add Paths")
    for item in list(payload.get("git_add_paths", [])):
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## Commands")
    lines.append("```bash")
    lines.append("git add -- <paths from list>")
    lines.append(f"git commit -m \"{payload.get('commit_message', '')}\"")
    lines.append(f"git tag -a {payload.get('tag_name', '')} -m \"IEEE release {payload.get('tag', '')}\"")
    lines.append("git push origin <branch>")
    lines.append(f"git push origin {payload.get('tag_name', '')}")
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def _git_add_paths(repo_root: Path, paths: List[str]) -> None:
    if not paths:
        return
    subprocess.run(["git", "add", "--", *paths], cwd=repo_root, check=True)


def _git_commit(repo_root: Path, message: str) -> None:
    subprocess.run(["git", "commit", "-m", message], cwd=repo_root, check=True)


def _git_tag(repo_root: Path, tag_name: str, tag_message: str) -> None:
    subprocess.run(["git", "tag", "-a", tag_name, "-m", tag_message], cwd=repo_root, check=True)


def _git_push(repo_root: Path, remote: str, branch: str, tag_name: str) -> None:
    subprocess.run(["git", "push", remote, branch], cwd=repo_root, check=True)
    subprocess.run(["git", "push", remote, tag_name], cwd=repo_root, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare/apply git release commit for IEEE frozen package.")
    parser.add_argument("--step28-dir", required=True)
    parser.add_argument("--ieee-root", default="paper/ieee_2026")
    parser.add_argument("--bundle-dir", default="", help="Default: <ieee-root>/submission_bundle/<tag>")
    parser.add_argument("--tag", default="", help="Default: <step28-dir-name>")
    parser.add_argument("--commit-message", default="")
    parser.add_argument("--tag-name", default="")
    parser.add_argument("--out-json", default="", help="Default: <step28-dir>/RELEASE_GIT_PLAN.json")
    parser.add_argument("--out-md", default="", help="Default: <step28-dir>/RELEASE_GIT_PLAN.md")
    parser.add_argument("--apply", action="store_true", help="Execute git add/commit/tag locally.")
    parser.add_argument("--push", action="store_true", help="Push commit/tag to remote (requires --apply).")
    parser.add_argument("--remote", default="origin")
    parser.add_argument("--branch", default="")
    parser.add_argument("--allow-dirty", action="store_true")
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
    bundle_dir = (
        Path(str(args.bundle_dir)).expanduser().resolve()
        if str(args.bundle_dir).strip()
        else (ieee_root / "submission_bundle" / tag)
    )
    if not bundle_dir.exists():
        raise FileNotFoundError(bundle_dir)

    commit_message = str(args.commit_message).strip() or f"chore(ieee): freeze submission candidate {tag}"
    tag_name = str(args.tag_name).strip() or f"ieee/{tag}"

    out_json = (
        Path(str(args.out_json)).expanduser().resolve()
        if str(args.out_json).strip()
        else (step28_dir / "RELEASE_GIT_PLAN.json")
    )
    out_md = (
        Path(str(args.out_md)).expanduser().resolve()
        if str(args.out_md).strip()
        else (step28_dir / "RELEASE_GIT_PLAN.md")
    )

    payload = _build_plan(
        repo_root=repo_root,
        step28_dir=step28_dir,
        ieee_root=ieee_root,
        bundle_dir=bundle_dir,
        tag=tag,
        commit_message=commit_message,
        tag_name=tag_name,
    )
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(_render_md(payload), encoding="utf-8")
    print(f"saved: {out_json}")
    print(f"saved: {out_md}")
    print(f"release_ready: {bool(payload.get('release_ready', False))}")

    if bool(args.strict) and not bool(payload.get("release_ready", False)):
        raise SystemExit(1)

    if not bool(args.apply):
        if bool(args.push):
            raise SystemExit("--push requires --apply")
        return

    if not bool(args.allow_dirty):
        status_lines = list(dict(payload.get("git", {})).get("status_porcelain", []))
        if status_lines:
            raise SystemExit("git worktree is dirty (use --allow-dirty to proceed)")

    paths = list(payload.get("git_add_paths", []))
    _git_add_paths(repo_root, [str(p) for p in paths])
    _git_commit(repo_root, commit_message)
    _git_tag(repo_root, tag_name, f"IEEE release {tag}")
    print(f"commit_created: {commit_message}")
    print(f"tag_created: {tag_name}")

    if bool(args.push):
        branch = str(args.branch).strip() or _safe_git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_root)
        if not branch:
            raise SystemExit("cannot determine branch for push")
        _git_push(repo_root, str(args.remote), branch, tag_name)
        print(f"pushed: remote={args.remote} branch={branch} tag={tag_name}")


if __name__ == "__main__":
    main()
