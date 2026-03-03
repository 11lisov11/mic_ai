from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Set


PATH_TOKEN_RE = re.compile(r"`([^`\n]+)`")


def _looks_like_path(token: str) -> bool:
    text = str(token).strip()
    if not text:
        return False
    if text.startswith("http://") or text.startswith("https://"):
        return False
    # Exclude scalar/equation snippets often wrapped in backticks in manuscript text.
    if any(ch in text for ch in ("%", "=", "<", ">", " ")):
        return False
    # Treat only explicit directory-like tokens as paths to avoid false positives.
    return ("/" in text) or ("\\" in text)


def _resolve_path(token: str, repo_root: Path, manuscript_dir: Path) -> Path:
    raw = str(token).strip().replace("\\", "/")
    cand = Path(raw)
    if cand.is_absolute():
        return cand
    # Prefer repo-root relative first for project docs.
    p1 = (repo_root / cand).resolve()
    if p1.exists():
        return p1
    # Fallback: relative to manuscript directory.
    return (manuscript_dir / cand).resolve()


def check_manuscript(manuscript: Path, *, repo_root: Path) -> Dict[str, object]:
    text = manuscript.read_text(encoding="utf-8")
    tokens = [m.group(1).strip() for m in PATH_TOKEN_RE.finditer(text)]
    path_tokens: List[str] = []
    seen: Set[str] = set()
    for t in tokens:
        if not _looks_like_path(t):
            continue
        key = t.strip()
        if key in seen:
            continue
        seen.add(key)
        path_tokens.append(key)

    exists_rows: List[Dict[str, object]] = []
    missing_rows: List[Dict[str, object]] = []
    manuscript_dir = manuscript.parent
    for token in path_tokens:
        resolved = _resolve_path(token, repo_root=repo_root, manuscript_dir=manuscript_dir)
        row = {
            "token": token,
            "resolved_path": str(resolved),
            "exists": bool(resolved.exists()),
        }
        if resolved.exists():
            exists_rows.append(row)
        else:
            missing_rows.append(row)

    has_fig = "fig." in text.lower() or "figure" in text.lower()
    has_table = "tab." in text.lower() or "table" in text.lower()

    payload: Dict[str, object] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "manuscript": str(manuscript),
        "paths_total": len(path_tokens),
        "paths_existing": len(exists_rows),
        "paths_missing": len(missing_rows),
        "has_figure_mentions": bool(has_fig),
        "has_table_mentions": bool(has_table),
        "ok": bool(len(missing_rows) == 0 and has_fig and has_table),
        "existing_paths": exists_rows,
        "missing_paths": missing_rows,
    }
    return payload


def _render_md(payload: Dict[str, object]) -> str:
    lines: List[str] = []
    lines.append("# Manuscript Consistency Report")
    lines.append("")
    lines.append(f"- manuscript: `{payload.get('manuscript', '')}`")
    lines.append(f"- ok: `{payload.get('ok', False)}`")
    lines.append(f"- paths_total: `{payload.get('paths_total', 0)}`")
    lines.append(f"- paths_existing: `{payload.get('paths_existing', 0)}`")
    lines.append(f"- paths_missing: `{payload.get('paths_missing', 0)}`")
    lines.append(f"- has_figure_mentions: `{payload.get('has_figure_mentions', False)}`")
    lines.append(f"- has_table_mentions: `{payload.get('has_table_mentions', False)}`")
    lines.append("")
    missing = payload.get("missing_paths", [])
    if isinstance(missing, list) and missing:
        lines.append("## Missing Paths")
        for row in missing:
            if not isinstance(row, dict):
                continue
            lines.append(f"- token=`{row.get('token','')}` resolved=`{row.get('resolved_path','')}`")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check IEEE manuscript consistency (path references and mentions).")
    parser.add_argument("--manuscript", default="paper/ieee_2026/manuscript.md")
    parser.add_argument("--repo-root", default="", help="Project root for resolving repo-relative paths.")
    parser.add_argument("--out-json", default="")
    parser.add_argument("--out-md", default="")
    parser.add_argument("--strict", action="store_true", help="Return non-zero when report.ok=false")
    args = parser.parse_args()

    repo_root = (
        Path(str(args.repo_root)).expanduser().resolve()
        if str(args.repo_root).strip()
        else Path(__file__).resolve().parents[1]
    )
    manuscript = Path(str(args.manuscript)).expanduser().resolve()
    if not manuscript.exists():
        raise FileNotFoundError(manuscript)
    out_json = (
        Path(str(args.out_json)).expanduser().resolve()
        if str(args.out_json).strip()
        else (manuscript.parent / "MANUSCRIPT_CONSISTENCY_REPORT.json")
    )
    out_md = (
        Path(str(args.out_md)).expanduser().resolve()
        if str(args.out_md).strip()
        else (manuscript.parent / "MANUSCRIPT_CONSISTENCY_REPORT.md")
    )

    payload = check_manuscript(manuscript, repo_root=repo_root)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(_render_md(payload), encoding="utf-8")
    print(f"saved: {out_json}")
    print(f"saved: {out_md}")
    print(f"ok: {bool(payload.get('ok', False))}")

    if bool(args.strict) and not bool(payload.get("ok", False)):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
