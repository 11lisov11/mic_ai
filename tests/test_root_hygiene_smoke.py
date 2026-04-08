from __future__ import annotations

from pathlib import Path


def test_root_contains_only_active_project_master_plan() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    root_entries = sorted(p.name for p in repo_root.glob("PROJECT_MASTER*") if p.is_file())
    assert root_entries == ["PROJECT_MASTER_PLAN.md"]


def test_plan_archive_readme_documents_root_rule() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    archive_readme = repo_root / "docs" / "plan_archive" / "README.md"
    assert archive_readme.exists()
    text = archive_readme.read_text(encoding="utf-8")
    assert "PROJECT_MASTER_PLAN.md" in text
    assert "do not create dated `PROJECT_MASTER_PLAN_*` files in root anymore" in text
