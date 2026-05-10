from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_report_plan_completion_smoke(tmp_path: Path) -> None:
    plan = tmp_path / "PLAN.md"
    plan.write_text(
        (
            "# Plan\n"
            "- [x] done item\n"
            "- [ ] todo item\n"
            "1. [~] in progress item\n"
            "Статус: `DONE`\n"
        ),
        encoding="utf-8",
    )
    out_json = tmp_path / "progress.json"
    out_md = tmp_path / "progress.md"
    cmd = [
        sys.executable,
        "tools/report_plan_completion.py",
        "--plan",
        str(plan),
        "--out-json",
        str(out_json),
        "--out-md",
        str(out_md),
    ]
    subprocess.run(cmd, check=True, cwd=Path(__file__).resolve().parents[1])
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    checkboxes = dict(payload.get("checkboxes", {}))
    assert int(checkboxes.get("total", 0)) == 3
    assert int(checkboxes.get("done", 0)) == 1
    assert int(checkboxes.get("todo", 0)) == 1
    assert int(checkboxes.get("in_progress", 0)) == 1


def test_project_master_plan_keeps_hardware_open_until_physical_acceptance() -> None:
    root = Path(__file__).resolve().parents[1]
    plan = root / "PROJECT_MASTER_PLAN.md"
    out = root / ".tmp_pytest" / "project_master_plan_guard.json"
    cmd = [
        sys.executable,
        "tools/report_plan_completion.py",
        "--plan",
        str(plan),
        "--out-json",
        str(out),
        "--out-md",
        str(out.with_suffix(".md")),
    ]
    subprocess.run(cmd, check=True, cwd=root)
    payload = json.loads(out.read_text(encoding="utf-8"))
    checkboxes = dict(payload.get("checkboxes", {}))
    open_items = list(payload.get("open_items", []))

    assert int(checkboxes.get("total", 0)) > 0
    assert int(checkboxes.get("todo", 0)) > 0
    assert bool(checkboxes.get("hard_ready", True)) is False
    assert float(checkboxes.get("completion_pct", 100.0)) < 100.0
    assert any("Stage 4 physical A/B comparison" in str(item.get("text", "")) for item in open_items)
