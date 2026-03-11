from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path


def test_robust_motor_hardening_dry_run(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = repo_root / "outputs" / "_pytest_robust_hardening" / tmp_path.name
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "tools/robust_motor_hardening.py",
        "--motors",
        "al31",
        "--out-dir",
        str(out_dir),
        "--dry-run",
    ]
    subprocess.run(cmd, check=True, cwd=repo_root)

    summary_json = out_dir / "robust_hardening_summary.json"
    assert summary_json.exists()
    payload = json.loads(summary_json.read_text(encoding="utf-8"))
    assert payload["motors"] == ["al31"]
    assert len(payload["results"]) == 1
    assert payload["results"][0]["dry_run"] is True
    shutil.rmtree(out_dir.parent, ignore_errors=True)
