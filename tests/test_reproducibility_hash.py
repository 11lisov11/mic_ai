from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _run_step27(out_dir: Path) -> None:
    cmd = [
        sys.executable,
        "tools/step27_pipeline.py",
        "--motors",
        "air56",
        "--seeds",
        "101",
        "--scenarios",
        "speed_step",
        "--skip-air56-tune",
        "--mic-mode",
        "rule",
        "--out-dir",
        str(out_dir),
    ]
    subprocess.run(cmd, check=True)


def test_step27_reproducibility_hash_is_stable_for_same_protocol(tmp_path: Path) -> None:
    out_dir = tmp_path / "step27_repro"

    _run_step27(out_dir)
    first = json.loads((out_dir / "step27_reproducibility.json").read_text(encoding="utf-8"))
    sha1 = str(first.get("table_sha256", ""))
    assert sha1
    assert first.get("stable_vs_previous") is None

    _run_step27(out_dir)
    second = json.loads((out_dir / "step27_reproducibility.json").read_text(encoding="utf-8"))
    sha2 = str(second.get("table_sha256", ""))
    assert sha2 == sha1
    assert second.get("previous_table_sha256") == sha1
    assert second.get("stable_vs_previous") is True
