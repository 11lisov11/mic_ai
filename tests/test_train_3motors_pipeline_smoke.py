from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_train_3motors_pipeline_smoke_separate_mode(tmp_path: Path) -> None:
    out_dir = tmp_path / "train3"
    cmd = [
        sys.executable,
        "tools/train_3motors_pipeline.py",
        "--mode",
        "separate-per-motor",
        "--motors",
        "air56",
        "--seeds",
        "101",
        "--episodes",
        "1",
        "--episode-steps",
        "20",
        "--fast",
        "--scenarios",
        "speed_step",
        "--out-dir",
        str(out_dir),
    ]
    subprocess.run(cmd, check=True)

    manifests = list(out_dir.rglob("training_manifest_3motors.json"))
    assert len(manifests) == 1
    payload = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert payload.get("mode") == "separate-per-motor"
    assert payload.get("motors") == ["air56"]
    runs = list(payload.get("runs", []))
    assert len(runs) == 1
    run0 = dict(runs[0])
    assert run0.get("motor") == "air56"
    best_ckpt = Path(str(run0.get("best_checkpoint", "")))
    assert best_ckpt.exists()
