from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_train_3motors_pipeline_smoke_separate_mode(tmp_path: Path) -> None:
    out_dir = tmp_path / "train3"
    ai_output_dir = tmp_path / "ai_outputs"
    results_root = tmp_path / "results_run"
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
        "--ai-output-dir",
        str(ai_output_dir),
        "--results-root",
        str(results_root),
        "--out-dir",
        str(out_dir),
    ]
    subprocess.run(cmd, check=True)

    manifests = list(out_dir.rglob("training_manifest_3motors.json"))
    assert len(manifests) == 1
    payload = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert payload.get("mode") == "separate-per-motor"
    assert payload.get("motors") == ["air56"]
    artifacts = dict(payload.get("artifacts", {}))
    assert "training_summaries_csv" in artifacts
    assert "training_acceptance_csv" in artifacts
    assert "training_eval_snapshots_csv" in artifacts
    assert "checkpoint_registry_json" in artifacts
    assert Path(str(artifacts["training_summaries_csv"])).exists()
    assert Path(str(artifacts["training_acceptance_csv"])).exists()
    assert Path(str(artifacts["training_eval_snapshots_csv"])).exists()
    assert Path(str(artifacts["checkpoint_registry_json"])).exists()

    runs = list(payload.get("runs", []))
    assert len(runs) == 1
    run0 = dict(runs[0])
    assert run0.get("motor") == "air56"
    best_ckpt = Path(str(run0.get("best_checkpoint", "")))
    assert best_ckpt.exists()
    acceptance = dict(payload.get("acceptance", {}))
    assert int(acceptance.get("total_runs", 0)) == 1
