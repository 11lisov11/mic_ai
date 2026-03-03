from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_train_3motors_pipeline_joint_and_finetune_smoke(tmp_path: Path) -> None:
    out_joint = tmp_path / "joint"
    ai_output_dir = tmp_path / "ai_outputs"
    results_root = tmp_path / "results_run"
    cmd_joint = [
        sys.executable,
        "tools/train_3motors_pipeline.py",
        "--mode",
        "joint-domain-randomized",
        "--motors",
        "air56,al31",
        "--seeds",
        "101",
        "--joint-cycles",
        "1",
        "--joint-cycle-episodes",
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
        str(out_joint),
    ]
    subprocess.run(cmd_joint, check=True, cwd=Path(__file__).resolve().parents[1])

    joint_manifests = list(out_joint.rglob("training_manifest_3motors.json"))
    assert len(joint_manifests) == 1
    joint_payload = json.loads(joint_manifests[0].read_text(encoding="utf-8"))
    assert joint_payload.get("mode") == "joint-domain-randomized"
    shared = dict(joint_payload.get("per_seed_shared_checkpoints", {}))
    assert str(shared.get("101", "")) != ""

    out_ft = tmp_path / "finetune"
    cmd_ft = [
        sys.executable,
        "tools/train_3motors_pipeline.py",
        "--mode",
        "fine_tune_per_motor",
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
        "--base-manifest",
        str(joint_manifests[0]),
        "--ai-output-dir",
        str(ai_output_dir),
        "--results-root",
        str(results_root),
        "--out-dir",
        str(out_ft),
    ]
    subprocess.run(cmd_ft, check=True, cwd=Path(__file__).resolve().parents[1])

    ft_manifests = list(out_ft.rglob("training_manifest_3motors.json"))
    assert len(ft_manifests) == 1
    ft_payload = json.loads(ft_manifests[0].read_text(encoding="utf-8"))
    assert ft_payload.get("mode") == "fine_tune_per_motor"
    runs = list(ft_payload.get("runs", []))
    assert len(runs) == 1
    assert str(runs[0].get("motor", "")) == "air56"
