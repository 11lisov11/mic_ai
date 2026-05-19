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
    motor_shared = dict(joint_payload.get("per_seed_motor_checkpoints", {}))
    seed_shared = dict(motor_shared.get("101", {}))
    assert sorted(seed_shared) == ["air56", "al31"]
    assert "air56" in str(seed_shared["air56"])
    assert "al31" in str(seed_shared["al31"])

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
    assert str(runs[0].get("init_checkpoint", "")) == str(seed_shared["air56"])


def test_train_3motors_pipeline_isolates_joint_cycle_artifacts(tmp_path: Path) -> None:
    out_joint = tmp_path / "joint_isolated"
    ai_output_dir = tmp_path / "ai_outputs"
    results_root = tmp_path / "results_run"
    cmd_joint = [
        sys.executable,
        "tools/train_3motors_pipeline.py",
        "--mode",
        "joint-domain-randomized",
        "--motors",
        "air56",
        "--seeds",
        "101",
        "--joint-cycles",
        "2",
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
    payload = json.loads(joint_manifests[0].read_text(encoding="utf-8"))
    runs = [dict(row) for row in payload.get("runs", [])]
    assert [row.get("stage") for row in runs] == ["joint_cycle_1", "joint_cycle_2"]

    best_checkpoints = [Path(str(row.get("best_checkpoint", ""))) for row in runs]
    episode_logs = [Path(str(row.get("episodes_log", ""))) for row in runs]
    run_dirs = [Path(str(row.get("run_dir", ""))) for row in runs]

    assert len({str(path) for path in best_checkpoints}) == len(best_checkpoints)
    assert len({str(path) for path in episode_logs}) == len(episode_logs)
    assert len({str(path) for path in run_dirs}) == len(run_dirs)
    for path in [*best_checkpoints, *episode_logs, *run_dirs]:
        assert path.exists()
