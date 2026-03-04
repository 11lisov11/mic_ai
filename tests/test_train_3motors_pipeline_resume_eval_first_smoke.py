from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_train_3motors_pipeline_resume_eval_first_smoke(tmp_path: Path) -> None:
    base_out = tmp_path / "base"
    ai_output_dir = tmp_path / "ai_outputs"
    results_root = tmp_path / "results_run"

    cmd_base = [
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
        "--accept-max-speed-error",
        "1000000",
        "--accept-min-eta-energy",
        "-1000000",
        "--accept-max-current-rms",
        "1000000",
        "--ai-output-dir",
        str(ai_output_dir),
        "--results-root",
        str(results_root),
        "--out-dir",
        str(base_out),
    ]
    subprocess.run(cmd_base, check=True, cwd=Path(__file__).resolve().parents[1])

    base_manifests = list(base_out.rglob("training_manifest_3motors.json"))
    assert len(base_manifests) == 1
    base_manifest = base_manifests[0]
    base_payload = json.loads(base_manifest.read_text(encoding="utf-8"))
    assert int(dict(base_payload.get("acceptance", {})).get("passed_runs", 0)) == 1

    eval_first_out = tmp_path / "eval_first"
    cmd_eval_first = [
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
        "--eval-first",
        "--resume-manifest",
        str(base_manifest),
        "--ai-output-dir",
        str(ai_output_dir),
        "--results-root",
        str(results_root),
        "--out-dir",
        str(eval_first_out),
    ]
    subprocess.run(cmd_eval_first, check=True, cwd=Path(__file__).resolve().parents[1])

    eval_manifests = list(eval_first_out.rglob("training_manifest_3motors.json"))
    assert len(eval_manifests) == 1
    eval_payload = json.loads(eval_manifests[0].read_text(encoding="utf-8"))
    assert bool(eval_payload.get("eval_first", False)) is True
    assert str(eval_payload.get("resume_manifest", "")) == str(base_manifest.resolve())
    artifacts = dict(eval_payload.get("artifacts", {}))
    assert Path(str(artifacts.get("training_protocol_json", ""))).exists()
    run_rows = list(eval_payload.get("runs", []))
    assert len(run_rows) == 1
    row0 = dict(run_rows[0])
    assert bool(row0.get("reused", False)) is True
    assert str(row0.get("reused_from_manifest", "")) == str(base_manifest.resolve())

    repro_files = list(eval_first_out.rglob("training_repro_package_3motors.json"))
    assert len(repro_files) == 1
    repro_payload = json.loads(repro_files[0].read_text(encoding="utf-8"))
    assert str(repro_payload.get("protocol_hash", "")).strip() != ""
