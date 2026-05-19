from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import tools.train_3motors_pipeline as train3
from tools.train_3motors_pipeline import _summarize_run


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


def test_train_3motors_acceptance_rejects_unphysical_eta(tmp_path: Path) -> None:
    episodes_log = tmp_path / "episodes.json"
    episodes_log.write_text(
        json.dumps(
            [
                {
                    "mean_speed_error": 1.0,
                    "eta_energy": 25.0,
                    "mean_p_in_pos": 10.0,
                    "mean_current_rms": 1.0,
                }
            ]
        ),
        encoding="utf-8",
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    best = tmp_path / "best_actor.pth"
    last = tmp_path / "last_actor.pth"
    best.write_bytes(b"best")
    last.write_bytes(b"last")

    args = SimpleNamespace(
        degradation_window=5,
        degradation_speed_delta=2.0,
        degradation_eta_delta=0.05,
        accept_max_speed_error=30.0,
        accept_min_eta_energy=0.0,
        accept_max_eta_energy=1.2,
        accept_max_current_rms=10.0,
        accept_max_p_in_pos=None,
    )
    row = {
        "motor": "air56",
        "seed": 101,
        "stage": "separate",
        "episodes_log": str(episodes_log),
        "run_dir": str(run_dir),
        "best_checkpoint": str(best),
        "last_checkpoint": str(last),
    }

    _, acceptance, _, _ = _summarize_run(row, args)

    assert acceptance["eta_min_ok"] is True
    assert acceptance["eta_max_ok"] is False
    assert acceptance["eta_ok"] is False
    assert acceptance["acceptance_pass"] is False


def test_train_3motors_acceptance_can_use_step27_source(tmp_path: Path) -> None:
    episodes_log = tmp_path / "episodes.json"
    episodes_log.write_text(
        json.dumps(
            [
                {
                    "mean_speed_error": 999.0,
                    "eta_energy": 0.5,
                    "mean_p_in_pos": 10.0,
                    "mean_current_rms": 1.0,
                }
            ]
        ),
        encoding="utf-8",
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    best = tmp_path / "best_actor_step27_train3.pth"
    last = tmp_path / "last_actor.pth"
    best.write_bytes(b"best")
    last.write_bytes(b"last")

    args = SimpleNamespace(
        step27_select=True,
        degradation_window=5,
        degradation_speed_delta=2.0,
        degradation_eta_delta=0.05,
        accept_max_speed_error=30.0,
        accept_min_eta_energy=0.0,
        accept_max_eta_energy=1.2,
        accept_max_current_rms=10.0,
        accept_max_p_in_pos=None,
    )
    row = {
        "motor": "al31",
        "seed": 101,
        "stage": "fine_tune",
        "episodes_log": str(episodes_log),
        "run_dir": str(run_dir),
        "best_checkpoint": str(best),
        "last_checkpoint": str(last),
        "step27_acceptance_pass": True,
        "step27_envelope_all_rows_pass": True,
    }

    _, acceptance, _, _ = _summarize_run(row, args)

    assert acceptance["acceptance_source"] == "step27"
    assert acceptance["training_episode_acceptance_pass"] is False
    assert acceptance["step27_acceptance_pass"] is True
    assert acceptance["acceptance_pass"] is True


def test_step27_selection_spec_resolves_candidate_and_canonical_checkpoint(tmp_path: Path, monkeypatch) -> None:
    candidate_json = tmp_path / "candidate.json"
    candidate_json.write_text("[]", encoding="utf-8")
    checkpoint = tmp_path / "actor_ep_canonical.pth"
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setitem(
        train3.CANONICAL_STEP27_SELECTION,
        "air56",
        {
            "ai_control_mode": "ai_id_ref_hybrid",
            "candidate_json": str(candidate_json),
            "candidate_tag": "candidate",
            "checkpoint_path": str(checkpoint),
        },
    )

    spec = train3._resolve_step27_selection_spec("air56", SimpleNamespace(step27_profile="canonical"))

    assert spec["candidate_json"] == str(candidate_json.resolve())
    assert spec["checkpoint_path"] == str(checkpoint.resolve())
