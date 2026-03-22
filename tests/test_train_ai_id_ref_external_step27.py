from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mic_ai.ai.train_ai_id_ref import _promote_external_step27_checkpoint, _run_external_step27_selection


def test_run_external_step27_selection_promotes_selected_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    eval_dir = run_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    selected_ckpt = eval_dir / "actor_ep003.pth"
    selected_ckpt.write_bytes(b"selected")
    init_ckpt = tmp_path / "init_actor.pth"
    init_ckpt.write_bytes(b"init")

    candidate_json = tmp_path / "candidate.json"
    candidate_json.write_text(json.dumps([{"tag": "base_current"}], indent=2), encoding="utf-8")

    def _fake_scan_checkpoints(**kwargs):
        out_dir = Path(kwargs["out_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        assert (eval_dir / "actor_ep_init.pth").exists()
        assert (eval_dir / "actor_ep_init.pth").read_bytes() == b"init"
        assert kwargs["min_avg_power_saving_pct"] == pytest.approx(0.5)
        assert kwargs["min_avg_eta_gain_pct"] == pytest.approx(0.0)
        assert kwargs["max_avg_eta_gain_pct"] == pytest.approx(25.0)
        assert kwargs["max_err_failures"] == pytest.approx(2.0)
        assert kwargs["min_start_stop_saving_pct"] == pytest.approx(-0.5)
        assert kwargs["max_start_stop_saving_pct"] == pytest.approx(20.0)
        assert kwargs["max_worst_current_peak_ratio"] == pytest.approx(1.3)
        assert kwargs["max_worst_current_mean_ratio"] == pytest.approx(1.2)
        assert kwargs["use_envelope_acceptance"] is True
        assert str(kwargs["acceptance_envelopes"]).endswith("acceptance_envelopes_3motors.json")
        assert kwargs["min_avg_power_saving_pct_min_seed"] == pytest.approx(0.5)
        assert kwargs["min_avg_eta_gain_pct_min_seed"] == pytest.approx(0.0)
        assert kwargs["max_err_failures_max_seed"] == pytest.approx(2.0)
        assert kwargs["min_start_stop_saving_pct_min_seed"] == pytest.approx(-0.5)
        summary = {
            "best": {
                "checkpoint": str(selected_ckpt.resolve()),
                "checkpoint_name": selected_ckpt.name,
                "rank": 1,
                "score": 1.25,
                "acceptance_pass": False,
            }
        }
        (out_dir / "ao2_checkpoint_scan_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (out_dir / "ao2_checkpoint_scan.json").write_text(json.dumps([summary["best"]], indent=2), encoding="utf-8")
        return summary

    monkeypatch.setattr("tools.scan_step27_checkpoints.scan_checkpoints", _fake_scan_checkpoints)

    payload = _run_external_step27_selection(
        run_dir=run_dir,
        motor="ao2",
        candidate_json=str(candidate_json),
        candidate_index=0,
        candidate_tag="",
        seeds="101,202,303",
        scenarios="speed_step,ramp,load_step,start_stop",
        seed_perturbation=True,
        seed_perturb_level=0.2,
        min_avg_power_saving_pct=0.5,
        min_avg_eta_gain_pct=0.0,
        max_avg_eta_gain_pct=25.0,
        max_err_failures=2.0,
        min_start_stop_saving_pct=-0.5,
        max_start_stop_saving_pct=20.0,
        max_worst_current_peak_ratio=1.3,
        max_worst_current_mean_ratio=1.2,
        use_envelope_acceptance=True,
        acceptance_envelopes="config/acceptance_envelopes_3motors.json",
        min_avg_power_saving_pct_min_seed=0.5,
        min_avg_eta_gain_pct_min_seed=0.0,
        max_err_failures_max_seed=2.0,
        min_start_stop_saving_pct_min_seed=-0.5,
        top_k=5,
        init_checkpoint=str(init_ckpt),
        include_init_checkpoint=True,
    )

    promoted = run_dir / "best_actor_step27.pth"
    assert promoted.exists()
    assert promoted.read_bytes() == b"selected"
    assert payload["selected_checkpoint"] == str(selected_ckpt.resolve())
    assert payload["promoted_checkpoint"] == str(promoted.resolve())
    assert payload["seeds"] == [101, 202, 303]
    assert payload["scenarios"] == ["speed_step", "ramp", "load_step", "start_stop"]
    assert payload["min_avg_power_saving_pct"] == pytest.approx(0.5)
    assert payload["min_avg_eta_gain_pct"] == pytest.approx(0.0)
    assert payload["max_avg_eta_gain_pct"] == pytest.approx(25.0)
    assert payload["max_err_failures"] == pytest.approx(2.0)
    assert payload["min_start_stop_saving_pct"] == pytest.approx(-0.5)
    assert payload["max_start_stop_saving_pct"] == pytest.approx(20.0)
    assert payload["max_worst_current_peak_ratio"] == pytest.approx(1.3)
    assert payload["max_worst_current_mean_ratio"] == pytest.approx(1.2)
    assert payload["use_envelope_acceptance"] is True
    assert payload["acceptance_envelopes"].endswith("acceptance_envelopes_3motors.json")
    assert payload["min_avg_power_saving_pct_min_seed"] == pytest.approx(0.5)
    assert payload["min_avg_eta_gain_pct_min_seed"] == pytest.approx(0.0)
    assert payload["max_err_failures_max_seed"] == pytest.approx(2.0)
    assert payload["min_start_stop_saving_pct_min_seed"] == pytest.approx(-0.5)
    assert payload["include_init_checkpoint"] is True
    assert payload["included_init_checkpoint"] == str((eval_dir / "actor_ep_init.pth").resolve())


def test_promote_external_step27_checkpoint_updates_registry_best(tmp_path: Path) -> None:
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    registry_best = ckpt_dir / "best_actor.pth"
    registry_best.write_bytes(b"train-best")

    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    selected = run_dir / "best_actor_step27.pth"
    selected.write_bytes(b"step27-best")

    payload = _promote_external_step27_checkpoint(
        ckpt_dir=ckpt_dir,
        external_step27_selection={"promoted_checkpoint": str(selected.resolve())},
    )

    assert payload is not None
    assert registry_best.read_bytes() == b"step27-best"
    assert (ckpt_dir / "best_actor_train_internal.pth").read_bytes() == b"train-best"
    assert payload["registry_best_checkpoint"] == str(registry_best.resolve())
