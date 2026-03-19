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

    candidate_json = tmp_path / "candidate.json"
    candidate_json.write_text(json.dumps([{"tag": "base_current"}], indent=2), encoding="utf-8")

    def _fake_scan_checkpoints(**kwargs):
        out_dir = Path(kwargs["out_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        assert kwargs["use_envelope_acceptance"] is True
        assert str(kwargs["acceptance_envelopes"]).endswith("acceptance_envelopes_3motors.json")
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
        use_envelope_acceptance=True,
        acceptance_envelopes="config/acceptance_envelopes_3motors.json",
        top_k=5,
    )

    promoted = run_dir / "best_actor_step27.pth"
    assert promoted.exists()
    assert promoted.read_bytes() == b"selected"
    assert payload["selected_checkpoint"] == str(selected_ckpt.resolve())
    assert payload["promoted_checkpoint"] == str(promoted.resolve())
    assert payload["seeds"] == [101, 202, 303]
    assert payload["scenarios"] == ["speed_step", "ramp", "load_step", "start_stop"]
    assert payload["use_envelope_acceptance"] is True
    assert payload["acceptance_envelopes"].endswith("acceptance_envelopes_3motors.json")


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
