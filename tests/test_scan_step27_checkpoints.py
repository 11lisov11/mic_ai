from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.scan_step27_checkpoints import _select_candidate, scan_checkpoints
from tools.tune_motor_step27 import DEFAULT_ACCEPTANCE_ENVELOPES


def _base_candidate() -> dict[str, object]:
    return {
        "tag": "baseline",
        "source": "config",
        "objective": "specific_power",
        "speed_tol_rel": 0.1,
        "speed_tol_abs": 0.0,
        "omega_min_pu": 0.1,
        "update_steps": 12,
        "dither_amp": 0.01,
        "bias_step": 0.005,
        "bias_max": 0.12,
        "shaft_eps": 10.0,
        "reset_decay": 0.98,
        "objective_clip": 10.0,
        "idle_enable": False,
        "idle_omega_pu": 0.05,
        "idle_action": -0.8,
        "idle_exit_boost_steps": 0,
        "idle_exit_action": 0.95,
        "idle_bias_decay": 0.96,
        "id_ref_alpha": 0.2,
        "delta_id_max": 0.08,
        "id_ref_gate_speed_tol_rel": 0.1,
        "id_ref_gate_min_scale": 0.1,
        "id_ref_gate_exponent": 1.0,
    }


def test_select_candidate_by_index(tmp_path: Path) -> None:
    path = tmp_path / "candidates.json"
    path.write_text(
        json.dumps(
            [
                {"tag": "c0", "id_ref_alpha": 0.21},
                {"tag": "c1", "id_ref_alpha": 0.31, "objective": "p_in"},
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    row, total = _select_candidate(path, base=_base_candidate(), candidate_index=1, candidate_tag="")
    assert total == 2
    assert row["tag"] == "c1"
    assert row["objective"] == "p_in"
    assert row["id_ref_alpha"] == 0.31
    assert row["delta_id_max"] == 0.08


def test_select_candidate_by_tag(tmp_path: Path) -> None:
    path = tmp_path / "candidates.json"
    path.write_text(
        json.dumps(
            {"candidates": [{"tag": "low"}, {"tag": "keep_me", "bias_max": 0.22}]},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    row, total = _select_candidate(path, base=_base_candidate(), candidate_index=0, candidate_tag="keep_me")
    assert total == 2
    assert row["tag"] == "keep_me"
    assert row["bias_max"] == 0.22
    assert row["objective"] == "specific_power"


def test_select_candidate_rejects_missing_tag(tmp_path: Path) -> None:
    path = tmp_path / "candidates.json"
    path.write_text(json.dumps([{"tag": "c0"}], ensure_ascii=False, indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match="was not found"):
        _select_candidate(path, base=_base_candidate(), candidate_index=0, candidate_tag="missing")


def test_scan_checkpoints_skips_missing_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    existing = tmp_path / "actor_ep000.pth"
    missing = tmp_path / "actor_ep001.pth"
    existing.write_bytes(b"checkpoint")

    candidate_json = tmp_path / "candidate.json"
    candidate_json.write_text(
        json.dumps([{"tag": "base_current"}], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    out_dir = tmp_path / "scan_out"
    calls: list[Path] = []

    monkeypatch.setattr(
        "tools.scan_step27_checkpoints._collect_checkpoint_paths",
        lambda _pattern: [existing, missing],
    )
    monkeypatch.setattr(
        "tools.scan_step27_checkpoints._load_env_and_agent",
        lambda *args, **kwargs: (object(), None, None),
    )
    monkeypatch.setattr(
        "tools.scan_step27_checkpoints._select_candidate",
        lambda *args, **kwargs: (_base_candidate(), 1),
    )
    monkeypatch.setattr(
        "tools.scan_step27_checkpoints._load_agent",
        lambda path: calls.append(Path(path)) or object(),
    )
    monkeypatch.setattr(
        "tools.scan_step27_checkpoints._eval_candidate",
        lambda **kwargs: {
            "avg_power_saving_pct": 1.0,
            "avg_eta_gain_pct": 2.0,
            "err_failures": 0.0,
            "start_stop_power_saving_pct": 3.0,
            "worst_current_peak_ratio": 1.0,
            "worst_current_mean_ratio": 1.0,
            "avg_power_saving_pct_min_seed": 1.0,
            "avg_eta_gain_pct_min_seed": 2.0,
            "err_failures_max_seed": 0.0,
            "start_stop_power_saving_pct_min_seed": 3.0,
        },
    )

    summary = scan_checkpoints(
        motor="ao2",
        checkpoint_glob="unused",
        candidate_json=str(candidate_json),
        seeds=[101],
        scenarios=["speed_step"],
        out_dir=out_dir,
        top_k=5,
    )

    assert calls == [existing]
    assert summary["scan_rows"] == 2
    assert summary["skipped_count"] == 1
    assert summary["best"]["checkpoint_name"] == "actor_ep000.pth"
    assert summary["best"]["status"] == "evaluated"
    assert summary["top_rows"][0]["checkpoint_name"] == "actor_ep000.pth"
    assert summary["skipped_rows"][0]["checkpoint_name"] == "actor_ep001.pth"
    assert summary["skipped_rows"][0]["skip_reason"] == "missing_file"


def test_scan_checkpoints_prefers_envelope_passing_row(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ckpt_bad = tmp_path / "actor_ep000.pth"
    ckpt_good = tmp_path / "actor_ep001.pth"
    ckpt_bad.write_bytes(b"bad")
    ckpt_good.write_bytes(b"good")

    candidate_json = tmp_path / "candidate.json"
    candidate_json.write_text(
        json.dumps([{"tag": "base_current"}], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "tools.scan_step27_checkpoints._collect_checkpoint_paths",
        lambda _pattern: [ckpt_bad, ckpt_good],
    )
    monkeypatch.setattr(
        "tools.scan_step27_checkpoints._load_env_and_agent",
        lambda *args, **kwargs: (object(), None, None),
    )
    monkeypatch.setattr(
        "tools.scan_step27_checkpoints._select_candidate",
        lambda *args, **kwargs: (_base_candidate(), 1),
    )
    monkeypatch.setattr(
        "tools.scan_step27_checkpoints._load_agent",
        lambda path: Path(path),
    )

    def _fake_eval_candidate(**kwargs):
        agent_path = Path(kwargs["agent"])
        if agent_path.name == "actor_ep000.pth":
            return {
                "avg_power_saving_pct": 5.0,
                "avg_eta_gain_pct": 5.0,
                "err_failures": 2.0,
                "start_stop_power_saving_pct": 5.0,
                "worst_current_peak_ratio": 1.0,
                "worst_current_mean_ratio": 1.0,
                "avg_power_saving_pct_min_seed": -1.0,
                "avg_eta_gain_pct_min_seed": -1.0,
                "err_failures_max_seed": 2.0,
                "start_stop_power_saving_pct_min_seed": -1.0,
                "envelope_all_rows_pass": False,
                "envelope_fail_count": 3,
                "envelope_gap_total": 2.5,
            }
        return {
            "avg_power_saving_pct": 0.1,
            "avg_eta_gain_pct": 0.1,
            "err_failures": 1.0,
            "start_stop_power_saving_pct": 0.2,
            "worst_current_peak_ratio": 1.0,
            "worst_current_mean_ratio": 1.0,
            "avg_power_saving_pct_min_seed": -0.1,
            "avg_eta_gain_pct_min_seed": -0.1,
            "err_failures_max_seed": 1.0,
            "start_stop_power_saving_pct_min_seed": 0.0,
            "envelope_all_rows_pass": True,
            "envelope_fail_count": 0,
            "envelope_gap_total": 0.0,
        }

    monkeypatch.setattr(
        "tools.scan_step27_checkpoints._eval_candidate",
        _fake_eval_candidate,
    )

    summary = scan_checkpoints(
        motor="ao2",
        checkpoint_glob="unused",
        candidate_json=str(candidate_json),
        seeds=[101],
        scenarios=["speed_step"],
        out_dir=tmp_path / "scan_out",
        use_envelope_acceptance=True,
        top_k=5,
    )

    assert summary["selector_mode"] == "canonical_envelope_then_aggregate"
    assert summary["use_envelope_acceptance"] is True
    assert summary["best"]["checkpoint_name"] == "actor_ep001.pth"
    assert summary["best"]["acceptance_pass"] is True
    assert summary["top_rows"][0]["checkpoint_name"] == "actor_ep001.pth"


def test_scan_checkpoints_defaults_acceptance_envelope_when_none(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ckpt = tmp_path / "actor_ep000.pth"
    ckpt.write_bytes(b"ok")

    candidate_json = tmp_path / "candidate.json"
    candidate_json.write_text(
        json.dumps([{"tag": "base_current"}], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    seen: dict[str, Path] = {}

    monkeypatch.setattr(
        "tools.scan_step27_checkpoints._collect_checkpoint_paths",
        lambda _pattern: [ckpt],
    )
    monkeypatch.setattr(
        "tools.scan_step27_checkpoints._load_env_and_agent",
        lambda *args, **kwargs: (object(), None, None),
    )
    monkeypatch.setattr(
        "tools.scan_step27_checkpoints._select_candidate",
        lambda *args, **kwargs: (_base_candidate(), 1),
    )
    monkeypatch.setattr(
        "tools.scan_step27_checkpoints._load_agent",
        lambda path: Path(path),
    )

    def _fake_eval_candidate(**kwargs):
        seen["acceptance_envelopes_path"] = Path(kwargs["acceptance_envelopes_path"]).resolve()
        return {
            "avg_power_saving_pct": 0.1,
            "avg_eta_gain_pct": 0.1,
            "err_failures": 0.0,
            "start_stop_power_saving_pct": 0.1,
            "worst_current_peak_ratio": 1.0,
            "worst_current_mean_ratio": 1.0,
            "avg_power_saving_pct_min_seed": 0.1,
            "avg_eta_gain_pct_min_seed": 0.1,
            "err_failures_max_seed": 0.0,
            "start_stop_power_saving_pct_min_seed": 0.1,
            "envelope_all_rows_pass": True,
            "envelope_fail_count": 0,
            "envelope_gap_total": 0.0,
        }

    monkeypatch.setattr(
        "tools.scan_step27_checkpoints._eval_candidate",
        _fake_eval_candidate,
    )

    scan_checkpoints(
        motor="ao2",
        checkpoint_glob="unused",
        candidate_json=str(candidate_json),
        seeds=[101],
        scenarios=["speed_step"],
        out_dir=tmp_path / "scan_out",
        use_envelope_acceptance=True,
        acceptance_envelopes=None,
        top_k=1,
    )

    assert seen["acceptance_envelopes_path"] == Path(DEFAULT_ACCEPTANCE_ENVELOPES).resolve()


def test_scan_checkpoints_keeps_aggregate_mode_by_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ckpt_bad = tmp_path / "actor_ep000.pth"
    ckpt_good = tmp_path / "actor_ep001.pth"
    ckpt_bad.write_bytes(b"bad")
    ckpt_good.write_bytes(b"good")

    candidate_json = tmp_path / "candidate.json"
    candidate_json.write_text(
        json.dumps([{"tag": "base_current"}], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "tools.scan_step27_checkpoints._collect_checkpoint_paths",
        lambda _pattern: [ckpt_bad, ckpt_good],
    )
    monkeypatch.setattr(
        "tools.scan_step27_checkpoints._load_env_and_agent",
        lambda *args, **kwargs: (object(), None, None),
    )
    monkeypatch.setattr(
        "tools.scan_step27_checkpoints._select_candidate",
        lambda *args, **kwargs: (_base_candidate(), 1),
    )
    monkeypatch.setattr(
        "tools.scan_step27_checkpoints._load_agent",
        lambda path: Path(path),
    )

    def _fake_eval_candidate(**kwargs):
        agent_path = Path(kwargs["agent"])
        if agent_path.name == "actor_ep000.pth":
            return {
                "avg_power_saving_pct": 5.0,
                "avg_eta_gain_pct": 5.0,
                "err_failures": 0.0,
                "start_stop_power_saving_pct": 5.0,
                "worst_current_peak_ratio": 1.0,
                "worst_current_mean_ratio": 1.0,
                "avg_power_saving_pct_min_seed": 5.0,
                "avg_eta_gain_pct_min_seed": 5.0,
                "err_failures_max_seed": 0.0,
                "start_stop_power_saving_pct_min_seed": 5.0,
                "envelope_all_rows_pass": False,
                "envelope_fail_count": 3,
                "envelope_gap_total": 2.5,
            }
        return {
            "avg_power_saving_pct": 0.1,
            "avg_eta_gain_pct": 0.1,
            "err_failures": 0.0,
            "start_stop_power_saving_pct": 0.2,
            "worst_current_peak_ratio": 1.0,
            "worst_current_mean_ratio": 1.0,
            "avg_power_saving_pct_min_seed": 0.1,
            "avg_eta_gain_pct_min_seed": 0.1,
            "err_failures_max_seed": 0.0,
            "start_stop_power_saving_pct_min_seed": 0.2,
                "envelope_all_rows_pass": True,
                "envelope_fail_count": 0,
                "envelope_gap_total": 0.0,
        }

    monkeypatch.setattr(
        "tools.scan_step27_checkpoints._eval_candidate",
        _fake_eval_candidate,
    )

    summary = scan_checkpoints(
        motor="ao2",
        checkpoint_glob="unused",
        candidate_json=str(candidate_json),
        seeds=[101],
        scenarios=["speed_step"],
        out_dir=tmp_path / "scan_out",
        top_k=5,
    )

    assert summary["selector_mode"] == "aggregate_only"
    assert summary["use_envelope_acceptance"] is False
    assert summary["best"]["checkpoint_name"] == "actor_ep000.pth"
    assert summary["best"]["acceptance_pass"] is True
