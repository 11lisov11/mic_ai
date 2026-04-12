from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.scan_step27_checkpoints import _scan_state_signature, _select_candidate, _select_candidates, scan_checkpoints
from tools.step27_pipeline import _load_agent
from tools.tune_motor_step27 import DEFAULT_ACCEPTANCE_ENVELOPES
from mic_ai.ai.agents.ppo_voltage import PPOVoltageAgent
from mic_ai.ai.train_ai_id_ref import build_feature_keys
from mic_ai.tools.drive_characteristics_ai import _load_ai_agent_from_checkpoint
from mic_ai.tools.scenario_compare import _resolve_feature_keys


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


def test_select_candidates_by_tags(tmp_path: Path) -> None:
    path = tmp_path / "candidates.json"
    path.write_text(
        json.dumps(
            [
                {"tag": "c0", "id_ref_alpha": 0.21},
                {"tag": "c1", "id_ref_alpha": 0.31},
                {"tag": "c2", "id_ref_alpha": 0.41},
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    rows, total = _select_candidates(
        path,
        base=_base_candidate(),
        candidate_index=0,
        candidate_tag="",
        candidate_tags=["c2", "c0"],
    )
    assert total == 3
    assert [(row["tag"], idx) for row, idx in rows] == [("c2", 2), ("c0", 0)]


def test_load_agent_falls_back_to_checkpoint_feature_keys_on_mismatch(tmp_path: Path) -> None:
    ckpt = tmp_path / "actor_ep000.pth"
    inferred_keys = build_feature_keys(True, False)
    requested_keys = build_feature_keys(False, False)
    agent = PPOVoltageAgent(feature_keys=inferred_keys, action_dim=1, device="cpu", hidden_sizes=(64, 64))
    torch_state = agent.net.state_dict()
    import torch

    torch.save(torch_state, ckpt)

    loaded = _load_agent(ckpt, feature_keys=requested_keys)
    assert loaded.feature_keys == inferred_keys


def test_load_agent_infers_two_action_actor_head_checkpoint(tmp_path: Path) -> None:
    ckpt = tmp_path / "actor_ep000.pth"
    inferred_keys = build_feature_keys(True, False)
    agent = PPOVoltageAgent(feature_keys=inferred_keys, action_dim=2, device="cpu", hidden_sizes=(64, 64))
    import torch

    torch.save(agent.net.state_dict(), ckpt)

    loaded = _load_agent(ckpt, feature_keys=inferred_keys)
    assert loaded.action_dim == 2


def test_load_agent_remaps_single_action_id_head_for_ai_speed(tmp_path: Path) -> None:
    ckpt = tmp_path / "actor_ep000.pth"
    inferred_keys = build_feature_keys(True, False)
    agent = PPOVoltageAgent(feature_keys=inferred_keys, action_dim=1, device="cpu", hidden_sizes=(64, 64))
    import torch

    state = agent.net.state_dict()
    weight_key = next(k for k in ("actor_mu.weight", "actor_head.weight") if k in state)
    bias_key = next(k for k in ("actor_mu.bias", "actor_head.bias") if k in state)
    state[weight_key].fill_(0.0)
    state[bias_key].fill_(0.0)
    state[weight_key][0, :3] = torch.tensor([1.0, 2.0, 3.0], dtype=state[weight_key].dtype)
    state[bias_key][0] = torch.tensor(4.0, dtype=state[bias_key].dtype)
    if "log_std" in state:
        state["log_std"].fill_(5.0)
    torch.save(state, ckpt)

    loaded = _load_agent(ckpt, feature_keys=inferred_keys, ai_control_mode="ai_speed")
    loaded_state = loaded.net.state_dict()
    assert loaded.action_dim == 2
    assert loaded_state[weight_key][0, :3].tolist() == [0.0, 0.0, 0.0]
    assert loaded_state[weight_key][1, :3].tolist() == [1.0, 2.0, 3.0]
    assert loaded_state[bias_key].tolist() == [0.0, 4.0]
    if "log_std" in loaded_state:
        assert loaded_state["log_std"].tolist() == [0.0, 5.0]
    assert tuple(loaded.net.log_std.shape) == (2,)
    assert tuple(loaded.net.actor_head.weight.shape) == (2, 64)


def test_load_agent_accepts_ai_id_ref_hybrid_alias(tmp_path: Path) -> None:
    ckpt = tmp_path / "actor_ep000.pth"
    inferred_keys = build_feature_keys(True, False)
    agent = PPOVoltageAgent(feature_keys=inferred_keys, action_dim=1, device="cpu", hidden_sizes=(64, 64))
    import torch

    torch.save(agent.net.state_dict(), ckpt)

    loaded = _load_agent(ckpt, feature_keys=inferred_keys, ai_control_mode="ai_id_ref_hybrid")
    assert loaded.action_dim == 1
    assert loaded.feature_keys == inferred_keys


def test_resolve_feature_keys_supports_episode_eta_checkpoint() -> None:
    import torch

    feature_keys = build_feature_keys(True, True)
    agent = PPOVoltageAgent(feature_keys=feature_keys, action_dim=1, device="cpu", hidden_sizes=(64, 64))
    state = agent.net.state_dict()

    resolved = _resolve_feature_keys(None, state)
    assert resolved == feature_keys


def test_drive_characteristics_loader_accepts_episode_eta_checkpoint(tmp_path: Path) -> None:
    ckpt = tmp_path / "actor_ep000.pth"
    feature_keys = build_feature_keys(True, True)
    agent = PPOVoltageAgent(feature_keys=feature_keys, action_dim=1, device="cpu", hidden_sizes=(64, 64))
    import torch

    torch.save(agent.net.state_dict(), ckpt)

    loaded = _load_ai_agent_from_checkpoint(ckpt, "ai_id_ref")
    assert loaded.feature_keys == feature_keys


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


def test_scan_checkpoints_uses_explicit_config_path_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ckpt = tmp_path / "actor_ep000.pth"
    ckpt.write_bytes(b"checkpoint")

    candidate_json = tmp_path / "candidate.json"
    candidate_json.write_text(
        json.dumps([{"tag": "base_current"}], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    seen: dict[str, object] = {}
    monkeypatch.setattr(
        "tools.scan_step27_checkpoints._collect_checkpoint_paths",
        lambda _pattern: [ckpt],
    )

    def _fake_load_env_and_agent(config_path, **kwargs):
        seen["config_path"] = config_path
        return object(), None, None

    monkeypatch.setattr(
        "tools.scan_step27_checkpoints._load_env_and_agent",
        _fake_load_env_and_agent,
    )
    monkeypatch.setattr(
        "tools.scan_step27_checkpoints._select_candidate",
        lambda *args, **kwargs: (_base_candidate(), 1),
    )
    monkeypatch.setattr(
        "tools.scan_step27_checkpoints._load_agent",
        lambda path: object(),
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
        config_path="config/env_backlog_ao2_nameplate_foc_tuned.py",
        checkpoint_glob=str(ckpt),
        candidate_json=str(candidate_json),
        seeds=[101],
        scenarios=["speed_step"],
        out_dir=tmp_path / "scan_out_cfg",
        top_k=3,
    )

    assert seen["config_path"] == "config/env_backlog_ao2_nameplate_foc_tuned.py"
    assert summary["best"]["checkpoint_name"] == "actor_ep000.pth"


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


def test_scan_checkpoints_ranks_shortlist_per_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ckpt_a = tmp_path / "actor_ep000.pth"
    ckpt_b = tmp_path / "actor_ep001.pth"
    ckpt_a.write_bytes(b"a")
    ckpt_b.write_bytes(b"b")

    candidate_json = tmp_path / "candidate.json"
    candidate_json.write_text(
        json.dumps([{"tag": "base_a"}, {"tag": "base_b"}], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "tools.scan_step27_checkpoints._collect_checkpoint_paths",
        lambda _pattern: [ckpt_a, ckpt_b],
    )
    monkeypatch.setattr(
        "tools.scan_step27_checkpoints._load_env_and_agent",
        lambda *args, **kwargs: (object(), None, None),
    )
    monkeypatch.setattr(
        "tools.scan_step27_checkpoints._load_agent",
        lambda path: Path(path),
    )

    def _fake_eval_candidate(**kwargs):
        agent_path = Path(kwargs["agent"])
        tag = str(kwargs["candidate"]["tag"])
        if agent_path.name == "actor_ep000.pth" and tag == "base_a":
            return {
                "avg_power_saving_pct": 0.8,
                "avg_eta_gain_pct": 0.1,
                "err_failures": 0.0,
                "start_stop_power_saving_pct": 0.0,
                "worst_current_peak_ratio": 1.0,
                "worst_current_mean_ratio": 1.0,
                "avg_power_saving_pct_min_seed": 0.8,
                "avg_eta_gain_pct_min_seed": 0.1,
                "err_failures_max_seed": 0.0,
                "start_stop_power_saving_pct_min_seed": 0.0,
                "envelope_all_rows_pass": True,
                "envelope_fail_count": 0,
                "envelope_gap_total": 0.0,
            }
        if agent_path.name == "actor_ep000.pth" and tag == "base_b":
            return {
                "avg_power_saving_pct": 1.2,
                "avg_eta_gain_pct": 0.3,
                "err_failures": 0.0,
                "start_stop_power_saving_pct": 0.0,
                "worst_current_peak_ratio": 1.0,
                "worst_current_mean_ratio": 1.0,
                "avg_power_saving_pct_min_seed": 1.2,
                "avg_eta_gain_pct_min_seed": 0.3,
                "err_failures_max_seed": 0.0,
                "start_stop_power_saving_pct_min_seed": 0.0,
                "envelope_all_rows_pass": True,
                "envelope_fail_count": 0,
                "envelope_gap_total": 0.0,
            }
        return {
            "avg_power_saving_pct": 0.6,
            "avg_eta_gain_pct": 0.2,
            "err_failures": 0.0,
            "start_stop_power_saving_pct": 0.0,
            "worst_current_peak_ratio": 1.0,
            "worst_current_mean_ratio": 1.0,
            "avg_power_saving_pct_min_seed": 0.6,
            "avg_eta_gain_pct_min_seed": 0.2,
            "err_failures_max_seed": 0.0,
            "start_stop_power_saving_pct_min_seed": 0.0,
            "envelope_all_rows_pass": True,
            "envelope_fail_count": 0,
            "envelope_gap_total": 0.0,
        }

    monkeypatch.setattr("tools.scan_step27_checkpoints._eval_candidate", _fake_eval_candidate)

    summary = scan_checkpoints(
        motor="ao2",
        checkpoint_glob="unused",
        candidate_json=str(candidate_json),
        candidate_tags=["base_a", "base_b"],
        seeds=[101],
        scenarios=["speed_step"],
        out_dir=tmp_path / "scan_out",
        use_envelope_acceptance=True,
        top_k=5,
    )

    assert summary["best"]["checkpoint_name"] == "actor_ep000.pth"
    assert summary["best"]["candidate_tag"] == "base_b"
    assert summary["best"]["candidate_variants_evaluated"] == 2
    assert summary["best"]["candidate_tags_evaluated"] == ["base_b", "base_a"]
    assert summary["candidate_tags"] == ["base_a", "base_b"]
    assert summary["top_rows"][0]["checkpoint_name"] == "actor_ep000.pth"


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


def test_scan_checkpoints_writes_incremental_progress_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ckpt0 = tmp_path / "actor_ep000.pth"
    ckpt1 = tmp_path / "actor_ep001.pth"
    ckpt0.write_bytes(b"c0")
    ckpt1.write_bytes(b"c1")

    candidate_json = tmp_path / "candidate.json"
    candidate_json.write_text(
        json.dumps([{"tag": "base_current"}], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    out_dir = tmp_path / "scan_out"
    call_count = {"value": 0}

    monkeypatch.setattr(
        "tools.scan_step27_checkpoints._collect_checkpoint_paths",
        lambda _pattern: [ckpt0, ckpt1],
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
        call_count["value"] += 1
        if call_count["value"] == 2:
            progress_path = out_dir / "ao2_checkpoint_scan_progress.json"
            state_path = out_dir / "ao2_checkpoint_scan_state.json"
            assert progress_path.exists()
            assert state_path.exists()
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            state = json.loads(state_path.read_text(encoding="utf-8"))
            assert progress["complete"] is False
            assert progress["processed_count"] == 1
            assert progress["evaluated_count"] == 1
            assert progress["last_checkpoint_name"] == "actor_ep000.pth"
            assert progress["best_so_far"]["checkpoint_name"] == "actor_ep000.pth"
            assert progress["top_rows"][0]["checkpoint_name"] == "actor_ep000.pth"
            assert state["complete"] is False
            assert state["last_checkpoint_name"] == "actor_ep000.pth"
            assert state["evaluated_rows"][0]["checkpoint_name"] == "actor_ep000.pth"
        agent_path = Path(kwargs["agent"])
        if agent_path.name == "actor_ep000.pth":
            return {
                "avg_power_saving_pct": 0.3,
                "avg_eta_gain_pct": 0.1,
                "err_failures": 0.0,
                "start_stop_power_saving_pct": 0.2,
                "worst_current_peak_ratio": 1.0,
                "worst_current_mean_ratio": 1.0,
                "avg_power_saving_pct_min_seed": 0.3,
                "avg_eta_gain_pct_min_seed": 0.1,
                "err_failures_max_seed": 0.0,
                "start_stop_power_saving_pct_min_seed": 0.2,
                "envelope_all_rows_pass": True,
                "envelope_fail_count": 0,
                "envelope_gap_total": 0.0,
            }
        return {
            "avg_power_saving_pct": 0.2,
            "avg_eta_gain_pct": 0.05,
            "err_failures": 0.0,
            "start_stop_power_saving_pct": 0.1,
            "worst_current_peak_ratio": 1.0,
            "worst_current_mean_ratio": 1.0,
            "avg_power_saving_pct_min_seed": 0.2,
            "avg_eta_gain_pct_min_seed": 0.05,
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

    summary = scan_checkpoints(
        motor="ao2",
        checkpoint_glob="unused",
        candidate_json=str(candidate_json),
        seeds=[101],
        scenarios=["speed_step"],
        out_dir=out_dir,
        top_k=5,
    )

    progress_path = out_dir / "ao2_checkpoint_scan_progress.json"
    state_path = out_dir / "ao2_checkpoint_scan_state.json"
    assert progress_path.exists()
    assert state_path.exists()
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert progress["complete"] is True
    assert progress["processed_count"] == 2
    assert progress["evaluated_count"] == 2
    assert progress["best_so_far"]["checkpoint_name"] == summary["best"]["checkpoint_name"]
    assert state["complete"] is True
    assert len(state["evaluated_rows"]) == 2
    assert state["signature"]["motor"] == "ao2"


def test_scan_checkpoints_resume_reuses_saved_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ckpt0 = tmp_path / "actor_ep000.pth"
    ckpt1 = tmp_path / "actor_ep001.pth"
    ckpt2 = tmp_path / "actor_ep002.pth"
    ckpt0.write_bytes(b"c0")
    ckpt1.write_bytes(b"c1")
    ckpt2.write_bytes(b"c2")

    candidate_json = tmp_path / "candidate.json"
    candidate_json.write_text(
        json.dumps([{"tag": "base_current"}], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    out_dir = tmp_path / "scan_out"
    out_dir.mkdir(parents=True, exist_ok=True)
    state_path = out_dir / "ao2_checkpoint_scan_state.json"
    signature = _scan_state_signature(
        motor="ao2",
        config_path="",
        ai_control_mode="ai_id_ref",
        checkpoint_glob="unused",
        candidate_json=str(candidate_json),
        candidate_index=0,
        candidate_tag="",
        candidate_tags=[],
        seeds=[101],
        scenarios=["speed_step"],
        use_envelope_acceptance=False,
    )
    state_path.write_text(
        json.dumps(
            {
                "signature": signature,
                "complete": False,
                "last_checkpoint_name": "actor_ep000.pth",
                "evaluated_rows": [
                    {
                        "rank_input": 1,
                        "checkpoint": str(ckpt0.resolve()),
                        "checkpoint_name": "actor_ep000.pth",
                        "status": "evaluated",
                        "skip_reason": None,
                        "avg_power_saving_pct": 0.3,
                        "avg_eta_gain_pct": 0.1,
                        "err_failures": 0.0,
                        "start_stop_power_saving_pct": 0.2,
                        "worst_current_peak_ratio": 1.0,
                        "worst_current_mean_ratio": 1.0,
                        "avg_power_saving_pct_min_seed": 0.3,
                        "avg_eta_gain_pct_min_seed": 0.1,
                        "err_failures_max_seed": 0.0,
                        "start_stop_power_saving_pct_min_seed": 0.2,
                        "score": 0.0,
                        "aggregate_score": 0.0,
                        "selector_score": 0.0,
                        "acceptance_pass_aggregate": True,
                        "acceptance_pass": True,
                    }
                ],
                "skipped_rows": [],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    calls: list[Path] = []

    monkeypatch.setattr(
        "tools.scan_step27_checkpoints._collect_checkpoint_paths",
        lambda _pattern: [ckpt0, ckpt1, ckpt2],
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
        lambda path: calls.append(Path(path)) or Path(path),
    )

    def _fake_eval_candidate(**kwargs):
        agent_path = Path(kwargs["agent"])
        if agent_path.name == "actor_ep001.pth":
            return {
                "avg_power_saving_pct": 0.2,
                "avg_eta_gain_pct": 0.05,
                "err_failures": 0.0,
                "start_stop_power_saving_pct": 0.1,
                "worst_current_peak_ratio": 1.0,
                "worst_current_mean_ratio": 1.0,
                "avg_power_saving_pct_min_seed": 0.2,
                "avg_eta_gain_pct_min_seed": 0.05,
                "err_failures_max_seed": 0.0,
                "start_stop_power_saving_pct_min_seed": 0.1,
            }
        return {
            "avg_power_saving_pct": 0.4,
            "avg_eta_gain_pct": 0.2,
            "err_failures": 0.0,
            "start_stop_power_saving_pct": 0.3,
            "worst_current_peak_ratio": 1.0,
            "worst_current_mean_ratio": 1.0,
            "avg_power_saving_pct_min_seed": 0.4,
            "avg_eta_gain_pct_min_seed": 0.2,
            "err_failures_max_seed": 0.0,
            "start_stop_power_saving_pct_min_seed": 0.3,
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
        out_dir=out_dir,
        top_k=5,
        resume=True,
    )

    assert calls == [ckpt1, ckpt2]
    assert summary["scan_rows"] == 3
    assert summary["best"]["checkpoint_name"] == "actor_ep002.pth"

    progress = json.loads((out_dir / "ao2_checkpoint_scan_progress.json").read_text(encoding="utf-8"))
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert progress["complete"] is True
    assert progress["processed_count"] == 3
    assert state["complete"] is True
    assert len(state["evaluated_rows"]) == 3


def test_scan_checkpoints_allows_ai_voltage_without_candidate_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ckpt = tmp_path / "actor_ep000.pth"
    ckpt.write_bytes(b"ok")

    monkeypatch.setattr(
        "tools.scan_step27_checkpoints._collect_checkpoint_paths",
        lambda _pattern: [ckpt],
    )
    monkeypatch.setattr(
        "tools.scan_step27_checkpoints._load_env_and_agent",
        lambda *args, **kwargs: (object(), None, None),
    )

    def _forbidden_select_candidate(*args, **kwargs):
        raise AssertionError("_select_candidate should not be called for ai_voltage")

    monkeypatch.setattr(
        "tools.scan_step27_checkpoints._select_candidate",
        _forbidden_select_candidate,
    )
    monkeypatch.setattr(
        "tools.scan_step27_checkpoints._load_agent",
        lambda path: Path(path),
    )

    seen: dict[str, object] = {}

    def _fake_eval_candidate(**kwargs):
        seen["ai_control_mode"] = kwargs["ai_control_mode"]
        seen["candidate_tag"] = kwargs["candidate"]["tag"]
        return {
            "avg_power_saving_pct": 0.2,
            "avg_eta_gain_pct": 0.1,
            "err_failures": 0.0,
            "start_stop_power_saving_pct": 0.1,
            "worst_current_peak_ratio": 1.0,
            "worst_current_mean_ratio": 1.0,
            "avg_power_saving_pct_min_seed": 0.2,
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

    summary = scan_checkpoints(
        motor="ao2",
        checkpoint_glob="unused",
        ai_control_mode="ai_voltage",
        candidate_json="",
        seeds=[101],
        scenarios=["speed_step"],
        out_dir=tmp_path / "scan_out",
        top_k=1,
    )

    assert seen["ai_control_mode"] == "ai_voltage"
    assert seen["candidate_tag"] == "baseline"
    assert summary["ai_control_mode"] == "ai_voltage"
    assert summary["candidate_count"] == 1
    assert summary["best"]["checkpoint_name"] == "actor_ep000.pth"
