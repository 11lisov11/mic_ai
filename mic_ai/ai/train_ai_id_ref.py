from __future__ import annotations

import sys
import argparse
import json
import os
import random
import shutil
import time
import subprocess
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mic_ai.ai.agents.ppo_voltage import PPOVoltageAgent
from mic_ai.ai.ai_env import AiEnvConfig, MicAiAIEnv
from mic_ai.ai.ai_voltage_config import get_curriculum_config, load_ai_voltage_config
from mic_ai.ai.id_ref_supervisor import AiIdRefSupervisor, AiIdRefSupervisorConfig
from mic_ai.ai.scenario_randomization import wrap_scenario_with_ranges
from mic_ai.core.env import make_env_from_config
from mic_ai.tools.checkpoint_adaptation import adapt_checkpoint_state_dict_for_model
from simulation.gym_env import InductionMotorEnv


BASE_FEATURE_KEYS = [
    "omega_norm",
    "omega_ref_norm",
    "err_norm",
    "id_norm",
    "iq_norm",
    "slip_norm",
    "load_torque_norm",
]

TWO_ACTION_CONTROL_MODES = {"ai_current", "foc_assist", "ai_speed"}


def build_feature_keys(include_energy_obs: bool, include_episode_eta_obs: bool = False) -> List[str]:
    keys = list(BASE_FEATURE_KEYS)
    if include_energy_obs:
        keys += ["p_in_norm", "p_el_filt", "p_shaft_norm", "eta_norm"]
    if include_episode_eta_obs:
        keys += ["eta_episode_norm"]
    # de-dup preserving order
    seen = set()
    out: List[str] = []
    for key in keys:
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


# Default feature set for id_ref policies.
# NOTE: Paper checkpoints were trained with energy-related observations enabled.
# Keep this in sync so evaluation tools (e.g. scenario_compare) load those checkpoints by default.
FEATURE_KEYS = build_feature_keys(include_energy_obs=True, include_episode_eta_obs=False)

OUTPUT_DIR = Path(os.environ.get("MIC_AI_ID_REF_OUTPUT_DIR", "outputs/ai_id_ref"))
EPISODE_LOG_DIR = OUTPUT_DIR / "episode_logs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
RESULTS_ROOT = Path(os.environ.get("MIC_AI_RESULTS_ROOT", "results_run"))


def _parse_scenarios(text: str) -> List[str]:
    names = [item.strip() for item in str(text).split(",") if item.strip()]
    return names


def _parse_int_csv(text: str) -> List[int]:
    return [int(item.strip()) for item in str(text).split(",") if item.strip()]


def _select_episode_seed(episode_index: int, episode_seed_cycle: List[int] | None) -> int | None:
    if not episode_seed_cycle:
        return None
    idx = int(episode_index) % len(episode_seed_cycle)
    return int(episode_seed_cycle[idx])


def _parse_range(text: str | None) -> tuple[float, float] | None:
    if not text:
        return None
    raw = str(text).strip().replace(":", ",")
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if len(parts) != 2:
        return None
    try:
        lo = float(parts[0])
        hi = float(parts[1])
    except ValueError:
        return None
    if hi < lo:
        lo, hi = hi, lo
    return lo, hi


def _parse_hidden_sizes(text: str | None) -> tuple[int, ...] | None:
    if not text:
        return None
    parts = [item.strip() for item in str(text).replace(";", ",").split(",") if item.strip()]
    sizes: List[int] = []
    for part in parts:
        value = int(part)
        if value <= 0:
            raise ValueError(f"hidden size must be positive, got {value}")
        sizes.append(value)
    return tuple(sizes) if sizes else None


_SCENARIO_REWARD_OVERRIDE_KEYS = {
    "w_speed",
    "w_power",
    "w_shaft",
    "w_eta",
    "w_eta_episode",
    "reward_start_frac",
    "terminal_energy_bonus",
    "ai_id_speed_tol",
    "ai_id_speed_tol_rel",
    "id_ref_gate_speed_tol",
    "id_ref_gate_speed_tol_rel",
    "id_ref_gate_min_scale",
    "id_ref_gate_exponent",
}


def _parse_scenario_reward_overrides(text: str | None) -> Dict[str, Dict[str, float]] | None:
    if not text:
        return None
    raw = str(text).strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    else:
        payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("scenario reward overrides must be a JSON object")
    normalized: Dict[str, Dict[str, float]] = {}
    for scenario_name, scenario_payload in payload.items():
        scenario = str(scenario_name).strip()
        if not scenario:
            continue
        if not isinstance(scenario_payload, dict):
            raise ValueError(f"scenario override for {scenario!r} must be an object")
        row: Dict[str, float] = {}
        for key, value in scenario_payload.items():
            key_norm = str(key).strip()
            if key_norm not in _SCENARIO_REWARD_OVERRIDE_KEYS:
                raise ValueError(f"unsupported scenario reward override key: {key_norm}")
            row[key_norm] = float(value)
        if row:
            normalized[scenario] = row
    return normalized or None


def _parse_seed_scenario_reward_overrides(text: str | None) -> Dict[int, Dict[str, Dict[str, float]]] | None:
    if not text:
        return None
    raw = str(text).strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    else:
        payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("seed-scenario reward overrides must be a JSON object")

    normalized: Dict[int, Dict[str, Dict[str, float]]] = {}
    for seed_key, seed_payload in payload.items():
        seed_int = int(seed_key)
        if not isinstance(seed_payload, dict):
            raise ValueError(f"seed override for {seed_key!r} must be an object")
        scenario_map: Dict[str, Dict[str, float]] = {}
        for scenario_name, scenario_payload in seed_payload.items():
            scenario = str(scenario_name).strip()
            if not scenario:
                continue
            if not isinstance(scenario_payload, dict):
                raise ValueError(f"seed scenario override for {seed_key!r}/{scenario!r} must be an object")
            row: Dict[str, float] = {}
            for key, value in scenario_payload.items():
                key_norm = str(key).strip()
                if key_norm not in _SCENARIO_REWARD_OVERRIDE_KEYS:
                    raise ValueError(f"unsupported seed scenario reward override key: {key_norm}")
                row[key_norm] = float(value)
            if row:
                scenario_map[scenario] = row
        if scenario_map:
            normalized[int(seed_int)] = scenario_map
    return normalized or None


def _estimate_scenario_activation_steps(scenario_name: str, *, t_end: float, dt: float) -> int | None:
    name = str(scenario_name or "").strip().split(":", 1)[0]
    t_end = float(max(t_end, 0.0))
    dt = float(max(dt, 1e-9))
    if not name:
        return None
    event_time: float | None = None
    if name == "speed_step":
        event_time = 0.1 * t_end
    elif name == "ramp":
        event_time = 0.6 * t_end
    elif name == "load_step":
        event_time = 0.3 * t_end
    elif name == "load_profile":
        event_time = 0.2 * t_end
    elif name == "start_stop":
        event_time = 0.2 * t_end
    elif name == "hold":
        event_time = 0.0
    if event_time is None:
        return None
    return int(math.ceil(max(event_time, 0.0) / dt))


def _collect_underhorizon_scenarios(
    scenarios: List[str],
    *,
    episode_steps: int,
    t_end: float,
    dt: float,
) -> List[Dict[str, float | int | str]]:
    warnings: List[Dict[str, float | int | str]] = []
    seen: set[str] = set()
    for scenario_name in scenarios:
        scenario_key = str(scenario_name or "").strip()
        if not scenario_key or scenario_key in seen:
            continue
        seen.add(scenario_key)
        required_steps = _estimate_scenario_activation_steps(scenario_key, t_end=t_end, dt=dt)
        if required_steps is None or int(episode_steps) >= int(required_steps):
            continue
        warnings.append(
            {
                "scenario": scenario_key,
                "episode_steps": int(episode_steps),
                "required_steps": int(required_steps),
                "episode_horizon_s": float(int(episode_steps) * float(dt)),
                "required_horizon_s": float(int(required_steps) * float(dt)),
            }
        )
    return warnings


def _normalize_range(value: object | None) -> tuple[float, float] | None:
    if value is None:
        return None
    if isinstance(value, (tuple, list)) and len(value) == 2:
        try:
            lo = float(value[0])
            hi = float(value[1])
        except Exception:
            return None
        if hi < lo:
            lo, hi = hi, lo
        return lo, hi
    if isinstance(value, str):
        return _parse_range(value)
    return None


def _curriculum_scale(episode: int, warmup_episodes: int, ramp_episodes: int) -> float:
    warmup = max(int(warmup_episodes), 0)
    ramp = max(int(ramp_episodes), 0)
    if warmup <= 0 and ramp <= 0:
        return 1.0
    if int(episode) < warmup:
        return 0.0
    if ramp <= 0:
        return 1.0
    if int(episode) < warmup + ramp:
        return float(int(episode) - warmup) / float(ramp)
    return 1.0


def _infer_hidden_sizes_from_state_dict(state: Dict[str, torch.Tensor]) -> tuple[int, ...] | None:
    layers: List[tuple[int, int]] = []
    for key, value in state.items():
        if not key.startswith("actor_body.") or not key.endswith(".weight"):
            continue
        parts = key.split(".")
        if len(parts) < 3 or not parts[1].isdigit():
            continue
        if not torch.is_tensor(value) or value.ndim != 2:
            continue
        layers.append((int(parts[1]), int(value.shape[0])))
    if not layers:
        return None
    layers.sort(key=lambda item: item[0])
    return tuple(out_dim for _, out_dim in layers)


def _prepare_output_file(path: Path) -> Path:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = path.with_name(f"{path.stem}_backup_{ts}{path.suffix}")
        path.rename(backup)
        print(f"[log] existing {path.name} -> backup {backup.name}")
    return path


def _run_eval(
    env_config: str,
    checkpoint_path: Path,
    out_dir: Path,
    scenarios: str,
    dt: float | None,
    t_end: float | None,
    window_frac: float,
    error_tol_rel: float,
    error_tol_abs: float,
    use_total_power: bool,
    ai_id_relative: bool,
    delta_id_max: float,
    id_ref_alpha: float,
    id_ref_rate_limit: float | None,
    id_ref_gate_speed_tol: float | None,
    id_ref_gate_speed_tol_rel: float | None,
    id_ref_gate_min_scale: float,
    id_ref_gate_exponent: float,
    feature_keys: List[str],
) -> None:
    cmd = [
        sys.executable,
        "-m",
        "mic_ai.tools.scenario_compare",
        "--env-config",
        str(env_config),
        "--ai-checkpoint",
        str(checkpoint_path),
        "--out-dir",
        str(out_dir),
        "--scenarios",
        str(scenarios),
        "--window-frac",
        str(window_frac),
        "--error-tol-rel",
        str(error_tol_rel),
        "--error-tol-abs",
        str(error_tol_abs),
        "--id-ref-alpha",
        str(id_ref_alpha),
        "--id-ref-gate-min-scale",
        str(id_ref_gate_min_scale),
        "--id-ref-gate-exponent",
        str(id_ref_gate_exponent),
    ]
    if feature_keys:
        cmd += ["--ai-feature-keys", ",".join(feature_keys)]
    if dt is not None:
        cmd += ["--dt", str(dt)]
    if t_end is not None:
        cmd += ["--t-end", str(t_end)]
    if use_total_power:
        cmd += ["--use-total-power"]
    if ai_id_relative:
        cmd += ["--ai-id-relative", "--delta-id-max", str(delta_id_max)]
    if id_ref_rate_limit is not None:
        cmd += ["--id-ref-rate-limit", str(id_ref_rate_limit)]
    if id_ref_gate_speed_tol is not None:
        cmd += ["--id-ref-gate-speed-tol", str(id_ref_gate_speed_tol)]
    if id_ref_gate_speed_tol_rel is not None:
        cmd += ["--id-ref-gate-speed-tol-rel", str(id_ref_gate_speed_tol_rel)]
    subprocess.run(cmd, check=False)


def _train_supervisor_from_env(env_cfg: object) -> AiIdRefSupervisorConfig | None:
    if not bool(getattr(env_cfg, "ai_eval_supervisor_enabled", False)):
        return None
    cfg = AiIdRefSupervisorConfig(
        enabled=True,
        speed_tol_rel=float(getattr(env_cfg, "ai_eval_sup_speed_tol_rel", 0.05)),
        speed_tol_abs=float(getattr(env_cfg, "ai_eval_sup_speed_tol_abs", 0.0)),
        omega_min_pu=float(getattr(env_cfg, "ai_eval_sup_omega_min", 0.1)),
        update_steps=int(getattr(env_cfg, "ai_eval_sup_update", 20)),
        dither_amp=float(getattr(env_cfg, "ai_eval_sup_dither", 0.04)),
        bias_step=float(getattr(env_cfg, "ai_eval_sup_step", 0.01)),
        bias_max=float(getattr(env_cfg, "ai_eval_sup_bias_max", 0.25)),
        objective=str(getattr(env_cfg, "ai_eval_sup_objective", "specific_power")),
        shaft_eps=float(getattr(env_cfg, "ai_eval_sup_shaft_eps", 10.0)),
        reset_decay=float(getattr(env_cfg, "ai_eval_sup_reset_decay", 0.98)),
        objective_clip=getattr(env_cfg, "ai_eval_sup_objective_clip", 10.0),
        idle_enable=bool(getattr(env_cfg, "ai_eval_sup_idle_enable", False)),
        idle_omega_pu=float(getattr(env_cfg, "ai_eval_sup_idle_omega_min", 0.05)),
        idle_action=float(getattr(env_cfg, "ai_eval_sup_idle_action", -1.0)),
        idle_blend=float(getattr(env_cfg, "ai_eval_sup_idle_blend", 1.0)),
        idle_exit_boost_steps=int(getattr(env_cfg, "ai_eval_sup_idle_exit_boost", 0)),
        idle_exit_action=float(getattr(env_cfg, "ai_eval_sup_idle_exit_action", 1.0)),
        idle_bias_decay=float(getattr(env_cfg, "ai_eval_sup_idle_bias_decay", 0.95)),
    )
    if cfg.objective_clip is not None:
        cfg.objective_clip = float(cfg.objective_clip)
    return cfg


def _apply_train_supervisor_action(
    action: object,
    *,
    obs: Dict[str, float],
    supervisor: AiIdRefSupervisor | None,
) -> tuple[object, bool]:
    if supervisor is None:
        return action, False
    action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
    if action_arr.size == 0:
        return action, False
    action_adj = action_arr.copy()
    action0, gate_open = supervisor.adjust_action(
        float(action_adj[0]),
        omega_ref=float(obs.get("omega_ref", 0.0)),
        omega=float(obs.get("omega", 0.0)),
    )
    action_adj[0] = np.float32(action0)
    return action_adj, bool(gate_open)


def _run_external_step27_selection(
    *,
    run_dir: Path,
    motor: str,
    ai_control_mode: str = "ai_id_ref",
    candidate_json: str | None,
    candidate_index: int,
    candidate_tag: str,
    candidate_tags: str = "",
    seeds: str,
    scenarios: str,
    seed_perturbation: bool,
    seed_perturb_level: float,
    min_avg_power_saving_pct: float,
    min_avg_eta_gain_pct: float,
    max_avg_eta_gain_pct: float,
    max_err_failures: float,
    min_start_stop_saving_pct: float,
    max_start_stop_saving_pct: float,
    max_worst_current_peak_ratio: float,
    max_worst_current_mean_ratio: float,
    use_envelope_acceptance: bool,
    acceptance_envelopes: str | None,
    min_avg_power_saving_pct_min_seed: float | None,
    min_avg_eta_gain_pct_min_seed: float | None,
    max_err_failures_max_seed: float | None,
    min_start_stop_saving_pct_min_seed: float | None,
    top_k: int,
    feature_keys: List[str] | None = None,
    init_checkpoint: str | None = None,
    include_init_checkpoint: bool = False,
    resume: bool = False,
) -> Dict[str, object]:
    from tools.scan_step27_checkpoints import scan_checkpoints

    scan_out_dir = (run_dir / "external_step27_scan").resolve()
    acceptance_envelopes_path = None if acceptance_envelopes is None else Path(str(acceptance_envelopes)).expanduser().resolve()
    included_init_path: Path | None = None
    if bool(include_init_checkpoint) and init_checkpoint is not None:
        init_path = Path(str(init_checkpoint)).expanduser().resolve()
        if not init_path.exists():
            raise FileNotFoundError(f"External Step27 init checkpoint not found: {init_path}")
        eval_dir = (run_dir / "eval").resolve()
        eval_dir.mkdir(parents=True, exist_ok=True)
        included_init_path = (eval_dir / "actor_ep_init.pth").resolve()
        if init_path != included_init_path:
            shutil.copyfile(init_path, included_init_path)
    candidate_json_arg = "" if candidate_json is None else str(candidate_json)
    summary = scan_checkpoints(
        motor=str(motor),
        ai_control_mode=str(ai_control_mode),
        checkpoint_glob=str((run_dir / "eval").resolve()),
        candidate_json=candidate_json_arg,
        candidate_index=int(candidate_index),
        candidate_tag=str(candidate_tag),
        candidate_tags=[part.strip() for part in str(candidate_tags).split(",") if part.strip()],
        seeds=_parse_int_csv(seeds),
        scenarios=_parse_scenarios(scenarios),
        out_dir=scan_out_dir,
        seed_perturbation=bool(seed_perturbation),
        seed_perturb_level=float(seed_perturb_level),
        min_avg_power_saving_pct=float(min_avg_power_saving_pct),
        min_avg_eta_gain_pct=float(min_avg_eta_gain_pct),
        max_avg_eta_gain_pct=float(max_avg_eta_gain_pct),
        max_err_failures=float(max_err_failures),
        min_start_stop_saving_pct=float(min_start_stop_saving_pct),
        max_start_stop_saving_pct=float(max_start_stop_saving_pct),
        max_worst_current_peak_ratio=float(max_worst_current_peak_ratio),
        max_worst_current_mean_ratio=float(max_worst_current_mean_ratio),
        use_envelope_acceptance=bool(use_envelope_acceptance),
        acceptance_envelopes=acceptance_envelopes_path,
        min_avg_power_saving_pct_min_seed=min_avg_power_saving_pct_min_seed,
        min_avg_eta_gain_pct_min_seed=min_avg_eta_gain_pct_min_seed,
        max_err_failures_max_seed=max_err_failures_max_seed,
        min_start_stop_saving_pct_min_seed=min_start_stop_saving_pct_min_seed,
        top_k=int(top_k),
        feature_keys=None if feature_keys is None else list(feature_keys),
        resume=bool(resume),
    )
    best = dict(summary.get("best") or {})
    selected_raw = str(best.get("checkpoint", "")).strip()
    if not selected_raw:
        raise ValueError(f"External Step27 scan produced no evaluated checkpoints for motor={motor}")
    selected_path = Path(selected_raw).resolve()
    if not selected_path.exists():
        raise FileNotFoundError(f"External Step27 selected checkpoint not found: {selected_path}")

    promoted_path = (run_dir / "best_actor_step27.pth").resolve()
    shutil.copyfile(selected_path, promoted_path)
    payload: Dict[str, object] = {
        "enabled": True,
        "motor": str(motor),
        "ai_control_mode": str(ai_control_mode),
        "scan_summary_json": str((scan_out_dir / f"{motor}_checkpoint_scan_summary.json").resolve()),
        "scan_rows_json": str((scan_out_dir / f"{motor}_checkpoint_scan.json").resolve()),
        "selected_checkpoint": str(selected_path),
        "selected_checkpoint_name": str(best.get("checkpoint_name", selected_path.name)),
        "selected_rank": int(best.get("rank", 0)),
        "selected_score": float(best.get("score", float("inf"))),
        "acceptance_pass": bool(best.get("acceptance_pass", False)),
        "promoted_checkpoint": str(promoted_path),
        "candidate_json": None
        if not str(candidate_json_arg).strip()
        else str(Path(candidate_json_arg).resolve()),
        "candidate_index": int(candidate_index),
        "candidate_tag": str(candidate_tag),
        "candidate_tags": [part.strip() for part in str(candidate_tags).split(",") if part.strip()],
        "seeds": _parse_int_csv(seeds),
        "scenarios": _parse_scenarios(scenarios),
        "seed_perturbation": bool(seed_perturbation),
        "seed_perturb_level": float(seed_perturb_level),
        "min_avg_power_saving_pct": float(min_avg_power_saving_pct),
        "min_avg_eta_gain_pct": float(min_avg_eta_gain_pct),
        "max_avg_eta_gain_pct": float(max_avg_eta_gain_pct),
        "max_err_failures": float(max_err_failures),
        "min_start_stop_saving_pct": float(min_start_stop_saving_pct),
        "max_start_stop_saving_pct": float(max_start_stop_saving_pct),
        "max_worst_current_peak_ratio": float(max_worst_current_peak_ratio),
        "max_worst_current_mean_ratio": float(max_worst_current_mean_ratio),
        "use_envelope_acceptance": bool(use_envelope_acceptance),
        "acceptance_envelopes": None if acceptance_envelopes_path is None else str(acceptance_envelopes_path),
        "min_avg_power_saving_pct_min_seed": None
        if min_avg_power_saving_pct_min_seed is None
        else float(min_avg_power_saving_pct_min_seed),
        "min_avg_eta_gain_pct_min_seed": None
        if min_avg_eta_gain_pct_min_seed is None
        else float(min_avg_eta_gain_pct_min_seed),
        "max_err_failures_max_seed": None
        if max_err_failures_max_seed is None
        else float(max_err_failures_max_seed),
        "min_start_stop_saving_pct_min_seed": None
        if min_start_stop_saving_pct_min_seed is None
        else float(min_start_stop_saving_pct_min_seed),
        "resume": bool(resume),
        "include_init_checkpoint": bool(include_init_checkpoint),
        "included_init_checkpoint": None if included_init_path is None else str(included_init_path),
        "best_metrics": best,
    }
    (run_dir / "external_step27_selection.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _promote_external_step27_checkpoint(
    *,
    ckpt_dir: Path,
    external_step27_selection: Dict[str, object] | None,
) -> Dict[str, object] | None:
    if external_step27_selection is None:
        return None

    promoted_path = Path(str(external_step27_selection["promoted_checkpoint"])).expanduser().resolve()
    if not promoted_path.exists():
        raise FileNotFoundError(f"Selected Step27 checkpoint does not exist: {promoted_path}")

    ckpt_dir = ckpt_dir.expanduser().resolve()
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    registry_best = (ckpt_dir / "best_actor.pth").resolve()
    train_best = (ckpt_dir / "best_actor_train_internal.pth").resolve()

    if registry_best.exists() and registry_best != promoted_path:
        shutil.copyfile(registry_best, train_best)

    shutil.copyfile(promoted_path, registry_best)

    payload = dict(external_step27_selection)
    payload["registry_best_checkpoint"] = str(registry_best)
    if train_best.exists():
        payload["train_internal_best_checkpoint"] = str(train_best)
    return payload


def _adapt_checkpoint_state_dict_for_model(
    state_dict: Dict[str, torch.Tensor],
    model_state_dict: Dict[str, torch.Tensor],
    *,
    target_control_mode: str | None = None,
) -> tuple[Dict[str, torch.Tensor], List[str]]:
    return adapt_checkpoint_state_dict_for_model(
        state_dict,
        model_state_dict,
        target_control_mode=target_control_mode,
    )


def build_env(
    env_config_path: str,
    episode_steps: int,
    control_mode: str,
    w_speed: float,
    w_power: float,
    w_current: float | None,
    w_smooth: float,
    w_mag: float,
    w_shaft: float,
    w_eta: float,
    w_eta_episode: float,
    eta_clip: float,
    override_load_torque: bool,
    override_omega_ref: bool,
    ai_id_ref_relative: bool,
    delta_id_max: float,
    id_ref_alpha: float,
    id_ref_rate_limit: float | None,
    ai_id_speed_tol: float,
    ai_id_speed_tol_rel: float | None,
    id_ref_gate_speed_tol: float | None,
    id_ref_gate_speed_tol_rel: float | None,
    id_ref_gate_min_scale: float,
    id_ref_gate_exponent: float,
    load_torque: float | None,
    omega_ref_override: float | None,
    feature_keys: List[str],
) -> MicAiAIEnv:
    env_sim = make_env_from_config(env_config_path)
    env_cfg = env_sim.env_config

    if omega_ref_override is None:
        omega_ref = float(2.0 * np.pi * 10.0 / max(env_cfg.motor.p, 1))
    else:
        omega_ref = float(omega_ref_override)
    i_base_nom = float(getattr(env_cfg.motor, "I_n", 1.0))
    foc_cfg = getattr(env_cfg, "foc", None)
    iq_limit_cfg = getattr(foc_cfg, "iq_limit", None)
    if iq_limit_cfg is None:
        iq_limit_cfg = i_base_nom * 8.0
    iq_limit = float(iq_limit_cfg)
    id_ref_base = float(getattr(foc_cfg, "id_ref", 0.0) or 0.0)
    mode = str(control_mode).lower()
    if mode in {"ai_current", "ai_speed"}:
        # Current-control mode: keep a wide normalization range because the agent can command iq/id directly.
        i_limit = float(max(iq_limit, i_base_nom * 8.0, 5.0))
        i_base = float(i_limit)
    else:
        # id_ref-supervision mode: normalize power and current to the realistic (iq,id) vector range.
        # Using i_base_nom*8 here makes p_in_norm almost zero for bigger motors and hurts learning,
        # while evaluation (scenario_compare) uses a much smaller i_max. Keep train/eval consistent.
        # Wider id_ref range improves learning for small motors where id_ref_base can be > I_n
        # due to dq vs line-current scaling.
        id_ref_max_est = float(max(i_base_nom * 1.5, id_ref_base, id_ref_base * 1.6))
        i_limit = float(max(math.hypot(iq_limit, id_ref_max_est), i_base_nom, 5.0))
        i_base = float(i_base_nom)
    id_ref_max = max(i_base * 1.5, id_ref_base, id_ref_base * 1.6)

    cfg = load_ai_voltage_config()
    curriculum_cfg = get_curriculum_config(cfg)
    piecewise_steps = curriculum_cfg.get("piecewise_steps", (150, 300))
    piecewise_multipliers = curriculum_cfg.get("piecewise_multipliers", (1.0, 0.8, 1.0))
    curriculum_stages = curriculum_cfg.get("omega_pu_stages", (0.3, 0.5))
    stage_boundaries = curriculum_cfg.get("stage_episode_boundaries", (150, 300))

    w_current_cfg = w_current
    if w_current_cfg is None:
        w_current_cfg = float(getattr(env_cfg, "ai_w_id_current", 0.0))
    tracking_mode = mode in {"ai_current", "ai_speed"}
    baseline_speed_err = float(getattr(env_cfg, "baseline_speed_err", getattr(env_cfg, "baseline_speed_error", 0.0)))
    baseline_current_rms = float(getattr(env_cfg, "baseline_current_rms", 0.0))

    ai_cfg = AiEnvConfig(
        episode_steps=int(episode_steps),
        dt=float(env_cfg.sim.dt),
        omega_ref=omega_ref,
        omega_ref_max=max(abs(omega_ref) * 1.2, 1e-3),
        w_speed_error=float(w_speed) if tracking_mode else 0.0,
        w_current_rms=float(w_current_cfg) if tracking_mode else 0.0,
        i_base=i_base,
        i_max=i_limit,
        control_mode=str(control_mode).lower(),
        reward_min=-10.0,
        reward_max=1.0,
        w_ai_id_speed=float(w_speed),
        w_ai_id_power=float(w_power),
        w_ai_id_current=float(w_current_cfg),
        w_ai_id_smooth=float(w_smooth),
        w_ai_id_mag=float(w_mag),
        w_ai_id_shaft=float(w_shaft),
        w_ai_id_eta=float(w_eta),
        w_ai_id_eta_episode=float(w_eta_episode),
        ai_id_eta_clip=float(eta_clip),
        baseline_speed_err=baseline_speed_err,
        baseline_current_rms=baseline_current_rms,
        ai_id_energy_gate_mode=str(getattr(env_cfg, "ai_id_energy_gate_mode", "hard")),
        ai_id_energy_gate_min_scale=float(getattr(env_cfg, "ai_id_energy_gate_min_scale", 0.0)),
        ai_id_energy_gate_exponent=float(getattr(env_cfg, "ai_id_energy_gate_exponent", 1.0)),
        ai_id_terminal_energy_bonus=float(getattr(env_cfg, "ai_id_terminal_energy_bonus", 0.0)),
        ai_id_terminal_eta_target=float(getattr(env_cfg, "ai_id_terminal_eta_target", 0.0)),
        ai_id_terminal_shaft_ratio_min=float(getattr(env_cfg, "ai_id_terminal_shaft_ratio_min", 0.0)),
        sigma_omega=float(getattr(env_cfg, "ai_sigma_omega", 0.05)),
        sigma_id=float(getattr(env_cfg, "ai_sigma_id", 0.03)),
        sigma_iq=float(getattr(env_cfg, "ai_sigma_iq", 0.03)),
        drift_every_episodes=int(getattr(env_cfg, "ai_drift_every_episodes", 5)),
        drift_scale=float(getattr(env_cfg, "ai_drift_scale", 0.04)),
        w_ext_scale=float(getattr(env_cfg, "ai_w_ext_scale", 1.0)),
        w_int_scale=float(getattr(env_cfg, "ai_w_int_scale", 0.0)),
        wm_lr=float(getattr(env_cfg, "ai_wm_lr", 1e-4)),
        curiosity_beta=float(getattr(env_cfg, "ai_curiosity_beta", 0.0)),
        foc_assist_reward_mode=str(getattr(env_cfg, "foc_assist_reward_mode", "baseline")),
        w_foc_speed=float(getattr(env_cfg, "w_foc_speed", 1.0)),
        w_foc_power=float(getattr(env_cfg, "w_foc_power", 0.5)),
        w_foc_current=float(getattr(env_cfg, "w_foc_current", 0.1)),
        w_foc_action=float(getattr(env_cfg, "w_foc_action", 0.01)),
        foc_speed_tol=float(getattr(env_cfg, "foc_speed_tol", 0.5)),
        p_el_tau=float(getattr(env_cfg, "p_el_tau", 0.02)),
        id_ref_alpha=float(id_ref_alpha),
        id_ref_rate_limit=None if id_ref_rate_limit is None else float(id_ref_rate_limit),
        id_ref_gate_speed_tol=None if id_ref_gate_speed_tol is None else float(id_ref_gate_speed_tol),
        id_ref_gate_speed_tol_rel=None if id_ref_gate_speed_tol_rel is None else float(id_ref_gate_speed_tol_rel),
        id_ref_gate_min_scale=float(id_ref_gate_min_scale),
        id_ref_gate_exponent=float(id_ref_gate_exponent),
        delta_id_max=float(delta_id_max),
        ai_id_speed_tol=float(ai_id_speed_tol),
        ai_id_speed_tol_rel=None if ai_id_speed_tol_rel is None else float(ai_id_speed_tol_rel),
        curriculum_omega_pu=tuple(float(x) for x in curriculum_stages),
        curriculum_stage_episodes=tuple(int(x) for x in stage_boundaries),
        omega_piecewise_steps=tuple(int(x) for x in piecewise_steps),
        omega_piecewise_multipliers=tuple(float(x) for x in piecewise_multipliers),
        id_ref_min=0.0,
        id_ref_max=float(id_ref_max),
        ai_id_ref_relative=bool(ai_id_ref_relative),
        # Keep some safety margin: phase current can exceed iq_limit because id_ref and iq add
        # in the current vector. Too-tight limit would truncate episodes near the end of start/stop.
        i_hard_limit=float(i_limit * 4.0),
        i_soft_limit=float(getattr(env_cfg, "i_soft_limit", 1.2)),
        i_soft_penalty=float(getattr(env_cfg, "i_soft_penalty", 0.5)),
        load_torque_override=None if load_torque is None else float(load_torque),
        override_load_torque=bool(override_load_torque),
        override_omega_ref=bool(override_omega_ref),
        enable_id_control=bool(str(control_mode).lower() in TWO_ACTION_CONTROL_MODES),
    )

    base_env = InductionMotorEnv(env_cfg)
    if bool(override_omega_ref):
        base_env.omega_ref_func = lambda _t, ref=omega_ref: ref
    if load_torque is not None:
        base_env.load_torque_func = lambda _t, load=load_torque: float(load)

    env = MicAiAIEnv(base_env, ai_cfg, curiosity=None, world_model=None, world_input_keys=feature_keys, world_target_keys=["omega_norm"])
    setattr(env, "_train_env_cfg", env_cfg)
    setattr(env, "_train_env_config_path", str(env_config_path))
    return env


def _apply_env_reward_overrides(
    env: MicAiAIEnv,
    *,
    energy_gate_mode: str | None = None,
    energy_gate_min_scale: float | None = None,
    energy_gate_exponent: float | None = None,
    terminal_energy_bonus: float | None = None,
    terminal_eta_target: float | None = None,
    terminal_shaft_ratio_min: float | None = None,
    i_soft_limit: float | None = None,
    i_soft_penalty: float | None = None,
) -> None:
    if energy_gate_mode is not None:
        env.cfg.ai_id_energy_gate_mode = str(energy_gate_mode)
    if energy_gate_min_scale is not None:
        env.cfg.ai_id_energy_gate_min_scale = float(energy_gate_min_scale)
    if energy_gate_exponent is not None:
        env.cfg.ai_id_energy_gate_exponent = float(energy_gate_exponent)
    if terminal_energy_bonus is not None:
        env.cfg.ai_id_terminal_energy_bonus = float(terminal_energy_bonus)
    if terminal_eta_target is not None:
        env.cfg.ai_id_terminal_eta_target = float(terminal_eta_target)
    if terminal_shaft_ratio_min is not None:
        env.cfg.ai_id_terminal_shaft_ratio_min = float(terminal_shaft_ratio_min)
    if i_soft_limit is not None:
        env.cfg.i_soft_limit = float(i_soft_limit)
    if i_soft_penalty is not None:
        env.cfg.i_soft_penalty = float(i_soft_penalty)


def _apply_scenario_reward_overrides(
    env: MicAiAIEnv,
    *,
    scenario_name: str,
    episode_seed: int | None,
    base_w_speed: float,
    base_w_power: float,
    base_w_shaft: float,
    base_w_eta: float,
    base_w_eta_episode: float,
    base_reward_start_frac: float,
    base_terminal_energy_bonus: float,
    base_ai_id_speed_tol: float,
    base_ai_id_speed_tol_rel: float | None,
    base_id_ref_gate_speed_tol: float | None,
    base_id_ref_gate_speed_tol_rel: float | None,
    base_id_ref_gate_min_scale: float,
    base_id_ref_gate_exponent: float,
    scenario_reward_overrides: Dict[str, Dict[str, float]] | None,
    seed_scenario_reward_overrides: Dict[int, Dict[str, Dict[str, float]]] | None,
) -> Dict[str, float]:
    scenario_key = str(scenario_name or "").strip()
    override = dict((scenario_reward_overrides or {}).get(scenario_key, {}))
    if episode_seed is not None:
        seed_override = (
            (seed_scenario_reward_overrides or {})
            .get(int(episode_seed), {})
            .get(scenario_key, {})
        )
        if seed_override:
            override.update(dict(seed_override))
    effective = {
        "w_speed": float(override.get("w_speed", base_w_speed)),
        "w_power": float(override.get("w_power", base_w_power)),
        "w_shaft": float(override.get("w_shaft", base_w_shaft)),
        "w_eta": float(override.get("w_eta", base_w_eta)),
        "w_eta_episode": float(override.get("w_eta_episode", base_w_eta_episode)),
        "reward_start_frac": float(np.clip(override.get("reward_start_frac", base_reward_start_frac), 0.0, 1.0)),
        "terminal_energy_bonus": float(override.get("terminal_energy_bonus", base_terminal_energy_bonus)),
        "ai_id_speed_tol": float(override.get("ai_id_speed_tol", base_ai_id_speed_tol)),
        "ai_id_speed_tol_rel": None
        if "ai_id_speed_tol_rel" not in override and base_ai_id_speed_tol_rel is None
        else float(override.get("ai_id_speed_tol_rel", base_ai_id_speed_tol_rel or 0.0)),
        "id_ref_gate_speed_tol": None
        if "id_ref_gate_speed_tol" not in override and base_id_ref_gate_speed_tol is None
        else float(override.get("id_ref_gate_speed_tol", base_id_ref_gate_speed_tol or 0.0)),
        "id_ref_gate_speed_tol_rel": None
        if "id_ref_gate_speed_tol_rel" not in override and base_id_ref_gate_speed_tol_rel is None
        else float(override.get("id_ref_gate_speed_tol_rel", base_id_ref_gate_speed_tol_rel or 0.0)),
        "id_ref_gate_min_scale": float(override.get("id_ref_gate_min_scale", base_id_ref_gate_min_scale)),
        "id_ref_gate_exponent": float(override.get("id_ref_gate_exponent", base_id_ref_gate_exponent)),
    }
    env.cfg.w_ai_id_speed = float(effective["w_speed"])
    env.cfg.w_ai_id_power = float(effective["w_power"])
    env.cfg.w_ai_id_shaft = float(effective["w_shaft"])
    env.cfg.w_ai_id_eta = float(effective["w_eta"])
    env.cfg.w_ai_id_eta_episode = float(effective["w_eta_episode"])
    env.cfg.ai_id_terminal_energy_bonus = float(effective["terminal_energy_bonus"])
    env.cfg.ai_id_speed_tol = float(effective["ai_id_speed_tol"])
    env.cfg.ai_id_speed_tol_rel = (
        None if effective["ai_id_speed_tol_rel"] is None else float(effective["ai_id_speed_tol_rel"])
    )
    env.cfg.id_ref_gate_speed_tol = (
        None if effective["id_ref_gate_speed_tol"] is None else float(effective["id_ref_gate_speed_tol"])
    )
    env.cfg.id_ref_gate_speed_tol_rel = (
        None if effective["id_ref_gate_speed_tol_rel"] is None else float(effective["id_ref_gate_speed_tol_rel"])
    )
    env.cfg.id_ref_gate_min_scale = float(effective["id_ref_gate_min_scale"])
    env.cfg.id_ref_gate_exponent = float(effective["id_ref_gate_exponent"])
    return effective


def train(
    env_config: str,
    episodes: int,
    episode_steps: int,
    control_mode: str,
    w_speed: float,
    w_power: float,
    w_current: float | None,
    w_smooth: float,
    w_mag: float,
    w_shaft: float,
    w_eta: float,
    w_eta_episode: float,
    eta_clip: float,
    id_ref_alpha: float,
    id_ref_rate_limit: float | None,
    ai_id_speed_tol: float,
    ai_id_speed_tol_rel: float | None,
    id_ref_gate_speed_tol: float | None,
    id_ref_gate_speed_tol_rel: float | None,
    id_ref_gate_min_scale: float,
    id_ref_gate_exponent: float,
    fast: bool,
    time_budget_min: float | None,
    override_load_torque: bool,
    override_omega_ref: bool,
    ai_id_ref_relative: bool,
    delta_id_max: float,
    load_torque: float | None,
    omega_ref_override: float | None,
    scenarios: List[str] | None,
    scenario_sample: str,
    omega_ref_range: tuple[float, float] | None,
    load_torque_range: tuple[float, float] | None,
    seed: int | None,
    sigma_start: float,
    sigma_end: float,
    sigma_decay_episodes: int,
    power_warmup_episodes: int,
    power_ramp_episodes: int,
    energy_warmup_episodes: int,
    energy_ramp_episodes: int,
    eval_interval: int,
    eval_scenarios: str,
    eval_dt: float | None,
    eval_t_end: float | None,
    eval_window_frac: float,
    eval_error_tol_rel: float,
    eval_error_tol_abs: float,
    eval_use_total_power: bool,
    include_energy_obs: bool,
    include_episode_eta_obs: bool,
    update_every_episodes: int,
    episode_seed_cycle: List[int] | None = None,
    lr: float = 5e-4,
    entropy_coef: float = 0.005,
    actor_anchor_coef: float = 0.0,
    external_step27_select: bool = False,
    external_step27_motor: str | None = None,
    external_step27_candidate_json: str | None = None,
    external_step27_candidate_index: int = 0,
    external_step27_candidate_tag: str = "",
    external_step27_candidate_tags: str = "",
    external_step27_seeds: str = "101,202,303",
    external_step27_scenarios: str = "speed_step,ramp,load_step,start_stop",
    external_step27_seed_perturbation: bool = False,
    external_step27_seed_perturb_level: float = 0.2,
    external_step27_min_avg_power_saving_pct: float = 0.0,
    external_step27_min_avg_eta_gain_pct: float = 0.0,
    external_step27_max_avg_eta_gain_pct: float = 25.0,
    external_step27_max_err_failures: float = 2.0,
    external_step27_min_start_stop_saving_pct: float = -0.5,
    external_step27_max_start_stop_saving_pct: float = 20.0,
    external_step27_max_worst_current_peak_ratio: float = 1.30,
    external_step27_max_worst_current_mean_ratio: float = 1.20,
    external_step27_use_envelope_acceptance: bool = False,
    external_step27_acceptance_envelopes: str | None = None,
    external_step27_min_avg_power_saving_pct_min_seed: float | None = None,
    external_step27_min_avg_eta_gain_pct_min_seed: float | None = None,
    external_step27_max_err_failures_max_seed: float | None = None,
    external_step27_min_start_stop_saving_pct_min_seed: float | None = None,
    external_step27_top_k: int = 10,
    external_step27_include_init_checkpoint: bool = False,
    external_step27_resume: bool = False,
    init_checkpoint: str | None = None,
    output_dir: str | None = None,
    results_root: str | None = None,
    energy_gate_mode: str | None = None,
    energy_gate_min_scale: float | None = None,
    energy_gate_exponent: float | None = None,
    terminal_energy_bonus: float | None = None,
    terminal_eta_target: float | None = None,
    terminal_shaft_ratio_min: float | None = None,
    i_soft_limit: float | None = None,
    i_soft_penalty: float | None = None,
    hidden_sizes_override: tuple[int, ...] | None = None,
    scenario_reward_overrides: Dict[str, Dict[str, float]] | None = None,
    seed_scenario_reward_overrides: Dict[int, Dict[str, Dict[str, float]]] | None = None,
) -> Dict[str, str]:
    feature_keys = build_feature_keys(include_energy_obs, include_episode_eta_obs)
    if seed is not None:
        random.seed(int(seed))
        np.random.seed(int(seed))
        torch.manual_seed(int(seed))

    init_path: Path | None = None
    init_state: Dict[str, torch.Tensor] | None = None
    if init_checkpoint:
        from mic_ai.tools.scenario_compare import _resolve_feature_keys

        init_path = Path(str(init_checkpoint)).resolve()
        if not init_path.exists():
            raise FileNotFoundError(f"Init checkpoint not found: {init_path}")
        state_raw = torch.load(init_path, map_location="cpu")
        if isinstance(state_raw, dict) and "state_dict" in state_raw and isinstance(state_raw.get("state_dict"), dict):
            state_raw = state_raw["state_dict"]
        if not isinstance(state_raw, dict):
            raise ValueError(f"Unsupported checkpoint format: {init_path}")
        init_state = state_raw
        inferred_feature_keys = list(_resolve_feature_keys(None, init_state))
        if len(feature_keys) < len(inferred_feature_keys) and feature_keys == inferred_feature_keys[: len(feature_keys)]:
            feature_keys = inferred_feature_keys
            print(
                "[train_ai_id_ref] inferred feature_keys from init checkpoint {} -> {}".format(
                    init_path.name,
                    feature_keys,
                )
            )

    env = build_env(
        env_config,
        episode_steps=episode_steps,
        control_mode=str(control_mode),
        w_speed=w_speed,
        w_power=w_power,
        w_current=w_current,
        w_smooth=w_smooth,
        w_mag=w_mag,
        w_shaft=w_shaft,
        w_eta=w_eta,
        w_eta_episode=w_eta_episode,
        eta_clip=eta_clip,
        override_load_torque=override_load_torque,
        override_omega_ref=override_omega_ref,
        ai_id_ref_relative=ai_id_ref_relative,
        delta_id_max=delta_id_max,
        id_ref_alpha=id_ref_alpha,
        id_ref_rate_limit=id_ref_rate_limit,
        ai_id_speed_tol=ai_id_speed_tol,
        ai_id_speed_tol_rel=ai_id_speed_tol_rel,
        id_ref_gate_speed_tol=id_ref_gate_speed_tol,
        id_ref_gate_speed_tol_rel=id_ref_gate_speed_tol_rel,
        id_ref_gate_min_scale=id_ref_gate_min_scale,
        id_ref_gate_exponent=id_ref_gate_exponent,
        load_torque=load_torque,
        omega_ref_override=omega_ref_override,
        feature_keys=feature_keys,
    )
    _apply_env_reward_overrides(
        env,
        energy_gate_mode=energy_gate_mode,
        energy_gate_min_scale=energy_gate_min_scale,
        energy_gate_exponent=energy_gate_exponent,
        terminal_energy_bonus=terminal_energy_bonus,
        terminal_eta_target=terminal_eta_target,
        terminal_shaft_ratio_min=terminal_shaft_ratio_min,
        i_soft_limit=i_soft_limit,
        i_soft_penalty=i_soft_penalty,
    )
    env_train_cfg = getattr(env, "_train_env_cfg", None)
    train_supervisor_cfg = _train_supervisor_from_env(env_train_cfg)
    train_supervisor: AiIdRefSupervisor | None = None
    if train_supervisor_cfg is not None:
        omega_nominal = float(
            max(
                abs(float(getattr(env, "_omega_nominal", 0.0))),
                abs(float(getattr(env.cfg, "omega_ref_max", 0.0) or 0.0)),
                abs(float(getattr(env.cfg, "omega_ref", 0.0) or 0.0)),
                1e-6,
            )
        )
        train_supervisor = AiIdRefSupervisor(train_supervisor_cfg, omega_nominal=omega_nominal)
        print(
            "[train_ai_id_ref] training with eval supervisor "
            f"objective={train_supervisor_cfg.objective} "
            f"update_steps={train_supervisor_cfg.update_steps} "
            f"dither={train_supervisor_cfg.dither_amp}"
        )
    base_terminal_energy_bonus = float(getattr(env.cfg, "ai_id_terminal_energy_bonus", 0.0))

    external_step27_motor_name = str(external_step27_motor or "").strip().lower()
    external_candidate_raw = None if external_step27_candidate_json is None else str(external_step27_candidate_json).strip()
    if external_candidate_raw is not None and external_candidate_raw.lower() in {"", "none", "null"}:
        external_candidate_raw = None
    external_step27_candidate_path = None if external_candidate_raw is None else Path(external_candidate_raw).expanduser().resolve()
    if external_step27_select:
        if not external_step27_motor_name:
            raise ValueError("--external-step27-motor is required when --external-step27-select is enabled")
        if str(control_mode).lower() == "ai_id_ref" and external_step27_candidate_path is None:
            raise ValueError("--external-step27-candidate-json is required when --external-step27-select is enabled")
        if external_step27_candidate_path is not None and not external_step27_candidate_path.exists():
            raise FileNotFoundError(f"External Step27 candidate json not found: {external_step27_candidate_path}")

    scenarios = [s for s in (scenarios or []) if s]
    scenario_sample = str(scenario_sample or "random").lower()
    rng = np.random.default_rng(seed)

    hidden_sizes = tuple(hidden_sizes_override) if hidden_sizes_override else ((64, 64) if fast else (128, 128))
    inferred_hidden_sizes = None if init_state is None else _infer_hidden_sizes_from_state_dict(init_state)
    if hidden_sizes_override:
        hidden_sizes = tuple(int(x) for x in hidden_sizes_override)
    elif inferred_hidden_sizes:
        hidden_sizes = tuple(int(x) for x in inferred_hidden_sizes)
        print(f"[train_ai_id_ref] inferred hidden_sizes={hidden_sizes} from init checkpoint {init_path.name}")
    train_epochs = 3 if fast else 5
    minibatch_frac = 0.5 if fast else 0.25
    action_dim = 2 if str(control_mode).lower() in TWO_ACTION_CONTROL_MODES else 1
    agent = PPOVoltageAgent(
        feature_keys=feature_keys,
        action_dim=action_dim,
        device="cpu",
        hidden_sizes=hidden_sizes,
        lr=float(lr),
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        entropy_coef=float(entropy_coef),
        value_coef=0.3,
        max_grad_norm=0.5,
        train_epochs=train_epochs,
        minibatch_frac=minibatch_frac,
    )
    if init_state is not None and init_path is not None:
        state, adjusted_keys = _adapt_checkpoint_state_dict_for_model(
            init_state,
            agent.net.state_dict(),
            target_control_mode=str(control_mode).lower(),
        )
        missing_keys, unexpected_keys = agent.net.load_state_dict(state, strict=False)
        print(
            "[train_ai_id_ref] warm-start checkpoint={} missing_keys={} unexpected_keys={} adjusted_keys={}".format(
                init_path,
                len(missing_keys),
                len(unexpected_keys),
                len(adjusted_keys),
            )
        )
    if float(actor_anchor_coef) > 0.0:
        if init_state is None:
            raise ValueError("--actor-anchor-coef requires --init-checkpoint")
        agent.set_actor_anchor_from_current(float(actor_anchor_coef))
        print(f"[train_ai_id_ref] actor anchor enabled coef={float(actor_anchor_coef):.6f}")

    output_root_path = OUTPUT_DIR if output_dir is None else Path(str(output_dir)).expanduser()
    if not output_root_path.is_absolute():
        output_root_path = (Path.cwd() / output_root_path).resolve()
    episode_log_dir = output_root_path / "episode_logs"
    checkpoint_root = output_root_path / "checkpoints"

    results_root_path = RESULTS_ROOT if results_root is None else Path(str(results_root)).expanduser()
    if not results_root_path.is_absolute():
        results_root_path = (Path.cwd() / results_root_path).resolve()

    env_name = Path(env_config).stem
    ckpt_dir = (checkpoint_root / env_name).resolve()
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if scenarios:
        train_env_cfg = getattr(env, "_train_env_cfg", None)
        sim_cfg = getattr(train_env_cfg, "sim", None)
        if sim_cfg is not None:
            underhorizon = _collect_underhorizon_scenarios(
                scenarios,
                episode_steps=int(episode_steps),
                t_end=float(getattr(sim_cfg, "t_end", 0.0) or 0.0),
                dt=float(getattr(sim_cfg, "dt", 0.0) or 0.0),
            )
            if underhorizon:
                print(
                    "[train_ai_id_ref] warning: episode horizon is shorter than scenario activation for {}".format(
                        ", ".join(
                            "{}(steps {} < {})".format(
                                row["scenario"],
                                row["episode_steps"],
                                row["required_steps"],
                            )
                            for row in underhorizon
                        )
                    )
                )

    mode_tag = str(control_mode).lower()
    run_dir = results_root_path / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{env_name}_{mode_tag}"
    run_dir.mkdir(parents=True, exist_ok=True)

    episodes_log: List[Dict[str, float]] = []
    best_score = float("inf")
    best_ckpt: Path | None = None
    t0 = time.perf_counter()
    max_seconds = None if time_budget_min is None else float(time_budget_min) * 60.0

    update_every = max(int(update_every_episodes), 1)
    for ep in range(int(episodes)):
        if max_seconds is not None and (time.perf_counter() - t0) >= max_seconds:
            print(f"[{env_name}] time budget reached at ep {ep}")
            break

        episode_seed = _select_episode_seed(ep, episode_seed_cycle)
        if episode_seed is not None:
            random.seed(int(episode_seed))
            np.random.seed(int(episode_seed))
            torch.manual_seed(int(episode_seed))

        scenario_name = ""
        if scenarios:
            if scenario_sample == "cycle":
                scenario_name = scenarios[ep % len(scenarios)]
            else:
                scenario_name = str(rng.choice(scenarios))
            env.set_scenario(scenario_name)

        obs = env.reset()
        if train_supervisor is not None:
            train_supervisor.reset()
        scenario_meta = {
            "omega_base_peak": float(getattr(env.cfg, "omega_ref", 0.0)),
            "load_base_peak": float(getattr(env.base_env, "load_torque_func", lambda _t: 0.0)(0.0)),
            "omega_scale": 1.0,
            "load_scale": 1.0,
            "omega_peak": float(getattr(env.cfg, "omega_ref", 0.0)),
            "load_peak": float(getattr(env.base_env, "load_torque_func", lambda _t: 0.0)(0.0)),
        }
        if scenarios or omega_ref_range is not None or load_torque_range is not None:
            sim_cfg = getattr(getattr(env.base_env, "env", None), "sim", None)
            t_end = float(getattr(sim_cfg, "t_end", 0.0) or 0.0)
            scenario_rng = rng if episode_seed is None else np.random.default_rng(int(episode_seed))
            wrapped_omega, wrapped_load, scenario_meta = wrap_scenario_with_ranges(
                getattr(env.base_env, "omega_ref_func", lambda _t: float(getattr(env.cfg, "omega_ref", 0.0))),
                getattr(env.base_env, "load_torque_func", lambda _t: 0.0),
                t_end=t_end,
                rng=scenario_rng,
                omega_ref_range=omega_ref_range,
                load_torque_range=load_torque_range,
            )
            env.base_env.omega_ref_func = wrapped_omega
            env.base_env.load_torque_func = wrapped_load
        scenario_meta["scenario"] = scenario_name
        done = False
        total_reward = 0.0
        steps = 0
        if sigma_decay_episodes <= 0:
            sigma = float(sigma_end)
        else:
            frac = min(1.0, ep / max(sigma_decay_episodes, 1))
            sigma = float(sigma_start + (sigma_end - sigma_start) * frac)
        agent.set_action_std(sigma)

        power_scale = _curriculum_scale(ep, power_warmup_episodes, power_ramp_episodes)
        energy_scale = _curriculum_scale(ep, energy_warmup_episodes, energy_ramp_episodes)
        effective_weights = _apply_scenario_reward_overrides(
            env,
            scenario_name=scenario_name,
            episode_seed=episode_seed,
            base_w_speed=float(w_speed),
            base_w_power=float(w_power) * float(power_scale) * float(energy_scale),
            base_w_shaft=float(w_shaft),
            base_w_eta=float(w_eta) * float(energy_scale),
            base_w_eta_episode=float(w_eta_episode) * float(energy_scale),
            base_reward_start_frac=0.0,
            base_terminal_energy_bonus=float(base_terminal_energy_bonus) * float(energy_scale),
            base_ai_id_speed_tol=float(ai_id_speed_tol),
            base_ai_id_speed_tol_rel=None if ai_id_speed_tol_rel is None else float(ai_id_speed_tol_rel),
            base_id_ref_gate_speed_tol=None if id_ref_gate_speed_tol is None else float(id_ref_gate_speed_tol),
            base_id_ref_gate_speed_tol_rel=None
            if id_ref_gate_speed_tol_rel is None
            else float(id_ref_gate_speed_tol_rel),
            base_id_ref_gate_min_scale=float(id_ref_gate_min_scale),
            base_id_ref_gate_exponent=float(id_ref_gate_exponent),
            scenario_reward_overrides=scenario_reward_overrides,
            seed_scenario_reward_overrides=seed_scenario_reward_overrides,
        )
        sim_cfg = getattr(getattr(env.base_env, "env", None), "sim", None)
        scenario_t_end = float(getattr(sim_cfg, "t_end", 0.0) or 0.0)
        reward_start_frac_eff = float(effective_weights["reward_start_frac"])
        reward_masked_steps = 0

        while not done and steps < int(episode_steps):
            action, logp, value = agent.act(obs)
            env_action, gate_open = _apply_train_supervisor_action(action, obs=obs, supervisor=train_supervisor)
            obs_next, reward, done, info = env.step(env_action)
            info_dict = info if isinstance(info, dict) else {}
            progress = 0.0
            t_now = info_dict.get("t")
            if t_now is not None and scenario_t_end > 0.0:
                progress = float(np.clip(float(t_now) / scenario_t_end, 0.0, 1.0))
            else:
                progress = float(np.clip(float(steps + 1) / max(int(episode_steps), 1), 0.0, 1.0))
            if progress < reward_start_frac_eff:
                reward = 0.0
                reward_masked_steps += 1
            if train_supervisor is not None:
                train_supervisor.update(
                    float(info_dict.get("p_in_pos", 0.0)),
                    float(info_dict.get("p_shaft_pos", 0.0)),
                    bool(gate_open),
                )
            agent.store(obs, action, logp, reward, done, value)
            total_reward += float(reward)
            obs = obs_next
            steps += 1

        losses = {
            "actor_loss": agent.last_actor_loss,
            "value_loss": agent.last_value_loss,
            "anchor_loss": getattr(agent, "last_anchor_loss", 0.0),
        }
        if (ep + 1) % update_every == 0 or ep == episodes - 1:
            with torch.no_grad():
                last_value = float(agent.net(agent._to_tensor(obs).unsqueeze(0))[2].item())
            losses = agent.update(last_value=last_value)
        m = env.episode_metrics()

        omega_ref_logged = float(getattr(env.cfg, "omega_ref", 0.0))
        load_logged = float(getattr(env.base_env, "load_torque_func", lambda _t: 0.0)(0.0))
        entry = {
            "episode": float(ep),
            "steps": float(m.get("steps", steps)),
            "mean_speed_error": float(m.get("mean_speed_error", 0.0)),
            "mean_p_in_pos": float(m.get("mean_p_in_pos", 0.0)),
            "mean_p_shaft_pos": float(m.get("mean_p_shaft_pos", 0.0)),
            "mean_p_shaft_target_pos": float(m.get("mean_p_shaft_target_pos", 0.0)),
            "mean_eta_inst": float(m.get("mean_eta_inst", 0.0)),
            "eta_energy": float(m.get("eta_energy", 0.0)),
            "mean_current_rms": float(m.get("mean_current_rms", 0.0)),
            "mean_action_norm": float(m.get("action_norm", 0.0)),
            "mean_reward": float(total_reward / max(steps, 1)),
            "actor_loss": float(losses.get("actor_loss", 0.0)),
            "value_loss": float(losses.get("value_loss", 0.0)),
            "anchor_loss": float(losses.get("anchor_loss", 0.0)),
            "scenario": scenario_name,
            "episode_seed": None if episode_seed is None else int(episode_seed),
            "omega_ref": omega_ref_logged,
            "load_torque": load_logged,
            "scenario_omega_scale": float(scenario_meta.get("omega_scale", 1.0)),
            "scenario_load_scale": float(scenario_meta.get("load_scale", 1.0)),
            "scenario_omega_peak": float(scenario_meta.get("omega_peak", omega_ref_logged)),
            "scenario_load_peak": float(scenario_meta.get("load_peak", load_logged)),
            "power_scale_eff": float(power_scale),
            "energy_scale_eff": float(energy_scale),
            "w_speed_eff": float(effective_weights["w_speed"]),
            "w_power_eff": float(effective_weights["w_power"]),
            "w_shaft_eff": float(effective_weights["w_shaft"]),
            "w_eta_eff": float(effective_weights["w_eta"]),
            "w_eta_episode_eff": float(effective_weights["w_eta_episode"]),
            "reward_start_frac_eff": float(reward_start_frac_eff),
            "reward_masked_steps": float(reward_masked_steps),
            "reward_masked_frac": float(reward_masked_steps / max(steps, 1)),
            "terminal_energy_bonus_eff": float(effective_weights["terminal_energy_bonus"]),
            "ai_id_speed_tol_eff": float(effective_weights["ai_id_speed_tol"]),
            "ai_id_speed_tol_rel_eff": effective_weights["ai_id_speed_tol_rel"],
            "id_ref_gate_speed_tol_eff": effective_weights["id_ref_gate_speed_tol"],
            "id_ref_gate_speed_tol_rel_eff": effective_weights["id_ref_gate_speed_tol_rel"],
            "id_ref_gate_min_scale_eff": float(effective_weights["id_ref_gate_min_scale"]),
            "id_ref_gate_exponent_eff": float(effective_weights["id_ref_gate_exponent"]),
            "scenario_reward_override_keys": sorted(
                list((scenario_reward_overrides or {}).get(str(scenario_name or "").strip(), {}).keys())
            ),
            "seed_scenario_reward_override_keys": sorted(
                list(
                    (seed_scenario_reward_overrides or {})
                    .get(int(episode_seed), {})
                    .get(str(scenario_name or "").strip(), {})
                    .keys()
                )
            )
            if episode_seed is not None
            else [],
            "exploration_sigma": float(sigma),
        }
        episodes_log.append(entry)

        # Score: minimize electric input, keep tracking quality, and avoid shaft-power deficit.
        shaft_deficit = max(0.0, entry["mean_p_shaft_target_pos"] - entry["mean_p_shaft_pos"])
        score = (
            entry["mean_p_in_pos"]
            + 50.0 * entry["mean_speed_error"]
            + 3.0 * shaft_deficit
            - 5.0 * entry["eta_energy"]
        )
        if score < best_score:
            best_score = score
            best_ckpt = ckpt_dir / "best_actor.pth"
            torch.save(agent.net.state_dict(), best_ckpt)

        # Always save per-episode checkpoints for offline, reproducible evaluation.
        eval_root = run_dir / "eval"
        eval_root.mkdir(parents=True, exist_ok=True)
        eval_ckpt = eval_root / f"actor_ep{ep:03d}.pth"
        torch.save(agent.net.state_dict(), eval_ckpt)
        if eval_interval > 0 and (ep % eval_interval == 0):
            _run_eval(
                env_config=env_config,
                checkpoint_path=eval_ckpt,
                out_dir=eval_root / f"ep_{ep:03d}",
                scenarios=eval_scenarios,
                dt=eval_dt,
                t_end=eval_t_end,
                window_frac=eval_window_frac,
                error_tol_rel=eval_error_tol_rel,
                error_tol_abs=eval_error_tol_abs,
                use_total_power=eval_use_total_power,
                ai_id_relative=bool(ai_id_ref_relative),
                delta_id_max=float(delta_id_max),
                id_ref_alpha=float(id_ref_alpha),
                id_ref_rate_limit=id_ref_rate_limit,
                id_ref_gate_speed_tol=id_ref_gate_speed_tol,
                id_ref_gate_speed_tol_rel=id_ref_gate_speed_tol_rel,
                id_ref_gate_min_scale=float(id_ref_gate_min_scale),
                id_ref_gate_exponent=float(id_ref_gate_exponent),
                feature_keys=feature_keys,
            )

        if ep % 10 == 0 or ep == episodes - 1:
            print(
                f"[{env_name}] ep {ep:03d} | mean_p_in_pos {entry['mean_p_in_pos']:.3f} | "
                f"mean_p_shaft_pos {entry['mean_p_shaft_pos']:.3f} | "
                f"eta {entry['eta_energy']:.3f} | mean|e_w| {entry['mean_speed_error']:.3f} | "
                f"act_norm {entry['mean_action_norm']:.3f}"
            )

    last_ckpt = ckpt_dir / "last_actor.pth"
    torch.save(agent.net.state_dict(), last_ckpt)

    episodes_path = _prepare_output_file(episode_log_dir / f"ai_id_ref_{env_name}_episodes.json")
    with episodes_path.open("w", encoding="utf-8") as f:
        json.dump(episodes_log, f, indent=2)

    with (run_dir / "training_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(episodes_log, f, indent=2)
    torch.save(agent.net.state_dict(), run_dir / "actor_critic.pth")
    external_step27_selection: Dict[str, object] | None = None
    if external_step27_select:
        external_step27_selection = _run_external_step27_selection(
            run_dir=run_dir,
            motor=external_step27_motor_name,
            ai_control_mode=str(control_mode).lower(),
            candidate_json=None if external_step27_candidate_path is None else str(external_step27_candidate_path),
            candidate_index=int(external_step27_candidate_index),
            candidate_tag=str(external_step27_candidate_tag),
            candidate_tags=str(external_step27_candidate_tags),
            seeds=str(external_step27_seeds),
            scenarios=str(external_step27_scenarios),
            seed_perturbation=bool(external_step27_seed_perturbation),
            seed_perturb_level=float(external_step27_seed_perturb_level),
            min_avg_power_saving_pct=float(external_step27_min_avg_power_saving_pct),
            min_avg_eta_gain_pct=float(external_step27_min_avg_eta_gain_pct),
            max_avg_eta_gain_pct=float(external_step27_max_avg_eta_gain_pct),
            max_err_failures=float(external_step27_max_err_failures),
            min_start_stop_saving_pct=float(external_step27_min_start_stop_saving_pct),
            max_start_stop_saving_pct=float(external_step27_max_start_stop_saving_pct),
            max_worst_current_peak_ratio=float(external_step27_max_worst_current_peak_ratio),
            max_worst_current_mean_ratio=float(external_step27_max_worst_current_mean_ratio),
            use_envelope_acceptance=bool(external_step27_use_envelope_acceptance),
            acceptance_envelopes=external_step27_acceptance_envelopes,
            min_avg_power_saving_pct_min_seed=external_step27_min_avg_power_saving_pct_min_seed,
            min_avg_eta_gain_pct_min_seed=external_step27_min_avg_eta_gain_pct_min_seed,
            max_err_failures_max_seed=external_step27_max_err_failures_max_seed,
            min_start_stop_saving_pct_min_seed=external_step27_min_start_stop_saving_pct_min_seed,
            top_k=int(external_step27_top_k),
            feature_keys=feature_keys,
            init_checkpoint=init_checkpoint,
            include_init_checkpoint=bool(external_step27_include_init_checkpoint),
            resume=bool(external_step27_resume),
        )
        external_step27_selection = _promote_external_step27_checkpoint(
            ckpt_dir=ckpt_dir,
            external_step27_selection=external_step27_selection,
        )
        best_ckpt = ckpt_dir / "best_actor.pth"
    run_config = {
        "env_config": str(env_config),
        "control_mode": str(control_mode).lower(),
        "episodes": int(episodes),
        "episode_steps": int(episode_steps),
        "weights": {
            "w_speed": float(w_speed),
            "w_power": float(w_power),
            "w_current": None if w_current is None else float(w_current),
            "w_smooth": float(w_smooth),
            "w_mag": float(w_mag),
            "w_shaft": float(w_shaft),
            "w_eta": float(w_eta),
            "w_eta_episode": float(w_eta_episode),
            "eta_clip": float(eta_clip),
        },
        "id_ref_alpha": float(id_ref_alpha),
        "id_ref_rate_limit": None if id_ref_rate_limit is None else float(id_ref_rate_limit),
        "ai_id_speed_tol": float(ai_id_speed_tol),
        "ai_id_speed_tol_rel": None if ai_id_speed_tol_rel is None else float(ai_id_speed_tol_rel),
        "id_ref_gate_speed_tol": None if id_ref_gate_speed_tol is None else float(id_ref_gate_speed_tol),
        "id_ref_gate_speed_tol_rel": None if id_ref_gate_speed_tol_rel is None else float(id_ref_gate_speed_tol_rel),
        "id_ref_gate_min_scale": float(id_ref_gate_min_scale),
        "id_ref_gate_exponent": float(id_ref_gate_exponent),
        "ai_id_ref_relative": bool(ai_id_ref_relative),
        "delta_id_max": float(delta_id_max),
        "load_torque_override": None if load_torque is None else float(load_torque),
        "omega_ref_override": None if omega_ref_override is None else float(omega_ref_override),
        "omega_ref_range": omega_ref_range,
        "load_torque_range": load_torque_range,
        "scenarios": scenarios or [],
        "scenario_sample": str(scenario_sample),
        "seed": None if seed is None else int(seed),
        "sigma_start": float(sigma_start),
        "sigma_end": float(sigma_end),
        "sigma_decay_episodes": int(sigma_decay_episodes),
        "power_warmup_episodes": int(power_warmup_episodes),
        "power_ramp_episodes": int(power_ramp_episodes),
        "energy_warmup_episodes": int(energy_warmup_episodes),
        "energy_ramp_episodes": int(energy_ramp_episodes),
        "eval_interval": int(eval_interval),
        "eval_scenarios": str(eval_scenarios),
        "eval_dt": None if eval_dt is None else float(eval_dt),
        "eval_t_end": None if eval_t_end is None else float(eval_t_end),
        "eval_window_frac": float(eval_window_frac),
        "eval_error_tol_rel": float(eval_error_tol_rel),
        "eval_error_tol_abs": float(eval_error_tol_abs),
        "eval_use_total_power": bool(eval_use_total_power),
        "include_energy_obs": bool(include_energy_obs),
        "include_episode_eta_obs": bool(include_episode_eta_obs),
        "update_every_episodes": int(update_every),
        "optimizer": {
            "lr": float(lr),
            "entropy_coef": float(entropy_coef),
            "actor_anchor_coef": float(actor_anchor_coef),
            "hidden_sizes": [int(x) for x in hidden_sizes],
        },
        "reward_overrides": {
            "energy_gate_mode": None if energy_gate_mode is None else str(energy_gate_mode),
            "energy_gate_min_scale": None if energy_gate_min_scale is None else float(energy_gate_min_scale),
            "energy_gate_exponent": None if energy_gate_exponent is None else float(energy_gate_exponent),
            "terminal_energy_bonus": None if terminal_energy_bonus is None else float(terminal_energy_bonus),
            "terminal_eta_target": None if terminal_eta_target is None else float(terminal_eta_target),
            "terminal_shaft_ratio_min": None if terminal_shaft_ratio_min is None else float(terminal_shaft_ratio_min),
            "i_soft_limit": None if i_soft_limit is None else float(i_soft_limit),
            "i_soft_penalty": None if i_soft_penalty is None else float(i_soft_penalty),
        },
        "scenario_reward_overrides": scenario_reward_overrides,
        "seed_scenario_reward_overrides": seed_scenario_reward_overrides,
        "external_step27_selection": external_step27_selection,
        "external_step27_min_avg_power_saving_pct": float(external_step27_min_avg_power_saving_pct),
        "external_step27_min_avg_eta_gain_pct": float(external_step27_min_avg_eta_gain_pct),
        "external_step27_max_avg_eta_gain_pct": float(external_step27_max_avg_eta_gain_pct),
        "external_step27_max_err_failures": float(external_step27_max_err_failures),
        "external_step27_min_start_stop_saving_pct": float(external_step27_min_start_stop_saving_pct),
        "external_step27_max_start_stop_saving_pct": float(external_step27_max_start_stop_saving_pct),
        "external_step27_max_worst_current_peak_ratio": float(external_step27_max_worst_current_peak_ratio),
        "external_step27_max_worst_current_mean_ratio": float(external_step27_max_worst_current_mean_ratio),
        "external_step27_use_envelope_acceptance": bool(external_step27_use_envelope_acceptance),
        "external_step27_acceptance_envelopes": None
        if external_step27_acceptance_envelopes is None
        else str(Path(external_step27_acceptance_envelopes).expanduser().resolve()),
        "external_step27_min_avg_power_saving_pct_min_seed": None
        if external_step27_min_avg_power_saving_pct_min_seed is None
        else float(external_step27_min_avg_power_saving_pct_min_seed),
        "external_step27_min_avg_eta_gain_pct_min_seed": None
        if external_step27_min_avg_eta_gain_pct_min_seed is None
        else float(external_step27_min_avg_eta_gain_pct_min_seed),
        "external_step27_max_err_failures_max_seed": None
        if external_step27_max_err_failures_max_seed is None
        else float(external_step27_max_err_failures_max_seed),
        "external_step27_min_start_stop_saving_pct_min_seed": None
        if external_step27_min_start_stop_saving_pct_min_seed is None
        else float(external_step27_min_start_stop_saving_pct_min_seed),
        "external_step27_include_init_checkpoint": bool(external_step27_include_init_checkpoint),
        "external_step27_resume": bool(external_step27_resume),
        "feature_keys": feature_keys,
        "init_checkpoint": None if init_checkpoint is None else str(Path(init_checkpoint).resolve()),
        "output_dir": str(output_root_path),
        "results_root": str(results_root_path),
    }
    (run_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    if best_ckpt is None:
        best_ckpt = ckpt_dir / "best_actor.pth"
        shutil.copyfile(last_ckpt, best_ckpt)

    print(f"Saved checkpoints: {best_ckpt} | {last_ckpt}")
    result = {
        "episodes": str(episodes_path),
        "best": str(best_ckpt),
        "last": str(last_ckpt),
        "run_dir": str(run_dir),
    }
    if external_step27_selection is not None:
        result["best_step27"] = str(external_step27_selection["promoted_checkpoint"])
        result["best_step27_selected"] = str(external_step27_selection["selected_checkpoint"])
    return result


def main() -> None:
    p = argparse.ArgumentParser(description="Train AI to adapt FOC id_ref for efficiency (minimize P_in).")
    p.add_argument("config", help="Env config path (.py)")
    p.add_argument("--control-mode", type=str, default="ai_id_ref", choices=["ai_id_ref", "ai_current", "foc_assist", "ai_speed"])
    p.add_argument("--episodes", type=int, default=400)
    p.add_argument("--episode-steps", type=int, default=200)
    p.add_argument("--w-speed", type=float, default=1.0)
    p.add_argument("--w-power", type=float, default=6.0)
    p.add_argument("--w-current", type=float, default=None, help="Penalty for current magnitude (defaults to config).")
    p.add_argument("--w-smooth", type=float, default=0.05)
    p.add_argument("--w-mag", type=float, default=0.0)
    p.add_argument("--w-shaft", type=float, default=2.0, help="Penalty for shaft-power deficit vs omega_ref*load.")
    p.add_argument("--w-eta", type=float, default=1.0, help="Penalty for low instantaneous efficiency.")
    p.add_argument("--w-eta-episode", type=float, default=0.0, help="Penalty for low running episode efficiency.")
    p.add_argument("--eta-clip", type=float, default=1.2, help="Upper clip for eta term in reward.")
    p.add_argument("--ai-id-speed-tol", type=float, default=0.5)
    p.add_argument("--ai-id-speed-tol-rel", type=float, default=None, help="Relative speed tol (e.g., 0.05).")
    p.add_argument("--id-ref-alpha", type=float, default=1.0)
    p.add_argument("--id-ref-rate-limit", type=float, default=None, help="Max d(id_ref)/dt, A/s.")
    p.add_argument("--id-ref-gate-speed-tol", type=float, default=None, help="Gate id_ref when |e_omega| exceeds tol.")
    p.add_argument("--id-ref-gate-speed-tol-rel", type=float, default=None, help="Relative gate tol (e.g., 0.05).")
    p.add_argument("--id-ref-gate-min-scale", type=float, default=0.0)
    p.add_argument("--id-ref-gate-exponent", type=float, default=1.0)
    p.add_argument("--fast", action="store_true")
    p.add_argument("--time-budget-min", type=float, default=None)
    p.add_argument("--override-load-torque", action="store_true", help="Force zero load during training.")
    p.add_argument("--no-override-omega-ref", dest="override_omega_ref", action="store_false", help="Use scenario omega_ref.")
    p.add_argument("--relative", action="store_true", help="Interpret action as delta around base id_ref.")
    p.add_argument("--delta-id-max", type=float, default=0.3, help="Relative id_ref delta scale.")
    p.add_argument("--load-torque", type=float, default=None, help="Override constant load torque, N*m.")
    p.add_argument("--omega-ref", type=float, default=None, help="Override omega_ref, rad/s.")
    p.add_argument("--omega-ref-pu", type=float, default=0.8, help="Omega_ref as pu of base omega (2*pi*10/p).")
    p.add_argument("--omega-ref-range", type=str, default=None, help="Random omega_ref range, e.g., 20,120 (rad/s).")
    p.add_argument("--omega-ref-pu-range", type=str, default=None, help="Random omega_ref range in pu, e.g., 0.2,1.2.")
    p.add_argument("--scenarios", type=str, default="", help="Comma-separated scenario list (e.g., speed_step,ramp,load_step,start_stop).")
    p.add_argument("--scenario-sample", type=str, default="random", choices=["random", "cycle"])
    p.add_argument("--episode-seeds", type=str, default="", help="Optional comma-separated episode seed cycle for deterministic replay of failing cases.")
    p.add_argument(
        "--scenario-reward-overrides-json",
        type=str,
        default=None,
        help="JSON file or inline JSON with per-scenario reward overrides, e.g. {\"load_step\": {\"w_speed\": 3.0, \"w_power\": 4.0}}.",
    )
    p.add_argument(
        "--seed-scenario-reward-overrides-json",
        type=str,
        default=None,
        help="JSON file or inline JSON with per-seed per-scenario overrides, e.g. {\"505\": {\"start_stop\": {\"w_speed\": 4.0}}}.",
    )
    p.add_argument("--load-torque-range", type=str, default=None, help="Random load torque range, N*m (min,max).")
    p.add_argument("--load-mult-range", type=str, default=None, help="Random load multiplier of env load (min,max).")
    p.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    p.add_argument("--sigma-start", type=float, default=0.2, help="Exploration sigma at episode 0.")
    p.add_argument("--sigma-end", type=float, default=0.05, help="Final exploration sigma.")
    p.add_argument("--sigma-decay-episodes", type=int, default=100, help="Episodes to decay sigma.")
    p.add_argument("--power-warmup-episodes", type=int, default=0, help="Episodes before enabling power penalty.")
    p.add_argument("--power-ramp-episodes", type=int, default=50, help="Episodes to ramp power penalty.")
    p.add_argument("--energy-warmup-episodes", type=int, default=0, help="Episodes before enabling eta/terminal energy terms.")
    p.add_argument("--energy-ramp-episodes", type=int, default=0, help="Episodes to ramp eta/terminal energy terms.")
    p.add_argument("--energy-gate-mode", type=str, default=None, choices=["hard", "soft"], help="Override ai_id_energy_gate_mode from env config.")
    p.add_argument("--energy-gate-min-scale", type=float, default=None, help="Override ai_id_energy_gate_min_scale from env config.")
    p.add_argument("--energy-gate-exponent", type=float, default=None, help="Override ai_id_energy_gate_exponent from env config.")
    p.add_argument("--terminal-energy-bonus", type=float, default=None, help="Override ai_id_terminal_energy_bonus from env config.")
    p.add_argument("--terminal-eta-target", type=float, default=None, help="Override ai_id_terminal_eta_target from env config.")
    p.add_argument("--terminal-shaft-ratio-min", type=float, default=None, help="Override ai_id_terminal_shaft_ratio_min from env config.")
    p.add_argument("--i-soft-limit", type=float, default=None, help="Override i_soft_limit from env config.")
    p.add_argument("--i-soft-penalty", type=float, default=None, help="Override i_soft_penalty from env config.")
    p.add_argument(
        "--include-energy-obs",
        action="store_true",
        help="Add p_in_norm, p_el_filt, p_shaft_norm and eta_norm to observations.",
    )
    p.add_argument(
        "--include-episode-eta-obs",
        action="store_true",
        help="Add running episode eta_energy to observations.",
    )
    p.add_argument("--update-every-episodes", type=int, default=1, help="PPO update frequency in episodes.")
    p.add_argument("--lr", type=float, default=5e-4, help="PPO optimizer learning rate.")
    p.add_argument("--entropy-coef", type=float, default=0.005, help="PPO entropy coefficient.")
    p.add_argument("--actor-anchor-coef", type=float, default=0.0, help="Penalty that anchors actor outputs to the warm-start policy.")
    p.add_argument("--hidden-sizes", type=str, default=None, help="Comma-separated hidden sizes, e.g. 64,64 or 96,96.")
    p.add_argument("--eval-interval", type=int, default=0, help="Run scenario_compare every N episodes (0 disables).")
    p.add_argument("--eval-scenarios", type=str, default="speed_step,ramp,load_step", help="Scenarios for eval.")
    p.add_argument("--eval-dt", type=float, default=None, help="Override dt for eval.")
    p.add_argument("--eval-t-end", type=float, default=None, help="Override t_end for eval.")
    p.add_argument("--eval-window-frac", type=float, default=0.25)
    p.add_argument("--eval-error-tol-rel", type=float, default=0.05)
    p.add_argument("--eval-error-tol-abs", type=float, default=0.0)
    p.add_argument("--eval-use-total-power", action="store_true")
    p.add_argument("--external-step27-select", action="store_true", help="Select the promoted checkpoint by external Step27 objective after training.")
    p.add_argument("--external-step27-motor", type=str, default=None, help="Motor key for Step27 selection (e.g. ao2).")
    p.add_argument("--external-step27-candidate-json", type=str, default=None, help="Candidate JSON used for external Step27 selection.")
    p.add_argument("--external-step27-candidate-index", type=int, default=0, help="Candidate index in --external-step27-candidate-json.")
    p.add_argument("--external-step27-candidate-tag", type=str, default="", help="Candidate tag in --external-step27-candidate-json.")
    p.add_argument("--external-step27-candidate-tags", type=str, default="", help="Comma-separated candidate tags to rank per checkpoint during external Step27 selection.")
    p.add_argument("--external-step27-seeds", type=str, default="101,202,303", help="Comma-separated seed list for external Step27 selection.")
    p.add_argument("--external-step27-scenarios", type=str, default="speed_step,ramp,load_step,start_stop", help="Comma-separated scenario list for external Step27 selection.")
    p.add_argument("--external-step27-seed-perturbation", action="store_true", help="Enable seed perturbation during external Step27 selection.")
    p.add_argument("--external-step27-seed-perturb-level", type=float, default=0.2, help="Seed perturbation level for external Step27 selection.")
    p.add_argument("--external-step27-min-avg-power-saving-pct", type=float, default=0.0, help="Minimum avg_power_saving_pct for external Step27 checkpoint selection.")
    p.add_argument("--external-step27-min-avg-eta-gain-pct", type=float, default=0.0, help="Minimum avg_eta_gain_pct for external Step27 checkpoint selection.")
    p.add_argument("--external-step27-max-avg-eta-gain-pct", type=float, default=25.0, help="Maximum avg_eta_gain_pct for external Step27 checkpoint selection.")
    p.add_argument("--external-step27-max-err-failures", type=float, default=2.0, help="Maximum err_failures for external Step27 checkpoint selection.")
    p.add_argument("--external-step27-min-start-stop-saving-pct", type=float, default=-0.5, help="Minimum start_stop_power_saving_pct for external Step27 checkpoint selection.")
    p.add_argument("--external-step27-max-start-stop-saving-pct", type=float, default=20.0, help="Maximum start_stop_power_saving_pct for external Step27 checkpoint selection.")
    p.add_argument("--external-step27-max-worst-current-peak-ratio", type=float, default=1.30, help="Maximum worst_current_peak_ratio for external Step27 checkpoint selection.")
    p.add_argument("--external-step27-max-worst-current-mean-ratio", type=float, default=1.20, help="Maximum worst_current_mean_ratio for external Step27 checkpoint selection.")
    p.add_argument("--external-step27-use-envelope-acceptance", action="store_true", help="Require canonical acceptance envelopes during external Step27 selection.")
    p.add_argument("--external-step27-acceptance-envelopes", type=str, default=None, help="Optional acceptance envelope JSON path for external Step27 selection.")
    p.add_argument("--external-step27-min-avg-power-saving-pct-min-seed", type=float, default=None, help="Optional minimum avg_power_saving_pct_min_seed for external Step27 checkpoint selection.")
    p.add_argument("--external-step27-min-avg-eta-gain-pct-min-seed", type=float, default=None, help="Optional minimum avg_eta_gain_pct_min_seed for external Step27 checkpoint selection.")
    p.add_argument("--external-step27-max-err-failures-max-seed", type=float, default=None, help="Optional maximum err_failures_max_seed for external Step27 checkpoint selection.")
    p.add_argument("--external-step27-min-start-stop-saving-pct-min-seed", type=float, default=None, help="Optional minimum start_stop_power_saving_pct_min_seed for external Step27 checkpoint selection.")
    p.add_argument("--external-step27-top-k", type=int, default=10, help="How many ranked rows to keep in external Step27 summary.")
    p.add_argument("--external-step27-include-init-checkpoint", action="store_true", help="Also rank the warm-start init checkpoint during external Step27 selection.")
    p.add_argument("--external-step27-resume", action="store_true", help="Resume a previously interrupted external Step27 checkpoint scan in the same run dir.")
    p.add_argument("--output-dir", type=str, default=None, help="Directory for shared checkpoints/episode logs.")
    p.add_argument("--results-root", type=str, default=None, help="Directory for per-run artifacts and eval snapshots.")
    p.add_argument("--init-checkpoint", type=str, default=None, help="Optional actor checkpoint to warm-start training.")
    p.set_defaults(override_omega_ref=True)
    args = p.parse_args()
    omega_ref_override = None
    if args.omega_ref is not None:
        omega_ref_override = float(args.omega_ref)
    elif args.omega_ref_pu is not None:
        env_cfg = make_env_from_config(args.config).env_config
        omega_base = float(2.0 * np.pi * 10.0 / max(env_cfg.motor.p, 1))
        omega_ref_override = float(args.omega_ref_pu) * omega_base

    scenarios = _parse_scenarios(args.scenarios)
    omega_ref_range = _parse_range(args.omega_ref_range)
    omega_ref_pu_range = _parse_range(args.omega_ref_pu_range)
    load_range = _parse_range(args.load_torque_range)
    load_mult_range = _parse_range(args.load_mult_range)
    hidden_sizes = _parse_hidden_sizes(args.hidden_sizes)
    scenario_reward_overrides = _parse_scenario_reward_overrides(args.scenario_reward_overrides_json)
    seed_scenario_reward_overrides = _parse_seed_scenario_reward_overrides(args.seed_scenario_reward_overrides_json)
    episode_seed_cycle = _parse_int_csv(args.episode_seeds) if str(args.episode_seeds).strip() else None
    override_omega_ref = bool(args.override_omega_ref)
    override_load_torque = bool(args.override_load_torque)
    if scenarios:
        override_omega_ref = False
        override_load_torque = False

    env_cfg = make_env_from_config(args.config).env_config
    if omega_ref_range is None and omega_ref_pu_range is not None:
        omega_base = float(2.0 * np.pi * 10.0 / max(env_cfg.motor.p, 1))
        omega_ref_range = (omega_ref_pu_range[0] * omega_base, omega_ref_pu_range[1] * omega_base)
    if load_range is None and load_mult_range is not None:
        base_load = float(getattr(env_cfg.sim, "load_torque", 0.0))
        load_range = (load_mult_range[0] * base_load, load_mult_range[1] * base_load)

    cfg_omega_range = _normalize_range(getattr(env_cfg, "ai_omega_ref_range", None))
    cfg_omega_pu_range = _normalize_range(getattr(env_cfg, "ai_omega_ref_pu_range", None))
    cfg_load_range = _normalize_range(getattr(env_cfg, "ai_load_torque_range", None))
    cfg_load_mult = _normalize_range(getattr(env_cfg, "ai_load_mult_range", None))
    if omega_ref_range is None and cfg_omega_range is not None:
        omega_ref_range = cfg_omega_range
    if omega_ref_range is None and cfg_omega_pu_range is not None:
        omega_base = float(2.0 * np.pi * 10.0 / max(env_cfg.motor.p, 1))
        omega_ref_range = (cfg_omega_pu_range[0] * omega_base, cfg_omega_pu_range[1] * omega_base)
    if load_range is None and cfg_load_range is not None:
        load_range = cfg_load_range
    if load_range is None and cfg_load_mult is not None:
        base_load = float(getattr(env_cfg.sim, "load_torque", 0.0))
        load_range = (cfg_load_mult[0] * base_load, cfg_load_mult[1] * base_load)

    train(
        env_config=args.config,
        episodes=args.episodes,
        episode_steps=args.episode_steps,
        control_mode=str(args.control_mode),
        w_speed=args.w_speed,
        w_power=args.w_power,
        w_current=args.w_current,
        w_smooth=args.w_smooth,
        w_mag=args.w_mag,
        w_shaft=args.w_shaft,
        w_eta=args.w_eta,
        w_eta_episode=args.w_eta_episode,
        eta_clip=args.eta_clip,
        id_ref_alpha=float(args.id_ref_alpha),
        id_ref_rate_limit=None if args.id_ref_rate_limit is None else float(args.id_ref_rate_limit),
        ai_id_speed_tol=float(args.ai_id_speed_tol),
        ai_id_speed_tol_rel=None if args.ai_id_speed_tol_rel is None else float(args.ai_id_speed_tol_rel),
        id_ref_gate_speed_tol=None if args.id_ref_gate_speed_tol is None else float(args.id_ref_gate_speed_tol),
        id_ref_gate_speed_tol_rel=None if args.id_ref_gate_speed_tol_rel is None else float(args.id_ref_gate_speed_tol_rel),
        id_ref_gate_min_scale=float(args.id_ref_gate_min_scale),
        id_ref_gate_exponent=float(args.id_ref_gate_exponent),
        fast=bool(args.fast),
        time_budget_min=args.time_budget_min,
        override_load_torque=override_load_torque,
        override_omega_ref=override_omega_ref,
        ai_id_ref_relative=bool(args.relative),
        delta_id_max=float(args.delta_id_max),
        load_torque=None if args.load_torque is None else float(args.load_torque),
        omega_ref_override=omega_ref_override,
        scenarios=scenarios,
        scenario_sample=str(args.scenario_sample),
        episode_seed_cycle=episode_seed_cycle,
        omega_ref_range=omega_ref_range,
        load_torque_range=load_range,
        seed=args.seed,
        sigma_start=float(args.sigma_start),
        sigma_end=float(args.sigma_end),
        sigma_decay_episodes=int(args.sigma_decay_episodes),
        power_warmup_episodes=int(args.power_warmup_episodes),
        power_ramp_episodes=int(args.power_ramp_episodes),
        energy_warmup_episodes=int(args.energy_warmup_episodes),
        energy_ramp_episodes=int(args.energy_ramp_episodes),
        eval_interval=int(args.eval_interval),
        eval_scenarios=str(args.eval_scenarios),
        eval_dt=None if args.eval_dt is None else float(args.eval_dt),
        eval_t_end=None if args.eval_t_end is None else float(args.eval_t_end),
        eval_window_frac=float(args.eval_window_frac),
        eval_error_tol_rel=float(args.eval_error_tol_rel),
        eval_error_tol_abs=float(args.eval_error_tol_abs),
        eval_use_total_power=bool(args.eval_use_total_power),
        include_energy_obs=bool(args.include_energy_obs),
        include_episode_eta_obs=bool(args.include_episode_eta_obs),
        update_every_episodes=int(args.update_every_episodes),
        lr=float(args.lr),
        entropy_coef=float(args.entropy_coef),
        actor_anchor_coef=float(args.actor_anchor_coef),
        external_step27_select=bool(args.external_step27_select),
        external_step27_motor=args.external_step27_motor,
        external_step27_candidate_json=args.external_step27_candidate_json,
        external_step27_candidate_index=int(args.external_step27_candidate_index),
        external_step27_candidate_tag=str(args.external_step27_candidate_tag),
        external_step27_candidate_tags=str(args.external_step27_candidate_tags),
        external_step27_seeds=str(args.external_step27_seeds),
        external_step27_scenarios=str(args.external_step27_scenarios),
        external_step27_seed_perturbation=bool(args.external_step27_seed_perturbation),
        external_step27_seed_perturb_level=float(args.external_step27_seed_perturb_level),
        external_step27_min_avg_power_saving_pct=float(args.external_step27_min_avg_power_saving_pct),
        external_step27_min_avg_eta_gain_pct=float(args.external_step27_min_avg_eta_gain_pct),
        external_step27_max_avg_eta_gain_pct=float(args.external_step27_max_avg_eta_gain_pct),
        external_step27_max_err_failures=float(args.external_step27_max_err_failures),
        external_step27_min_start_stop_saving_pct=float(args.external_step27_min_start_stop_saving_pct),
        external_step27_max_start_stop_saving_pct=float(args.external_step27_max_start_stop_saving_pct),
        external_step27_max_worst_current_peak_ratio=float(args.external_step27_max_worst_current_peak_ratio),
        external_step27_max_worst_current_mean_ratio=float(args.external_step27_max_worst_current_mean_ratio),
        external_step27_use_envelope_acceptance=bool(args.external_step27_use_envelope_acceptance),
        external_step27_acceptance_envelopes=None
        if args.external_step27_acceptance_envelopes is None
        else str(args.external_step27_acceptance_envelopes),
        external_step27_min_avg_power_saving_pct_min_seed=None
        if args.external_step27_min_avg_power_saving_pct_min_seed is None
        else float(args.external_step27_min_avg_power_saving_pct_min_seed),
        external_step27_min_avg_eta_gain_pct_min_seed=None
        if args.external_step27_min_avg_eta_gain_pct_min_seed is None
        else float(args.external_step27_min_avg_eta_gain_pct_min_seed),
        external_step27_max_err_failures_max_seed=None
        if args.external_step27_max_err_failures_max_seed is None
        else float(args.external_step27_max_err_failures_max_seed),
        external_step27_min_start_stop_saving_pct_min_seed=None
        if args.external_step27_min_start_stop_saving_pct_min_seed is None
        else float(args.external_step27_min_start_stop_saving_pct_min_seed),
        external_step27_top_k=int(args.external_step27_top_k),
        external_step27_include_init_checkpoint=bool(args.external_step27_include_init_checkpoint),
        external_step27_resume=bool(args.external_step27_resume),
        init_checkpoint=args.init_checkpoint,
        output_dir=args.output_dir,
        results_root=args.results_root,
        energy_gate_mode=args.energy_gate_mode,
        energy_gate_min_scale=None if args.energy_gate_min_scale is None else float(args.energy_gate_min_scale),
        energy_gate_exponent=None if args.energy_gate_exponent is None else float(args.energy_gate_exponent),
        terminal_energy_bonus=None if args.terminal_energy_bonus is None else float(args.terminal_energy_bonus),
        terminal_eta_target=None if args.terminal_eta_target is None else float(args.terminal_eta_target),
        terminal_shaft_ratio_min=None if args.terminal_shaft_ratio_min is None else float(args.terminal_shaft_ratio_min),
        i_soft_limit=None if args.i_soft_limit is None else float(args.i_soft_limit),
        i_soft_penalty=None if args.i_soft_penalty is None else float(args.i_soft_penalty),
        hidden_sizes_override=hidden_sizes,
        scenario_reward_overrides=scenario_reward_overrides,
        seed_scenario_reward_overrides=seed_scenario_reward_overrides,
    )


if __name__ == "__main__":
    main()
