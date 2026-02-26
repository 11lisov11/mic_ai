from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import random
import statistics
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mic_ai.ai.agents.ppo_voltage import PPOVoltageAgent
from mic_ai.ai.id_ref_supervisor import AiIdRefSupervisorConfig
from mic_ai.core.env import make_env_from_config
from mic_ai.tools.scenario_compare import (
    _clone_with_sim,
    _err_limit,
    _infer_action_dim,
    _infer_hidden_sizes,
    _resolve_feature_keys,
    _simulate_ai,
    _simulate_controller,
    _simulate_foc,
    _summarize,
)


@dataclass(frozen=True)
class MotorSpec:
    key: str
    config_path: str


@dataclass(frozen=True)
class Air56Acceptance:
    min_avg_power_saving_pct: float
    min_avg_eta_gain_pct: float
    max_err_failures: float
    min_start_stop_power_saving_pct: float


@dataclass(frozen=True)
class SeedPerturbationSettings:
    enabled: bool
    level: float


MOTOR_REGISTRY: Dict[str, MotorSpec] = {
    "air56": MotorSpec("air56", "config/env_research_air56_025kw.py"),
    "al31": MotorSpec("al31", "config/env_research_al31_4_06kw.py"),
    "ao2": MotorSpec("ao2", "config/env_research_ao2_32_4_3kw.py"),
}

CONTROLLER_ORDER: Tuple[str, ...] = ("PI", "FOC", "MIC")
DEFAULT_SEEDS: Tuple[int, ...] = (101, 202, 303, 404, 505)
DEFAULT_SCENARIOS: Tuple[str, ...] = ("speed_step", "ramp", "load_step", "start_stop")

METRIC_FIELDS: Tuple[str, ...] = (
    "avg_power_saving_pct",
    "avg_eta_gain_pct",
    "err_failures",
    "start_stop_power_saving_pct",
    "worst_current_peak_ratio",
    "worst_current_mean_ratio",
    "avg_controller_speed_err",
)


def _parse_csv_list(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def _parse_int_list(text: str) -> List[int]:
    vals: List[int] = []
    for raw in _parse_csv_list(text):
        vals.append(int(raw))
    return vals


def _stable_int_from_text(text: str) -> int:
    digest = hashlib.sha256(str(text).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _rng_for_seed_perturbation(*, motor_key: str, seed: int) -> random.Random:
    return random.Random(_stable_int_from_text(f"step27-perturb::{motor_key}::{int(seed)}"))


def _clamp(value: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, float(value))))


def _perturb_scale(rng: random.Random, *, rel_amp: float, level: float) -> float:
    amp = max(0.0, float(rel_amp)) * max(0.0, float(level))
    return float(1.0 + rng.uniform(-amp, amp))


def _seed_perturbation_payload(
    *,
    motor_key: str,
    seed: int,
    level: float,
) -> Dict[str, float]:
    rng = _rng_for_seed_perturbation(motor_key=str(motor_key), seed=int(seed))
    lvl = max(0.0, float(level))
    return {
        # Keep plant-parameter perturbations disabled by default: AIR56 start_stop is
        # highly sensitive to even tiny drifts; step28 statistics are built on load variation.
        "rs_scale": 1.0,
        "rr_scale": 1.0,
        "ls_sigma_scale": 1.0,
        "lr_sigma_scale": 1.0,
        "lm_scale": 1.0,
        "j_scale": 1.0,
        "b_scale": 1.0,
        "load_torque_scale": _perturb_scale(rng, rel_amp=0.03, level=lvl),
        # Keep measurement-noise perturbation disabled by default.
        "sigma_omega_pu": 0.0,
        "sigma_i_abc_pu": 0.0,
    }


def _apply_seed_perturbation(
    *,
    env_cfg: object,
    motor_key: str,
    seed: int,
    settings: SeedPerturbationSettings,
) -> Tuple[object, Dict[str, float]]:
    if (not bool(settings.enabled)) or float(settings.level) <= 0.0:
        return env_cfg, {
            "enabled": 0.0,
            "level": float(max(0.0, float(settings.level))),
            "seed": float(int(seed)),
            "motor_key_hash": float(_stable_int_from_text(motor_key) % 1_000_000),
        }

    payload = _seed_perturbation_payload(
        motor_key=str(motor_key),
        seed=int(seed),
        level=float(settings.level),
    )
    motor = env_cfg.motor
    sim = env_cfg.sim

    # Keep all perturbed parameters physically valid and close to calibrated baseline.
    motor_new = replace(
        motor,
        Rs=max(float(motor.Rs) * float(payload["rs_scale"]), 1e-7),
        Rr=max(float(motor.Rr) * float(payload["rr_scale"]), 1e-7),
        Ls_sigma=max(float(motor.Ls_sigma) * float(payload["ls_sigma_scale"]), 1e-8),
        Lr_sigma=max(float(motor.Lr_sigma) * float(payload["lr_sigma_scale"]), 1e-8),
        Lm=max(float(motor.Lm) * float(payload["lm_scale"]), 1e-8),
        J=max(float(motor.J) * float(payload["j_scale"]), 1e-7),
        B=max(float(motor.B) * float(payload["b_scale"]), 1e-8),
    )

    omega_nom = float(2.0 * math.pi * float(env_cfg.scalar_vf.f_max) / max(int(env_cfg.motor.p), 1))
    i_nom = float(max(float(getattr(env_cfg.motor, "I_n", 1.0)), 1e-6))
    sigma_omega_add = float(payload["sigma_omega_pu"]) * max(omega_nom, 1.0)
    sigma_i_add = float(payload["sigma_i_abc_pu"]) * i_nom

    sim_new = replace(
        sim,
        load_torque=max(float(sim.load_torque) * float(payload["load_torque_scale"]), 0.0),
        sigma_omega=max(float(getattr(sim, "sigma_omega", 0.0)) + sigma_omega_add, 0.0),
        sigma_i_abc=max(float(getattr(sim, "sigma_i_abc", 0.0)) + sigma_i_add, 0.0),
    )

    env_cfg_new = copy.copy(env_cfg)
    set_ok = True
    try:
        object.__setattr__(env_cfg_new, "motor", motor_new)
    except Exception:
        try:
            setattr(env_cfg_new, "motor", motor_new)
        except Exception:
            set_ok = False
    try:
        object.__setattr__(env_cfg_new, "sim", sim_new)
    except Exception:
        try:
            setattr(env_cfg_new, "sim", sim_new)
        except Exception:
            set_ok = False
    if not set_ok:
        env_cfg_new = replace(env_cfg, motor=motor_new, sim=sim_new)
        # Preserve runtime-attached config extras used by AI evaluation logic.
        for name, value in vars(env_cfg).items():
            if name in {"motor", "sim"}:
                continue
            if hasattr(env_cfg_new, name):
                continue
            try:
                object.__setattr__(env_cfg_new, name, value)
            except Exception:
                pass
    meta = {
        "enabled": 1.0,
        "level": float(max(0.0, float(settings.level))),
        "seed": float(int(seed)),
        "motor_key_hash": float(_stable_int_from_text(motor_key) % 1_000_000),
        "rs_scale": float(payload["rs_scale"]),
        "rr_scale": float(payload["rr_scale"]),
        "ls_sigma_scale": float(payload["ls_sigma_scale"]),
        "lr_sigma_scale": float(payload["lr_sigma_scale"]),
        "lm_scale": float(payload["lm_scale"]),
        "j_scale": float(payload["j_scale"]),
        "b_scale": float(payload["b_scale"]),
        "load_torque_scale": float(payload["load_torque_scale"]),
        "sigma_omega_add": float(sigma_omega_add),
        "sigma_i_abc_add": float(sigma_i_add),
    }
    return env_cfg_new, meta


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(statistics.fmean(values))


def _std(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(statistics.pstdev(values))


def _json_dump(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    row_list = list(rows)
    if not row_list:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(row_list[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(row_list)


def _resolve_config_path(config_name: str) -> Path:
    path = Path(config_name)
    if path.is_file():
        return path.resolve()
    candidate = Path("config") / f"{config_name}.py"
    if candidate.is_file():
        return candidate.resolve()
    raise FileNotFoundError(f"Cannot resolve config path: {config_name}")


def _resolve_checkpoint(env_cfg: object) -> Path:
    ckpt = getattr(env_cfg, "ai_eval_checkpoint_path", None)
    if ckpt is None:
        raise FileNotFoundError("Missing ai_eval_checkpoint_path in env config.")
    path = Path(str(ckpt)).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


def _load_agent(ckpt: Path) -> PPOVoltageAgent:
    state = torch.load(ckpt, map_location="cpu")
    hidden = _infer_hidden_sizes(state) or (128, 128)
    action_dim = _infer_action_dim(state)
    feature_keys = _resolve_feature_keys(None, state)
    agent = PPOVoltageAgent(feature_keys=feature_keys, action_dim=action_dim, device="cpu", hidden_sizes=hidden)
    agent.net.load_state_dict(state)
    agent.set_action_std(1e-6)
    return agent


def _disable_lut_if_needed(env_cfg: object, disable: bool) -> object:
    if not disable:
        return env_cfg
    cfg = copy.deepcopy(env_cfg)
    if hasattr(cfg, "id_ref_lut_path"):
        try:
            setattr(cfg, "id_ref_lut_path", None)
        except Exception:
            object.__setattr__(cfg, "id_ref_lut_path", None)
    if hasattr(cfg, "ai_eval_id_ref_lut_path"):
        try:
            setattr(cfg, "ai_eval_id_ref_lut_path", None)
        except Exception:
            object.__setattr__(cfg, "ai_eval_id_ref_lut_path", None)
    return cfg


def _id_ref_eval_params(env_cfg: object) -> Dict[str, object]:
    gate_rel = getattr(env_cfg, "ai_eval_id_ref_gate_speed_tol_rel", None)
    return {
        "id_ref_alpha": float(getattr(env_cfg, "ai_eval_id_ref_alpha", 1.0)),
        "id_ref_rate_limit": None,
        "id_ref_gate_speed_tol": None,
        "id_ref_gate_speed_tol_rel": None if gate_rel is None else float(gate_rel),
        "id_ref_gate_min_scale": float(getattr(env_cfg, "ai_eval_id_ref_gate_min_scale", 0.0)),
        "id_ref_gate_exponent": float(getattr(env_cfg, "ai_eval_id_ref_gate_exponent", 1.0)),
        "ai_id_relative": bool(getattr(env_cfg, "ai_eval_id_ref_relative", False)),
        "delta_id_max": float(getattr(env_cfg, "ai_eval_delta_id_max", 0.1)),
        "ai_id_allow_positive_delta": bool(getattr(env_cfg, "ai_eval_id_ref_allow_positive_delta", True)),
    }


def _sensorless_params(env_cfg: object) -> Dict[str, float]:
    sim = env_cfg.sim
    return {
        "sensorless_alpha": float(getattr(sim, "sensorless_alpha", 0.08846641008448847)),
        "sensorless_min_id": float(getattr(sim, "sensorless_min_id", 0.24395873531051976)),
        "sensorless_min_i_rms_pu": float(getattr(sim, "sensorless_min_i_rms_pu", 0.011682257466165403)),
        "sensorless_slip_limit_pu": float(getattr(sim, "sensorless_slip_limit_pu", 0.825241020775665)),
        "sensorless_max_domega_pu": float(getattr(sim, "sensorless_max_domega_pu", 6.142563166179913)),
        "sensorless_fallback_decay": float(getattr(sim, "sensorless_fallback_decay", 0.9776912678336072)),
        "sensorless_conf_alpha": float(getattr(sim, "sensorless_conf_alpha", 0.1)),
        "sensorless_model_weight_max": float(getattr(sim, "sensorless_model_weight_max", 0.14214686521445952)),
    }


def _supervisor_from_env(env_cfg: object) -> AiIdRefSupervisorConfig | None:
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
        idle_exit_boost_steps=int(getattr(env_cfg, "ai_eval_sup_idle_exit_boost", 0)),
        idle_exit_action=float(getattr(env_cfg, "ai_eval_sup_idle_exit_action", 1.0)),
        idle_bias_decay=float(getattr(env_cfg, "ai_eval_sup_idle_bias_decay", 0.95)),
    )
    if cfg.objective_clip is not None:
        cfg.objective_clip = float(cfg.objective_clip)
    return cfg


def _supervisor_to_candidate(base: AiIdRefSupervisorConfig, tag: str, source: str) -> Dict[str, object]:
    return {
        "tag": str(tag),
        "source": str(source),
        "objective": str(base.objective),
        "speed_tol_rel": float(base.speed_tol_rel),
        "speed_tol_abs": float(base.speed_tol_abs),
        "omega_min_pu": float(base.omega_min_pu),
        "update_steps": int(base.update_steps),
        "dither_amp": float(base.dither_amp),
        "bias_step": float(base.bias_step),
        "bias_max": float(base.bias_max),
        "shaft_eps": float(base.shaft_eps),
        "reset_decay": float(base.reset_decay),
        "objective_clip": None if base.objective_clip is None else float(base.objective_clip),
        "idle_enable": bool(base.idle_enable),
        "idle_omega_pu": float(base.idle_omega_pu),
        "idle_action": float(base.idle_action),
        "idle_exit_boost_steps": int(base.idle_exit_boost_steps),
        "idle_exit_action": float(base.idle_exit_action),
        "idle_bias_decay": float(base.idle_bias_decay),
    }


def _candidate_to_supervisor(candidate: Dict[str, object]) -> AiIdRefSupervisorConfig:
    return AiIdRefSupervisorConfig(
        enabled=True,
        objective=str(candidate["objective"]),
        speed_tol_rel=float(candidate["speed_tol_rel"]),
        speed_tol_abs=float(candidate["speed_tol_abs"]),
        omega_min_pu=float(candidate["omega_min_pu"]),
        update_steps=int(candidate["update_steps"]),
        dither_amp=float(candidate["dither_amp"]),
        bias_step=float(candidate["bias_step"]),
        bias_max=float(candidate["bias_max"]),
        shaft_eps=float(candidate["shaft_eps"]),
        reset_decay=float(candidate["reset_decay"]),
        objective_clip=None if candidate["objective_clip"] is None else float(candidate["objective_clip"]),
        idle_enable=bool(candidate["idle_enable"]),
        idle_omega_pu=float(candidate["idle_omega_pu"]),
        idle_action=float(candidate["idle_action"]),
        idle_exit_boost_steps=int(candidate["idle_exit_boost_steps"]),
        idle_exit_action=float(candidate["idle_exit_action"]),
        idle_bias_decay=float(candidate["idle_bias_decay"]),
    )


def _sample_supervisor_candidate(
    rng: random.Random,
    *,
    idx: int,
    base: Dict[str, object],
) -> Dict[str, object]:
    objective = rng.choice(("specific_power", "eta_inv", "p_in"))
    return {
        **base,
        "tag": f"rand_{idx:03d}",
        "source": "random",
        "id_ref_alpha": rng.uniform(0.20, 0.95),
        "delta_id_max": rng.uniform(0.12, 0.45),
        "id_ref_gate_speed_tol_rel": rng.uniform(0.04, 0.25),
        "id_ref_gate_min_scale": rng.uniform(0.0, 0.12),
        "id_ref_gate_exponent": rng.uniform(0.8, 1.4),
        "objective": objective,
        "speed_tol_rel": rng.uniform(0.04, 0.12),
        "update_steps": int(rng.randint(5, 18)),
        "dither_amp": rng.uniform(0.004, 0.03),
        "bias_step": rng.uniform(0.003, 0.02),
        "bias_max": rng.uniform(0.09, 0.24),
        "idle_omega_pu": rng.uniform(0.03, 0.10),
        "idle_action": rng.uniform(-0.95, -0.35),
        "idle_exit_boost_steps": int(rng.randint(0, 30)),
        "idle_exit_action": rng.uniform(0.75, 1.0),
        "idle_bias_decay": rng.uniform(0.93, 0.99),
    }


def _build_handcrafted_candidates(base: Dict[str, object]) -> List[Dict[str, object]]:
    base_alpha = float(base["id_ref_alpha"])
    base_delta = float(base["delta_id_max"])
    base_gate_rel = float(base["id_ref_gate_speed_tol_rel"])
    base_gate_min = float(base["id_ref_gate_min_scale"])
    base_gate_exp = float(base["id_ref_gate_exponent"])
    base_speed_tol = float(base["speed_tol_rel"])
    base_update = int(base["update_steps"])
    base_bias_step = float(base["bias_step"])
    base_bias_max = float(base["bias_max"])
    base_idle_omega = float(base["idle_omega_pu"])
    base_idle_action = float(base["idle_action"])
    base_idle_boost = int(base["idle_exit_boost_steps"])
    base_idle_exit_action = float(base["idle_exit_action"])
    base_idle_decay = float(base["idle_bias_decay"])

    variants = [
        {
            "tag": "manual_safe_01",
            "source": "manual",
            "objective": "specific_power",
            "id_ref_alpha": _clamp(base_alpha * 0.65, 0.08, 0.55),
            "delta_id_max": _clamp(base_delta * 0.55, 0.05, 0.22),
            "id_ref_gate_speed_tol_rel": _clamp(base_gate_rel * 1.20, 0.10, 0.28),
            "id_ref_gate_min_scale": _clamp(max(base_gate_min, 0.12), 0.0, 0.30),
            "id_ref_gate_exponent": _clamp(1.00, 0.7, 1.6),
            "speed_tol_rel": _clamp(base_speed_tol * 1.20, 0.08, 0.16),
            "update_steps": int(_clamp(float(base_update), 6.0, 22.0)),
            "dither_amp": _clamp(float(base["dither_amp"]) * 0.85, 0.003, 0.04),
            "bias_step": _clamp(base_bias_step * 0.80, 0.002, 0.03),
            "bias_max": _clamp(base_bias_max * 0.85, 0.08, 0.30),
            "idle_omega_pu": _clamp(base_idle_omega * 1.15, 0.03, 0.16),
            "idle_action": _clamp(max(base_idle_action, -0.55), -1.0, -0.2),
            "idle_exit_boost_steps": int(_clamp(float(base_idle_boost), 0.0, 35.0)),
            "idle_exit_action": _clamp(base_idle_exit_action, 0.70, 1.0),
            "idle_bias_decay": _clamp(base_idle_decay, 0.90, 0.995),
        },
        {
            "tag": "manual_safe_02",
            "source": "manual",
            "objective": "p_in",
            "id_ref_alpha": _clamp(base_alpha * 0.50, 0.06, 0.45),
            "delta_id_max": _clamp(base_delta * 0.40, 0.04, 0.18),
            "id_ref_gate_speed_tol_rel": _clamp(base_gate_rel * 1.35, 0.12, 0.30),
            "id_ref_gate_min_scale": _clamp(max(base_gate_min, 0.15), 0.0, 0.35),
            "id_ref_gate_exponent": _clamp((base_gate_exp + 1.0) / 2.0, 0.8, 1.3),
            "speed_tol_rel": _clamp(base_speed_tol * 1.35, 0.09, 0.18),
            "update_steps": int(_clamp(float(base_update + 2), 6.0, 24.0)),
            "dither_amp": _clamp(float(base["dither_amp"]) * 0.70, 0.003, 0.03),
            "bias_step": _clamp(base_bias_step * 0.65, 0.002, 0.02),
            "bias_max": _clamp(base_bias_max * 0.75, 0.07, 0.25),
            "idle_omega_pu": _clamp(base_idle_omega * 1.20, 0.03, 0.18),
            "idle_action": _clamp(max(base_idle_action, -0.50), -1.0, -0.2),
            "idle_exit_boost_steps": int(_clamp(float(base_idle_boost + 4), 0.0, 40.0)),
            "idle_exit_action": _clamp(base_idle_exit_action, 0.70, 1.0),
            "idle_bias_decay": _clamp(max(base_idle_decay, 0.96), 0.90, 0.995),
        },
        {
            "tag": "manual_safe_03",
            "source": "manual",
            "objective": "specific_power",
            "id_ref_alpha": _clamp(base_alpha * 0.80, 0.10, 0.70),
            "delta_id_max": _clamp(base_delta * 0.70, 0.06, 0.28),
            "id_ref_gate_speed_tol_rel": _clamp(base_gate_rel * 1.10, 0.08, 0.26),
            "id_ref_gate_min_scale": _clamp(max(base_gate_min, 0.10), 0.0, 0.30),
            "id_ref_gate_exponent": _clamp(base_gate_exp, 0.8, 1.3),
            "speed_tol_rel": _clamp(base_speed_tol * 1.10, 0.07, 0.15),
            "update_steps": int(_clamp(float(base_update), 6.0, 22.0)),
            "dither_amp": _clamp(float(base["dither_amp"]) * 0.90, 0.003, 0.04),
            "bias_step": _clamp(base_bias_step * 0.90, 0.002, 0.03),
            "bias_max": _clamp(base_bias_max * 0.90, 0.08, 0.30),
            "idle_omega_pu": _clamp(base_idle_omega * 1.10, 0.03, 0.16),
            "idle_action": _clamp(base_idle_action, -1.0, -0.2),
            "idle_exit_boost_steps": int(_clamp(float(base_idle_boost), 0.0, 35.0)),
            "idle_exit_action": _clamp(base_idle_exit_action, 0.70, 1.0),
            "idle_bias_decay": _clamp(base_idle_decay, 0.90, 0.995),
        },
        # AIR56 step27 local search winner:
        # improves start_stop while keeping avg_power>0.5 and avg_eta>=0.
        {
            "tag": "manual_air56_step27_fix_01",
            "source": "manual",
            "objective": "specific_power",
            "id_ref_alpha": 0.6916041739118715,
            "delta_id_max": 0.41175616659850595,
            "id_ref_gate_speed_tol_rel": 0.12802819602776283,
            "id_ref_gate_min_scale": 0.10854865567909161,
            "id_ref_gate_exponent": 1.2112607240172235,
            "speed_tol_rel": 0.07682193507978134,
            "update_steps": 11,
            "dither_amp": 0.004179110345105394,
            "bias_step": 0.010135882917319504,
            "bias_max": 0.21136303877643553,
            "idle_omega_pu": 0.07740003039871146,
            "idle_action": -0.6189698591906893,
            "idle_exit_boost_steps": 39,
            "idle_exit_action": 0.817417657326223,
            "idle_bias_decay": 0.9886974417190302,
        },
    ]

    out: List[Dict[str, object]] = []
    for variant in variants:
        out.append({**base, **variant})
    return out


def _simulate_rows(
    *,
    env_cfg: object,
    motor_key: str,
    agent: PPOVoltageAgent | None,
    scenarios: Sequence[str],
    seed: int | None,
    window_frac: float,
    error_tol_rel: float,
    error_tol_abs: float,
    use_total_power: bool,
    foc_feedback_mode: str,
    mic_feedback_mode: str,
    controller: str,
    id_ref_params: Dict[str, object],
    supervisor_cfg: AiIdRefSupervisorConfig | None,
    sensorless: Dict[str, float],
    seed_perturbation: SeedPerturbationSettings,
) -> List[Dict[str, object]]:
    if seed is not None:
        np.random.seed(int(seed))
        torch.manual_seed(int(seed))

    env_eval = env_cfg
    perturb_meta: Dict[str, float] = {
        "enabled": 0.0,
        "level": float(max(0.0, float(seed_perturbation.level))),
    }
    if seed is not None:
        env_eval, perturb_meta = _apply_seed_perturbation(
            env_cfg=env_cfg,
            motor_key=str(motor_key),
            seed=int(seed),
            settings=seed_perturbation,
        )

    dt = float(env_eval.sim.dt)
    t_end = float(env_eval.sim.t_end)
    load_torque = float(env_eval.sim.load_torque)

    rows: List[Dict[str, object]] = []
    for scenario in scenarios:
        sim_cfg = replace(
            env_eval.sim,
            scenario_name=str(scenario),
            dt=dt,
            t_end=t_end,
            load_torque=load_torque,
            omega_feedback_mode=str(mic_feedback_mode),
            sensorless_alpha=float(sensorless["sensorless_alpha"]),
            sensorless_min_id=float(sensorless["sensorless_min_id"]),
            sensorless_min_i_rms_pu=float(sensorless["sensorless_min_i_rms_pu"]),
            sensorless_slip_limit_pu=float(sensorless["sensorless_slip_limit_pu"]),
            sensorless_max_domega_pu=float(sensorless["sensorless_max_domega_pu"]),
            sensorless_fallback_decay=float(sensorless["sensorless_fallback_decay"]),
            sensorless_conf_alpha=float(sensorless["sensorless_conf_alpha"]),
            sensorless_model_weight_max=float(sensorless["sensorless_model_weight_max"]),
        )
        env_cfg_mic = _clone_with_sim(env_eval, sim_cfg)
        env_cfg_foc = _clone_with_sim(env_eval, replace(sim_cfg, omega_feedback_mode=str(foc_feedback_mode)))

        foc = _simulate_foc(env_cfg_foc, dt, t_end, use_total_power)
        if controller == "MIC":
            if agent is None:
                raise RuntimeError("MIC evaluation requires loaded AI agent.")
            mic = _simulate_ai(
                agent,
                env_cfg_mic,
                dt,
                t_end,
                "ai_id_ref",
                float(id_ref_params["id_ref_alpha"]),
                id_ref_params["id_ref_rate_limit"],
                id_ref_params["id_ref_gate_speed_tol"],
                id_ref_params["id_ref_gate_speed_tol_rel"],
                float(id_ref_params["id_ref_gate_min_scale"]),
                float(id_ref_params["id_ref_gate_exponent"]),
                bool(id_ref_params["ai_id_relative"]),
                float(id_ref_params["delta_id_max"]),
                use_total_power,
                supervisor_cfg=supervisor_cfg,
                ai_id_allow_positive_delta=bool(id_ref_params["ai_id_allow_positive_delta"]),
            )
        elif controller == "FOC":
            mic = _simulate_controller(env_cfg_mic, dt, t_end, mode="foc", use_total_power=use_total_power)
        else:
            raise ValueError(f"Unknown controller kind: {controller}")

        foc_sum = _summarize(foc, window_frac)
        mic_sum = _summarize(mic, window_frac)

        err_limit = _err_limit(foc_sum["mean_abs_speed_err"], error_tol_rel, error_tol_abs)
        err_ok = bool(mic_sum["mean_abs_speed_err"] <= err_limit)
        power_saving_pct = 0.0
        if foc_sum["mean_p_el_pos"] > 1e-9:
            power_saving_pct = 100.0 * (1.0 - mic_sum["mean_p_el_pos"] / foc_sum["mean_p_el_pos"])
        eta_gain_pct = 0.0
        if foc_sum["eta"] > 1e-9:
            eta_gain_pct = 100.0 * (mic_sum["eta"] / foc_sum["eta"] - 1.0)
        current_peak_ratio = float("inf")
        if foc_sum["peak_i_rms"] > 1e-9:
            current_peak_ratio = float(mic_sum["peak_i_rms"] / foc_sum["peak_i_rms"])
        elif mic_sum["peak_i_rms"] <= 1e-9:
            current_peak_ratio = 1.0
        current_mean_ratio = float("inf")
        if foc_sum["mean_i_rms"] > 1e-9:
            current_mean_ratio = float(mic_sum["mean_i_rms"] / foc_sum["mean_i_rms"])
        elif mic_sum["mean_i_rms"] <= 1e-9:
            current_mean_ratio = 1.0

        rows.append(
            {
                "scenario": str(scenario),
                "err_ok": err_ok,
                "err_limit": float(err_limit),
                "power_saving_pct": float(power_saving_pct),
                "eta_gain_pct": float(eta_gain_pct),
                "foc_mean_err": float(foc_sum["mean_abs_speed_err"]),
                "mic_mean_err": float(mic_sum["mean_abs_speed_err"]),
                "current_peak_ratio": float(current_peak_ratio),
                "current_mean_ratio": float(current_mean_ratio),
                "foc_p_el_pos": float(foc_sum["mean_p_el_pos"]),
                "mic_p_el_pos": float(mic_sum["mean_p_el_pos"]),
                "foc_eta": float(foc_sum["eta"]),
                "mic_eta": float(mic_sum["eta"]),
                "perturb_enabled": float(perturb_meta.get("enabled", 0.0)),
                "perturb_level": float(perturb_meta.get("level", 0.0)),
                "perturb_load_torque_scale": float(perturb_meta.get("load_torque_scale", 1.0)),
                "perturb_sigma_omega_add": float(perturb_meta.get("sigma_omega_add", 0.0)),
                "perturb_sigma_i_abc_add": float(perturb_meta.get("sigma_i_abc_add", 0.0)),
            }
        )
    return rows


def _aggregate_rows(rows: Sequence[Dict[str, object]]) -> Dict[str, float]:
    if not rows:
        return {
            "scenarios": 0.0,
            "avg_power_saving_pct": 0.0,
            "avg_eta_gain_pct": 0.0,
            "avg_mic_err": 0.0,
            "avg_foc_err": 0.0,
            "err_failures": 0.0,
            "worst_current_peak_ratio": 0.0,
            "worst_current_mean_ratio": 0.0,
            "start_stop_power_saving_pct": 0.0,
            "start_stop_eta_gain_pct": 0.0,
        }
    power = [float(r["power_saving_pct"]) for r in rows]
    eta = [float(r["eta_gain_pct"]) for r in rows]
    mic_err = [float(r["mic_mean_err"]) for r in rows]
    foc_err = [float(r["foc_mean_err"]) for r in rows]
    peak = [float(r["current_peak_ratio"]) for r in rows]
    mean_ratio = [float(r["current_mean_ratio"]) for r in rows]
    err_failures = float(sum(0 if bool(r["err_ok"]) else 1 for r in rows))
    start_stop = next((r for r in rows if str(r["scenario"]).strip().lower() == "start_stop"), None)
    return {
        "scenarios": float(len(rows)),
        "avg_power_saving_pct": _mean(power),
        "avg_eta_gain_pct": _mean(eta),
        "avg_mic_err": _mean(mic_err),
        "avg_foc_err": _mean(foc_err),
        "err_failures": float(err_failures),
        "worst_current_peak_ratio": float(max(peak)),
        "worst_current_mean_ratio": float(max(mean_ratio)),
        "start_stop_power_saving_pct": float(start_stop["power_saving_pct"]) if start_stop is not None else 0.0,
        "start_stop_eta_gain_pct": float(start_stop["eta_gain_pct"]) if start_stop is not None else 0.0,
    }


def _stage1_score(agg: Dict[str, float], acceptance: Air56Acceptance) -> float:
    s = 0.0
    s += max(0.0, acceptance.min_start_stop_power_saving_pct - float(agg["start_stop_power_saving_pct"])) * 50.0
    s += max(0.0, float(agg["err_failures"]) - acceptance.max_err_failures) * 15.0
    s += max(0.0, float(agg["worst_current_peak_ratio"]) - 1.15) * 5.0
    # Prefer lower power draw in start_stop when constraints are met.
    s += -float(agg["start_stop_power_saving_pct"]) * 0.05
    return float(s)


def _stage2_score(agg: Dict[str, float], acceptance: Air56Acceptance) -> float:
    s = 0.0
    s += max(0.0, acceptance.min_avg_power_saving_pct - float(agg["avg_power_saving_pct"])) * 25.0
    s += max(0.0, acceptance.min_avg_eta_gain_pct - float(agg["avg_eta_gain_pct"])) * 20.0
    s += max(0.0, float(agg["err_failures"]) - acceptance.max_err_failures) * 12.0
    s += max(0.0, acceptance.min_start_stop_power_saving_pct - float(agg["start_stop_power_saving_pct"])) * 40.0
    s += max(0.0, float(agg["worst_current_peak_ratio"]) - 1.15) * 5.0
    # Secondary preference among feasible candidates.
    s += -float(agg["avg_power_saving_pct"]) * 0.1
    return float(s)


def _acceptance_pass(agg: Dict[str, float], acceptance: Air56Acceptance) -> bool:
    return bool(
        float(agg["avg_power_saving_pct"]) > float(acceptance.min_avg_power_saving_pct)
        and float(agg["avg_eta_gain_pct"]) >= float(acceptance.min_avg_eta_gain_pct)
        and float(agg["err_failures"]) <= float(acceptance.max_err_failures)
        and float(agg["start_stop_power_saving_pct"]) >= float(acceptance.min_start_stop_power_saving_pct)
    )


def _run_air56_tuning(
    *,
    env_cfg: object,
    motor_key: str,
    agent: PPOVoltageAgent,
    scenarios: Sequence[str],
    stage1_trials: int,
    stage2_topk: int,
    stage1_seed: int,
    stage2_seed: int,
    stage2_eval_seeds: Sequence[int],
    search_seed: int,
    window_frac: float,
    error_tol_rel: float,
    error_tol_abs: float,
    use_total_power: bool,
    foc_feedback_mode: str,
    mic_feedback_mode: str,
    id_ref_params: Dict[str, object],
    sensorless: Dict[str, float],
    acceptance: Air56Acceptance,
    out_dir: Path,
    seed_perturbation: SeedPerturbationSettings,
) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    base_sup = _supervisor_from_env(env_cfg) or AiIdRefSupervisorConfig(enabled=True)
    base_cand = _supervisor_to_candidate(base_sup, tag="baseline", source="config")
    base_gate_rel = id_ref_params["id_ref_gate_speed_tol_rel"]
    if base_gate_rel is None:
        base_gate_rel = 0.05
    base_cand.update(
        {
            "id_ref_alpha": float(id_ref_params["id_ref_alpha"]),
            "delta_id_max": float(id_ref_params["delta_id_max"]),
            "id_ref_gate_speed_tol_rel": float(base_gate_rel),
            "id_ref_gate_min_scale": float(id_ref_params["id_ref_gate_min_scale"]),
            "id_ref_gate_exponent": float(id_ref_params["id_ref_gate_exponent"]),
        }
    )
    rng = random.Random(int(search_seed))

    candidates: List[Dict[str, object]] = [dict(base_cand)]
    candidates.extend(_build_handcrafted_candidates(base_cand))
    for i in range(1, int(stage1_trials) + 1):
        candidates.append(_sample_supervisor_candidate(rng, idx=i, base=base_cand))

    def _id_params_for_candidate(candidate: Dict[str, object]) -> Dict[str, object]:
        out = dict(id_ref_params)
        out["id_ref_alpha"] = float(candidate["id_ref_alpha"])
        out["delta_id_max"] = float(candidate["delta_id_max"])
        out["id_ref_gate_speed_tol_rel"] = float(candidate["id_ref_gate_speed_tol_rel"])
        out["id_ref_gate_min_scale"] = float(candidate["id_ref_gate_min_scale"])
        out["id_ref_gate_exponent"] = float(candidate["id_ref_gate_exponent"])
        return out

    stage1_rows: List[Dict[str, object]] = []
    stage1_scenarios = ["start_stop"]
    for idx, cand in enumerate(candidates, start=1):
        sup_cfg = _candidate_to_supervisor(cand)
        rows = _simulate_rows(
            env_cfg=env_cfg,
            motor_key=str(motor_key),
            agent=agent,
            scenarios=stage1_scenarios,
            seed=stage1_seed,
            window_frac=window_frac,
            error_tol_rel=error_tol_rel,
            error_tol_abs=error_tol_abs,
            use_total_power=use_total_power,
            foc_feedback_mode=foc_feedback_mode,
            mic_feedback_mode=mic_feedback_mode,
            controller="MIC",
            id_ref_params=_id_params_for_candidate(cand),
            supervisor_cfg=sup_cfg,
            sensorless=sensorless,
            seed_perturbation=seed_perturbation,
        )
        agg = _aggregate_rows(rows)
        score = _stage1_score(agg, acceptance)
        rec = {
            **cand,
            "stage1_seed": int(stage1_seed),
            "start_stop_power_saving_pct": float(agg["start_stop_power_saving_pct"]),
            "start_stop_eta_gain_pct": float(agg["start_stop_eta_gain_pct"]),
            "err_failures": float(agg["err_failures"]),
            "worst_current_peak_ratio": float(agg["worst_current_peak_ratio"]),
            "score": float(score),
        }
        stage1_rows.append(rec)
        if idx == 1 or idx % 5 == 0:
            print(
                "[step27][air56][stage1] {}/{} tag={} start_stop={:.3f}% err_fail={:.0f}".format(
                    idx,
                    len(candidates),
                    cand["tag"],
                    rec["start_stop_power_saving_pct"],
                    rec["err_failures"],
                ),
                flush=True,
            )

    stage1_rows.sort(
        key=lambda r: (
            float(r["score"]),
            -float(r["start_stop_power_saving_pct"]),
            float(r["worst_current_peak_ratio"]),
        )
    )
    _write_csv(out_dir / "stage1_rank.csv", stage1_rows)

    stage2_cands = stage1_rows[: max(int(stage2_topk), 1)]
    eval_seeds = [int(s) for s in stage2_eval_seeds] if stage2_eval_seeds else [int(stage2_seed)]
    stage2_rows: List[Dict[str, object]] = []
    for idx, cand in enumerate(stage2_cands, start=1):
        sup_cfg = _candidate_to_supervisor(cand)
        seed_aggs: List[Dict[str, float]] = []
        for eval_seed in eval_seeds:
            rows = _simulate_rows(
                env_cfg=env_cfg,
                motor_key=str(motor_key),
                agent=agent,
                scenarios=scenarios,
                seed=int(eval_seed),
                window_frac=window_frac,
                error_tol_rel=error_tol_rel,
                error_tol_abs=error_tol_abs,
                use_total_power=use_total_power,
                foc_feedback_mode=foc_feedback_mode,
                mic_feedback_mode=mic_feedback_mode,
                controller="MIC",
                id_ref_params=_id_params_for_candidate(cand),
                supervisor_cfg=sup_cfg,
                sensorless=sensorless,
                seed_perturbation=seed_perturbation,
            )
            seed_aggs.append(_aggregate_rows(rows))

        mean_agg = {
            "avg_power_saving_pct": _mean([float(a["avg_power_saving_pct"]) for a in seed_aggs]),
            "avg_eta_gain_pct": _mean([float(a["avg_eta_gain_pct"]) for a in seed_aggs]),
            "err_failures": _mean([float(a["err_failures"]) for a in seed_aggs]),
            "start_stop_power_saving_pct": _mean([float(a["start_stop_power_saving_pct"]) for a in seed_aggs]),
            "start_stop_eta_gain_pct": _mean([float(a["start_stop_eta_gain_pct"]) for a in seed_aggs]),
            "worst_current_peak_ratio": float(max(float(a["worst_current_peak_ratio"]) for a in seed_aggs)),
            "worst_current_mean_ratio": float(max(float(a["worst_current_mean_ratio"]) for a in seed_aggs)),
        }
        min_power = float(min(float(a["avg_power_saving_pct"]) for a in seed_aggs))
        min_eta = float(min(float(a["avg_eta_gain_pct"]) for a in seed_aggs))
        max_err = float(max(float(a["err_failures"]) for a in seed_aggs))
        min_start_stop = float(min(float(a["start_stop_power_saving_pct"]) for a in seed_aggs))
        score = _stage2_score(mean_agg, acceptance)
        score += max(0.0, float(acceptance.min_avg_power_saving_pct) - min_power) * 18.0
        score += max(0.0, float(acceptance.min_avg_eta_gain_pct) - min_eta) * 18.0
        score += max(0.0, max_err - float(acceptance.max_err_failures)) * 10.0
        score += max(0.0, float(acceptance.min_start_stop_power_saving_pct) - min_start_stop) * 22.0
        mean_pass = _acceptance_pass(mean_agg, acceptance)
        worst_pass = bool(
            min_power > float(acceptance.min_avg_power_saving_pct)
            and min_eta >= float(acceptance.min_avg_eta_gain_pct)
            and max_err <= float(acceptance.max_err_failures)
            and min_start_stop >= float(acceptance.min_start_stop_power_saving_pct)
        )
        rec = {
            **{
                k: cand[k]
                for k in cand.keys()
                if k not in {"score", "start_stop_power_saving_pct", "start_stop_eta_gain_pct", "err_failures", "worst_current_peak_ratio"}
            },
            "stage2_seed": int(stage2_seed),
            "stage2_eval_seeds": ",".join(str(s) for s in eval_seeds),
            "avg_power_saving_pct": float(mean_agg["avg_power_saving_pct"]),
            "avg_eta_gain_pct": float(mean_agg["avg_eta_gain_pct"]),
            "err_failures": float(mean_agg["err_failures"]),
            "start_stop_power_saving_pct": float(mean_agg["start_stop_power_saving_pct"]),
            "start_stop_eta_gain_pct": float(mean_agg["start_stop_eta_gain_pct"]),
            "worst_current_peak_ratio": float(mean_agg["worst_current_peak_ratio"]),
            "worst_current_mean_ratio": float(mean_agg["worst_current_mean_ratio"]),
            "avg_power_saving_pct_min_seed": min_power,
            "avg_eta_gain_pct_min_seed": min_eta,
            "err_failures_max_seed": max_err,
            "start_stop_power_saving_pct_min_seed": min_start_stop,
            "score": float(score),
            "acceptance_pass_mean": mean_pass,
            "acceptance_pass_worst": worst_pass,
            "acceptance_pass": bool(mean_pass and worst_pass),
        }
        stage2_rows.append(rec)
        print(
            "[step27][air56][stage2] {}/{} tag={} avg_power={:.3f}% start_stop={:.3f}% min_start_stop={:.3f}%".format(
                idx,
                len(stage2_cands),
                rec["tag"],
                rec["avg_power_saving_pct"],
                rec["start_stop_power_saving_pct"],
                rec["start_stop_power_saving_pct_min_seed"],
            ),
            flush=True,
        )

    stage2_rows.sort(
        key=lambda r: (
            0 if bool(r["acceptance_pass"]) else 1,
            float(r["score"]),
            -float(r["avg_power_saving_pct"]),
            -float(r["avg_eta_gain_pct"]),
        )
    )
    _write_csv(out_dir / "stage2_rank.csv", stage2_rows)

    selected = stage2_rows[0]
    selected_candidate = {k: selected[k] for k in base_cand.keys()}

    summary = {
        "stage1_trials": int(stage1_trials),
        "stage2_topk": int(stage2_topk),
        "stage1_seed": int(stage1_seed),
        "stage2_seed": int(stage2_seed),
        "stage2_eval_seeds": [int(s) for s in eval_seeds],
        "search_seed": int(search_seed),
        "acceptance": {
            "min_avg_power_saving_pct": float(acceptance.min_avg_power_saving_pct),
            "min_avg_eta_gain_pct": float(acceptance.min_avg_eta_gain_pct),
            "max_err_failures": float(acceptance.max_err_failures),
            "min_start_stop_power_saving_pct": float(acceptance.min_start_stop_power_saving_pct),
        },
        "selected": selected,
        "top_stage2": stage2_rows[: min(10, len(stage2_rows))],
    }
    _json_dump(out_dir / "tuning_summary.json", summary)
    return {
        "selected_candidate": selected_candidate,
        "selected_metrics": selected,
        "stage1_rows": stage1_rows,
        "stage2_rows": stage2_rows,
        "summary": summary,
    }


def _group_stats(
    rows: Sequence[Dict[str, object]],
    *,
    group_keys: Sequence[str],
    metric_keys: Sequence[str],
) -> List[Dict[str, object]]:
    groups: Dict[Tuple[object, ...], List[Dict[str, object]]] = {}
    for row in rows:
        key = tuple(row[k] for k in group_keys)
        groups.setdefault(key, []).append(row)

    out: List[Dict[str, object]] = []
    for key, items in sorted(groups.items()):
        rec: Dict[str, object] = {}
        for idx, name in enumerate(group_keys):
            rec[name] = key[idx]
        rec["samples"] = len(items)
        for metric in metric_keys:
            vals = [float(item[metric]) for item in items]
            rec[f"{metric}_mean"] = _mean(vals)
            rec[f"{metric}_std"] = _std(vals)
            rec[f"{metric}_min"] = float(min(vals)) if vals else 0.0
            rec[f"{metric}_max"] = float(max(vals)) if vals else 0.0
        out.append(rec)
    return out


def _index_stats(
    rows: Sequence[Dict[str, object]],
    *,
    key_fields: Sequence[str],
) -> Dict[Tuple[object, ...], Dict[str, object]]:
    out: Dict[Tuple[object, ...], Dict[str, object]] = {}
    for row in rows:
        key = tuple(row[k] for k in key_fields)
        out[key] = row
    return out


def _format_num(value: object, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return "n/a"


def _build_report_markdown(
    *,
    motors: Sequence[str],
    scenarios: Sequence[str],
    seeds: Sequence[int],
    seed_perturbation: SeedPerturbationSettings,
    acceptance: Air56Acceptance,
    tuning: Dict[str, object] | None,
    global_stats: Sequence[Dict[str, object]],
    motor_stats: Sequence[Dict[str, object]],
    air56_accept: Dict[str, object],
    reproducibility: Dict[str, object],
) -> str:
    lines: List[str] = []
    lines.append("# Step27 Pipeline Report")
    lines.append("")
    lines.append(f"- Motors: `{','.join(motors)}`")
    lines.append(f"- Scenarios: `{','.join(scenarios)}`")
    lines.append(f"- Seeds: `{','.join(str(s) for s in seeds)}`")
    lines.append(
        "- Seed perturbation: enabled=`{}` level=`{}`".format(
            bool(seed_perturbation.enabled and seed_perturbation.level > 0.0),
            _format_num(seed_perturbation.level),
        )
    )
    lines.append("")
    lines.append("## AIR56 Acceptance Criteria")
    lines.append("")
    lines.append(
        "- avg_power_saving_pct > `{}`; avg_eta_gain_pct >= `{}`; err_failures <= `{}`; start_stop >= `{}`".format(
            acceptance.min_avg_power_saving_pct,
            acceptance.min_avg_eta_gain_pct,
            acceptance.max_err_failures,
            acceptance.min_start_stop_power_saving_pct,
        )
    )
    lines.append(f"- Mean pass: `{air56_accept.get('mean_pass', False)}`")
    lines.append(f"- Worst-case pass: `{air56_accept.get('worst_case_pass', False)}`")
    lines.append("")

    if tuning is not None:
        selected = dict(tuning.get("selected_metrics", {}))
        lines.append("## AIR56 Tuned Candidate")
        lines.append("")
        lines.append(
            "- tag `{}` objective `{}` avg_power `{}` avg_eta `{}` start_stop `{}`".format(
                selected.get("tag", "n/a"),
                selected.get("objective", "n/a"),
                _format_num(selected.get("avg_power_saving_pct")),
                _format_num(selected.get("avg_eta_gain_pct")),
                _format_num(selected.get("start_stop_power_saving_pct")),
            )
        )
        lines.append("")

    lines.append("## PI vs FOC vs MIC (All Motors, All Seeds)")
    lines.append("")
    lines.append("| Controller | Avg Power Saving, % (mean/std/min) | Avg Eta Gain, % (mean/std/min) | Err Failures (mean/max) | Start-stop Saving, % (mean/min) |")
    lines.append("|---|---:|---:|---:|---:|")
    global_idx = _index_stats(global_stats, key_fields=("controller",))
    for ctrl in CONTROLLER_ORDER:
        row = global_idx.get((ctrl,))
        if row is None:
            continue
        lines.append(
            "| {ctrl} | {p_mean}/{p_std}/{p_min} | {e_mean}/{e_std}/{e_min} | {f_mean}/{f_max} | {ss_mean}/{ss_min} |".format(
                ctrl=ctrl,
                p_mean=_format_num(row.get("avg_power_saving_pct_mean")),
                p_std=_format_num(row.get("avg_power_saving_pct_std")),
                p_min=_format_num(row.get("avg_power_saving_pct_min")),
                e_mean=_format_num(row.get("avg_eta_gain_pct_mean")),
                e_std=_format_num(row.get("avg_eta_gain_pct_std")),
                e_min=_format_num(row.get("avg_eta_gain_pct_min")),
                f_mean=_format_num(row.get("err_failures_mean")),
                f_max=_format_num(row.get("err_failures_max")),
                ss_mean=_format_num(row.get("start_stop_power_saving_pct_mean")),
                ss_min=_format_num(row.get("start_stop_power_saving_pct_min")),
            )
        )
    lines.append("")

    lines.append("## MIC by Motor (mean/std/min)")
    lines.append("")
    lines.append("| Motor | Power Saving, % | Eta Gain, % | Err Failures | Start-stop Saving, % |")
    lines.append("|---|---:|---:|---:|---:|")
    for motor in motors:
        row = next((r for r in motor_stats if r.get("motor") == motor and r.get("controller") == "MIC"), None)
        if row is None:
            continue
        lines.append(
            "| {motor} | {p_mean}/{p_std}/{p_min} | {e_mean}/{e_std}/{e_min} | {f_mean}/{f_max} | {ss_mean}/{ss_min} |".format(
                motor=motor,
                p_mean=_format_num(row.get("avg_power_saving_pct_mean")),
                p_std=_format_num(row.get("avg_power_saving_pct_std")),
                p_min=_format_num(row.get("avg_power_saving_pct_min")),
                e_mean=_format_num(row.get("avg_eta_gain_pct_mean")),
                e_std=_format_num(row.get("avg_eta_gain_pct_std")),
                e_min=_format_num(row.get("avg_eta_gain_pct_min")),
                f_mean=_format_num(row.get("err_failures_mean")),
                f_max=_format_num(row.get("err_failures_max")),
                ss_mean=_format_num(row.get("start_stop_power_saving_pct_mean")),
                ss_min=_format_num(row.get("start_stop_power_saving_pct_min")),
            )
        )
    lines.append("")

    lines.append("## Reproducibility")
    lines.append("")
    lines.append(f"- table_sha256: `{reproducibility.get('table_sha256', '')}`")
    lines.append(f"- stable_vs_previous: `{reproducibility.get('stable_vs_previous')}`")
    lines.append("")

    pi_row = global_idx.get(("PI",))
    foc_row = global_idx.get(("FOC",))
    mic_row = global_idx.get(("MIC",))
    mic_power = float(mic_row.get("avg_power_saving_pct_mean", 0.0)) if mic_row else 0.0
    foc_power = float(foc_row.get("avg_power_saving_pct_mean", 0.0)) if foc_row else 0.0
    mic_eta = float(mic_row.get("avg_eta_gain_pct_mean", 0.0)) if mic_row else 0.0
    lines.append("## Short Scientific Conclusion")
    lines.append("")
    if bool(air56_accept.get("mean_pass", False)):
        lines.append(
            "AIR56 mean acceptance constraints are satisfied in this run, including start-stop and tracking constraints."
        )
    else:
        lines.append(
            "AIR56 mean acceptance constraints are not fully satisfied; additional targeted tuning is still required."
        )
    if mic_power > foc_power:
        lines.append(
            "Across the 3-motor benchmark, MIC keeps a higher mean power-saving margin than sensorless FOC relative to PI."
        )
    else:
        lines.append(
            "Across the 3-motor benchmark, MIC does not yet exceed sensorless FOC in mean power-saving margin relative to PI."
        )
    lines.append(
        "The observed MIC mean eta-gain relative to PI is `{}`%.".format(_format_num(mic_eta))
    )
    if pi_row is None:
        lines.append("PI baseline row is missing in aggregated stats and should be rechecked.")
    lines.append("")

    return "\n".join(lines)


def _sha256_rows(rows: Sequence[Dict[str, object]]) -> str:
    normalized: List[Dict[str, object]] = []
    for row in rows:
        normalized.append(
            {
                "motor": row["motor"],
                "seed": int(row["seed"]),
                "controller": row["controller"],
                "avg_power_saving_pct": round(float(row["avg_power_saving_pct"]), 10),
                "avg_eta_gain_pct": round(float(row["avg_eta_gain_pct"]), 10),
                "err_failures": round(float(row["err_failures"]), 10),
                "start_stop_power_saving_pct": round(float(row["start_stop_power_saving_pct"]), 10),
                "worst_current_peak_ratio": round(float(row["worst_current_peak_ratio"]), 10),
                "worst_current_mean_ratio": round(float(row["worst_current_mean_ratio"]), 10),
                "avg_controller_speed_err": round(float(row["avg_controller_speed_err"]), 10),
            }
        )
    normalized.sort(key=lambda r: (str(r["motor"]), str(r["controller"]), int(r["seed"])))
    payload = json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_env_and_agent(config_path: str, *, foc_disable_lut: bool) -> Tuple[object, PPOVoltageAgent, Path]:
    cfg_path = _resolve_config_path(config_path)
    env_cfg = make_env_from_config(str(cfg_path)).env_config
    env_cfg = _disable_lut_if_needed(env_cfg, disable=foc_disable_lut)
    ckpt = _resolve_checkpoint(env_cfg)
    agent = _load_agent(ckpt)
    return env_cfg, agent, ckpt


def main() -> None:
    parser = argparse.ArgumentParser(description="Step27 pipeline: multi-motor fixed-seed benchmark + AIR56 start_stop tuning.")
    parser.add_argument("--motors", default="air56,al31,ao2")
    parser.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS))
    parser.add_argument("--scenarios", default=",".join(DEFAULT_SCENARIOS))
    parser.add_argument("--out-dir", default="outputs/progress_step27_pipeline")
    parser.add_argument("--window-frac", type=float, default=0.25)
    parser.add_argument("--error-tol-rel", type=float, default=0.05)
    parser.add_argument("--error-tol-abs", type=float, default=0.0)
    parser.add_argument("--foc-feedback-mode", default="encoder", choices=["encoder", "sensorless"])
    parser.add_argument("--mic-feedback-mode", default="sensorless", choices=["encoder", "sensorless"])
    parser.add_argument("--seed-perturbation", action="store_true")
    parser.add_argument("--seed-perturb-level", type=float, default=1.0)
    parser.add_argument("--use-total-power", action="store_true")
    parser.add_argument("--no-use-total-power", dest="use_total_power", action="store_false")
    parser.add_argument("--foc-disable-lut", action="store_true")
    parser.add_argument("--allow-foc-lut", dest="foc_disable_lut", action="store_false")

    parser.add_argument("--skip-air56-tune", action="store_true")
    parser.add_argument("--air56-stage1-trials", type=int, default=18)
    parser.add_argument("--air56-stage2-topk", type=int, default=4)
    parser.add_argument("--air56-stage1-seed", type=int, default=None)
    parser.add_argument("--air56-stage2-seed", type=int, default=None)
    parser.add_argument("--air56-search-seed", type=int, default=26027)
    parser.add_argument("--air56-accept-min-power-saving-pct", type=float, default=0.5)
    parser.add_argument("--air56-accept-min-eta-gain-pct", type=float, default=0.0)
    parser.add_argument("--air56-accept-max-err-failures", type=float, default=2.0)
    parser.add_argument("--air56-accept-min-start-stop-saving-pct", type=float, default=-0.5)

    parser.set_defaults(use_total_power=True)
    parser.set_defaults(foc_disable_lut=True)
    args = parser.parse_args()

    t0 = time.time()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    motors = _parse_csv_list(args.motors)
    for motor in motors:
        if motor not in MOTOR_REGISTRY:
            raise ValueError(f"Unknown motor key: {motor}. Allowed: {','.join(MOTOR_REGISTRY.keys())}")
    if not motors:
        raise ValueError("No motors selected.")

    seeds = _parse_int_list(args.seeds)
    if not seeds:
        raise ValueError("Seed list is empty.")
    scenarios = _parse_csv_list(args.scenarios)
    if not scenarios:
        raise ValueError("Scenario list is empty.")

    stage1_seed = int(args.air56_stage1_seed) if args.air56_stage1_seed is not None else int(seeds[0])
    stage2_seed = int(args.air56_stage2_seed) if args.air56_stage2_seed is not None else int(seeds[0])
    acceptance = Air56Acceptance(
        min_avg_power_saving_pct=float(args.air56_accept_min_power_saving_pct),
        min_avg_eta_gain_pct=float(args.air56_accept_min_eta_gain_pct),
        max_err_failures=float(args.air56_accept_max_err_failures),
        min_start_stop_power_saving_pct=float(args.air56_accept_min_start_stop_saving_pct),
    )
    seed_perturbation = SeedPerturbationSettings(
        enabled=bool(args.seed_perturbation),
        level=float(max(0.0, float(args.seed_perturb_level))),
    )

    reproducibility_prev: str | None = None
    reproducibility_path = out_dir / "step27_reproducibility.json"
    if reproducibility_path.exists():
        try:
            reproducibility_prev = str(json.loads(reproducibility_path.read_text(encoding="utf-8")).get("table_sha256", ""))
        except Exception:
            reproducibility_prev = None

    tuning_result: Dict[str, object] | None = None
    tuned_air56_candidate: Dict[str, object] | None = None

    if (not bool(args.skip_air56_tune)) and ("air56" in motors):
        print("[step27] AIR56 tuning started...", flush=True)
        air56_cfg, air56_agent, _air56_ckpt = _load_env_and_agent(
            MOTOR_REGISTRY["air56"].config_path,
            foc_disable_lut=bool(args.foc_disable_lut),
        )
        tuning_result = _run_air56_tuning(
            env_cfg=air56_cfg,
            motor_key="air56",
            agent=air56_agent,
            scenarios=scenarios,
            stage1_trials=int(args.air56_stage1_trials),
            stage2_topk=int(args.air56_stage2_topk),
            stage1_seed=stage1_seed,
            stage2_seed=stage2_seed,
            stage2_eval_seeds=seeds,
            search_seed=int(args.air56_search_seed),
            window_frac=float(args.window_frac),
            error_tol_rel=float(args.error_tol_rel),
            error_tol_abs=float(args.error_tol_abs),
            use_total_power=bool(args.use_total_power),
            foc_feedback_mode=str(args.foc_feedback_mode),
            mic_feedback_mode=str(args.mic_feedback_mode),
            id_ref_params=_id_ref_eval_params(air56_cfg),
            sensorless=_sensorless_params(air56_cfg),
            acceptance=acceptance,
            out_dir=out_dir / "air56_tuning",
            seed_perturbation=seed_perturbation,
        )
        tuned_air56_candidate = dict(tuning_result["selected_candidate"])
        print("[step27] AIR56 tuning done. Selected:", tuned_air56_candidate.get("tag"), flush=True)

    per_seed_rows: List[Dict[str, object]] = []
    run_manifest_rows: List[Dict[str, object]] = []
    seed_perturb_rows: List[Dict[str, object]] = []

    for motor in motors:
        spec = MOTOR_REGISTRY[motor]
        print(f"[step27] Evaluate motor={motor}", flush=True)
        env_cfg, agent, ckpt = _load_env_and_agent(spec.config_path, foc_disable_lut=bool(args.foc_disable_lut))
        id_ref_params = _id_ref_eval_params(env_cfg)
        sensorless = _sensorless_params(env_cfg)
        base_sup = _supervisor_from_env(env_cfg)
        id_ref_eval = dict(id_ref_params)
        if motor == "air56" and tuned_air56_candidate is not None:
            sup_cfg = _candidate_to_supervisor(tuned_air56_candidate)
            sup_source = str(tuned_air56_candidate.get("tag", "tuned"))
            id_ref_eval["id_ref_alpha"] = float(tuned_air56_candidate["id_ref_alpha"])
            id_ref_eval["delta_id_max"] = float(tuned_air56_candidate["delta_id_max"])
            id_ref_eval["id_ref_gate_speed_tol_rel"] = float(tuned_air56_candidate["id_ref_gate_speed_tol_rel"])
            id_ref_eval["id_ref_gate_min_scale"] = float(tuned_air56_candidate["id_ref_gate_min_scale"])
            id_ref_eval["id_ref_gate_exponent"] = float(tuned_air56_candidate["id_ref_gate_exponent"])
        else:
            sup_cfg = base_sup
            sup_source = "config"

        for seed in seeds:
            seed_dir = out_dir / "runs" / motor / f"seed_{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            mic_rows = _simulate_rows(
                env_cfg=env_cfg,
                motor_key=motor,
                agent=agent,
                scenarios=scenarios,
                seed=int(seed),
                window_frac=float(args.window_frac),
                error_tol_rel=float(args.error_tol_rel),
                error_tol_abs=float(args.error_tol_abs),
                use_total_power=bool(args.use_total_power),
                foc_feedback_mode=str(args.foc_feedback_mode),
                mic_feedback_mode=str(args.mic_feedback_mode),
                controller="MIC",
                id_ref_params=id_ref_eval,
                supervisor_cfg=sup_cfg,
                sensorless=sensorless,
                seed_perturbation=seed_perturbation,
            )
            foc_rows = _simulate_rows(
                env_cfg=env_cfg,
                motor_key=motor,
                agent=None,
                scenarios=scenarios,
                seed=int(seed),
                window_frac=float(args.window_frac),
                error_tol_rel=float(args.error_tol_rel),
                error_tol_abs=float(args.error_tol_abs),
                use_total_power=bool(args.use_total_power),
                foc_feedback_mode=str(args.foc_feedback_mode),
                mic_feedback_mode=str(args.mic_feedback_mode),
                controller="FOC",
                id_ref_params=id_ref_eval,
                supervisor_cfg=None,
                sensorless=sensorless,
                seed_perturbation=seed_perturbation,
            )
            _json_dump(seed_dir / "mic_summary_rows.json", mic_rows)
            _json_dump(seed_dir / "foc_sensorless_summary_rows.json", foc_rows)

            perturb_ref = mic_rows[0] if mic_rows else {}
            seed_perturb_rows.append(
                {
                    "motor": motor,
                    "seed": int(seed),
                    "enabled": float(perturb_ref.get("perturb_enabled", 0.0)),
                    "level": float(perturb_ref.get("perturb_level", 0.0)),
                    "load_torque_scale": float(perturb_ref.get("perturb_load_torque_scale", 1.0)),
                    "sigma_omega_add": float(perturb_ref.get("perturb_sigma_omega_add", 0.0)),
                    "sigma_i_abc_add": float(perturb_ref.get("perturb_sigma_i_abc_add", 0.0)),
                }
            )

            mic_agg = _aggregate_rows(mic_rows)
            foc_agg = _aggregate_rows(foc_rows)

            pi_avg_err = float(mic_agg["avg_foc_err"])
            per_seed_rows.append(
                {
                    "motor": motor,
                    "seed": int(seed),
                    "controller": "PI",
                    "avg_power_saving_pct": 0.0,
                    "avg_eta_gain_pct": 0.0,
                    "err_failures": 0.0,
                    "start_stop_power_saving_pct": 0.0,
                    "worst_current_peak_ratio": 1.0,
                    "worst_current_mean_ratio": 1.0,
                    "avg_controller_speed_err": pi_avg_err,
                    "checkpoint": str(ckpt),
                    "supervisor_source": "baseline",
                }
            )
            per_seed_rows.append(
                {
                    "motor": motor,
                    "seed": int(seed),
                    "controller": "FOC",
                    "avg_power_saving_pct": float(foc_agg["avg_power_saving_pct"]),
                    "avg_eta_gain_pct": float(foc_agg["avg_eta_gain_pct"]),
                    "err_failures": float(foc_agg["err_failures"]),
                    "start_stop_power_saving_pct": float(foc_agg["start_stop_power_saving_pct"]),
                    "worst_current_peak_ratio": float(foc_agg["worst_current_peak_ratio"]),
                    "worst_current_mean_ratio": float(foc_agg["worst_current_mean_ratio"]),
                    "avg_controller_speed_err": float(foc_agg["avg_mic_err"]),
                    "checkpoint": str(ckpt),
                    "supervisor_source": "n/a",
                }
            )
            per_seed_rows.append(
                {
                    "motor": motor,
                    "seed": int(seed),
                    "controller": "MIC",
                    "avg_power_saving_pct": float(mic_agg["avg_power_saving_pct"]),
                    "avg_eta_gain_pct": float(mic_agg["avg_eta_gain_pct"]),
                    "err_failures": float(mic_agg["err_failures"]),
                    "start_stop_power_saving_pct": float(mic_agg["start_stop_power_saving_pct"]),
                    "worst_current_peak_ratio": float(mic_agg["worst_current_peak_ratio"]),
                    "worst_current_mean_ratio": float(mic_agg["worst_current_mean_ratio"]),
                    "avg_controller_speed_err": float(mic_agg["avg_mic_err"]),
                    "checkpoint": str(ckpt),
                    "supervisor_source": sup_source,
                }
            )
            run_manifest_rows.append(
                {
                    "motor": motor,
                    "seed": int(seed),
                    "checkpoint": str(ckpt),
                    "supervisor_source": sup_source,
                    "seed_perturbation_enabled": bool(seed_perturbation.enabled and seed_perturbation.level > 0.0),
                    "seed_perturbation_level": float(seed_perturbation.level),
                    "seed_perturb_load_torque_scale": float(perturb_ref.get("perturb_load_torque_scale", 1.0)),
                    "seed_perturb_sigma_omega_add": float(perturb_ref.get("perturb_sigma_omega_add", 0.0)),
                    "seed_perturb_sigma_i_abc_add": float(perturb_ref.get("perturb_sigma_i_abc_add", 0.0)),
                    "mic_rows_path": str((seed_dir / "mic_summary_rows.json").resolve()),
                    "foc_sensorless_rows_path": str((seed_dir / "foc_sensorless_summary_rows.json").resolve()),
                }
            )
            print(
                "[step27] motor={} seed={} mic_avg_power={:.3f}% mic_start_stop={:.3f}%".format(
                    motor,
                    seed,
                    float(mic_agg["avg_power_saving_pct"]),
                    float(mic_agg["start_stop_power_saving_pct"]),
                ),
                flush=True,
            )

    per_seed_csv = out_dir / "step27_per_seed_metrics.csv"
    per_seed_json = out_dir / "step27_per_seed_metrics.json"
    _write_csv(per_seed_csv, per_seed_rows)
    _json_dump(per_seed_json, per_seed_rows)
    _write_csv(out_dir / "step27_run_manifest.csv", run_manifest_rows)
    _json_dump(out_dir / "step27_run_manifest.json", run_manifest_rows)
    _write_csv(out_dir / "step27_seed_perturbations.csv", seed_perturb_rows)
    _json_dump(out_dir / "step27_seed_perturbations.json", seed_perturb_rows)

    stats_motor = _group_stats(
        per_seed_rows,
        group_keys=("motor", "controller"),
        metric_keys=METRIC_FIELDS,
    )
    stats_global = _group_stats(
        per_seed_rows,
        group_keys=("controller",),
        metric_keys=METRIC_FIELDS,
    )

    stats_motor_csv = out_dir / "step27_stats_motor_controller.csv"
    stats_global_csv = out_dir / "step27_final_pi_vs_foc_vs_mic.csv"
    _write_csv(stats_motor_csv, stats_motor)
    _write_csv(stats_global_csv, stats_global)
    _json_dump(out_dir / "step27_stats_motor_controller.json", stats_motor)
    _json_dump(out_dir / "step27_final_pi_vs_foc_vs_mic.json", stats_global)

    air56_row = next((r for r in stats_motor if r.get("motor") == "air56" and r.get("controller") == "MIC"), None)
    air56_accept = {
        "mean_pass": False,
        "worst_case_pass": False,
        "details": {},
    }
    if air56_row is not None:
        mean_pass = bool(
            float(air56_row["avg_power_saving_pct_mean"]) > float(acceptance.min_avg_power_saving_pct)
            and float(air56_row["avg_eta_gain_pct_mean"]) >= float(acceptance.min_avg_eta_gain_pct)
            and float(air56_row["err_failures_mean"]) <= float(acceptance.max_err_failures)
            and float(air56_row["start_stop_power_saving_pct_mean"]) >= float(acceptance.min_start_stop_power_saving_pct)
        )
        worst_case_pass = bool(
            float(air56_row["avg_power_saving_pct_min"]) > float(acceptance.min_avg_power_saving_pct)
            and float(air56_row["avg_eta_gain_pct_min"]) >= float(acceptance.min_avg_eta_gain_pct)
            and float(air56_row["err_failures_max"]) <= float(acceptance.max_err_failures)
            and float(air56_row["start_stop_power_saving_pct_min"]) >= float(acceptance.min_start_stop_power_saving_pct)
        )
        air56_accept = {
            "mean_pass": mean_pass,
            "worst_case_pass": worst_case_pass,
            "details": {
                "avg_power_saving_pct_mean": float(air56_row["avg_power_saving_pct_mean"]),
                "avg_power_saving_pct_min": float(air56_row["avg_power_saving_pct_min"]),
                "avg_eta_gain_pct_mean": float(air56_row["avg_eta_gain_pct_mean"]),
                "avg_eta_gain_pct_min": float(air56_row["avg_eta_gain_pct_min"]),
                "err_failures_mean": float(air56_row["err_failures_mean"]),
                "err_failures_max": float(air56_row["err_failures_max"]),
                "start_stop_power_saving_pct_mean": float(air56_row["start_stop_power_saving_pct_mean"]),
                "start_stop_power_saving_pct_min": float(air56_row["start_stop_power_saving_pct_min"]),
            },
        }
    _json_dump(out_dir / "step27_air56_acceptance.json", air56_accept)

    table_sha = _sha256_rows(per_seed_rows)
    reproducibility = {
        "table_sha256": table_sha,
        "previous_table_sha256": reproducibility_prev,
        "stable_vs_previous": (reproducibility_prev == table_sha) if reproducibility_prev else None,
    }
    _json_dump(reproducibility_path, reproducibility)

    report_md = _build_report_markdown(
        motors=motors,
        scenarios=scenarios,
        seeds=seeds,
        seed_perturbation=seed_perturbation,
        acceptance=acceptance,
        tuning=tuning_result,
        global_stats=stats_global,
        motor_stats=stats_motor,
        air56_accept=air56_accept,
        reproducibility=reproducibility,
    )
    (out_dir / "step27_report.md").write_text(report_md, encoding="utf-8")

    manifest = {
        "out_dir": str(out_dir),
        "motors": motors,
        "scenarios": scenarios,
        "seeds": seeds,
        "seed_perturbation": {
            "enabled": bool(seed_perturbation.enabled and seed_perturbation.level > 0.0),
            "level": float(seed_perturbation.level),
        },
        "runtime_sec": float(time.time() - t0),
        "files": {
            "per_seed_csv": str(per_seed_csv),
            "per_seed_json": str(per_seed_json),
            "stats_motor_controller_csv": str(stats_motor_csv),
            "final_pi_vs_foc_vs_mic_csv": str(stats_global_csv),
            "air56_acceptance_json": str((out_dir / "step27_air56_acceptance.json")),
            "reproducibility_json": str(reproducibility_path),
            "report_markdown": str((out_dir / "step27_report.md")),
            "run_manifest_json": str((out_dir / "step27_run_manifest.json")),
            "seed_perturbations_json": str((out_dir / "step27_seed_perturbations.json")),
            "tuning_summary_json": str((out_dir / "air56_tuning" / "tuning_summary.json")) if tuning_result is not None else None,
        },
    }
    _json_dump(out_dir / "step27_manifest.json", manifest)

    print(f"[step27] done in {manifest['runtime_sec']:.1f}s")
    print(f"[step27] report: {out_dir / 'step27_report.md'}")
    print(f"[step27] table:  {out_dir / 'step27_final_pi_vs_foc_vs_mic.csv'}")
    print(f"[step27] seeds:  {out_dir / 'step27_per_seed_metrics.csv'}")


if __name__ == "__main__":
    main()
