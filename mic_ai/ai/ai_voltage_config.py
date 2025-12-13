from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Dict

DEFAULT_CONFIG_PATH = Path("config/ai_voltage_config.json")
DEFAULT_CONFIG: Dict[str, Dict] = {
    "reward": {
        "motor1": {"w_speed": 1.0, "w_current": 1.0, "w_power": 2.0, "w_action": 0.1},
        "motor2": {"w_speed": 1.0, "w_current": 1.5, "w_power": 2.0, "w_action": 0.1},
    },
    "voltage_scale": {"motor1": 0.20, "motor2": 0.20},
    "success": {
        "w_speed": 1.0,
        "w_current": 5.0,
        "speed_tol": 0.5,
        "current_tol": 0.2,
        "I_nom": {"motor1": 1.2, "motor2": 1.0},
    },
    "curriculum": {
        "omega_pu_stages": [0.3, 0.5],
        "stage_episode_boundaries": [300, 600],
        "piecewise_steps": [150, 300],
        "piecewise_multipliers": [1.0, 0.8, 1.0],
    },
    "exploration": {"sigma_start": 0.25, "sigma_end": 0.05, "sigma_decay_episodes": 300},
    "eval": {"episodes": 5},
}


def _deep_update(base: Dict, update: Dict) -> Dict:
    """Recursively merge ``update`` into ``base`` and return the merged dict."""
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_ai_voltage_config(path: str | Path | None = None) -> Dict:
    """Load ai_voltage configuration (reward, success thresholds, eval) with sane defaults."""
    cfg = deepcopy(DEFAULT_CONFIG)
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if cfg_path.is_file():
        try:
            with cfg_path.open("r", encoding="utf-8") as f:
                disk_cfg = json.load(f)
            if isinstance(disk_cfg, dict):
                cfg = _deep_update(cfg, disk_cfg)
        except Exception:
            # Fallback to defaults if config cannot be parsed.
            pass
    return cfg


def get_reward_weights(config: Dict, motor_key: str) -> Dict[str, float]:
    """Return reward weights for the given motor, falling back to defaults."""
    reward_cfg = config.get("reward", {})
    weights = reward_cfg.get(motor_key, {}) if isinstance(reward_cfg, dict) else {}
    default_weights = DEFAULT_CONFIG["reward"].get(
        motor_key, {"w_speed": 1.0, "w_current": 1.0, "w_power": 2.0, "w_action": 0.1}
    )
    merged = deepcopy(default_weights)
    if isinstance(weights, dict):
        merged.update({k: float(v) for k, v in weights.items() if isinstance(v, (int, float))})
    return merged


def get_success_config(config: Dict) -> Dict[str, float | Dict[str, float]]:
    """Return success criteria config merged with defaults."""
    success_cfg = config.get("success", {}) if isinstance(config, dict) else {}
    merged = deepcopy(DEFAULT_CONFIG["success"])
    if isinstance(success_cfg, dict):
        merged.update({k: v for k, v in success_cfg.items() if k != "I_nom"})
        if isinstance(success_cfg.get("I_nom"), dict):
            merged["I_nom"].update({k: float(v) for k, v in success_cfg["I_nom"].items()})
    return merged


def get_voltage_scale(config: Dict, motor_key: str) -> float:
    """Return per-motor voltage scale (per-unit of base_v_limit)."""
    volt_cfg = config.get("voltage_scale", {}) if isinstance(config, dict) else {}
    default_scale = DEFAULT_CONFIG["voltage_scale"].get(motor_key, 0.1)
    if isinstance(volt_cfg, dict) and motor_key in volt_cfg:
        try:
            return float(volt_cfg[motor_key])
        except Exception:
            return default_scale
    return default_scale


def get_curriculum_config(config: Dict) -> Dict:
    """Return curriculum config merged with defaults."""
    merged = deepcopy(DEFAULT_CONFIG["curriculum"])
    user_cfg = config.get("curriculum", {}) if isinstance(config, dict) else {}
    if isinstance(user_cfg, dict):
        for key in ("omega_pu_stages", "stage_episode_boundaries", "piecewise_steps", "piecewise_multipliers"):
            if key in user_cfg and isinstance(user_cfg[key], list):
                merged[key] = list(user_cfg[key])
    return merged


def get_exploration_config(config: Dict) -> Dict[str, float]:
    """Return exploration sigma schedule merged with defaults."""
    merged = deepcopy(DEFAULT_CONFIG["exploration"])
    user_cfg = config.get("exploration", {}) if isinstance(config, dict) else {}
    if isinstance(user_cfg, dict):
        for key in ("sigma_start", "sigma_end", "sigma_decay_episodes"):
            if key in user_cfg:
                merged[key] = float(user_cfg[key]) if "episodes" not in key else int(user_cfg[key])
    return merged


__all__ = [
    "load_ai_voltage_config",
    "get_reward_weights",
    "get_success_config",
    "get_curriculum_config",
    "get_exploration_config",
    "get_voltage_scale",
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_CONFIG",
]
