from __future__ import annotations

from dataclasses import is_dataclass, replace
from typing import Any

from .io import load_estimated_params
from .motor_params import MotorParamsEstimated


def apply_estimated_params_to_env_config(env_cfg: Any, estimated: MotorParamsEstimated) -> Any:
    """
    Apply estimated motor parameters to an EnvConfig-like object.

    Supports dataclass configs from `config/env.py` and plain objects.
    Inductances in `estimated` are treated as total Ls/Lr; if the motor config
    uses sigma notation (Ls_sigma/Lr_sigma + Lm), the sigma parts are derived.
    """
    if not hasattr(env_cfg, "motor"):
        raise ValueError("env_cfg must have .motor attribute")
    motor = env_cfg.motor

    updates: dict[str, float] = {}
    for key in ("Rs", "Rr", "Lm", "J", "B"):
        val = getattr(estimated, key, None)
        if val is not None and hasattr(motor, key):
            updates[key] = float(val)

    # Inductance mapping.
    has_sigma = all(hasattr(motor, k) for k in ("Ls_sigma", "Lr_sigma", "Lm"))
    if has_sigma and (estimated.Ls is not None or estimated.Lr is not None or estimated.Lm is not None):
        lm_val = updates.get("Lm", float(getattr(motor, "Lm")))
        if estimated.Ls is not None:
            updates["Ls_sigma"] = max(float(estimated.Ls) - lm_val, 1e-6)
        if estimated.Lr is not None:
            updates["Lr_sigma"] = max(float(estimated.Lr) - lm_val, 1e-6)
    else:
        if estimated.Ls is not None and hasattr(motor, "Ls"):
            updates["Ls"] = float(estimated.Ls)
        if estimated.Lr is not None and hasattr(motor, "Lr"):
            updates["Lr"] = float(estimated.Lr)

    if not updates:
        return env_cfg

    if is_dataclass(motor):
        motor_new = replace(motor, **updates)
    else:
        motor_new = motor
        for k, v in updates.items():
            setattr(motor_new, k, v)

    if is_dataclass(env_cfg):
        return replace(env_cfg, motor=motor_new)

    env_cfg.motor = motor_new
    return env_cfg


def load_and_apply_ident(env_cfg: Any, ident_path: str) -> Any:
    """
    Convenience: load MotorParamsEstimated from JSON and apply to env_cfg.
    """
    estimated = load_estimated_params(ident_path)
    return apply_estimated_params_to_env_config(env_cfg, estimated)


__all__ = ["apply_estimated_params_to_env_config", "load_and_apply_ident"]

