from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Mapping

import numpy as np

from config.env import EnvConfig, NAMEPLATE_N_RATED, create_default_env
from sim.load_servo import ServoLoadConfig
from sim.safety import SafetyLimits, SafetySupervisor


@dataclass(frozen=True)
class IdentificationConfig:
    """Configuration for the identification sequence."""

    env: EnvConfig | None = None
    dt: float = 0.001
    duration: float = 5.0
    pulse_start: float = 1.0
    pulse_end: float = 3.0
    pulse_v_d: float = 0.0
    pulse_v_q: float | None = None
    rs_nom: float | None = None
    rs_max: float | None = None
    current_ratio: float = 0.5
    i_inst_max: float | None = None
    i_rms_max: float | None = None
    omega_max: float | None = None
    vdc_max: float | None = None
    log_root: str | Path = "logs"
    run_name: str | None = None
    seed: int | None = None


class _PulsePolicy:
    def __init__(
        self,
        pulse_v_d: float,
        pulse_v_q: float,
        pulse_start: float,
        pulse_end: float,
        base_policy: object | None,
        dt: float,
    ) -> None:
        self._pulse_v_d = pulse_v_d
        self._pulse_v_q = pulse_v_q
        self._pulse_start = pulse_start
        self._pulse_end = pulse_end
        self._dt = dt
        self._base_policy = _wrap_base_policy(base_policy, dt) if base_policy else None

    def reset(self) -> None:
        if self._base_policy is not None and hasattr(self._base_policy, "reset"):
            self._base_policy.reset()

    def act(self, state: Mapping[str, object]) -> dict[str, float]:
        t = float(state["t"])
        base = {"v_d": 0.0, "v_q": 0.0, "theta_e": 0.0, "omega_syn": 0.0}
        if self._base_policy is not None:
            base = self._base_policy.act(state)
        v_d = float(base["v_d"])
        v_q = float(base["v_q"])
        if self._pulse_start <= t <= self._pulse_end:
            v_d += self._pulse_v_d
            v_q += self._pulse_v_q
        return {
            "v_d": v_d,
            "v_q": v_q,
            "theta_e": float(base["theta_e"]),
            "omega_syn": float(base["omega_syn"]),
        }


class _StepPolicyAdapter:
    def __init__(self, policy: object) -> None:
        self._policy = policy

    def reset(self) -> None:
        if hasattr(self._policy, "reset"):
            self._policy.reset()

    def act(self, state: Mapping[str, object]) -> dict[str, float]:
        v_d, v_q, theta_e, omega_syn, _ = self._policy.step(
            float(state["t"]),
            float(state["omega_ref"]),
            float(state["omega_m"]),
            tuple(state["i_abc"]),
            float(state["torque_e"]),
            float(state["theta_mech"]),
        )
        return {"v_d": v_d, "v_q": v_q, "theta_e": theta_e, "omega_syn": omega_syn}


def run_id_sequence(
    orchestrator: object, id_config: IdentificationConfig, base_policy: object | None = None
) -> dict[str, object]:
    """Run identification sequence and estimate parameters/aging."""
    env = id_config.env or create_default_env()
    config = _normalize_config(id_config, env)
    _validate_config(config)

    default_cfg = orchestrator.default_experiment_config(env)
    safety = SafetySupervisor(
        SafetyLimits(
            i_inst_max=config.i_inst_max,
            i_rms_max=config.i_rms_max,
            omega_max=config.omega_max,
            vdc_max=config.vdc_max,
        )
    )

    speed_profile = orchestrator.ProfileConfig(kind="constant", value=0.0)
    load_profile = orchestrator.ProfileConfig(kind="constant", value=0.0)
    load_config = ServoLoadConfig(mode="torque", tau_load=0.02, t_max=0.1)
    run_name = config.run_name or f"identify_{_now_stamp()}"

    exp_cfg = replace(
        default_cfg,
        dt=config.dt,
        duration=config.duration,
        controller="foc",
        speed_profile=speed_profile,
        load_profile=load_profile,
        load_config=load_config,
        safety=safety,
        seed=config.seed,
        log_root=config.log_root,
        run_name=run_name,
    )

    policy = _PulsePolicy(
        config.pulse_v_d,
        config.pulse_v_q,
        config.pulse_start,
        config.pulse_end,
        base_policy,
        config.dt,
    )
    result = orchestrator.run_experiment(exp_cfg, policy=policy)

    metrics, rs_est = _estimate_rs(result.npz_path, config)
    aging = _compute_aging(rs_est, config.rs_nom, config.rs_max)

    return {
        "est_params": {"Rs_est": rs_est, "J_est": None, "B_est": None},
        "aging": aging,
        "id_metrics": metrics,
        "id_log_path": str(result.npz_path) if result.npz_path else None,
    }


def _wrap_base_policy(base_policy: object, dt: float) -> object:
    if hasattr(base_policy, "act"):
        return base_policy
    if hasattr(base_policy, "step"):
        return _StepPolicyAdapter(base_policy)
    raise ValueError("base_policy must provide act() or step()")


def _normalize_config(id_config: IdentificationConfig, env: EnvConfig) -> IdentificationConfig:
    rs_nom = id_config.rs_nom if id_config.rs_nom is not None else env.motor.Rs
    rs_max = id_config.rs_max if id_config.rs_max is not None else rs_nom * 2.0
    pulse_v_q = id_config.pulse_v_q
    if pulse_v_q is None:
        target_current = max(id_config.current_ratio, 0.1) * env.motor.I_n
        pulse_v_q = rs_nom * target_current
    v_limit = 0.1 * env.inverter.Vdc
    pulse_v_q = float(max(-v_limit, min(v_limit, pulse_v_q)))

    i_inst_max = id_config.i_inst_max or 1.2 * env.motor.I_n
    i_rms_max = id_config.i_rms_max or 0.8 * env.motor.I_n
    omega_max = id_config.omega_max or 0.2 * _rated_omega()
    vdc_max = id_config.vdc_max or 1.05 * env.inverter.Vdc

    return IdentificationConfig(
        env=env,
        dt=id_config.dt,
        duration=id_config.duration,
        pulse_start=id_config.pulse_start,
        pulse_end=id_config.pulse_end,
        pulse_v_d=id_config.pulse_v_d,
        pulse_v_q=float(pulse_v_q),
        rs_nom=rs_nom,
        rs_max=rs_max,
        current_ratio=id_config.current_ratio,
        i_inst_max=i_inst_max,
        i_rms_max=i_rms_max,
        omega_max=omega_max,
        vdc_max=vdc_max,
        log_root=id_config.log_root,
        run_name=id_config.run_name,
        seed=id_config.seed,
    )


def _validate_config(config: IdentificationConfig) -> None:
    if config.dt <= 0.0:
        raise ValueError("id_config.dt must be positive")
    if config.duration <= 0.0:
        raise ValueError("id_config.duration must be positive")
    if config.pulse_start < 0.0 or config.pulse_end < 0.0:
        raise ValueError("pulse_start/end must be non-negative")
    if config.pulse_end <= config.pulse_start:
        raise ValueError("pulse_end must be greater than pulse_start")
    if config.pulse_end > config.duration:
        raise ValueError("pulse_end must be within duration")


def _estimate_rs(npz_path: Path | None, config: IdentificationConfig) -> tuple[dict[str, float], float]:
    if npz_path is None:
        raise ValueError("identification run did not produce npz logs")
    data = np.load(npz_path)
    t = np.asarray(data["t"], dtype=float)
    v_q = np.asarray(data["v_q"], dtype=float)
    i_q = np.asarray(data["i_q"], dtype=float)

    window_end = min(config.pulse_end, float(t[-1]) if t.size else config.pulse_end)
    window_start = max(config.pulse_start, window_end - 0.2 * (config.pulse_end - config.pulse_start))
    mask = (t >= window_start) & (t <= window_end)
    if not np.any(mask):
        rs_est = float(config.rs_nom or 0.0)
        return {"status": "insufficient_data"}, rs_est

    v_mean = float(np.mean(np.abs(v_q[mask])))
    i_mean = float(np.mean(np.abs(i_q[mask])))
    eps = 1e-6
    rs_est = v_mean / max(i_mean, eps)
    metrics = {
        "v_q_mean": v_mean,
        "i_q_mean": i_mean,
        "window_start": float(window_start),
        "window_end": float(window_end),
    }
    return metrics, rs_est


def _compute_aging(rs_est: float, rs_nom: float | None, rs_max: float | None) -> float:
    if rs_nom is None or rs_max is None or rs_max <= rs_nom:
        return 0.0
    aging = (rs_est - rs_nom) / (rs_max - rs_nom)
    return float(min(1.0, max(0.0, aging)))


def _rated_omega() -> float:
    return float(2.0 * np.pi * NAMEPLATE_N_RATED / 60.0)


def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


__all__ = ["IdentificationConfig", "run_id_sequence"]
