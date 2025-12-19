from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Mapping, Sequence

import json
import math
import numpy as np

from config.env import (
    EnvConfig,
    FocParams,
    InverterParams,
    MotorParams,
    NAMEPLATE_N_RATED,
    NAMEPLATE_P_KW,
    ScalarVfParams,
    create_default_env,
)
from control.scalar_vf import ScalarVfController
from control.vector_foc import FocController
from models.induction_motor import InductionMotorModel
from models.inverter_ideal import IdealInverter
from models.transformations import dq_to_abc
from sim.load_servo import ServoLoadConfig, ServoLoadModel
from sim.safety import SafetyLimits, SafetySupervisor
from sim.sensors import SensorConfig, SensorModel, SensorReading

SignalFn = Callable[[float, int], tuple[Mapping[str, float], Mapping[str, object]]]

BENCH_VERSION = "0.1.0"
DEFAULT_I_INST_MULT = 10.0
DEFAULT_I_RMS_MULT = 5.0
DEFAULT_OMEGA_MULT = 3.0
DEFAULT_VDC_MULT = 1.2
LCG_A = 1664525
LCG_C = 1013904223
LCG_M = 2**32
ACTION_TRACE_LEN = 200


@dataclass(frozen=True)
class ProfileConfig:
    """Parametric profile for speed/load commands."""

    kind: str
    value: float
    initial: float = 0.0
    step_time: float = 0.0
    amplitude: float = 0.0
    frequency: float = 0.0
    step_interval: float = 0.0
    seed: int | None = None


@dataclass(frozen=True)
class ParamDriftConfig:
    """Linear parameter drift from start_time to end_time."""

    start_time: float
    end_time: float
    rs_scale_end: float = 1.0
    rr_scale_end: float = 1.0
    lm_scale_end: float = 1.0
    j_scale_end: float = 1.0
    b_scale_end: float = 1.0


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for a single experiment run."""

    dt: float
    duration: float
    safety: SafetySupervisor | None = None
    signal_fn: SignalFn | None = None
    motor: MotorParams | None = None
    inverter: InverterParams | None = None
    scalar_vf: ScalarVfParams | None = None
    foc: FocParams | None = None
    controller: str = "foc"
    speed_profile: ProfileConfig | None = None
    load_profile: ProfileConfig | None = None
    disturbance_profile: ProfileConfig | None = None
    load_config: ServoLoadConfig | None = None
    param_drift: ParamDriftConfig | None = None
    context: Mapping[str, object] | None = None
    identification: Mapping[str, object] | None = None
    sensor: SensorConfig = field(default_factory=SensorConfig)
    seed: int | None = None
    log_root: str | Path = "logs"
    run_name: str | None = None


@dataclass(frozen=True)
class ExperimentResult:
    """Outcome of a single experiment run."""

    ok: bool
    terminated_early: bool
    safety_trip: bool
    reason: str
    steps_run: int
    duration: float
    log_dir: Path | None = None
    npz_path: Path | None = None
    json_path: Path | None = None


class PolicyAdapter:
    """Adapter that turns policy.act(state)->action into controller.step()."""

    def __init__(self, policy: object, dt: float, context: Mapping[str, object] | None) -> None:
        self._policy = policy
        self._dt = dt
        self._context = dict(context) if context is not None else {}

    def step(
        self,
        t: float,
        omega_ref: float,
        omega_m: float,
        i_abc: Sequence[float],
        torque_e: float,
        theta_mech: float,
    ) -> tuple[float, float, float, float, dict]:
        state = {
            "t": t,
            "dt": self._dt,
            "omega_ref": omega_ref,
            "omega_m": omega_m,
            "i_abc": tuple(float(v) for v in i_abc),
            "torque_e": torque_e,
            "theta_mech": theta_mech,
        }
        state["aging"] = float(self._context.get("aging", 0.0))
        state["est_params"] = self._context.get("est_params", {})
        state["context"] = self._context
        action = self._policy.act(state)
        v_d, v_q, theta_e, omega_syn = _extract_action(action)
        return v_d, v_q, theta_e, omega_syn, {}


def default_experiment_config(env: EnvConfig | None = None) -> ExperimentConfig:
    """Create a default bench config from an EnvConfig."""
    env = env or create_default_env()
    omega_rated = _rated_omega()
    speed_profile = ProfileConfig(
        kind="step",
        initial=0.0,
        value=omega_rated,
        step_time=0.1 * env.sim.t_end,
    )
    load_profile = ProfileConfig(kind="constant", value=env.sim.load_torque)
    load_config = ServoLoadConfig(
        mode="torque",
        tau_load=0.05,
        t_max=_rated_torque(),
    )
    safety = SafetySupervisor(
        SafetyLimits(
            i_inst_max=DEFAULT_I_INST_MULT * env.motor.I_n,
            i_rms_max=DEFAULT_I_RMS_MULT * env.motor.I_n,
            omega_max=DEFAULT_OMEGA_MULT * omega_rated,
            vdc_max=DEFAULT_VDC_MULT * env.inverter.Vdc,
        )
    )
    return ExperimentConfig(
        dt=env.sim.dt,
        duration=env.sim.t_end,
        safety=safety,
        motor=env.motor,
        inverter=env.inverter,
        scalar_vf=env.scalar_vf,
        foc=env.foc,
        controller=env.sim.mode,
        speed_profile=speed_profile,
        load_profile=load_profile,
        disturbance_profile=None,
        load_config=load_config,
        param_drift=None,
        sensor=SensorConfig(),
        seed=None,
        log_root="logs",
        run_name=None,
    )


def run_experiment(config: ExperimentConfig, policy: object | None = None) -> ExperimentResult:
    """Run an experiment loop with optional safety supervision."""
    if config.dt <= 0.0:
        raise ValueError("dt must be positive")
    if config.duration <= 0.0:
        raise ValueError("duration must be positive")
    if config.param_drift is not None:
        _validate_param_drift(config.param_drift)

    if config.signal_fn is not None:
        return _run_signal_experiment(config)
    return _run_sim_experiment(config, policy)


def _run_signal_experiment(config: ExperimentConfig) -> ExperimentResult:
    steps_total = int(config.duration / config.dt)
    if steps_total <= 0:
        raise ValueError("duration must be at least one dt")

    steps_run = 0
    for step in range(steps_total):
        t = step * config.dt
        state, signals = config.signal_fn(t, step)
        steps_run = step + 1
        if config.safety is not None:
            ok, reason = config.safety.check(state, signals)
            if not ok:
                return ExperimentResult(
                    ok=False,
                    terminated_early=True,
                    safety_trip=True,
                    reason=reason,
                    steps_run=steps_run,
                    duration=t,
                )

    return ExperimentResult(
        ok=True,
        terminated_early=False,
        safety_trip=False,
        reason="",
        steps_run=steps_run,
        duration=config.duration,
    )


def _run_sim_experiment(config: ExperimentConfig, policy: object | None) -> ExperimentResult:
    env = _require_env(config)
    controller = _init_controller(config, policy)
    inverter = IdealInverter(env.inverter)
    motor = InductionMotorModel(env.motor)
    base_params = env.motor
    load = ServoLoadModel(config.load_config)
    sensors = SensorModel(_with_seed(config.sensor, config.seed))

    vdc = env.inverter.Vdc
    omega_m = 0.0
    torque_e = 0.0
    theta_mech = 0.0
    i_abc = (0.0, 0.0, 0.0)
    measurement = sensors.measure(omega_m, i_abc, vdc)

    data = _init_timeseries()
    steps_total = int(config.duration / config.dt)
    if steps_total <= 0:
        raise ValueError("duration must be at least one dt")

    terminated_early = False
    reason = ""
    steps_run = 0
    for step in range(steps_total):
        t = step * config.dt
        if config.param_drift is not None:
            motor.update_params(_drift_params(base_params, config.param_drift, t))
        omega_ref = _eval_profile(config.speed_profile, t, "speed_profile")
        load_cmd = _eval_profile(config.load_profile, t, "load_profile")
        disturbance = 0.0
        if config.disturbance_profile is not None:
            disturbance = _eval_profile(config.disturbance_profile, t, "disturbance_profile")
        load_cmd += disturbance

        v_d, v_q, theta_e, omega_syn, _ = controller.step(
            t,
            omega_ref,
            measurement.omega,
            measurement.i_abc,
            torque_e,
            theta_mech,
        )

        _, v_dq = inverter.output(v_d, v_q, theta_e)
        t_load = load.step(config.dt, omega_m, load_cmd)
        state, i_ds, i_qs, torque_e, omega_m = motor.step(
            v_dq[0], v_dq[1], t_load, config.dt, omega_syn
        )
        i_abc = dq_to_abc(i_ds, i_qs, theta_e)
        theta_mech += omega_m * config.dt
        measurement = sensors.measure(omega_m, i_abc, vdc)

        ok, safety_flag, reason = _check_safety(config.safety, omega_m, measurement, reason)
        steps_run = step + 1
        _append_timeseries(
            data,
            t + config.dt,
            omega_m,
            omega_ref,
            i_ds,
            i_qs,
            i_abc,
            v_dq,
            t_load,
            torque_e,
            safety_flag,
        )

        if not ok:
            terminated_early = True
            break

    log_dir, npz_path, json_path = _write_logs(config, data, steps_run)
    return ExperimentResult(
        ok=not terminated_early,
        terminated_early=terminated_early,
        safety_trip=terminated_early,
        reason=reason,
        steps_run=steps_run,
        duration=steps_run * config.dt,
        log_dir=log_dir,
        npz_path=npz_path,
        json_path=json_path,
    )


def _require_env(config: ExperimentConfig) -> EnvConfig:
    if config.motor is None or config.inverter is None:
        raise ValueError("motor and inverter config must be set for simulation mode")
    if config.scalar_vf is None or config.foc is None:
        raise ValueError("scalar_vf and foc config must be set for simulation mode")
    if config.speed_profile is None or config.load_profile is None:
        raise ValueError("speed_profile and load_profile must be set for simulation mode")
    if config.load_config is None:
        raise ValueError("load_config must be set for simulation mode")
    return EnvConfig(
        motor=config.motor,
        inverter=config.inverter,
        scalar_vf=config.scalar_vf,
        foc=config.foc,
        sim=create_default_env().sim,
    )


def _init_controller(config: ExperimentConfig, policy: object | None) -> object:
    if policy is None:
        if config.controller == "scalar":
            return ScalarVfController(
                config.scalar_vf, config.dt, config.motor.p, config.inverter.Vdc
            )
        if config.controller == "foc":
            return FocController(config.foc, config.motor, config.dt)
        raise ValueError(f"Unsupported controller type: {config.controller!r}")
    if hasattr(policy, "reset"):
        policy.reset()
    if hasattr(policy, "act") and not hasattr(policy, "step"):
        return PolicyAdapter(policy, config.dt, config.context)
    if not hasattr(policy, "step"):
        raise ValueError("policy must provide step() method or act()")
    return policy


def _eval_profile(profile: ProfileConfig, t: float, name: str) -> float:
    if profile is None:
        raise ValueError(f"{name} is required for simulation mode")
    if profile.kind == "constant":
        return profile.value
    if profile.kind == "step":
        return profile.initial if t < profile.step_time else profile.value
    if profile.kind == "sin":
        return profile.value + profile.amplitude * np.sin(2.0 * np.pi * profile.frequency * t)
    if profile.kind == "random_step":
        if profile.step_interval <= 0.0:
            raise ValueError(f"{name} step_interval must be positive for random_step")
        step_idx = int(t / profile.step_interval)
        seed = 0 if profile.seed is None else int(profile.seed)
        rnd = _random_step_value(seed, step_idx)
        return profile.value + profile.amplitude * rnd
    raise ValueError(f"Unsupported {name} kind: {profile.kind!r}")


def _with_seed(config: SensorConfig, seed: int | None) -> SensorConfig:
    if seed is None:
        return config
    return SensorConfig(
        omega=config.omega,
        currents=config.currents,
        vdc=config.vdc,
        seed=seed,
    )


def _check_safety(
    safety: SafetySupervisor | None,
    omega_m: float,
    measurement: SensorReading,
    last_reason: str,
) -> tuple[bool, int, str]:
    if safety is None:
        return True, 0, ""
    i_rms = _calc_i_rms(measurement.i_abc)
    ok, reason = safety.check(
        {"omega": omega_m},
        {"i_abc": measurement.i_abc, "vdc": measurement.vdc, "i_rms": i_rms},
    )
    if ok:
        return True, 0, ""
    return False, 1, reason or last_reason


def _calc_i_rms(i_abc: Sequence[float]) -> float:
    ia, ib, ic = i_abc
    return math.sqrt((ia * ia + ib * ib + ic * ic) / 3.0)


def _init_timeseries() -> dict[str, list[float]]:
    return {
        "t": [],
        "omega_m": [],
        "omega_ref": [],
        "i_d": [],
        "i_q": [],
        "i_a": [],
        "i_b": [],
        "i_c": [],
        "v_d": [],
        "v_q": [],
        "t_load": [],
        "t_e": [],
        "flag_safety": [],
    }


def _append_timeseries(
    data: dict[str, list[float]],
    t: float,
    omega_m: float,
    omega_ref: float,
    i_d: float,
    i_q: float,
    i_abc: Sequence[float],
    v_dq: Sequence[float],
    t_load: float,
    t_e: float,
    flag_safety: int,
) -> None:
    data["t"].append(float(t))
    data["omega_m"].append(float(omega_m))
    data["omega_ref"].append(float(omega_ref))
    data["i_d"].append(float(i_d))
    data["i_q"].append(float(i_q))
    data["i_a"].append(float(i_abc[0]))
    data["i_b"].append(float(i_abc[1]))
    data["i_c"].append(float(i_abc[2]))
    data["v_d"].append(float(v_dq[0]))
    data["v_q"].append(float(v_dq[1]))
    data["t_load"].append(float(t_load))
    data["t_e"].append(float(t_e))
    data["flag_safety"].append(float(flag_safety))


def _write_logs(
    config: ExperimentConfig, data: dict[str, list[float]], steps_run: int
) -> tuple[Path, Path, Path]:
    log_root = Path(config.log_root)
    log_root.mkdir(parents=True, exist_ok=True)
    run_dir = _resolve_run_dir(log_root, config.run_name)

    npz_path = run_dir / "timeseries.npz"
    np.savez(npz_path, **{k: np.asarray(v) for k, v in data.items()})

    action_trace_path, action_trace_len = _write_action_trace(run_dir, data, ACTION_TRACE_LEN)

    meta = {
        "bench_version": BENCH_VERSION,
        "npz_path": str(npz_path),
        "action_trace_path": str(action_trace_path),
        "action_trace_len": action_trace_len,
        "steps": steps_run,
        "config": _config_to_dict(config),
        "metrics": {},
        "seed": config.seed,
        "context": config.context,
        "identification": config.identification,
    }
    json_path = run_dir / "run_meta.json"
    with json_path.open("w", encoding="utf-8") as handle:
        handle.write(_safe_json(meta))
    return run_dir, npz_path, json_path


def _resolve_run_dir(log_root: Path, run_name: str | None) -> Path:
    if run_name:
        run_dir = log_root / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = log_root / f"run_{stamp}"
    if not base.exists():
        base.mkdir(parents=True)
        return base
    idx = 1
    while True:
        candidate = log_root / f"run_{stamp}_{idx}"
        if not candidate.exists():
            candidate.mkdir(parents=True)
            return candidate
        idx += 1


def _write_action_trace(
    run_dir: Path, data: dict[str, list[float]], max_steps: int
) -> tuple[Path, int]:
    t = data.get("t", [])
    v_d = data.get("v_d", [])
    v_q = data.get("v_q", [])
    steps = min(len(t), len(v_d), len(v_q), max_steps)
    trace = [
        {"t": float(t[idx]), "v_d": float(v_d[idx]), "v_q": float(v_q[idx])}
        for idx in range(steps)
    ]
    path = run_dir / "action_trace.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(trace, handle, indent=2, sort_keys=True, ensure_ascii=True)
        handle.write("\n")
    return path, steps


def _config_to_dict(config: ExperimentConfig) -> dict[str, object]:
    payload = {
        "dt": config.dt,
        "duration": config.duration,
        "controller": config.controller,
        "speed_profile": asdict(config.speed_profile) if config.speed_profile else None,
        "load_profile": asdict(config.load_profile) if config.load_profile else None,
        "disturbance_profile": (
            asdict(config.disturbance_profile) if config.disturbance_profile else None
        ),
        "load_config": asdict(config.load_config) if config.load_config else None,
        "param_drift": asdict(config.param_drift) if config.param_drift else None,
        "sensor": asdict(config.sensor),
        "log_root": str(config.log_root),
        "run_name": config.run_name,
    }
    if config.motor is not None:
        payload["motor"] = asdict(config.motor)
    if config.inverter is not None:
        payload["inverter"] = asdict(config.inverter)
    if config.scalar_vf is not None:
        payload["scalar_vf"] = asdict(config.scalar_vf)
    if config.foc is not None:
        payload["foc"] = asdict(config.foc)
    if config.safety is not None:
        payload["safety"] = asdict(config.safety.limits)
    return payload


def _random_step_value(seed: int, step_idx: int) -> float:
    state = (seed + step_idx * LCG_A + LCG_C) & (LCG_M - 1)
    return (state / float(LCG_M - 1)) * 2.0 - 1.0


def _drift_params(base: MotorParams, drift: ParamDriftConfig, t: float) -> MotorParams:
    rs_scale = _linear_scale(t, drift.start_time, drift.end_time, drift.rs_scale_end)
    rr_scale = _linear_scale(t, drift.start_time, drift.end_time, drift.rr_scale_end)
    lm_scale = _linear_scale(t, drift.start_time, drift.end_time, drift.lm_scale_end)
    j_scale = _linear_scale(t, drift.start_time, drift.end_time, drift.j_scale_end)
    b_scale = _linear_scale(t, drift.start_time, drift.end_time, drift.b_scale_end)
    return MotorParams(
        Rs=base.Rs * rs_scale,
        Rr=base.Rr * rr_scale,
        Ls_sigma=base.Ls_sigma,
        Lr_sigma=base.Lr_sigma,
        Lm=base.Lm * lm_scale,
        J=base.J * j_scale,
        B=base.B * b_scale,
        p=base.p,
        I_n=base.I_n,
    )


def _linear_scale(t: float, start: float, end: float, end_scale: float) -> float:
    if end <= start:
        return end_scale
    if t <= start:
        return 1.0
    if t >= end:
        return end_scale
    ratio = (t - start) / (end - start)
    return 1.0 + (end_scale - 1.0) * ratio


def _validate_param_drift(drift: ParamDriftConfig) -> None:
    if drift.start_time < 0.0 or drift.end_time < 0.0:
        raise ValueError("param_drift times must be non-negative")
    for name, value in (
        ("rs_scale_end", drift.rs_scale_end),
        ("rr_scale_end", drift.rr_scale_end),
        ("lm_scale_end", drift.lm_scale_end),
        ("j_scale_end", drift.j_scale_end),
        ("b_scale_end", drift.b_scale_end),
    ):
        if value <= 0.0:
            raise ValueError(f"param_drift {name} must be positive")


def _extract_action(action: object) -> tuple[float, float, float, float]:
    if isinstance(action, Mapping):
        v_d = float(action.get("v_d", 0.0))
        v_q = float(action.get("v_q", 0.0))
        theta_e = float(action.get("theta_e", 0.0))
        omega_syn = float(action.get("omega_syn", 0.0))
        return v_d, v_q, theta_e, omega_syn
    for attr in ("v_d", "v_q", "theta_e", "omega_syn"):
        if not hasattr(action, attr):
            raise ValueError("policy action must provide v_d, v_q, theta_e, omega_syn")
    return (
        float(getattr(action, "v_d")),
        float(getattr(action, "v_q")),
        float(getattr(action, "theta_e")),
        float(getattr(action, "omega_syn")),
    )


def _safe_json(payload: Mapping[str, object]) -> str:
    return (
        json_dumps(payload)
        + "\n"
    )


def json_dumps(payload: Mapping[str, object]) -> str:
    import json

    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True)


def _rated_omega() -> float:
    return float(2.0 * np.pi * NAMEPLATE_N_RATED / 60.0)


def _rated_torque() -> float:
    omega = _rated_omega()
    if omega <= 0.0:
        raise ValueError("Rated omega must be positive")
    return float(NAMEPLATE_P_KW * 1000.0 / omega)


__all__ = [
    "ProfileConfig",
    "ParamDriftConfig",
    "ExperimentConfig",
    "ExperimentResult",
    "default_experiment_config",
    "run_experiment",
]
