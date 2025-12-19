from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Mapping

import numpy as np

import bench.orchestrator as orchestrator
from bench.identification import IdentificationConfig, run_id_sequence
from bench.orchestrator import (
    ExperimentConfig,
    ParamDriftConfig,
    ProfileConfig,
    default_experiment_config,
    run_experiment,
)
from config.env import NAMEPLATE_N_RATED, NAMEPLATE_P_KW, EnvConfig, create_default_env
from sim.load_servo import ServoLoadConfig


@dataclass(frozen=True)
class TestsuiteConfig:
    """Configuration for building a testsuite."""

    __test__ = False
    duration: float = 60.0
    dt: float | None = None
    controller: str | None = None
    seed: int = 1
    log_root: str | Path = "logs"
    run_prefix: str = "testsuite"
    with_identification: bool = False
    identify_once_per_suite: bool = True
    identify_per_case: bool = False
    id_config: IdentificationConfig | None = None


@dataclass(frozen=True)
class TestCase:
    """Single testsuite case definition."""

    __test__ = False
    name: str
    description: str
    config: ExperimentConfig


@dataclass(frozen=True)
class TestsuiteResult:
    """Result bundle for a testsuite run."""

    __test__ = False
    run_dir: Path
    results: list


def build_testsuite(
    env: EnvConfig | None = None,
    suite_cfg: TestsuiteConfig | None = None,
    run_dir: Path | None = None,
    safety: object | None = None,
) -> list[TestCase]:
    env = env or create_default_env()
    suite_cfg = suite_cfg or TestsuiteConfig()
    run_dir = run_dir or _resolve_testsuite_dir(Path(suite_cfg.log_root), suite_cfg.run_prefix)

    base = default_experiment_config(env)
    dt = suite_cfg.dt if suite_cfg.dt is not None else base.dt
    duration = suite_cfg.duration
    controller = suite_cfg.controller if suite_cfg.controller is not None else base.controller

    omega_rated = _rated_omega()
    omega_base = 0.8 * omega_rated
    omega_low = 0.1 * omega_rated
    t_rated = _rated_torque()
    t_max = 1.5 * t_rated

    load_config = ServoLoadConfig(mode="torque", tau_load=0.05, t_max=t_max)

    speed_step = ProfileConfig(kind="step", initial=0.0, value=omega_base, step_time=0.1 * duration)
    speed_const = ProfileConfig(kind="constant", value=omega_base)
    speed_low = ProfileConfig(kind="step", initial=0.0, value=omega_low, step_time=0.1 * duration)

    load_zero = ProfileConfig(kind="constant", value=0.0)
    load_step = ProfileConfig(
        kind="step",
        initial=0.0,
        value=0.6 * t_rated,
        step_time=0.2 * duration,
    )
    load_sin = ProfileConfig(
        kind="sin",
        value=0.3 * t_rated,
        amplitude=0.2 * t_rated,
        frequency=0.2,
    )
    load_random = ProfileConfig(
        kind="random_step",
        value=0.0,
        amplitude=0.5 * t_rated,
        step_interval=1.0,
        seed=suite_cfg.seed,
    )
    load_low = ProfileConfig(kind="constant", value=0.2 * t_rated)

    disturbance = ProfileConfig(
        kind="step",
        initial=0.0,
        value=0.5 * t_rated,
        step_time=0.6 * duration,
    )

    drift = ParamDriftConfig(
        start_time=0.3 * duration,
        end_time=0.9 * duration,
        rs_scale_end=1.2,
        rr_scale_end=1.2,
        lm_scale_end=0.9,
        j_scale_end=1.1,
        b_scale_end=1.2,
    )

    cases: list[TestCase] = []
    cases.append(
        TestCase(
            name="E1",
            description="step speed, no load",
            config=_case_config(
                base, dt, duration, controller, run_dir, "E1", speed_step, load_zero, load_config, safety=safety
            ),
        )
    )
    cases.append(
        TestCase(
            name="E2",
            description="constant speed + step load torque",
            config=_case_config(
                base, dt, duration, controller, run_dir, "E2", speed_const, load_step, load_config, safety=safety
            ),
        )
    )
    cases.append(
        TestCase(
            name="E3",
            description="constant speed + sinus load",
            config=_case_config(
                base, dt, duration, controller, run_dir, "E3", speed_const, load_sin, load_config, safety=safety
            ),
        )
    )
    cases.append(
        TestCase(
            name="E4",
            description="constant speed + random-step load",
            config=_case_config(
                base,
                dt,
                duration,
                controller,
                run_dir,
                "E4",
                speed_const,
                load_random,
                load_config,
                safety=safety,
            ),
        )
    )
    cases.append(
        TestCase(
            name="E5",
            description="low speed tracking",
            config=_case_config(
                base, dt, duration, controller, run_dir, "E5", speed_low, load_low, load_config, safety=safety
            ),
        )
    )
    cases.append(
        TestCase(
            name="E6",
            description="parameter drift during run",
            config=_case_config(
                base,
                dt,
                duration,
                controller,
                run_dir,
                "E6",
                speed_const,
                load_low,
                load_config,
                param_drift=drift,
                safety=safety,
            ),
        )
    )
    cases.append(
        TestCase(
            name="E7",
            description="disturbance injection (additive load)",
            config=_case_config(
                base,
                dt,
                duration,
                controller,
                run_dir,
                "E7",
                speed_const,
                load_low,
                load_config,
                disturbance_profile=disturbance,
                safety=safety,
            ),
        )
    )
    return cases


def run_testsuite(
    env: EnvConfig | None = None,
    suite_cfg: TestsuiteConfig | None = None,
    policy: object | None = None,
    safety: object | None = None,
) -> TestsuiteResult:
    env = env or create_default_env()
    suite_cfg = suite_cfg or TestsuiteConfig()
    run_dir = _resolve_testsuite_dir(Path(suite_cfg.log_root), suite_cfg.run_prefix)
    cases = build_testsuite(env=env, suite_cfg=suite_cfg, run_dir=run_dir, safety=safety)
    cases = _apply_identification(cases, env, suite_cfg, run_dir)

    results = []
    for case in cases:
        results.append(run_experiment(case.config, policy=policy))
    return TestsuiteResult(run_dir=run_dir, results=results)


def _case_config(
    base: ExperimentConfig,
    dt: float,
    duration: float,
    controller: str,
    run_dir: Path,
    run_name: str,
    speed_profile: ProfileConfig,
    load_profile: ProfileConfig,
    load_config: ServoLoadConfig,
    disturbance_profile: ProfileConfig | None = None,
    param_drift: ParamDriftConfig | None = None,
    safety: object | None = None,
) -> ExperimentConfig:
    return replace(
        base,
        dt=dt,
        duration=duration,
        controller=controller,
        speed_profile=speed_profile,
        load_profile=load_profile,
        disturbance_profile=disturbance_profile,
        load_config=load_config,
        param_drift=param_drift,
        safety=safety if safety is not None else base.safety,
        log_root=run_dir,
        run_name=run_name,
    )


def _resolve_testsuite_dir(log_root: Path, prefix: str) -> Path:
    log_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = log_root / f"{prefix}_{stamp}"
    if not base.exists():
        base.mkdir(parents=True)
        return base
    idx = 1
    while True:
        candidate = log_root / f"{prefix}_{stamp}_{idx}"
        if not candidate.exists():
            candidate.mkdir(parents=True)
            return candidate
        idx += 1


def _rated_omega() -> float:
    return float(2.0 * np.pi * NAMEPLATE_N_RATED / 60.0)


def _rated_torque() -> float:
    omega = _rated_omega()
    if omega <= 0.0:
        raise ValueError("rated omega must be positive")
    return float(NAMEPLATE_P_KW * 1000.0 / omega)


def _apply_identification(
    cases: list[TestCase],
    env: EnvConfig,
    suite_cfg: TestsuiteConfig,
    run_dir: Path,
) -> list[TestCase]:
    if not suite_cfg.with_identification:
        return cases
    if suite_cfg.identify_per_case:
        updated = []
        for case in cases:
            id_cfg = _resolve_id_config(env, suite_cfg, run_dir, run_name=f"{case.name}_identify")
            id_result = run_id_sequence(orchestrator, id_cfg, base_policy=None)
            context = {"aging": id_result["aging"], "est_params": id_result["est_params"]}
            updated.append(_attach_context(case, context, id_result))
        return updated

    if suite_cfg.identify_once_per_suite:
        id_cfg = _resolve_id_config(env, suite_cfg, run_dir, run_name="identify")
        id_result = run_id_sequence(orchestrator, id_cfg, base_policy=None)
        context = {"aging": id_result["aging"], "est_params": id_result["est_params"]}
        return [_attach_context(case, context, id_result) for case in cases]
    return cases


def _resolve_id_config(
    env: EnvConfig, suite_cfg: TestsuiteConfig, run_dir: Path, run_name: str
) -> IdentificationConfig:
    if suite_cfg.id_config is None:
        id_cfg = IdentificationConfig(env=env)
    else:
        id_cfg = suite_cfg.id_config
        if id_cfg.env is None:
            id_cfg = replace(id_cfg, env=env)
    dt = suite_cfg.dt if suite_cfg.dt is not None else id_cfg.dt
    return replace(
        id_cfg,
        dt=dt,
        log_root=str(run_dir),
        run_name=run_name,
        seed=suite_cfg.seed,
    )


def _attach_context(
    case: TestCase, context: Mapping[str, object], identification: Mapping[str, object]
) -> TestCase:
    return replace(
        case,
        config=replace(case.config, context=context, identification=identification),
    )


__all__ = [
    "TestsuiteConfig",
    "TestCase",
    "TestsuiteResult",
    "build_testsuite",
    "run_testsuite",
]
