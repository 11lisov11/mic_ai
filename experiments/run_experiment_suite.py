from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.env import EnvConfig, create_default_env
from drivers import BaseDriver, HwDriverStub, SimDriver
from metrics.score import (
    DEFAULT_FAULT_PENALTY,
    DEFAULT_SCORE_WEIGHTS,
    compute_metrics,
    score_from_metrics,
)
from mic_ai.analysis.metrics import calc_i_rms, calc_p_el, calc_p_mech
from tools.report import build_report
from mic_ai.tools.plot_style import apply_vak_style, ensure_matplotlib, save_figure
from optim.cmaes import normalize_param_space, optimize as cmaes_optimize
from control.vector_foc import FocController


SCENARIOS = ("speed_step", "ramp", "load_step")
OMEGA_MAX_MULT = 1.5
DEFAULT_PARAM_SPACE = [
    {"name": "k_vd", "low": 0.6, "high": 1.4, "init": 1.0},
    {"name": "k_vq", "low": 0.6, "high": 1.4, "init": 1.0},
]


def _load_env_config(path: Optional[str]) -> EnvConfig:
    if path is None:
        return create_default_env()
    from mic_ai.core.env import make_env_from_config

    env = make_env_from_config(path)
    return env.env_config


def _config_hash(env_cfg: EnvConfig) -> str:
    payload = json.dumps(asdict(env_cfg), sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _with_scenario(env_cfg: EnvConfig, scenario: str) -> EnvConfig:
    sim_cfg = replace(env_cfg.sim, scenario_name=str(scenario))
    return replace(env_cfg, sim=sim_cfg)


def _default_limits(env_cfg: EnvConfig) -> Dict[str, Optional[float]]:
    v_max = float(env_cfg.inverter.Vdc) / math.sqrt(3.0) if env_cfg.inverter.Vdc > 0.0 else 0.0
    i_max = getattr(env_cfg.foc, "iq_limit", None)
    if i_max is None or i_max <= 0.0:
        i_max = float(getattr(env_cfg.motor, "I_n", 0.0) or 0.0)
    omega_base = 2.0 * math.pi * env_cfg.scalar_vf.f_max / env_cfg.motor.p
    omega_max = OMEGA_MAX_MULT * omega_base if omega_base > 0.0 else 0.0
    return {
        "i_max": float(i_max) if i_max and i_max > 0.0 else None,
        "v_max": float(v_max) if v_max and v_max > 0.0 else None,
        "omega_max": float(omega_max) if omega_max and omega_max > 0.0 else None,
    }


class MicPolicy:
    def __init__(
        self,
        env_cfg: EnvConfig,
        limits: Dict[str, Optional[float]],
        policy: str = "zero",
        checkpoint: Optional[str] = None,
        action_std: Optional[float] = None,
        device: str = "cpu",
    ) -> None:
        self.policy = str(policy).lower()
        self.checkpoint = checkpoint
        self.device = device
        self._v_max = max(float(limits.get("v_max") or 0.0), 1e-6)
        self._omega_base = 2.0 * math.pi * env_cfg.scalar_vf.f_max / env_cfg.motor.p
        self._i_base = max(float(getattr(env_cfg.motor, "I_n", 1.0)), 1e-6)
        self._last_action = (0.0, 0.0)
        self._agent = None
        self._feature_keys: List[str] = []
        self._action_std = action_std
        self._load_agent_if_needed()

    def _load_agent_if_needed(self) -> None:
        if self.policy != "ppo_voltage":
            return
        if self.checkpoint is None:
            self.policy = "zero"
            return
        try:
            import torch
            from mic_ai.ai.agents.ppo_voltage import PPOVoltageAgent
            from mic_ai.ai.train_ai_voltage import FEATURE_KEYS
        except Exception:
            self.policy = "zero"
            return

        self._feature_keys = list(FEATURE_KEYS)
        self._agent = PPOVoltageAgent(feature_keys=self._feature_keys, action_dim=2, device=self.device)
        state = torch.load(self.checkpoint, map_location=self.device)
        self._agent.net.load_state_dict(state)
        if self._action_std is not None:
            self._agent.set_action_std(float(self._action_std))

    def reset(self) -> None:
        self._last_action = (0.0, 0.0)

    def _build_features(self, obs: Dict[str, float]) -> Dict[str, float]:
        omega = float(obs.get("omega", 0.0))
        omega_ref = float(obs.get("omega_ref", 0.0))
        i_d = float(obs.get("id", 0.0))
        i_q = float(obs.get("iq", 0.0))
        omega_syn = float(obs.get("omega_syn", omega))

        slip = omega_syn - omega
        slip_base = max(abs(omega_syn), abs(omega_ref), 1e-6)

        def _clip(val: float, low: float = -1.0, high: float = 1.0) -> float:
            return float(max(low, min(high, val))) if math.isfinite(val) else 0.0

        return {
            "omega_norm": _clip(omega / max(self._omega_base, 1e-6)),
            "omega_ref_norm": _clip(omega_ref / max(self._omega_base, 1e-6)),
            "err_norm": _clip((omega_ref - omega) / max(self._omega_base, 1e-6)),
            "id_norm": _clip(i_d / self._i_base, low=-5.0, high=5.0),
            "iq_norm": _clip(i_q / self._i_base, low=-5.0, high=5.0),
            "slip_norm": _clip(slip / slip_base),
            "last_action_vd": float(self._last_action[0] / self._v_max),
            "last_action_vq": float(self._last_action[1] / self._v_max),
        }

    def act(self, obs: Dict[str, float]) -> Tuple[float, float]:
        if self.policy != "ppo_voltage" or self._agent is None:
            self._last_action = (0.0, 0.0)
            return self._last_action
        features = self._build_features(obs)
        action, _logp, _value = self._agent.act(features)
        vd = float(action[0]) * self._v_max
        vq = float(action[1]) * self._v_max
        self._last_action = (vd, vq)
        return self._last_action

    def describe(self) -> Dict[str, str]:
        return {
            "policy": self.policy,
            "checkpoint": str(self.checkpoint) if self.checkpoint else "",
            "device": self.device,
        }


class ParametricPolicy:
    def __init__(self, env_cfg: EnvConfig, params: Dict[str, float]) -> None:
        self._params = dict(params)
        self._dt = float(env_cfg.sim.dt)
        self._controller = FocController(env_cfg.foc, env_cfg.motor, self._dt)
        self._theta_mech = 0.0
        self._last_action = (0.0, 0.0)

    def reset(self) -> None:
        self._controller.reset()
        self._theta_mech = 0.0
        self._last_action = (0.0, 0.0)

    def act(self, obs: Dict[str, float]) -> Tuple[float, float]:
        omega = float(obs.get("omega", 0.0))
        omega_ref = float(obs.get("omega_ref", 0.0))
        i_abc = (
            float(obs.get("ia", 0.0)),
            float(obs.get("ib", 0.0)),
            float(obs.get("ic", 0.0)),
        )
        torque = float(obs.get("torque", 0.0))
        if "theta_e" in obs and obs["theta_e"] is not None:
            self._controller.theta_e = float(obs["theta_e"])
        v_d, v_q, _theta_e, _omega_syn, _info = self._controller.step(
            t=float(obs.get("t", 0.0)),
            omega_ref=omega_ref,
            omega_m=omega,
            i_abc=i_abc,
            torque_e=torque,
            theta_mech=self._theta_mech,
        )
        self._theta_mech += omega * self._dt

        k_vd = float(self._params.get("k_vd", 1.0))
        k_vq = float(self._params.get("k_vq", 1.0))
        v_d *= k_vd
        v_q *= k_vq
        self._last_action = (float(v_d), float(v_q))
        return self._last_action

    def describe(self) -> Dict[str, object]:
        return {"policy": "parametric", "params": dict(self._params)}


def _run_episode(
    driver: BaseDriver,
    mode: str,
    policy: Optional[object],
    t_end: float,
    dt: float,
) -> Tuple[Dict[str, np.ndarray], Optional[str]]:
    series: Dict[str, List[float]] = {
        "t": [],
        "omega": [],
        "omega_ref": [],
        "id": [],
        "iq": [],
        "ia": [],
        "ib": [],
        "ic": [],
        "v_d": [],
        "v_q": [],
        "v_a": [],
        "v_b": [],
        "v_c": [],
        "torque": [],
        "load_torque": [],
        "p_el": [],
        "p_mech": [],
        "i_rms": [],
        "fault": [],
    }

    n_steps = int(max(t_end / dt, 1))
    obs = driver.read_obs()
    fault_reason = None
    if policy is not None and hasattr(policy, "reset"):
        policy.reset()

    for _ in range(n_steps):
        if mode == "MIC" and policy is not None:
            vd, vq = policy.act(obs)
            driver.apply_action(vd, vq)
        driver.step()
        obs = driver.read_obs()
        fault_reason = driver.get_last_fault()

        i_abc = (obs.get("ia", 0.0), obs.get("ib", 0.0), obs.get("ic", 0.0))
        v_abc = (obs.get("v_a", 0.0), obs.get("v_b", 0.0), obs.get("v_c", 0.0))
        i_rms = calc_i_rms(i_abc)
        p_el = calc_p_el(v_abc, i_abc)
        p_mech = calc_p_mech(obs.get("omega", 0.0), obs.get("torque", 0.0))

        series["t"].append(float(obs.get("t", 0.0)))
        series["omega"].append(float(obs.get("omega", 0.0)))
        series["omega_ref"].append(float(obs.get("omega_ref", 0.0)))
        series["id"].append(float(obs.get("id", 0.0)))
        series["iq"].append(float(obs.get("iq", 0.0)))
        series["ia"].append(float(obs.get("ia", 0.0)))
        series["ib"].append(float(obs.get("ib", 0.0)))
        series["ic"].append(float(obs.get("ic", 0.0)))
        series["v_d"].append(float(obs.get("v_d", 0.0)))
        series["v_q"].append(float(obs.get("v_q", 0.0)))
        series["v_a"].append(float(obs.get("v_a", 0.0)))
        series["v_b"].append(float(obs.get("v_b", 0.0)))
        series["v_c"].append(float(obs.get("v_c", 0.0)))
        series["torque"].append(float(obs.get("torque", 0.0)))
        series["load_torque"].append(float(obs.get("load_torque", 0.0)))
        series["p_el"].append(float(p_el))
        series["p_mech"].append(float(p_mech))
        series["i_rms"].append(float(i_rms))
        series["fault"].append(1.0 if fault_reason else 0.0)

        if fault_reason:
            break

    series_np = {k: np.asarray(v, dtype=float) for k, v in series.items()}
    return series_np, fault_reason


def _save_run(
    out_dir: Path,
    run_id: str,
    series: Dict[str, np.ndarray],
    meta: Dict[str, object],
) -> Tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"run_{run_id}.json"
    npz_path = out_dir / f"run_{run_id}.npz"

    json_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    meta_bytes = json.dumps(meta, sort_keys=True).encode("utf-8")
    np.savez(npz_path, **series, meta=np.array(meta_bytes, dtype=np.bytes_))
    return json_path, npz_path


def _update_leaderboard(path: Path, entry: Dict[str, object]) -> None:
    data: List[Dict[str, object]] = []
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            data = []
    if not isinstance(data, list):
        data = []
    data.append(entry)
    data.sort(key=lambda x: float(x.get("score", 0.0)))
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _load_json_input(text: Optional[str]) -> Optional[object]:
    if text is None:
        return None
    candidate = Path(text)
    if candidate.is_file():
        return json.loads(candidate.read_text(encoding="utf-8"))
    return json.loads(text)


def _make_policy(
    env_cfg: EnvConfig,
    limits: Dict[str, Optional[float]],
    mic_policy: str,
    mic_checkpoint: Optional[str],
    mic_action_std: Optional[float],
    params: Optional[Dict[str, float]],
) -> Optional[object]:
    mode = str(mic_policy).lower()
    if mode == "parametric":
        return ParametricPolicy(env_cfg, params or {})
    return MicPolicy(
        env_cfg,
        limits,
        policy=mic_policy,
        checkpoint=mic_checkpoint,
        action_std=mic_action_std,
    )


def _save_optimization_plot(history: List[Dict[str, object]], path: Path) -> None:
    if not history:
        return
    iters = [int(item.get("iteration", 0)) for item in history]
    scores = [float(item.get("score", 0.0)) for item in history]
    plt = apply_vak_style(ensure_matplotlib())
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(iters, scores, marker="o", color="black")
    ax.set_xlabel("iteration")
    ax.set_ylabel("score")
    ax.set_title("optimization history")
    ax.grid(True, alpha=0.35)
    fig.tight_layout()
    save_figure(fig, path)
    plt.close(fig)


def _run_suite_once(
    suite_dir: Path,
    base_env_cfg: EnvConfig,
    scenarios: List[str],
    seed: int,
    limits: Dict[str, Optional[float]],
    weights: Dict[str, float],
    fault_penalty: float,
    mic_policy: str,
    mic_checkpoint: Optional[str],
    mic_action_std: Optional[float],
    mic_params: Optional[Dict[str, float]],
    config_path: str,
    tag: str,
    driver_factory: Callable[[EnvConfig], BaseDriver],
    driver_name: str,
) -> Dict[str, object]:
    run_records: Dict[str, List[Dict[str, object]]] = {"FOC": [], "MIC": []}

    for scenario in scenarios:
        env_cfg = _with_scenario(base_env_cfg, scenario)
        dt = float(env_cfg.sim.dt)
        t_end = float(env_cfg.sim.t_end)

        for mode in ("FOC", "MIC"):
            driver = driver_factory(env_cfg)
            driver.reset(seed=seed)
            driver.set_limits(limits)
            driver.set_mode(mode)

            policy = None
            if mode == "MIC":
                policy = _make_policy(
                    env_cfg,
                    limits,
                    mic_policy,
                    mic_checkpoint,
                    mic_action_std,
                    mic_params,
                )

            series, fault_reason = _run_episode(driver, mode, policy, t_end, dt)
            metrics = compute_metrics(
                series["t"],
                series["omega"],
                series["omega_ref"],
                series["i_rms"],
                series["p_el"],
            )
            score = score_from_metrics(metrics, fault_reason=fault_reason, weights=weights, fault_penalty=fault_penalty)
            metrics_with_score = dict(metrics)
            metrics_with_score["score"] = score

            run_id = f"{tag}_{scenario}_{mode.lower()}_{seed}"
            meta = {
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "driver": driver_name,
                "scenario": scenario,
                "mode": mode,
                "seed": seed,
                "config_path": config_path,
                "config_hash": _config_hash(env_cfg),
                "limits": limits,
                "metrics": metrics_with_score,
                "fault_reason": fault_reason or "",
                "mic_policy": policy.describe() if policy is not None else {},
                "mic_params": mic_params or {},
            }
            json_path, npz_path = _save_run(suite_dir, run_id, series, meta)
            run_records[mode].append(
                {
                    "scenario": scenario,
                    "score": score,
                    "metrics": metrics_with_score,
                    "fault_reason": fault_reason or "",
                    "log_json": str(json_path),
                    "log_npz": str(npz_path),
                }
            )

            driver.close()

        foc_npz = Path(run_records["FOC"][-1]["log_npz"])
        mic_npz = Path(run_records["MIC"][-1]["log_npz"])
        build_report(
            foc_npz,
            mic_npz,
            suite_dir,
            title=f"{scenario} ({tag})",
            metrics_foc=run_records["FOC"][-1]["metrics"],
            metrics_mic=run_records["MIC"][-1]["metrics"],
        )

    foc_scores = [float(r["score"]) for r in run_records["FOC"]]
    mic_scores = [float(r["score"]) for r in run_records["MIC"]]
    return {
        "run_records": run_records,
        "foc_score": float(np.mean(foc_scores)) if foc_scores else float("inf"),
        "mic_score": float(np.mean(mic_scores)) if mic_scores else float("inf"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FOC vs MIC experiment suite.")
    parser.add_argument("--driver", default="sim", choices=["sim", "hw_stub"])
    parser.add_argument("--env-config", default="config/env_demo_true_motor1.py")
    parser.add_argument("--out-dir", default="outputs/experiments")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--scenarios", default=",".join(SCENARIOS))
    parser.add_argument("--mic-policy", default="zero", choices=["zero", "ppo_voltage", "parametric"])
    parser.add_argument("--mic-checkpoint", default=None)
    parser.add_argument("--mic-action-std", type=float, default=None)
    parser.add_argument("--mic-params", default=None, help="JSON string or file with MIC params.")
    parser.add_argument("--optimize", default=None, choices=[None, "cmaes"], help="Enable optimizer.")
    parser.add_argument("--budget", type=int, default=0, help="Optimizer evaluation budget.")
    parser.add_argument("--opt-seed", type=int, default=0, help="Optimizer seed.")
    parser.add_argument("--param-space", default=None, help="JSON string or file describing parameter space.")
    parser.add_argument("--w-speed", type=float, default=DEFAULT_SCORE_WEIGHTS["w_speed"])
    parser.add_argument("--w-current", type=float, default=DEFAULT_SCORE_WEIGHTS["w_current"])
    parser.add_argument("--w-power", type=float, default=DEFAULT_SCORE_WEIGHTS["w_power"])
    parser.add_argument("--fault-penalty", type=float, default=DEFAULT_FAULT_PENALTY)
    args = parser.parse_args()

    driver_factories = {
        "sim": SimDriver,
        "hw_stub": HwDriverStub,
    }
    if args.driver not in driver_factories:
        raise ValueError(f"Unknown driver '{args.driver}'.")

    if args.optimize and args.mic_policy != "parametric":
        raise ValueError("--optimize requires --mic-policy parametric.")

    base_env_cfg = _load_env_config(args.env_config)
    base_hash = _config_hash(base_env_cfg)

    limits = _default_limits(base_env_cfg)
    weights = {"w_speed": args.w_speed, "w_current": args.w_current, "w_power": args.w_power}

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suite_dir = Path(args.out_dir) / f"suite_{timestamp}"
    suite_dir.mkdir(parents=True, exist_ok=True)

    scenarios = [s.strip() for s in str(args.scenarios).split(",") if s.strip()]
    mic_params = _load_json_input(args.mic_params)

    if not args.optimize:
        result = _run_suite_once(
            suite_dir,
            base_env_cfg,
            scenarios,
            args.seed,
            limits,
            weights,
            args.fault_penalty,
            args.mic_policy,
            args.mic_checkpoint,
            args.mic_action_std,
            mic_params,
            str(args.env_config),
            "run",
            driver_factories[args.driver],
            args.driver,
        )
        entry = {
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "candidate": {"type": "MIC", "mic_policy": args.mic_policy, "mic_params": mic_params or {}},
            "conditions": {
                "driver": args.driver,
                "config_path": str(args.env_config),
                "config_hash": base_hash,
                "limits": limits,
                "scenarios": scenarios,
                "seed": args.seed,
            },
            "score": result["mic_score"],
            "baseline_score": result["foc_score"],
            "runs": result["run_records"],
        }
        _update_leaderboard(Path("leaderboard.json"), entry)
        print(f"[suite] completed: {suite_dir}")
        return

    if args.budget <= 0:
        raise ValueError("--budget must be > 0 for optimization.")

    param_space_input = _load_json_input(args.param_space) if args.param_space else DEFAULT_PARAM_SPACE
    param_specs = normalize_param_space(param_space_input)
    default_params = {spec.name: spec.init for spec in param_specs}

    best_path = suite_dir / "best_candidate.json"
    history_path = suite_dir / "optimization_history.json"
    history: List[Dict[str, object]] = []

    def _append_history(iter_idx: int, params: Dict[str, float], mic_score: float, foc_score: float) -> None:
        history.append(
            {
                "iteration": int(iter_idx),
                "score": float(mic_score),
                "baseline_score": float(foc_score),
                "params": dict(params),
            }
        )
        history_path.write_text(json.dumps(history, indent=2, sort_keys=True), encoding="utf-8")
        _save_optimization_plot(history, suite_dir / "optimization_history.png")

    def score_fn(params: Dict[str, float]) -> float:
        iter_idx = max(1, len(history))
        tag = f"iter{iter_idx:03d}"
        result = _run_suite_once(
            suite_dir,
            base_env_cfg,
            scenarios,
            args.seed,
            limits,
            weights,
            args.fault_penalty,
            "parametric",
            None,
            None,
            params,
            str(args.env_config),
            tag,
            driver_factories[args.driver],
            args.driver,
        )
        entry = {
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "candidate": {"type": "MIC_parametric", "params": dict(params)},
            "conditions": {
                "driver": args.driver,
                "config_path": str(args.env_config),
                "config_hash": base_hash,
                "limits": limits,
                "scenarios": scenarios,
                "seed": args.seed,
            },
            "score": result["mic_score"],
            "baseline_score": result["foc_score"],
            "runs": result["run_records"],
        }
        _update_leaderboard(Path("leaderboard.json"), entry)
        _append_history(iter_idx, params, result["mic_score"], result["foc_score"])

        best = None
        if best_path.exists():
            try:
                best = json.loads(best_path.read_text(encoding="utf-8"))
            except Exception:
                best = None
        if best is None or result["mic_score"] < float(best.get("score", float("inf"))):
            best_payload = {
                "score": result["mic_score"],
                "baseline_score": result["foc_score"],
                "params": dict(params),
                "timestamp": entry["timestamp"],
            }
            best_path.write_text(json.dumps(best_payload, indent=2, sort_keys=True), encoding="utf-8")

        return float(result["mic_score"])

    def callback(eval_idx: int, params: Dict[str, float], score: float) -> None:
        return None

    baseline_result = _run_suite_once(
        suite_dir,
        base_env_cfg,
        scenarios,
        args.seed,
        limits,
        weights,
        args.fault_penalty,
        "parametric",
        None,
        None,
        default_params,
        str(args.env_config),
        "iter000",
        driver_factories[args.driver],
        args.driver,
    )
    baseline_entry = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "candidate": {"type": "MIC_parametric", "params": dict(default_params), "label": "baseline"},
        "conditions": {
            "driver": args.driver,
            "config_path": str(args.env_config),
            "config_hash": base_hash,
            "limits": limits,
            "scenarios": scenarios,
            "seed": args.seed,
        },
        "score": baseline_result["mic_score"],
        "baseline_score": baseline_result["foc_score"],
        "runs": baseline_result["run_records"],
    }
    _update_leaderboard(Path("leaderboard.json"), baseline_entry)
    _append_history(0, default_params, baseline_result["mic_score"], baseline_result["foc_score"])
    best_path.write_text(
        json.dumps(
            {
                "score": baseline_result["mic_score"],
                "baseline_score": baseline_result["foc_score"],
                "params": dict(default_params),
                "timestamp": baseline_entry["timestamp"],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    cmaes_optimize(
        score_fn=score_fn,
        param_space=param_space_input,
        budget=int(args.budget),
        seed=int(args.opt_seed) if args.opt_seed else None,
        callback=callback,
    )

    print(f"[suite] optimization completed: {suite_dir}")


if __name__ == "__main__":
    main()
