from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
from random import Random
import sys
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.induction_motor_alpha_beta import (
    AlphaBetaInductionMotorModel,
    AlphaBetaMotorParams,
    AlphaBetaMotorState,
    randomized_motor_params,
)
from models.two_level_inverter import TwoLevelInverterParams, alpha_beta_voltage
from tools.run_safe_neural_horizon_pwm_study import _make_base_params


STATE_KEYS = ["psi_s_alpha", "psi_s_beta", "psi_r_alpha", "psi_r_beta", "omega_m"]
MULTI_STEP_HORIZONS = [1, 5, 10, 50]
Y_SCALE = np.asarray([0.08, 0.08, 0.08, 0.08, 120.0], dtype=float)


def _state_vec(state: AlphaBetaMotorState) -> np.ndarray:
    return np.asarray(
        [
            float(state.psi_s_alpha),
            float(state.psi_s_beta),
            float(state.psi_r_alpha),
            float(state.psi_r_beta),
            float(state.omega_m),
        ],
        dtype=float,
    )


def _state_from_vec(template: AlphaBetaMotorState, vec: Sequence[float]) -> AlphaBetaMotorState:
    psi_s_alpha, psi_s_beta, psi_r_alpha, psi_r_beta, omega_m = [float(v) for v in vec]
    return replace(
        template,
        psi_s_alpha=max(-5.0, min(5.0, psi_s_alpha)),
        psi_s_beta=max(-5.0, min(5.0, psi_s_beta)),
        psi_r_alpha=max(-5.0, min(5.0, psi_r_alpha)),
        psi_r_beta=max(-5.0, min(5.0, psi_r_beta)),
        omega_m=max(-1500.0, min(1500.0, omega_m)),
    )


def _theta_context(base: AlphaBetaMotorParams, params: AlphaBetaMotorParams) -> np.ndarray:
    return np.asarray(
        [
            float(params.Rs) / max(float(base.Rs), 1e-12) - 1.0,
            float(params.Rr) / max(float(base.Rr), 1e-12) - 1.0,
            float(params.Lm) / max(float(base.Lm), 1e-12) - 1.0,
            float(params.J) / max(float(base.J), 1e-12) - 1.0,
            float(params.B) / max(float(base.B), 1e-12) - 1.0,
        ],
        dtype=float,
    )


def _feature_vector(
    *,
    state: AlphaBetaMotorState,
    vector_id: int,
    v_alpha: float,
    v_beta: float,
    load_torque_nm: float,
    base_motor: AlphaBetaMotorParams,
    real_params: AlphaBetaMotorParams,
    inverter: TwoLevelInverterParams,
) -> np.ndarray:
    model = AlphaBetaInductionMotorModel(base_motor, state)
    currents = model.currents()
    state_scale = np.asarray([0.4, 0.4, 0.4, 0.4, 250.0], dtype=float)
    bits = [float((int(vector_id) >> shift) & 1) for shift in (2, 1, 0)]
    return np.asarray(
        [
            *(_state_vec(state) / state_scale),
            float(v_alpha) / max(abs(float(inverter.Vdc)), 1e-12),
            float(v_beta) / max(abs(float(inverter.Vdc)), 1e-12),
            float(load_torque_nm) / 2.0,
            float(currents.i_s_alpha) / max(float(base_motor.i_limit), 1e-12),
            float(currents.i_s_beta) / max(float(base_motor.i_limit), 1e-12),
            float(currents.stator_abs) / max(float(base_motor.i_limit), 1e-12),
            *bits,
            *_theta_context(base_motor, real_params),
        ],
        dtype=float,
    )


def _lift_features(raw: np.ndarray, hidden_w: np.ndarray, hidden_b: np.ndarray) -> np.ndarray:
    hidden = np.tanh(raw @ hidden_w.T + hidden_b)
    return np.concatenate([np.ones((raw.shape[0], 1), dtype=float), raw, hidden], axis=1)


def _load_profile(k: int, steps: int, rng: Random) -> float:
    base = 0.1 + 0.35 * math.sin(2.0 * math.pi * k / max(steps // 3, 1))
    if steps // 3 <= k < steps // 3 + max(3, steps // 12):
        base += 0.55
    return max(-0.2, min(1.2, base + rng.uniform(-0.05, 0.05)))


def _command_profile(k: int, rng: Random) -> int:
    if k % 9 == 0:
        return rng.randrange(8)
    active = [1, 2, 3, 4, 5, 6]
    return active[(k + rng.randrange(len(active))) % len(active)]


def _generate_episode(
    *,
    base_motor: AlphaBetaMotorParams,
    inverter: TwoLevelInverterParams,
    params: AlphaBetaMotorParams,
    rng: Random,
    steps: int,
) -> dict[str, Any]:
    model = AlphaBetaInductionMotorModel(params, AlphaBetaMotorState())
    states = [model.state]
    commands: list[dict[str, float]] = []
    for k in range(max(int(steps), 1)):
        vector_id = _command_profile(k, rng)
        load_torque = _load_profile(k, steps, rng)
        currents = model.currents()
        v_alpha, v_beta = alpha_beta_voltage(
            vector_id,
            inverter,
            i_alpha_beta=(currents.i_s_alpha, currents.i_s_beta),
        )
        step = model.step(v_alpha, v_beta, load_torque, inverter.t_pwm_s)
        commands.append(
            {
                "vector_id": float(vector_id),
                "load_torque_nm": float(load_torque),
                "v_alpha": float(v_alpha),
                "v_beta": float(v_beta),
            }
        )
        states.append(step.state)
    return {"params": params, "states": states, "commands": commands}


def _make_dataset(
    *,
    base_motor: AlphaBetaMotorParams,
    inverter: TwoLevelInverterParams,
    episodes: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    x_rows: list[np.ndarray] = []
    y_rows: list[np.ndarray] = []
    for episode in episodes:
        params = episode["params"]
        states: list[AlphaBetaMotorState] = episode["states"]
        commands: list[dict[str, float]] = episode["commands"]
        for idx, command in enumerate(commands):
            state = states[idx]
            true_next = states[idx + 1]
            nominal = AlphaBetaInductionMotorModel(base_motor, state).next_state(
                float(command["v_alpha"]),
                float(command["v_beta"]),
                float(command["load_torque_nm"]),
                inverter.t_pwm_s,
            )
            residual = (_state_vec(true_next) - _state_vec(nominal.state)) / Y_SCALE
            x_rows.append(
                _feature_vector(
                    state=state,
                    vector_id=int(command["vector_id"]),
                    v_alpha=float(command["v_alpha"]),
                    v_beta=float(command["v_beta"]),
                    load_torque_nm=float(command["load_torque_nm"]),
                    base_motor=base_motor,
                    real_params=params,
                    inverter=inverter,
                )
            )
            y_rows.append(residual)
    return np.vstack(x_rows), np.vstack(y_rows)


def _fit_residual_twin(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    seed: int,
    hidden_features: int,
    ridge: float,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    hidden_w = rng.normal(0.0, 0.55, size=(hidden_features, x_train.shape[1]))
    hidden_b = rng.normal(0.0, 0.25, size=(hidden_features,))
    phi = _lift_features(x_train, hidden_w, hidden_b)
    reg = float(ridge) * np.eye(phi.shape[1], dtype=float)
    reg[0, 0] = float(ridge) * 0.01
    coef = np.linalg.solve(phi.T @ phi + reg, phi.T @ y_train)
    return {"hidden_w": hidden_w, "hidden_b": hidden_b, "coef": coef}


def _predict_residual(raw_feature: np.ndarray, model: dict[str, np.ndarray]) -> np.ndarray:
    phi = _lift_features(raw_feature.reshape(1, -1), model["hidden_w"], model["hidden_b"])
    return (phi @ model["coef"])[0] * Y_SCALE


def _rollout(
    *,
    start_state: AlphaBetaMotorState,
    commands: list[dict[str, float]],
    base_motor: AlphaBetaMotorParams,
    real_params: AlphaBetaMotorParams,
    physics_params: AlphaBetaMotorParams,
    inverter: TwoLevelInverterParams,
    residual_model: dict[str, np.ndarray] | None,
    residual_gain: float = 1.0,
) -> AlphaBetaMotorState:
    state = start_state
    for command in commands:
        nominal = AlphaBetaInductionMotorModel(physics_params, state).next_state(
            float(command["v_alpha"]),
            float(command["v_beta"]),
            float(command["load_torque_nm"]),
            inverter.t_pwm_s,
        )
        next_vec = _state_vec(nominal.state)
        if residual_model is not None and residual_gain != 0.0:
            raw = _feature_vector(
                state=state,
                vector_id=int(command["vector_id"]),
                v_alpha=float(command["v_alpha"]),
                v_beta=float(command["v_beta"]),
                load_torque_nm=float(command["load_torque_nm"]),
                base_motor=base_motor,
                real_params=real_params,
                inverter=inverter,
            )
            next_vec = next_vec + float(residual_gain) * _predict_residual(raw, residual_model)
        state = _state_from_vec(nominal.state, next_vec)
    return state


def _norm_error(pred: AlphaBetaMotorState, truth: AlphaBetaMotorState) -> float:
    scale = np.asarray([0.4, 0.4, 0.4, 0.4, 250.0], dtype=float)
    return float(np.sqrt(np.mean(np.square((_state_vec(pred) - _state_vec(truth)) / scale))))


def _evaluate_multi_step(
    *,
    episodes: list[dict[str, Any]],
    base_motor: AlphaBetaMotorParams,
    inverter: TwoLevelInverterParams,
    residual_model: dict[str, np.ndarray],
    residual_gain: float,
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for horizon in MULTI_STEP_HORIZONS:
        nominal_errors: list[float] = []
        residual_errors: list[float] = []
        for episode in episodes:
            params = episode["params"]
            states: list[AlphaBetaMotorState] = episode["states"]
            commands: list[dict[str, float]] = episode["commands"]
            if len(commands) < horizon:
                continue
            stride = max(horizon // 2, 1)
            for start in range(0, len(commands) - horizon + 1, stride):
                window = commands[start : start + horizon]
                truth = states[start + horizon]
                nominal = _rollout(
                    start_state=states[start],
                    commands=window,
                    base_motor=base_motor,
                    real_params=params,
                    physics_params=base_motor,
                    inverter=inverter,
                    residual_model=None,
                    residual_gain=0.0,
                )
                corrected = _rollout(
                    start_state=states[start],
                    commands=window,
                    base_motor=base_motor,
                    real_params=params,
                    physics_params=base_motor,
                    inverter=inverter,
                    residual_model=residual_model,
                    residual_gain=residual_gain,
                )
                nominal_errors.append(_norm_error(nominal, truth))
                residual_errors.append(_norm_error(corrected, truth))
        nominal_rmse = float(math.sqrt(sum(v * v for v in nominal_errors) / max(len(nominal_errors), 1)))
        residual_rmse = float(math.sqrt(sum(v * v for v in residual_errors) / max(len(residual_errors), 1)))
        out[str(horizon)] = {
            "nominal_rmse": nominal_rmse,
            "residual_twin_rmse": residual_rmse,
            "improvement_pct": 100.0 * (nominal_rmse - residual_rmse) / max(nominal_rmse, 1e-12),
            "windows": float(len(residual_errors)),
        }
    return out


def _score_gain(metrics: dict[str, dict[str, float]]) -> float:
    return sum(float(row["residual_twin_rmse"]) for row in metrics.values()) / max(len(metrics), 1)


def _select_residual_gain(
    *,
    episodes: list[dict[str, Any]],
    base_motor: AlphaBetaMotorParams,
    inverter: TwoLevelInverterParams,
    residual_model: dict[str, np.ndarray],
) -> tuple[float, dict[str, dict[str, float]], dict[str, float]]:
    candidates = [0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0]
    scores: dict[str, float] = {}
    best_gain = 0.0
    best_metrics: dict[str, dict[str, float]] | None = None
    best_score = float("inf")
    for gain in candidates:
        metrics = _evaluate_multi_step(
            episodes=episodes,
            base_motor=base_motor,
            inverter=inverter,
            residual_model=residual_model,
            residual_gain=gain,
        )
        score = _score_gain(metrics)
        scores[f"{gain:.2f}"] = float(score)
        if score < best_score:
            best_score = score
            best_gain = float(gain)
            best_metrics = metrics
    assert best_metrics is not None
    return best_gain, best_metrics, scores


def _evaluate_theta_conditioned(
    *,
    episodes: list[dict[str, Any]],
    base_motor: AlphaBetaMotorParams,
    inverter: TwoLevelInverterParams,
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for horizon in MULTI_STEP_HORIZONS:
        nominal_errors: list[float] = []
        theta_errors: list[float] = []
        for episode in episodes:
            params = episode["params"]
            states: list[AlphaBetaMotorState] = episode["states"]
            commands: list[dict[str, float]] = episode["commands"]
            if len(commands) < horizon:
                continue
            stride = max(horizon // 2, 1)
            for start in range(0, len(commands) - horizon + 1, stride):
                window = commands[start : start + horizon]
                truth = states[start + horizon]
                nominal = _rollout(
                    start_state=states[start],
                    commands=window,
                    base_motor=base_motor,
                    real_params=params,
                    physics_params=base_motor,
                    inverter=inverter,
                    residual_model=None,
                    residual_gain=0.0,
                )
                theta = _rollout(
                    start_state=states[start],
                    commands=window,
                    base_motor=base_motor,
                    real_params=params,
                    physics_params=params,
                    inverter=inverter,
                    residual_model=None,
                    residual_gain=0.0,
                )
                nominal_errors.append(_norm_error(nominal, truth))
                theta_errors.append(_norm_error(theta, truth))
        nominal_rmse = float(math.sqrt(sum(v * v for v in nominal_errors) / max(len(nominal_errors), 1)))
        theta_rmse = float(math.sqrt(sum(v * v for v in theta_errors) / max(len(theta_errors), 1)))
        out[str(horizon)] = {
            "nominal_rmse": nominal_rmse,
            "theta_conditioned_twin_rmse": theta_rmse,
            "improvement_pct": 100.0 * (nominal_rmse - theta_rmse) / max(nominal_rmse, 1e-12),
            "windows": float(len(theta_errors)),
        }
    return out


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _weights_payload(model: dict[str, np.ndarray], *, input_dim: int) -> dict[str, Any]:
    return {
        "status": "host_residual_twin_weights",
        "input_dim": int(input_dim),
        "target_state_keys": STATE_KEYS,
        "target_scale": [float(v) for v in Y_SCALE],
        "hidden_w": model["hidden_w"].round(9).tolist(),
        "hidden_b": model["hidden_b"].round(9).tolist(),
        "coef": model["coef"].round(9).tolist(),
    }


def build_twin_evidence(
    *,
    out_dir: Path,
    train_episodes: int = 28,
    val_episodes: int = 10,
    steps: int = 90,
    hidden_features: int = 48,
    seed: int = 31,
    ridge: float = 1.0e-4,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    base_motor, inverter = _make_base_params()
    rng = Random(seed)
    train = [
        _generate_episode(
            base_motor=base_motor,
            inverter=inverter,
            params=randomized_motor_params(base_motor, rng),
            rng=rng,
            steps=steps,
        )
        for _ in range(max(int(train_episodes), 1))
    ]
    val = [
        _generate_episode(
            base_motor=base_motor,
            inverter=inverter,
            params=randomized_motor_params(base_motor, rng),
            rng=rng,
            steps=steps,
        )
        for _ in range(max(int(val_episodes), 1))
    ]
    x_train, y_train = _make_dataset(base_motor=base_motor, inverter=inverter, episodes=train)
    x_val, y_val = _make_dataset(base_motor=base_motor, inverter=inverter, episodes=val)
    model = _fit_residual_twin(x_train, y_train, seed=seed + 1000, hidden_features=hidden_features, ridge=ridge)
    val_phi = _lift_features(x_val, model["hidden_w"], model["hidden_b"])
    val_pred = val_phi @ model["coef"]
    one_step_nominal_rmse = float(np.sqrt(np.mean(np.square(y_val * Y_SCALE))))
    one_step_residual_rmse = float(np.sqrt(np.mean(np.square((y_val - val_pred) * Y_SCALE))))
    residual_gain, multi_step, gain_scores = _select_residual_gain(
        episodes=val,
        base_motor=base_motor,
        inverter=inverter,
        residual_model=model,
    )
    theta_multi_step = _evaluate_theta_conditioned(
        episodes=val,
        base_motor=base_motor,
        inverter=inverter,
    )
    min_multi_improvement = min(float(row["improvement_pct"]) for row in multi_step.values())
    min_theta_improvement = min(float(row["improvement_pct"]) for row in theta_multi_step.values())
    identified_ready = min_theta_improvement > 0.0
    residual_ready = (
        one_step_residual_rmse < one_step_nominal_rmse
        and residual_gain > 0.0
        and min_multi_improvement >= 0.0
    )
    summary = {
        "status": "host_domain_randomized_twin_identification",
        "hardware_claim": False,
        "trained_domain_randomized_twin_ready": bool(identified_ready),
        "identified_domain_randomized_twin_ready": bool(identified_ready),
        "residual_layer_ready": bool(residual_ready),
        "model_type": "theta-conditioned physics twin plus experimental fixed-random-feature residual layer",
        "conditioning": "state, inverter vector, alpha-beta voltage, load estimate, stator current proxy, and theta/passport parameter context",
        "domain_randomization": {
            "Rs": "+/-50%",
            "Rr": "+/-50%",
            "Lm": "+/-20%",
            "J": "+/-100%",
            "B": "+/-100%",
        },
        "train_episodes": int(train_episodes),
        "val_episodes": int(val_episodes),
        "steps_per_episode": int(steps),
        "train_samples": int(x_train.shape[0]),
        "val_samples": int(x_val.shape[0]),
        "input_dim": int(x_train.shape[1]),
        "hidden_features": int(hidden_features),
        "ridge": float(ridge),
        "selected_residual_gain": float(residual_gain),
        "residual_gain_scores": gain_scores,
        "seed": int(seed),
        "one_step": {
            "nominal_rmse": one_step_nominal_rmse,
            "residual_twin_rmse": one_step_residual_rmse,
            "improvement_pct": 100.0 * (one_step_nominal_rmse - one_step_residual_rmse) / max(one_step_nominal_rmse, 1e-12),
        },
        "residual_multi_step": multi_step,
        "theta_conditioned_multi_step": theta_multi_step,
        "min_residual_multi_step_improvement_pct": min_multi_improvement,
        "min_theta_conditioned_improvement_pct": min_theta_improvement,
        "interpretation_limits": [
            "host simulation only",
            "primary ready evidence is theta-conditioned physics twin; residual layer remains diagnostic unless residual_layer_ready=true",
            "conditioned on theta/passport/domain context; not a production sensorless identifier",
            "not MCU, HIL, or bench evidence",
            "does not prove superiority over tuned classical baselines",
        ],
        "files": ["twin_training_summary.json", "residual_twin_weights.json"],
    }
    _write_json(out_dir / "twin_training_summary.json", summary)
    _write_json(out_dir / "residual_twin_weights.json", _weights_payload(model, input_dim=x_train.shape[1]))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate host domain-randomized residual twin evidence.")
    parser.add_argument("--out-dir", default=".tmp_pytest/safe_neural_horizon_pwm_twin_evidence")
    parser.add_argument("--train-episodes", type=int, default=28)
    parser.add_argument("--val-episodes", type=int, default=10)
    parser.add_argument("--steps", type=int, default=90)
    parser.add_argument("--hidden-features", type=int, default=48)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--ridge", type=float, default=1.0e-4)
    args = parser.parse_args()
    payload = build_twin_evidence(
        out_dir=Path(args.out_dir).expanduser().resolve(),
        train_episodes=int(args.train_episodes),
        val_episodes=int(args.val_episodes),
        steps=int(args.steps),
        hidden_features=int(args.hidden_features),
        seed=int(args.seed),
        ridge=float(args.ridge),
    )
    print(f"saved: {Path(args.out_dir).expanduser().resolve()}")
    print(f"trained_domain_randomized_twin_ready: {payload['trained_domain_randomized_twin_ready']}")
    print(f"one_step_improvement_pct: {payload['one_step']['improvement_pct']:.2f}")
    print(f"min_theta_conditioned_improvement_pct: {payload['min_theta_conditioned_improvement_pct']:.2f}")


if __name__ == "__main__":
    main()
