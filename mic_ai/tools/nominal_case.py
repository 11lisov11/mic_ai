from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mic_ai.ai.agents.ppo_voltage import PPOVoltageAgent
from mic_ai.ai.ai_env import AiEnvConfig, MicAiAIEnv
from mic_ai.ai.ai_voltage_config import get_voltage_scale, load_ai_voltage_config
from mic_ai.ai.train_ai_foc_assist import FEATURE_KEYS as FOC_FEATURE_KEYS
from mic_ai.ai.train_ai_id_ref import FEATURE_KEYS as ID_FEATURE_KEYS
from mic_ai.ai.train_ai_voltage import FEATURE_KEYS as VOLT_FEATURE_KEYS
from mic_ai.analysis.metrics import calc_i_rms, calc_p_el, calc_p_mech
from mic_ai.core.env import make_env_from_config
from mic_ai.tools.plot_style import apply_vak_style, ensure_matplotlib, save_figure
from simulation.gym_env import InductionMotorEnv


def _resolve_config_path(config_name: str) -> Path:
    path = Path(config_name)
    if path.is_file():
        return path.resolve()
    candidate = Path("config") / f"{config_name}.py"
    if candidate.is_file():
        return candidate.resolve()
    raise FileNotFoundError(f"Cannot find config file for {config_name}")


def _infer_hidden_sizes(state: Dict[str, torch.Tensor]) -> tuple[int, ...] | None:
    w0 = state.get("actor_body.0.weight")
    w2 = state.get("actor_body.2.weight")
    if w0 is None or w2 is None:
        return None
    try:
        return int(w0.shape[0]), int(w2.shape[0])
    except Exception:
        return None


def _steady_slice(n: int, window_frac: float) -> slice:
    if n <= 0:
        return slice(0, 0)
    window_frac = float(max(min(window_frac, 0.95), 0.05))
    start = int(max(0, n * (1.0 - window_frac)))
    return slice(start, n)


def _summarize(values: Dict[str, np.ndarray], window_frac: float) -> Dict[str, float]:
    n = int(values["t"].size)
    sl = _steady_slice(n, window_frac)
    p_el = values["p_el"][sl]
    p_mech = values["p_mech"][sl]
    p_el_mean = float(np.mean(p_el)) if p_el.size else 0.0
    p_mech_mean = float(np.mean(p_mech)) if p_mech.size else 0.0
    eta = float(p_mech_mean / p_el_mean) if p_el_mean > 1e-9 else 0.0
    return {
        "omega_ss": float(np.mean(values["omega"][sl])) if n else 0.0,
        "i_rms": float(np.mean(values["i_rms"][sl])) if n else 0.0,
        "p_el": p_el_mean,
        "p_mech": p_mech_mean,
        "eta": eta,
        "t_start": float(values["t"][sl][0]) if n and values["t"][sl].size else 0.0,
        "t_end": float(values["t"][sl][-1]) if n and values["t"][sl].size else 0.0,
    }


def _simulate_foc(
    env_cfg: object,
    omega_ref: float,
    load_torque: float,
    dt: float,
    t_end: float,
) -> Dict[str, np.ndarray]:
    env = InductionMotorEnv(env_cfg)
    env.omega_ref_func = lambda _t, ref=omega_ref: ref
    env.load_torque_func = lambda _t, load=load_torque: load
    env.reset()
    env.omega_ref_func = lambda _t, ref=omega_ref: ref
    env.load_torque_func = lambda _t, load=load_torque: load

    steps = int(max(t_end / dt, 1))
    t = np.zeros(steps, dtype=float)
    omega = np.zeros(steps, dtype=float)
    torque = np.zeros(steps, dtype=float)
    load = np.zeros(steps, dtype=float)
    i_rms = np.zeros(steps, dtype=float)
    p_el = np.zeros(steps, dtype=float)
    p_mech = np.zeros(steps, dtype=float)

    for k in range(steps):
        obs, _r, done, info = env.step(None)
        t[k] = float(env.t)
        omega[k] = float(obs[0]) if hasattr(obs, "__len__") else float(info.get("omega_meas", 0.0))
        torque[k] = float(info.get("torque_e", obs[2] if hasattr(obs, "__len__") else 0.0))
        load[k] = float(load_torque)
        i_abc = np.asarray(info.get("i_abc", (0.0, 0.0, 0.0)), dtype=float)
        v_abc = np.asarray(info.get("v_abc", (0.0, 0.0, 0.0)), dtype=float)
        i_rms[k] = calc_i_rms(i_abc)
        p_el[k] = calc_p_el(v_abc, i_abc)
        p_mech[k] = calc_p_mech(omega[k], torque[k])
        if done:
            t = t[: k + 1]
            omega = omega[: k + 1]
            torque = torque[: k + 1]
            load = load[: k + 1]
            i_rms = i_rms[: k + 1]
            p_el = p_el[: k + 1]
            p_mech = p_mech[: k + 1]
            break
    return {
        "t": t,
        "omega": omega,
        "torque": torque,
        "load": load,
        "i_rms": i_rms,
        "p_el": p_el,
        "p_mech": p_mech,
    }


def _build_ai_env(
    env_cfg: object,
    omega_ref: float,
    load_torque: float,
    dt: float,
    t_end: float,
    ai_mode: str,
    v_scale: float | None,
    ai_id_relative: bool,
    delta_id_max: float,
    id_ref_alpha: float,
    id_ref_rate_limit: float | None,
    id_ref_gate_speed_tol: float | None,
    id_ref_gate_speed_tol_rel: float | None,
    id_ref_gate_min_scale: float,
    id_ref_gate_exponent: float,
) -> MicAiAIEnv:
    i_base = float(getattr(env_cfg.motor, "I_n", 1.0))
    iq_limit = float(getattr(getattr(env_cfg, "foc", None), "iq_limit", i_base * 8.0))
    i_limit = max(iq_limit, i_base)
    steps = int(max(t_end / dt, 1))
    mode = str(ai_mode).lower()

    if mode == "ai_id_ref":
        ai_cfg = AiEnvConfig(
            episode_steps=steps,
            dt=dt,
            omega_ref=omega_ref,
            omega_ref_max=max(abs(omega_ref), 1e-3),
            w_speed_error=0.0,
            w_current_rms=0.0,
            control_mode="ai_id_ref",
            i_base=i_base,
            i_max=i_limit,
            i_hard_limit=float(i_limit * 1.2),
            sigma_omega=0.0,
            sigma_id=0.0,
            sigma_iq=0.0,
            w_ai_id_speed=0.0,
            w_ai_id_power=0.0,
            w_ai_id_smooth=0.0,
            ai_id_ref_relative=bool(ai_id_relative),
            delta_id_max=float(delta_id_max),
            id_ref_alpha=float(id_ref_alpha),
            id_ref_rate_limit=None if id_ref_rate_limit is None else float(id_ref_rate_limit),
            id_ref_gate_speed_tol=None if id_ref_gate_speed_tol is None else float(id_ref_gate_speed_tol),
            id_ref_gate_speed_tol_rel=None if id_ref_gate_speed_tol_rel is None else float(id_ref_gate_speed_tol_rel),
            id_ref_gate_min_scale=float(id_ref_gate_min_scale),
            id_ref_gate_exponent=float(id_ref_gate_exponent),
            id_ref_min=0.0,
            id_ref_max=float(i_base * 1.5),
            curriculum_omega_pu=(1.0,),
            curriculum_stage_episodes=(),
            omega_piecewise_steps=(),
            omega_piecewise_multipliers=(1.0,),
            load_torque_override=float(load_torque),
            override_load_torque=True,
        )
    elif mode == "foc_assist":
        ai_cfg = AiEnvConfig(
            episode_steps=steps,
            dt=dt,
            omega_ref=omega_ref,
            omega_ref_max=max(abs(omega_ref), 1e-3),
            w_speed_error=0.0,
            w_current_rms=0.0,
            control_mode="foc_assist",
            enable_id_control=True,
            delta_iq_max=float(getattr(env_cfg, "ai_delta_iq_max", 0.2)),
            delta_id_max=float(getattr(env_cfg, "ai_delta_id_max", 0.3)),
            i_base=i_base,
            i_max=i_limit,
            i_hard_limit=float(i_limit * 1.2),
            sigma_omega=0.0,
            sigma_id=0.0,
            sigma_iq=0.0,
            foc_assist_reward_mode="energy",
            w_foc_speed=0.0,
            w_foc_power=0.0,
            w_foc_current=0.0,
            w_foc_action=0.0,
            foc_speed_tol=0.0,
            p_el_tau=0.0,
            curriculum_omega_pu=(1.0,),
            curriculum_stage_episodes=(),
            omega_piecewise_steps=(),
            omega_piecewise_multipliers=(1.0,),
            load_torque_override=float(load_torque),
            override_load_torque=True,
        )
    else:
        ai_cfg = AiEnvConfig(
            episode_steps=steps,
            dt=dt,
            omega_ref=omega_ref,
            omega_ref_max=max(abs(omega_ref), 1e-3),
            w_speed_error=0.0,
            w_current_rms=0.0,
            control_mode="ai_voltage",
            i_base=i_base,
            i_max=i_limit,
            v_max=v_scale,
            sigma_omega=0.0,
            sigma_id=0.0,
            sigma_iq=0.0,
            w_ai_voltage_speed=0.0,
            w_ai_voltage_current=0.0,
            w_ai_voltage_power=0.0,
            w_ai_voltage_action=0.0,
            ai_voltage_speed_tol=0.0,
            load_torque_override=float(load_torque),
            override_load_torque=True,
        )

    base_env = InductionMotorEnv(env_cfg)
    base_env.omega_ref_func = lambda _t, ref=omega_ref: ref
    base_env.load_torque_func = lambda _t, load=load_torque: load

    env = MicAiAIEnv(
        base_env,
        ai_cfg,
        curiosity=None,
        world_model=None,
        world_input_keys=[],
        world_target_keys=[],
    )

    env._omega_piecewise_steps = tuple()
    env._omega_piecewise_multipliers = (1.0,)
    env._curriculum_ref = omega_ref
    env._omega_nominal = max(abs(omega_ref), 1e-6)
    env._omega_norm_base = env._omega_nominal
    env._omega_ref_max = env._omega_nominal
    env._slip_max = max(env._omega_ref_max, 1e-6)
    return env


def _simulate_ai(
    agent: PPOVoltageAgent,
    env_cfg: object,
    omega_ref: float,
    load_torque: float,
    dt: float,
    t_end: float,
    ai_mode: str,
    v_scale: float | None,
    ai_id_relative: bool,
    delta_id_max: float,
    id_ref_alpha: float,
    id_ref_rate_limit: float | None,
    id_ref_gate_speed_tol: float | None,
    id_ref_gate_speed_tol_rel: float | None,
    id_ref_gate_min_scale: float,
    id_ref_gate_exponent: float,
) -> Dict[str, np.ndarray]:
    env = _build_ai_env(
        env_cfg,
        omega_ref,
        load_torque,
        dt,
        t_end,
        ai_mode,
        v_scale,
        ai_id_relative,
        delta_id_max,
        id_ref_alpha,
        id_ref_rate_limit,
        id_ref_gate_speed_tol,
        id_ref_gate_speed_tol_rel,
        id_ref_gate_min_scale,
        id_ref_gate_exponent,
    )
    obs = env.reset()

    steps = int(max(t_end / dt, 1))
    t = np.zeros(steps, dtype=float)
    omega = np.zeros(steps, dtype=float)
    torque = np.zeros(steps, dtype=float)
    load = np.zeros(steps, dtype=float)
    i_rms = np.zeros(steps, dtype=float)
    p_el = np.zeros(steps, dtype=float)
    p_mech = np.zeros(steps, dtype=float)

    for k in range(steps):
        action, _lp, _v = agent.act(obs)
        obs, _r, done, info = env.step(action)
        t[k] = float(getattr(env.base_env, "t", k * dt))
        omega[k] = float(info.get("omega_meas", obs.get("omega", 0.0)))
        torque[k] = float(info.get("torque_e", getattr(env.base_env, "last_torque", 0.0)))
        load[k] = float(load_torque)
        i_abc = np.asarray(info.get("i_abc", (0.0, 0.0, 0.0)), dtype=float)
        v_abc = np.asarray(info.get("v_abc", (0.0, 0.0, 0.0)), dtype=float)
        i_rms[k] = calc_i_rms(i_abc)
        p_el[k] = calc_p_el(v_abc, i_abc)
        p_mech[k] = calc_p_mech(omega[k], torque[k])
        if done:
            t = t[: k + 1]
            omega = omega[: k + 1]
            torque = torque[: k + 1]
            load = load[: k + 1]
            i_rms = i_rms[: k + 1]
            p_el = p_el[: k + 1]
            p_mech = p_mech[: k + 1]
            break
    return {
        "t": t,
        "omega": omega,
        "torque": torque,
        "load": load,
        "i_rms": i_rms,
        "p_el": p_el,
        "p_mech": p_mech,
    }


def _save_csv(path: Path, series: Dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(series.keys())
    n = int(series[keys[0]].size) if keys else 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(keys)
        for idx in range(n):
            writer.writerow([float(series[k][idx]) for k in keys])


def _save_summary(path: Path, rows: List[Dict[str, float]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_timeseries(out_path: Path, foc: Dict[str, np.ndarray], mic: Dict[str, np.ndarray]) -> None:
    plt = apply_vak_style(ensure_matplotlib())
    fig, axes = plt.subplots(3, 2, figsize=(11.2, 7.6), sharex=True)

    axes[0, 0].plot(foc["t"], foc["omega"], color="black", label="FOC")
    axes[0, 0].plot(mic["t"], mic["omega"], color="0.35", linestyle="--", label="MIC AI")
    axes[0, 0].set_ylabel("ω, рад/с")

    axes[0, 1].plot(foc["t"], foc["load"], color="black")
    axes[0, 1].plot(mic["t"], mic["load"], color="0.35", linestyle="--")
    axes[0, 1].set_ylabel("M_нагрузки, Н·м")

    axes[1, 0].plot(foc["t"], foc["i_rms"], color="black")
    axes[1, 0].plot(mic["t"], mic["i_rms"], color="0.35", linestyle="--")
    axes[1, 0].set_ylabel("I_rms, А")

    axes[1, 1].plot(foc["t"], foc["p_el"], color="black")
    axes[1, 1].plot(mic["t"], mic["p_el"], color="0.35", linestyle="--")
    axes[1, 1].set_ylabel("P_эл, Вт")

    axes[2, 0].plot(foc["t"], foc["p_mech"], color="black")
    axes[2, 0].plot(mic["t"], mic["p_mech"], color="0.35", linestyle="--")
    axes[2, 0].set_ylabel("P_мех, Вт")

    axes[2, 1].plot(foc["t"], foc["torque"], color="black")
    axes[2, 1].plot(mic["t"], mic["torque"], color="0.35", linestyle="--")
    axes[2, 1].set_ylabel("M_эл, Н·м")

    for ax in axes[-1, :]:
        ax.set_xlabel("t, с")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
    save_figure(fig, out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Single nominal case: FOC vs MIC AI time-series.")
    parser.add_argument("--env-config", default="config/env_demo_true_motor1.py")
    parser.add_argument("--ai-checkpoint", required=True)
    parser.add_argument("--ai-mode", choices=["ai_voltage", "ai_id_ref", "foc_assist"], default="ai_id_ref")
    parser.add_argument("--ai-id-relative", action="store_true", help="Use relative id_ref around base for ai_id_ref.")
    parser.add_argument("--delta-id-max", type=float, default=0.1)
    parser.add_argument("--id-ref-alpha", type=float, default=1.0)
    parser.add_argument("--id-ref-rate-limit", type=float, default=None, help="Max d(id_ref)/dt, A/s.")
    parser.add_argument("--id-ref-gate-speed-tol", type=float, default=None)
    parser.add_argument("--id-ref-gate-speed-tol-rel", type=float, default=None, help="Relative gate tol (e.g., 0.05).")
    parser.add_argument("--id-ref-gate-min-scale", type=float, default=0.0)
    parser.add_argument("--id-ref-gate-exponent", type=float, default=1.0)
    parser.add_argument("--omega-ref", type=float, default=None, help="Omega_ref, rad/s.")
    parser.add_argument("--omega-ref-rpm", type=float, default=None, help="Omega_ref, rpm.")
    parser.add_argument("--load-torque", type=float, required=True, help="Load torque, N*m.")
    parser.add_argument("--t-end", type=float, default=None)
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--window-frac", type=float, default=0.25, help="Steady window fraction from the end.")
    parser.add_argument("--out-dir", default="outputs/nominal_case")
    args = parser.parse_args()

    env_path = _resolve_config_path(args.env_config)
    env_cfg = make_env_from_config(str(env_path)).env_config

    dt = float(args.dt) if args.dt is not None else float(env_cfg.sim.dt)
    t_end = float(args.t_end) if args.t_end is not None else float(env_cfg.sim.t_end)
    env_cfg = replace(env_cfg, sim=replace(env_cfg.sim, dt=dt, t_end=t_end))

    omega_ref = None
    if args.omega_ref is not None:
        omega_ref = float(args.omega_ref)
    elif args.omega_ref_rpm is not None:
        omega_ref = float(args.omega_ref_rpm) * 2.0 * math.pi / 60.0
    if omega_ref is None:
        raise ValueError("Provide --omega-ref (rad/s) or --omega-ref-rpm (rpm).")
    load_torque = float(args.load_torque)

    v_scale = None
    if str(args.ai_mode).lower() == "ai_voltage":
        motor_key = "custom"
        ai_cfg = load_ai_voltage_config()
        v_scale = float(get_voltage_scale(ai_cfg, motor_key))

    ckpt = Path(args.ai_checkpoint)
    state = torch.load(ckpt, map_location="cpu")
    hidden = _infer_hidden_sizes(state) or (128, 128)

    mode = str(args.ai_mode).lower()
    if mode == "ai_id_ref":
        feature_keys = ID_FEATURE_KEYS
        action_dim = 1
    elif mode == "foc_assist":
        feature_keys = FOC_FEATURE_KEYS
        action_dim = 2
    else:
        feature_keys = VOLT_FEATURE_KEYS
        action_dim = 2

    agent = PPOVoltageAgent(feature_keys=feature_keys, action_dim=action_dim, device="cpu", hidden_sizes=hidden)
    agent.net.load_state_dict(state)
    agent.set_action_std(1e-6)

    foc = _simulate_foc(env_cfg, omega_ref, load_torque, dt, t_end)
    mic = _simulate_ai(
        agent,
        env_cfg,
        omega_ref,
        load_torque,
        dt,
        t_end,
        args.ai_mode,
        v_scale,
        bool(args.ai_id_relative),
        float(args.delta_id_max),
        float(args.id_ref_alpha),
        None if args.id_ref_rate_limit is None else float(args.id_ref_rate_limit),
        None if args.id_ref_gate_speed_tol is None else float(args.id_ref_gate_speed_tol),
        None if args.id_ref_gate_speed_tol_rel is None else float(args.id_ref_gate_speed_tol_rel),
        float(args.id_ref_gate_min_scale),
        float(args.id_ref_gate_exponent),
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _save_csv(out_dir / "timeseries_foc.csv", foc)
    _save_csv(out_dir / "timeseries_mic_ai.csv", mic)
    _plot_timeseries(out_dir / "timeseries_compare.png", foc, mic)

    summary_rows = []
    for label, series in (("FOC", foc), ("MIC_AI", mic)):
        row = {"policy": label, "omega_ref": omega_ref, "load_torque": load_torque}
        row.update(_summarize(series, float(args.window_frac)))
        summary_rows.append(row)
    _save_summary(out_dir / "summary.csv", summary_rows)
    (out_dir / "summary.json").write_text(
        json.dumps(summary_rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    meta = {
        "env_config": str(env_path),
        "ai_checkpoint": str(ckpt.resolve()),
        "ai_mode": str(args.ai_mode),
        "omega_ref": omega_ref,
        "load_torque": load_torque,
        "dt": dt,
        "t_end": t_end,
        "window_frac": float(args.window_frac),
        "ai_id_relative": bool(args.ai_id_relative),
        "delta_id_max": float(args.delta_id_max),
    }
    (out_dir / "run_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Saved outputs to {out_dir}")


if __name__ == "__main__":
    main()
