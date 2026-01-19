# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.env import NAMEPLATE_N_RATED, NAMEPLATE_P_KW
from mic_ai.ai.agents.ppo_voltage import PPOVoltageAgent
from mic_ai.ai.ai_env import AiEnvConfig, MicAiAIEnv
from mic_ai.ai.ai_voltage_config import get_voltage_scale, load_ai_voltage_config
from mic_ai.ai.train_ai_foc_assist import FEATURE_KEYS as FOC_FEATURE_KEYS
from mic_ai.ai.train_ai_id_ref import FEATURE_KEYS as ID_FEATURE_KEYS
from mic_ai.ai.train_ai_voltage import FEATURE_KEYS as VOLT_FEATURE_KEYS, _motor_key_from_config, resolve_config_path
from mic_ai.analysis.metrics import calc_i_rms, calc_p_el, calc_p_mech
from mic_ai.core.env import make_env_from_config
from mic_ai.tools.plot_style import apply_vak_style, ensure_matplotlib, save_figure
from simulation.gym_env import InductionMotorEnv


def _rated_omega() -> float:
    return float(2.0 * math.pi * NAMEPLATE_N_RATED / 60.0)


def _omega_nominal(env_cfg: object, source: str) -> float:
    if source == "base":
        pole_pairs = int(getattr(getattr(env_cfg, "motor", None), "p", 1) or 1)
        return float(2.0 * math.pi * 10.0 / pole_pairs)
    return _rated_omega()


def _rated_torque() -> float:
    omega = _rated_omega()
    if omega <= 0.0:
        raise ValueError("rated omega must be positive")
    return float(NAMEPLATE_P_KW * 1000.0 / omega)


def _infer_hidden_sizes(state: Dict[str, torch.Tensor]) -> Tuple[int, ...] | None:
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


def _summarize_window(values: Dict[str, np.ndarray], window_frac: float) -> Dict[str, float]:
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


def _speed_valid(omega_ss: float, omega_ref: float, tol_rel: float, tol_abs: float | None) -> tuple[bool, float, float]:
    ref = max(abs(omega_ref), 1e-6)
    err_abs = abs(float(omega_ss) - float(omega_ref))
    lim = float(tol_abs) if tol_abs is not None else float(tol_rel) * ref
    return err_abs <= lim, err_abs, err_abs / ref


def _simulate_foc_case(
    env_cfg: object,
    omega_ref: float,
    load_torque: float,
    dt: float,
    t_end: float,
) -> Dict[str, np.ndarray]:
    env = InductionMotorEnv(env_cfg)
    env.omega_ref_func = lambda _t, ref=omega_ref: ref
    env.load_torque_func = lambda _t, load=load_torque: load
    obs = env.reset()
    env.omega_ref_func = lambda _t, ref=omega_ref: ref
    env.load_torque_func = lambda _t, load=load_torque: load

    steps = int(max(t_end / dt, 1))
    t = np.zeros(steps, dtype=float)
    omega = np.zeros(steps, dtype=float)
    i_rms = np.zeros(steps, dtype=float)
    p_el = np.zeros(steps, dtype=float)
    p_mech = np.zeros(steps, dtype=float)

    for k in range(steps):
        obs, _r, done, info = env.step(None)
        t[k] = float(env.t)
        omega[k] = float(obs[0]) if hasattr(obs, "__len__") else float(info.get("omega_meas", 0.0))
        i_abc = np.asarray(info.get("i_abc", (0.0, 0.0, 0.0)), dtype=float)
        v_abc = np.asarray(info.get("v_abc", (0.0, 0.0, 0.0)), dtype=float)
        torque = float(info.get("torque_e", obs[2] if hasattr(obs, "__len__") else 0.0))
        i_rms[k] = calc_i_rms(i_abc)
        p_el[k] = calc_p_el(v_abc, i_abc)
        p_mech[k] = calc_p_mech(omega[k], torque)
        if done:
            t = t[: k + 1]
            omega = omega[: k + 1]
            i_rms = i_rms[: k + 1]
            p_el = p_el[: k + 1]
            p_mech = p_mech[: k + 1]
            break
    return {"t": t, "omega": omega, "i_rms": i_rms, "p_el": p_el, "p_mech": p_mech}


def _simulate_mic_rule_case(
    env_cfg: object,
    omega_ref: float,
    load_torque: float,
    dt: float,
    t_end: float,
    id_ref_low: float,
    id_ref_high: float,
    speed_tol_rel: float,
    omega_min_pu: float,
    omega_nom: float,
) -> Dict[str, np.ndarray]:
    env = InductionMotorEnv(env_cfg)
    env.omega_ref_func = lambda _t, ref=omega_ref: ref
    env.load_torque_func = lambda _t, load=load_torque: load
    obs = env.reset()
    env.omega_ref_func = lambda _t, ref=omega_ref: ref
    env.load_torque_func = lambda _t, load=load_torque: load

    steps = int(max(t_end / dt, 1))
    t = np.zeros(steps, dtype=float)
    omega = np.zeros(steps, dtype=float)
    i_rms = np.zeros(steps, dtype=float)
    p_el = np.zeros(steps, dtype=float)
    p_mech = np.zeros(steps, dtype=float)

    for k in range(steps):
        omega_ref_k = float(omega_ref)
        omega_meas = float(getattr(getattr(env.motor, "state", None), "omega_m", 0.0))
        omega_ref_scale = max(abs(omega_ref_k), 1e-6)
        err = abs(omega_ref_k - omega_meas)
        id_ref_target = float(id_ref_high)
        if abs(omega_ref_k) >= float(omega_min_pu) * float(omega_nom) and err <= float(speed_tol_rel) * omega_ref_scale:
            id_ref_target = float(id_ref_low)
        env.controller.params = replace(env.controller.params, id_ref=id_ref_target)

        obs, _r, done, info = env.step(None)
        t[k] = float(env.t)
        omega[k] = float(obs[0]) if hasattr(obs, "__len__") else float(info.get("omega_meas", 0.0))
        i_abc = np.asarray(info.get("i_abc", (0.0, 0.0, 0.0)), dtype=float)
        v_abc = np.asarray(info.get("v_abc", (0.0, 0.0, 0.0)), dtype=float)
        torque = float(info.get("torque_e", obs[2] if hasattr(obs, "__len__") else 0.0))
        i_rms[k] = calc_i_rms(i_abc)
        p_el[k] = calc_p_el(v_abc, i_abc)
        p_mech[k] = calc_p_mech(omega[k], torque)
        if done:
            t = t[: k + 1]
            omega = omega[: k + 1]
            i_rms = i_rms[: k + 1]
            p_el = p_el[: k + 1]
            p_mech = p_mech[: k + 1]
            break
    return {"t": t, "omega": omega, "i_rms": i_rms, "p_el": p_el, "p_mech": p_mech}


def _build_ai_env(
    env_cfg: object,
    omega_ref: float,
    load_torque: float,
    dt: float,
    t_end: float,
    v_scale: float | None,
    ai_mode: str,
    override_load_torque: bool,
    ai_id_relative: bool,
    delta_id_max: float,
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
            sigma_omega=0.0,
            sigma_id=0.0,
            sigma_iq=0.0,
            w_ai_id_speed=0.0,
            w_ai_id_power=0.0,
            w_ai_id_smooth=0.0,
            id_ref_min=0.0,
            id_ref_max=float(i_base * 1.5),
            ai_id_ref_relative=bool(ai_id_relative),
            delta_id_max=float(delta_id_max),
            i_hard_limit=float(i_limit * 1.2),
            curriculum_omega_pu=(1.0,),
            curriculum_stage_episodes=(),
            omega_piecewise_steps=(),
            omega_piecewise_multipliers=(1.0,),
            load_torque_override=float(load_torque),
            override_load_torque=bool(override_load_torque),
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
            override_load_torque=bool(override_load_torque),
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


def _simulate_ai_case(
    agent: PPOVoltageAgent,
    env_cfg: object,
    omega_ref: float,
    load_torque: float,
    dt: float,
    t_end: float,
    v_scale: float | None,
    ai_mode: str,
    override_load_torque: bool,
    ai_id_relative: bool,
    delta_id_max: float,
) -> Dict[str, np.ndarray]:
    env = _build_ai_env(
        env_cfg,
        omega_ref,
        load_torque,
        dt,
        t_end,
        v_scale,
        ai_mode,
        override_load_torque,
        ai_id_relative,
        delta_id_max,
    )
    obs = env.reset()
    steps = int(max(t_end / dt, 1))

    t = np.zeros(steps, dtype=float)
    omega = np.zeros(steps, dtype=float)
    i_rms = np.zeros(steps, dtype=float)
    p_el = np.zeros(steps, dtype=float)
    p_mech = np.zeros(steps, dtype=float)

    for k in range(steps):
        action, _lp, _v = agent.act(obs)
        obs, _r, done, info = env.step(action)
        t[k] = float(getattr(env.base_env, "t", k * dt))
        omega[k] = float(info.get("omega_meas", obs.get("omega", 0.0)))
        i_abc = np.asarray(info.get("i_abc", (0.0, 0.0, 0.0)), dtype=float)
        v_abc = np.asarray(info.get("v_abc", (0.0, 0.0, 0.0)), dtype=float)
        torque = float(getattr(env.base_env, "last_torque", 0.0))
        i_rms[k] = calc_i_rms(i_abc)
        p_el[k] = calc_p_el(v_abc, i_abc)
        p_mech[k] = calc_p_mech(omega[k], torque)
        if done:
            t = t[: k + 1]
            omega = omega[: k + 1]
            i_rms = i_rms[: k + 1]
            p_el = p_el[: k + 1]
            p_mech = p_mech[: k + 1]
            break
    return {"t": t, "omega": omega, "i_rms": i_rms, "p_el": p_el, "p_mech": p_mech}


def _save_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _mask_values(values: List[float], mask: List[bool]) -> List[float]:
    return [v if ok else float("nan") for v, ok in zip(values, mask)]


def _plot_load_characteristics(
    out_path: Path,
    loads: np.ndarray,
    foc: List[Dict[str, float]],
    mic: List[Dict[str, float]],
    valid_mask: List[bool] | None = None,
) -> None:
    plt = apply_vak_style(ensure_matplotlib())

    fig, axes = plt.subplots(2, 2, figsize=(10.6, 7.4))
    if valid_mask is None:
        valid_mask = [True] * len(loads)
    ax = axes[0, 0]
    ax.plot(
        loads,
        _mask_values([x["omega_ss"] for x in foc], valid_mask),
        color="black",
        marker="o",
        linestyle="-",
        label="FOC",
    )
    ax.plot(
        loads,
        _mask_values([x["omega_ss"] for x in mic], valid_mask),
        color="0.35",
        marker="s",
        linestyle="--",
        label="MIC AI",
    )
    ax.set_xlabel("M_нагрузки, Н·м")
    ax.set_ylabel("ω_уст, рад/с")

    ax = axes[0, 1]
    ax.plot(loads, _mask_values([x["i_rms"] for x in foc], valid_mask), color="black", marker="o", linestyle="-")
    ax.plot(loads, _mask_values([x["i_rms"] for x in mic], valid_mask), color="0.35", marker="s", linestyle="--")
    ax.set_xlabel("M_нагрузки, Н·м")
    ax.set_ylabel("I_rms, А")

    ax = axes[1, 0]
    ax.plot(loads, _mask_values([x["p_el"] for x in foc], valid_mask), color="black", marker="o", linestyle="-")
    ax.plot(loads, _mask_values([x["p_el"] for x in mic], valid_mask), color="0.35", marker="s", linestyle="--")
    ax.set_xlabel("M_нагрузки, Н·м")
    ax.set_ylabel("P_эл, Вт")

    ax = axes[1, 1]
    ax.plot(loads, _mask_values([x["eta"] for x in foc], valid_mask), color="black", marker="o", linestyle="-")
    ax.plot(loads, _mask_values([x["eta"] for x in mic], valid_mask), color="0.35", marker="s", linestyle="--")
    ax.set_xlabel("M_нагрузки, Н·м")
    ax.set_ylabel("η = P_мех / P_эл")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.94])
    save_figure(fig, out_path)
    plt.close(fig)


def _plot_working_characteristics(
    out_path: Path,
    speeds: np.ndarray,
    loads: np.ndarray,
    foc_grid: List[List[Dict[str, float]]],
    mic_grid: List[List[Dict[str, float]]],
    valid_grid: List[List[bool]] | None = None,
) -> None:
    plt = apply_vak_style(ensure_matplotlib())

    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.4), sharex=True)
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i / max(len(loads) - 1, 1)) for i in range(len(loads))]

    for idx, load in enumerate(loads):
        color = colors[idx]
        foc = foc_grid[idx]
        mic = mic_grid[idx]
        valid_row = valid_grid[idx] if valid_grid is not None else [True] * len(speeds)
        foc_i = _mask_values([x["i_rms"] for x in foc], valid_row)
        mic_i = _mask_values([x["i_rms"] for x in mic], valid_row)
        foc_pel = _mask_values([x["p_el"] for x in foc], valid_row)
        mic_pel = _mask_values([x["p_el"] for x in mic], valid_row)
        foc_pmech = _mask_values([x["p_mech"] for x in foc], valid_row)
        mic_pmech = _mask_values([x["p_mech"] for x in mic], valid_row)
        axes[0].plot(
            speeds,
            foc_i,
            color=color,
            linestyle="-",
            marker="o",
        )
        axes[0].plot(
            speeds,
            mic_i,
            color=color,
            linestyle="--",
            marker="s",
        )
        axes[1].plot(speeds, foc_pel, color=color, linestyle="-", marker="o")
        axes[1].plot(speeds, mic_pel, color=color, linestyle="--", marker="s")
        axes[2].plot(speeds, foc_pmech, color=color, linestyle="-", marker="o")
        axes[2].plot(speeds, mic_pmech, color=color, linestyle="--", marker="s")

    axes[0].set_ylabel("I_rms, А")
    axes[1].set_ylabel("P_эл, Вт")
    axes[2].set_ylabel("P_мех, Вт")
    for ax in axes:
        ax.set_xlabel("ω_зад, рад/с")

    from matplotlib.lines import Line2D

    method_handles = [
        Line2D([0], [0], color="black", linestyle="-", marker="o", label="FOC"),
        Line2D([0], [0], color="black", linestyle="--", marker="s", label="MIC AI"),
    ]
    load_handles = [
        Line2D([0], [0], color=colors[idx], linestyle="-", marker="o", label=f"{load:.2f}")
        for idx, load in enumerate(loads)
    ]
    fig.legend(method_handles, [h.get_label() for h in method_handles], loc="upper center", ncol=2, frameon=False)
    fig.legend(
        load_handles,
        [h.get_label() for h in load_handles],
        loc="lower center",
        ncol=3,
        frameon=False,
        title="M_нагрузки, Н·м",
    )
    fig.tight_layout(rect=[0.0, 0.08, 1.0, 0.92])
    save_figure(fig, out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Load and working characteristics for FOC vs MIC AI.")
    parser.add_argument("--env-config", default="config/env_demo_true_motor1.py")
    parser.add_argument("--ai-checkpoint", default=None, help="Path to MIC AI (RL) checkpoint .pth")
    parser.add_argument("--mic-id-ref", type=float, default=None, help="Use fixed id_ref for MIC curve.")
    parser.add_argument("--mic-id-ref-low", type=float, default=None, help="Low id_ref for MIC rule.")
    parser.add_argument("--mic-id-ref-high", type=float, default=None, help="High id_ref for MIC rule.")
    parser.add_argument("--mic-id-ref-speed-tol-rel", type=float, default=0.05, help="Speed error tol (rel).")
    parser.add_argument("--mic-id-ref-omega-min", type=float, default=0.1, help="Min omega_ref pu for low id_ref.")
    parser.add_argument("--out-dir", default="outputs/drive_characteristics")
    parser.add_argument(
        "--ai-mode",
        choices=["ai_voltage", "ai_id_ref", "foc_assist"],
        default="ai_voltage",
        help="AI control mode: direct vd/vq (ai_voltage), id_ref on top of FOC (ai_id_ref), or FOC assist (foc_assist).",
    )
    parser.add_argument("--omega-ref-pu", type=float, default=0.8, help="Fixed omega_ref as pu of omega_nom")
    parser.add_argument("--load-points", type=int, default=6, help="Number of load points from 0..1.2 M_nom")
    parser.add_argument("--load-values", default=None, help="Explicit load list in N*m, comma-separated.")
    parser.add_argument("--m-nom", type=float, default=None, help="Override nominal torque for load sweep, N*m.")
    parser.add_argument("--speed-pu", default="0.3,0.5,0.7,0.9", help="Speed pu list for working curves")
    parser.add_argument("--window-frac", type=float, default=0.25, help="Steady window fraction from the end")
    parser.add_argument("--t-end", type=float, default=None, help="Override simulation duration, s")
    parser.add_argument("--dt", type=float, default=None, help="Override simulation dt, s")
    parser.add_argument("--voltage-scale", type=float, default=None, help="Per-unit voltage scale for AI")
    parser.add_argument(
        "--omega-nom-source",
        choices=["nameplate", "base"],
        default="nameplate",
        help="Nominal omega source: nameplate (default) or base (2*pi*10/p).",
    )
    parser.add_argument("--i-max", type=float, default=None, help="Override current limit for both FOC and AI, A.")
    parser.add_argument("--ai-id-relative", action="store_true", help="Use relative id_ref around base when ai_id_ref.")
    parser.add_argument("--delta-id-max", type=float, default=0.3, help="Relative id_ref delta scale.")
    parser.add_argument("--speed-tol", type=float, default=0.05, help="Relative speed tolerance for valid points.")
    parser.add_argument("--speed-tol-abs", type=float, default=None, help="Absolute speed tolerance for valid points, rad/s.")
    args = parser.parse_args()

    mic_id_ref = None if args.mic_id_ref is None else float(args.mic_id_ref)
    mic_id_ref_low = None if args.mic_id_ref_low is None else float(args.mic_id_ref_low)
    mic_id_ref_high = None if args.mic_id_ref_high is None else float(args.mic_id_ref_high)
    mic_rule = False
    if mic_id_ref_low is not None or mic_id_ref_high is not None:
        if mic_id_ref_low is None or mic_id_ref_high is None:
            raise ValueError("Provide both --mic-id-ref-low and --mic-id-ref-high.")
        mic_rule = True
    use_ai = mic_id_ref is None and not mic_rule
    ai_mode = str(args.ai_mode).lower()

    env_path = resolve_config_path(args.env_config)
    env_cfg = make_env_from_config(str(env_path)).env_config
    if args.i_max is not None:
        env_cfg = replace(env_cfg, foc=replace(env_cfg.foc, iq_limit=float(args.i_max)))
    v_scale = None
    if use_ai and ai_mode == "ai_voltage":
        motor_key = _motor_key_from_config(str(env_path))
        vdc = float(getattr(getattr(env_cfg, "inverter", None), "Vdc", 0.0) or 0.0)
        ai_cfg = load_ai_voltage_config()
        v_scale = (
            float(args.voltage_scale)
            if args.voltage_scale is not None
            else float(get_voltage_scale(ai_cfg, motor_key))
        )
        if vdc <= 0.0:
            raise ValueError("Vdc must be positive for AI voltage scaling")

    dt = float(args.dt) if args.dt is not None else float(env_cfg.sim.dt)
    t_end = float(args.t_end) if args.t_end is not None else float(env_cfg.sim.t_end)
    env_cfg = replace(env_cfg, sim=replace(env_cfg.sim, dt=dt, t_end=t_end))

    omega_nom = _omega_nominal(env_cfg, args.omega_nom_source)
    m_nom = float(args.m_nom) if args.m_nom is not None else _rated_torque()
    omega_ref = float(args.omega_ref_pu) * omega_nom

    load_values = None
    if args.load_values is not None:
        load_values = [float(x) for x in str(args.load_values).split(",") if str(x).strip()]
        if not load_values:
            raise ValueError("--load-values provided but parsed list is empty")
    if load_values is None:
        loads = np.linspace(0.0, 1.2 * m_nom, int(max(args.load_points, 2)))
    else:
        loads = np.asarray(load_values, dtype=float)
    speed_pu = [float(x) for x in str(args.speed_pu).split(",") if str(x).strip()]
    speeds = np.asarray(speed_pu, dtype=float) * omega_nom

    agent = None
    ckpt = None
    if use_ai:
        if args.ai_checkpoint is None:
            raise ValueError("Provide --ai-checkpoint or use --mic-id-ref / --mic-id-ref-low+--mic-id-ref-high.")
        ckpt = Path(args.ai_checkpoint)
        if not ckpt.exists():
            raise FileNotFoundError(f"AI checkpoint not found: {ckpt}")
        state = torch.load(ckpt, map_location="cpu")
        hidden = _infer_hidden_sizes(state) or (128, 128)
        if ai_mode == "ai_id_ref":
            feature_keys = ID_FEATURE_KEYS
            action_dim = 1
        elif ai_mode == "foc_assist":
            feature_keys = FOC_FEATURE_KEYS
            action_dim = 2
        else:
            feature_keys = VOLT_FEATURE_KEYS
            action_dim = 2
        agent = PPOVoltageAgent(feature_keys=feature_keys, action_dim=action_dim, device="cpu", hidden_sizes=hidden)
        agent.net.load_state_dict(state)
        agent.set_action_std(1e-6)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    load_rows: List[Dict[str, object]] = []
    foc_load_stats: List[Dict[str, float]] = []
    mic_load_stats: List[Dict[str, float]] = []
    load_valid_mask: List[bool] = []
    load_rows_filtered: List[Dict[str, object]] = []
    for load in loads:
        foc_vals = _simulate_foc_case(env_cfg, omega_ref, float(load), dt, t_end)
        foc_stats = _summarize_window(foc_vals, args.window_frac)
        if mic_rule:
            mic_vals = _simulate_mic_rule_case(
                env_cfg,
                omega_ref,
                float(load),
                dt,
                t_end,
                float(mic_id_ref_low),
                float(mic_id_ref_high),
                float(args.mic_id_ref_speed_tol_rel),
                float(args.mic_id_ref_omega_min),
                float(omega_nom),
            )
        elif mic_id_ref is not None:
            foc_mic = replace(env_cfg.foc, id_ref=float(mic_id_ref))
            mic_vals = _simulate_foc_case(replace(env_cfg, foc=foc_mic), omega_ref, float(load), dt, t_end)
        else:
            mic_vals = _simulate_ai_case(
                agent,
                env_cfg,
                omega_ref,
                float(load),
                dt,
                t_end,
                v_scale,
                args.ai_mode,
                False,
                bool(args.ai_id_relative),
                float(args.delta_id_max),
            )
        mic_stats = _summarize_window(mic_vals, args.window_frac)
        foc_load_stats.append(foc_stats)
        mic_load_stats.append(mic_stats)
        foc_valid, foc_err_abs, foc_err_rel = _speed_valid(
            foc_stats["omega_ss"], omega_ref, args.speed_tol, args.speed_tol_abs
        )
        mic_valid, mic_err_abs, mic_err_rel = _speed_valid(
            mic_stats["omega_ss"], omega_ref, args.speed_tol, args.speed_tol_abs
        )
        load_valid_mask.append(bool(foc_valid and mic_valid))
        load_rows.append(
            {
                "policy": "FOC",
                "load_torque": float(load),
                "omega_ref": omega_ref,
                "speed_err_abs": float(foc_err_abs),
                "speed_err_rel": float(foc_err_rel),
                "valid_speed": int(foc_valid),
                **foc_stats,
            }
        )
        load_rows.append(
            {
                "policy": "MIC_AI",
                "load_torque": float(load),
                "omega_ref": omega_ref,
                "speed_err_abs": float(mic_err_abs),
                "speed_err_rel": float(mic_err_rel),
                "valid_speed": int(mic_valid),
                **mic_stats,
            }
        )
        if foc_valid and mic_valid:
            load_rows_filtered.append(
                {
                    "policy": "FOC",
                    "load_torque": float(load),
                    "omega_ref": omega_ref,
                    "speed_err_abs": float(foc_err_abs),
                    "speed_err_rel": float(foc_err_rel),
                    "valid_speed": 1,
                    **foc_stats,
                }
            )
            load_rows_filtered.append(
                {
                    "policy": "MIC_AI",
                    "load_torque": float(load),
                    "omega_ref": omega_ref,
                    "speed_err_abs": float(mic_err_abs),
                    "speed_err_rel": float(mic_err_rel),
                    "valid_speed": 1,
                    **mic_stats,
                }
            )

    _save_csv(out_dir / "load_characteristics.csv", load_rows)
    _plot_load_characteristics(out_dir / "load_characteristics.png", loads, foc_load_stats, mic_load_stats)
    _save_csv(out_dir / "load_characteristics_filtered.csv", load_rows_filtered)
    _plot_load_characteristics(out_dir / "load_characteristics_valid.png", loads, foc_load_stats, mic_load_stats, load_valid_mask)

    work_rows: List[Dict[str, object]] = []
    foc_grid: List[List[Dict[str, float]]] = []
    mic_grid: List[List[Dict[str, float]]] = []
    valid_grid: List[List[bool]] = []
    work_rows_filtered: List[Dict[str, object]] = []
    for load in loads:
        foc_row: List[Dict[str, float]] = []
        mic_row: List[Dict[str, float]] = []
        valid_row: List[bool] = []
        for speed in speeds:
            foc_vals = _simulate_foc_case(env_cfg, float(speed), float(load), dt, t_end)
            foc_stats = _summarize_window(foc_vals, args.window_frac)
            if mic_rule:
                mic_vals = _simulate_mic_rule_case(
                    env_cfg,
                    float(speed),
                    float(load),
                    dt,
                    t_end,
                    float(mic_id_ref_low),
                    float(mic_id_ref_high),
                    float(args.mic_id_ref_speed_tol_rel),
                    float(args.mic_id_ref_omega_min),
                    float(omega_nom),
                )
            elif mic_id_ref is not None:
                foc_mic = replace(env_cfg.foc, id_ref=float(mic_id_ref))
                mic_vals = _simulate_foc_case(replace(env_cfg, foc=foc_mic), float(speed), float(load), dt, t_end)
            else:
                mic_vals = _simulate_ai_case(
                    agent,
                    env_cfg,
                    float(speed),
                    float(load),
                    dt,
                    t_end,
                    v_scale,
                    args.ai_mode,
                    False,
                    bool(args.ai_id_relative),
                    float(args.delta_id_max),
                )
            mic_stats = _summarize_window(mic_vals, args.window_frac)
            foc_valid, foc_err_abs, foc_err_rel = _speed_valid(
                foc_stats["omega_ss"], float(speed), args.speed_tol, args.speed_tol_abs
            )
            mic_valid, mic_err_abs, mic_err_rel = _speed_valid(
                mic_stats["omega_ss"], float(speed), args.speed_tol, args.speed_tol_abs
            )
            valid_row.append(bool(foc_valid and mic_valid))
            foc_row.append(foc_stats)
            mic_row.append(mic_stats)
            work_rows.append(
                {
                    "policy": "FOC",
                    "load_torque": float(load),
                    "omega_ref": float(speed),
                    "speed_err_abs": float(foc_err_abs),
                    "speed_err_rel": float(foc_err_rel),
                    "valid_speed": int(foc_valid),
                    **foc_stats,
                }
            )
            work_rows.append(
                {
                    "policy": "MIC_AI",
                    "load_torque": float(load),
                    "omega_ref": float(speed),
                    "speed_err_abs": float(mic_err_abs),
                    "speed_err_rel": float(mic_err_rel),
                    "valid_speed": int(mic_valid),
                    **mic_stats,
                }
            )
            if foc_valid and mic_valid:
                work_rows_filtered.append(
                    {
                        "policy": "FOC",
                        "load_torque": float(load),
                        "omega_ref": float(speed),
                        "speed_err_abs": float(foc_err_abs),
                        "speed_err_rel": float(foc_err_rel),
                        "valid_speed": 1,
                        **foc_stats,
                    }
                )
                work_rows_filtered.append(
                    {
                        "policy": "MIC_AI",
                        "load_torque": float(load),
                        "omega_ref": float(speed),
                        "speed_err_abs": float(mic_err_abs),
                        "speed_err_rel": float(mic_err_rel),
                        "valid_speed": 1,
                        **mic_stats,
                    }
                )
        foc_grid.append(foc_row)
        mic_grid.append(mic_row)
        valid_grid.append(valid_row)

    _save_csv(out_dir / "working_characteristics.csv", work_rows)
    _plot_working_characteristics(out_dir / "working_characteristics.png", speeds, loads, foc_grid, mic_grid)
    _save_csv(out_dir / "working_characteristics_filtered.csv", work_rows_filtered)
    _plot_working_characteristics(out_dir / "working_characteristics_valid.png", speeds, loads, foc_grid, mic_grid, valid_grid)

    mic_policy = "ai"
    if mic_rule:
        mic_policy = "rule"
    elif mic_id_ref is not None:
        mic_policy = "fixed_id"

    meta = {
        "env_config": str(env_path),
        "ai_checkpoint": None if ckpt is None else str(ckpt.resolve()),
        "mic_policy": mic_policy,
        "mic_id_ref": None if mic_id_ref is None else float(mic_id_ref),
        "mic_id_ref_low": None if mic_id_ref_low is None else float(mic_id_ref_low),
        "mic_id_ref_high": None if mic_id_ref_high is None else float(mic_id_ref_high),
        "mic_id_ref_speed_tol_rel": float(args.mic_id_ref_speed_tol_rel),
        "mic_id_ref_omega_min": float(args.mic_id_ref_omega_min),
        "omega_ref_pu": float(args.omega_ref_pu),
        "ai_mode": str(args.ai_mode),
        "omega_nom_source": str(args.omega_nom_source),
        "omega_nominal": float(omega_nom),
        "m_nom": float(m_nom),
        "loads": loads.tolist(),
        "load_values": load_values,
        "speed_pu": speed_pu,
        "dt": dt,
        "t_end": t_end,
        "window_frac": float(args.window_frac),
        "voltage_scale": None if v_scale is None else float(v_scale),
        "i_max": None if args.i_max is None else float(args.i_max),
        "speed_tol_rel": float(args.speed_tol),
        "speed_tol_abs": None if args.speed_tol_abs is None else float(args.speed_tol_abs),
        "ai_id_relative": bool(args.ai_id_relative),
        "delta_id_max": float(args.delta_id_max),
        "plot_style": "vak_ru",
        "plot_formats": ["png", "pdf", "svg"],
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"Saved outputs to {out_dir}")


if __name__ == "__main__":
    main()
