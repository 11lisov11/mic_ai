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
from mic_ai.analysis.metrics import calc_cos_phi, calc_i_rms, calc_p_el, calc_p_mech, calc_v_rms
from mic_ai.core.env import make_env_from_config
from mic_ai.tools.plot_style import apply_vak_style, ensure_matplotlib, save_figure
from simulation.gym_env import InductionMotorEnv


def _extract_nameplate(env_cfg: object) -> Dict[str, float] | None:
    try:
        attrs = vars(env_cfg)
    except Exception:
        attrs = {}
    candidates: List[tuple[int, str, dict]] = []
    for name, value in attrs.items():
        if not isinstance(value, dict):
            continue
        if "P_n" not in value:
            continue
        score = 0
        if "n_rated" in value:
            score += 3
        if "f_n" in value:
            score += 2
        if "p" in value:
            score += 2
        if str(name).startswith("NAMEPLATE_"):
            score += 1
        if str(name) == "NAMEPLATE_DEFAULT":
            score -= 2
        candidates.append((score, str(name), value))
    if not candidates:
        return None
    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return dict(candidates[0][2])


def _rated_omega(env_cfg: object | None = None) -> float:
    if env_cfg is not None:
        plate = _extract_nameplate(env_cfg)
        if plate is not None:
            try:
                n_rated = float(plate.get("n_rated", 0.0))
            except Exception:
                n_rated = 0.0
            if n_rated > 1e-9:
                return float(2.0 * math.pi * n_rated / 60.0)
    return float(2.0 * math.pi * NAMEPLATE_N_RATED / 60.0)


def _omega_nominal(env_cfg: object, source: str) -> float:
    if source == "base":
        pole_pairs = int(getattr(getattr(env_cfg, "motor", None), "p", 1) or 1)
        return float(2.0 * math.pi * 10.0 / pole_pairs)
    return _rated_omega(env_cfg)


def _rated_torque(env_cfg: object | None = None) -> float:
    omega = _rated_omega(env_cfg)
    if omega <= 0.0:
        raise ValueError("rated omega must be positive")
    if env_cfg is not None:
        plate = _extract_nameplate(env_cfg)
        if plate is not None:
            try:
                p_n = float(plate.get("P_n", 0.0))
            except Exception:
                p_n = 0.0
            if p_n > 1e-9:
                return float(p_n / omega)
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


def _replace_cfg_keep_extras(env_cfg: object, **kwargs):
    fields = set(getattr(env_cfg, "__dataclass_fields__", {}).keys())
    extras = {}
    try:
        for name, value in vars(env_cfg).items():
            if name not in fields:
                extras[name] = value
    except Exception:
        extras = {}
    out = replace(env_cfg, **kwargs)
    for name, value in extras.items():
        if hasattr(out, name):
            continue
        try:
            object.__setattr__(out, name, value)
        except Exception:
            try:
                setattr(out, name, value)
            except Exception:
                pass
    return out


def _calibrate_cosphi_to_nameplate(
    foc_rows: List[Dict[str, float]],
    mic_rows: List[Dict[str, float]],
    cos_phi_nom: float | None,
) -> tuple[float, float]:
    """
    Scale cosphi curves so that FOC nominal-load point matches passport cos_phi_n.
    Returns (scale, cosphi_nominal_raw).
    """
    if cos_phi_nom is None or not np.isfinite(float(cos_phi_nom)) or float(cos_phi_nom) <= 0.0:
        return 1.0, float("nan")
    if not foc_rows:
        return 1.0, float("nan")
    idx_nom = int(
        np.nanargmin(
            np.asarray([abs(float(r.get("load_factor", 0.0)) - 1.0) for r in foc_rows], dtype=float)
        )
    )
    cos_nom_raw = float(foc_rows[idx_nom].get("cos_phi", float("nan")))
    if not np.isfinite(cos_nom_raw) or cos_nom_raw <= 1e-9:
        return 1.0, cos_nom_raw
    scale = float(cos_phi_nom) / cos_nom_raw
    for rows in (foc_rows, mic_rows):
        for r in rows:
            raw = float(r.get("cos_phi", float("nan")))
            r["cos_phi_raw"] = raw
            if np.isfinite(raw):
                r["cos_phi"] = float(np.clip(scale * raw, 0.0, 1.0))
    return scale, cos_nom_raw


def _vll_rms_from_vabc(v_abc: np.ndarray) -> float:
    va, vb, vc = float(v_abc[0]), float(v_abc[1]), float(v_abc[2])
    vab = va - vb
    vbc = vb - vc
    vca = vc - va
    return float(math.sqrt((vab * vab + vbc * vbc + vca * vca) / 3.0))


def _summarize_window(values: Dict[str, np.ndarray], window_frac: float, debug_name: str = "") -> Dict[str, float]:
    n = int(values["t"].size)
    sl = _steady_slice(n, window_frac)
    p_el = values["p_el"][sl]
    p_el_motor = values.get("p_el_motor", values["p_el"])[sl]
    p_mech = values["p_mech"][sl]
    p_shaft = values.get("p_shaft", values["p_mech"])[sl]
    i_abc_w = values.get("i_abc", np.zeros((0, 3), dtype=float))[sl]
    v_abc_w = values.get("v_abc", np.zeros((0, 3), dtype=float))[sl]
    i_rms_w = values.get("i_rms", np.zeros(0, dtype=float))[sl]
    vll_rms_w = values.get("vll_rms", np.zeros(0, dtype=float))[sl]
    omega_w = values["omega"][sl]
    p_el_mean = float(np.mean(p_el)) if p_el.size else 0.0
    p_el_pos_mean = float(np.mean(np.maximum(p_el, 0.0))) if p_el.size else 0.0
    p_el_motor_pos_mean = float(np.mean(np.maximum(p_el_motor, 0.0))) if p_el_motor.size else 0.0
    p_mech_mean = float(np.mean(p_mech)) if p_mech.size else 0.0
    p_shaft_mean = float(np.mean(p_shaft)) if p_shaft.size else 0.0
    # Publication KPI: shaft efficiency against total electrical input power.
    eta_raw = float(p_shaft_mean / p_el_mean) if p_el_mean > 1e-9 else 0.0
    eta = float(np.clip(eta_raw, 0.0, 1.02))
    if p_shaft_mean > p_el_mean + 1e-6:
        print(
            f"[WARN][sanity]{'[' + debug_name + ']' if debug_name else ''} "
            f"P2>P1_total: P2={p_shaft_mean:.3f} W, P1={p_el_mean:.3f} W"
        )
    if i_abc_w.size:
        i_rms_mean = float(calc_i_rms(i_abc_w))
    else:
        i_rms_mean = float(np.mean(i_rms_w)) if i_rms_w.size else 0.0
    if v_abc_w.size:
        v_rms_phase_mean = float(calc_v_rms(v_abc_w))
    else:
        v_rms_phase_mean = float("nan")
    vll_rms_mean = float(np.mean(vll_rms_w)) if vll_rms_w.size else float("nan")
    omega_ss = float(np.mean(omega_w)) if omega_w.size else 0.0
    m2 = float(p_shaft_mean / max(abs(omega_ss), 1e-9))
    n2_rpm = float(abs(omega_ss) * 60.0 / (2.0 * math.pi))
    p2_kw = float(p_shaft_mean / 1000.0)
    cos_phi = float("nan")
    cos_diag: Dict[str, float | str] = {"method": "none", "warning": ""}
    if v_abc_w.size and i_abc_w.size:
        cos_phi, cos_diag = calc_cos_phi(v_abc_w, i_abc_w, window_slice=None)
        warn_tag = str(cos_diag.get("warning", ""))
        method_tag = str(cos_diag.get("method", ""))
        if warn_tag and method_tag != "phase":
            print(
                f"[WARN][cosphi]{'[' + debug_name + ']' if debug_name else ''} "
                f"{warn_tag} | "
                f"P={float(cos_diag.get('p_mean', float('nan'))):.3f} W, "
                f"Vrms_phase={float(cos_diag.get('v_rms_phase', float('nan'))):.3f} V, "
                f"Irms={float(cos_diag.get('i_rms_phase', float('nan'))):.3f} A, "
                f"S_phase={float(cos_diag.get('s_phase', float('nan'))):.3f} VA, "
                f"S_line={float(cos_diag.get('s_line', float('nan'))):.3f} VA"
            )
    else:
        denom = math.sqrt(3.0) * max(vll_rms_mean, 0.0) * max(i_rms_mean, 0.0)
        if denom > 1e-9:
            cos_phi = float(np.clip(p_el_motor_pos_mean / denom, 0.0, 1.0))
            cos_diag = {"method": "line_fallback", "warning": "raw_abc_absent"}
    if not (0.0 <= float(cos_phi) <= 1.0):
        print(f"[WARN][sanity]{'[' + debug_name + ']' if debug_name else ''} cosphi out of range: {cos_phi}")
        cos_phi = float(np.clip(cos_phi, 0.0, 1.0))
    return {
        "omega_ss": omega_ss,
        "n2_rpm": n2_rpm,
        "i_rms": i_rms_mean,
        "v_rms_phase": v_rms_phase_mean,
        "vll_rms": vll_rms_mean,
        "p_el": p_el_mean,
        "p_el_pos": p_el_pos_mean,
        "p_el_motor_pos": p_el_motor_pos_mean,
        "p_mech": p_mech_mean,
        "p_shaft": p_shaft_mean,
        "p2_kw": p2_kw,
        "m2": m2,
        "eta": eta,
        "eta_raw": eta_raw,
        "eta_pct": float(eta * 100.0),
        "cos_phi": cos_phi,
        "cos_phi_method": str(cos_diag.get("method", "none")),
        "cos_phi_warning": str(cos_diag.get("warning", "")),
        "cos_phi_phase_raw": float(cos_diag.get("cos_phase_raw", float("nan"))),
        "cos_phi_line_raw": float(cos_diag.get("cos_line_raw", float("nan"))),
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
    vll_rms = np.zeros(steps, dtype=float)
    i_abc_hist = np.zeros((steps, 3), dtype=float)
    v_abc_hist = np.zeros((steps, 3), dtype=float)
    p_el_motor = np.zeros(steps, dtype=float)
    p_el = np.zeros(steps, dtype=float)
    p_mech = np.zeros(steps, dtype=float)
    p_shaft = np.zeros(steps, dtype=float)

    for k in range(steps):
        obs, _r, done, info = env.step(None)
        t[k] = float(env.t)
        omega[k] = float(obs[0]) if hasattr(obs, "__len__") else float(info.get("omega_meas", 0.0))
        i_abc = np.asarray(info.get("i_abc", (0.0, 0.0, 0.0)), dtype=float)
        v_abc = np.asarray(info.get("v_abc", (0.0, 0.0, 0.0)), dtype=float)
        i_abc_hist[k, :] = i_abc
        v_abc_hist[k, :] = v_abc
        torque = float(info.get("torque_e", obs[2] if hasattr(obs, "__len__") else 0.0))
        i_rms[k] = calc_i_rms(i_abc)
        vll_rms[k] = _vll_rms_from_vabc(v_abc)
        p_el_motor[k] = calc_p_el(v_abc, i_abc)
        p_el[k] = float(info.get("p_in_total", p_el_motor[k]))
        p_mech[k] = calc_p_mech(omega[k], torque)
        p_shaft[k] = float(max(0.0, float(load_torque) * float(omega[k])))
        if done:
            t = t[: k + 1]
            omega = omega[: k + 1]
            i_rms = i_rms[: k + 1]
            vll_rms = vll_rms[: k + 1]
            i_abc_hist = i_abc_hist[: k + 1, :]
            v_abc_hist = v_abc_hist[: k + 1, :]
            p_el_motor = p_el_motor[: k + 1]
            p_el = p_el[: k + 1]
            p_mech = p_mech[: k + 1]
            p_shaft = p_shaft[: k + 1]
            break
    return {
        "t": t,
        "omega": omega,
        "i_rms": i_rms,
        "vll_rms": vll_rms,
        "i_abc": i_abc_hist,
        "v_abc": v_abc_hist,
        "p_el_motor": p_el_motor,
        "p_el": p_el,
        "p_mech": p_mech,
        "p_shaft": p_shaft,
    }


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
    vll_rms = np.zeros(steps, dtype=float)
    i_abc_hist = np.zeros((steps, 3), dtype=float)
    v_abc_hist = np.zeros((steps, 3), dtype=float)
    p_el_motor = np.zeros(steps, dtype=float)
    p_el = np.zeros(steps, dtype=float)
    p_mech = np.zeros(steps, dtype=float)
    p_shaft = np.zeros(steps, dtype=float)
    # Smooth id_ref scheduler to avoid hard switching near the speed-error boundary.
    id_low = float(min(id_ref_low, id_ref_high))
    id_high = float(max(id_ref_low, id_ref_high))
    err_band_low = 0.2
    err_band_high = 1.0
    id_ref_cmd = float(id_high)
    tau_id = 0.004

    for k in range(steps):
        omega_ref_k = float(omega_ref)
        omega_meas = float(getattr(getattr(env.motor, "state", None), "omega_m", 0.0))
        omega_ref_scale = max(abs(omega_ref_k), 1e-6)
        err = abs(omega_ref_k - omega_meas)
        id_ref_target = float(id_high)
        if abs(omega_ref_k) >= float(omega_min_pu) * float(omega_nom):
            err_tol = max(float(speed_tol_rel) * omega_ref_scale, 1e-6)
            err_rel = float(err / err_tol)
            u = (err_rel - err_band_low) / max(err_band_high - err_band_low, 1e-9)
            u = float(np.clip(u, 0.0, 1.0))
            blend = float(u * u * (3.0 - 2.0 * u))  # smoothstep
            id_ref_target = float(id_low + (id_high - id_low) * blend)
        alpha = float(np.clip(dt / max(tau_id, 1e-6), 0.0, 1.0))
        id_ref_cmd = float((1.0 - alpha) * id_ref_cmd + alpha * id_ref_target)
        env.controller.params = replace(env.controller.params, id_ref=float(id_ref_cmd))

        obs, _r, done, info = env.step(None)
        t[k] = float(env.t)
        omega[k] = float(obs[0]) if hasattr(obs, "__len__") else float(info.get("omega_meas", 0.0))
        i_abc = np.asarray(info.get("i_abc", (0.0, 0.0, 0.0)), dtype=float)
        v_abc = np.asarray(info.get("v_abc", (0.0, 0.0, 0.0)), dtype=float)
        i_abc_hist[k, :] = i_abc
        v_abc_hist[k, :] = v_abc
        torque = float(info.get("torque_e", obs[2] if hasattr(obs, "__len__") else 0.0))
        i_rms[k] = calc_i_rms(i_abc)
        vll_rms[k] = _vll_rms_from_vabc(v_abc)
        p_el_motor[k] = calc_p_el(v_abc, i_abc)
        p_el[k] = float(info.get("p_in_total", p_el_motor[k]))
        p_mech[k] = calc_p_mech(omega[k], torque)
        p_shaft[k] = float(max(0.0, float(load_torque) * float(omega[k])))
        if done:
            t = t[: k + 1]
            omega = omega[: k + 1]
            i_rms = i_rms[: k + 1]
            vll_rms = vll_rms[: k + 1]
            i_abc_hist = i_abc_hist[: k + 1, :]
            v_abc_hist = v_abc_hist[: k + 1, :]
            p_el_motor = p_el_motor[: k + 1]
            p_el = p_el[: k + 1]
            p_mech = p_mech[: k + 1]
            p_shaft = p_shaft[: k + 1]
            break
    return {
        "t": t,
        "omega": omega,
        "i_rms": i_rms,
        "vll_rms": vll_rms,
        "i_abc": i_abc_hist,
        "v_abc": v_abc_hist,
        "p_el_motor": p_el_motor,
        "p_el": p_el,
        "p_mech": p_mech,
        "p_shaft": p_shaft,
    }


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
    vll_rms = np.zeros(steps, dtype=float)
    i_abc_hist = np.zeros((steps, 3), dtype=float)
    v_abc_hist = np.zeros((steps, 3), dtype=float)
    p_el_motor = np.zeros(steps, dtype=float)
    p_el = np.zeros(steps, dtype=float)
    p_mech = np.zeros(steps, dtype=float)
    p_shaft = np.zeros(steps, dtype=float)

    for k in range(steps):
        action, _lp, _v = agent.act(obs)
        obs, _r, done, info = env.step(action)
        t[k] = float(getattr(env.base_env, "t", k * dt))
        omega[k] = float(info.get("omega_meas", obs.get("omega", 0.0)))
        i_abc = np.asarray(info.get("i_abc", (0.0, 0.0, 0.0)), dtype=float)
        v_abc = np.asarray(info.get("v_abc", (0.0, 0.0, 0.0)), dtype=float)
        i_abc_hist[k, :] = i_abc
        v_abc_hist[k, :] = v_abc
        torque = float(getattr(env.base_env, "last_torque", 0.0))
        i_rms[k] = calc_i_rms(i_abc)
        vll_rms[k] = _vll_rms_from_vabc(v_abc)
        p_el_motor[k] = calc_p_el(v_abc, i_abc)
        p_el[k] = float(info.get("p_in_total", p_el_motor[k]))
        p_mech[k] = calc_p_mech(omega[k], torque)
        p_shaft[k] = float(max(0.0, float(load_torque) * float(omega[k])))
        if done:
            t = t[: k + 1]
            omega = omega[: k + 1]
            i_rms = i_rms[: k + 1]
            vll_rms = vll_rms[: k + 1]
            i_abc_hist = i_abc_hist[: k + 1, :]
            v_abc_hist = v_abc_hist[: k + 1, :]
            p_el_motor = p_el_motor[: k + 1]
            p_el = p_el[: k + 1]
            p_mech = p_mech[: k + 1]
            p_shaft = p_shaft[: k + 1]
            break
    return {
        "t": t,
        "omega": omega,
        "i_rms": i_rms,
        "vll_rms": vll_rms,
        "i_abc": i_abc_hist,
        "v_abc": v_abc_hist,
        "p_el_motor": p_el_motor,
        "p_el": p_el,
        "p_mech": p_mech,
        "p_shaft": p_shaft,
    }


def _save_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _save_trace_csv(
    path: Path,
    values: Dict[str, np.ndarray],
    omega_ref: float,
    load_torque: float,
    policy: str,
) -> None:
    t = values.get("t", np.zeros(0, dtype=float))
    n = int(t.size)
    if n == 0:
        return
    i_abc = values.get("i_abc", np.zeros((n, 3), dtype=float))
    v_abc = values.get("v_abc", np.zeros((n, 3), dtype=float))
    rows: List[Dict[str, object]] = []
    for k in range(n):
        rows.append(
            {
                "policy": policy,
                "t": float(t[k]),
                "omega": float(values.get("omega", np.zeros(n, dtype=float))[k]),
                "omega_ref": float(omega_ref),
                "load_torque": float(load_torque),
                "i_a": float(i_abc[k, 0]),
                "i_b": float(i_abc[k, 1]),
                "i_c": float(i_abc[k, 2]),
                "v_a": float(v_abc[k, 0]),
                "v_b": float(v_abc[k, 1]),
                "v_c": float(v_abc[k, 2]),
                "p_el_motor": float(values.get("p_el_motor", np.zeros(n, dtype=float))[k]),
                "p_el_total": float(values.get("p_el", np.zeros(n, dtype=float))[k]),
                "p_mech": float(values.get("p_mech", np.zeros(n, dtype=float))[k]),
                "p_shaft": float(values.get("p_shaft", np.zeros(n, dtype=float))[k]),
            }
        )
    _save_csv(path, rows)


def _sanity_check_curves(label: str, rows: List[Dict[str, float]]) -> None:
    if not rows:
        return
    rows_s = sorted(rows, key=lambda r: float(r.get("load_factor", 0.0)))
    m2 = np.asarray([float(r.get("m2", 0.0)) for r in rows_s], dtype=float)
    i1 = np.asarray([float(r.get("i_rms", 0.0)) for r in rows_s], dtype=float)
    n2 = np.asarray([float(r.get("n2_rpm", 0.0)) for r in rows_s], dtype=float)
    eta = np.asarray([float(r.get("eta", 0.0)) for r in rows_s], dtype=float)
    cos = np.asarray([float(r.get("cos_phi", 0.0)) for r in rows_s], dtype=float)
    p1 = np.asarray([float(r.get("p_el", 0.0)) for r in rows_s], dtype=float)
    p2 = np.asarray([float(r.get("p_shaft", 0.0)) for r in rows_s], dtype=float)

    if np.any((eta < -1e-6) | (eta > 1.02 + 1e-6)):
        print(f"[WARN][sanity][{label}] eta out of [0..1.02]")
    if np.any((cos < -1e-6) | (cos > 1.0 + 1e-6)):
        print(f"[WARN][sanity][{label}] cosphi out of [0..1]")
    if np.any(p2 > p1 + 1e-6):
        print(f"[WARN][sanity][{label}] P2 > P1_total at some points")
    if np.any(np.diff(m2) < -0.02 * np.maximum(1.0, np.max(np.abs(m2)))):
        print(f"[WARN][sanity][{label}] M2 is not monotonic increasing")
    if np.any(np.diff(i1) < -0.03 * np.maximum(1.0, np.max(np.abs(i1)))):
        print(f"[WARN][sanity][{label}] I1 has strong non-monotonic drop")
    if np.any(np.diff(n2) > 0.02 * np.maximum(1.0, np.max(np.abs(n2)))):
        print(f"[WARN][sanity][{label}] n2 increases with load unexpectedly")


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
    ax.set_xlabel("M_РЅР°РіСЂСѓР·РєРё, РќВ·Рј")
    ax.set_ylabel("П‰_СѓСЃС‚, СЂР°Рґ/СЃ")

    ax = axes[0, 1]
    ax.plot(loads, _mask_values([x["i_rms"] for x in foc], valid_mask), color="black", marker="o", linestyle="-")
    ax.plot(loads, _mask_values([x["i_rms"] for x in mic], valid_mask), color="0.35", marker="s", linestyle="--")
    ax.set_xlabel("M_РЅР°РіСЂСѓР·РєРё, РќВ·Рј")
    ax.set_ylabel("I_rms, Рђ")

    ax = axes[1, 0]
    ax.plot(loads, _mask_values([x["p_el"] for x in foc], valid_mask), color="black", marker="o", linestyle="-")
    ax.plot(loads, _mask_values([x["p_el"] for x in mic], valid_mask), color="0.35", marker="s", linestyle="--")
    ax.set_xlabel("M_РЅР°РіСЂСѓР·РєРё, РќВ·Рј")
    ax.set_ylabel("P_СЌР», Р’С‚")

    ax = axes[1, 1]
    ax.plot(loads, _mask_values([x["eta"] for x in foc], valid_mask), color="black", marker="o", linestyle="-")
    ax.plot(loads, _mask_values([x["eta"] for x in mic], valid_mask), color="0.35", marker="s", linestyle="--")
    ax.set_xlabel("M_РЅР°РіСЂСѓР·РєРё, РќВ·Рј")
    ax.set_ylabel("О· = P_РјРµС… / P_СЌР»")

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

    axes[0].set_ylabel("I_rms, Рђ")
    axes[1].set_ylabel("P_СЌР», Р’С‚")
    axes[2].set_ylabel("P_РјРµС…, Р’С‚")
    for ax in axes:
        ax.set_xlabel("П‰_Р·Р°Рґ, СЂР°Рґ/СЃ")

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
        title="M_РЅР°РіСЂСѓР·РєРё, РќВ·Рј",
    )
    fig.tight_layout(rect=[0.0, 0.08, 1.0, 0.92])
    save_figure(fig, out_path)
    plt.close(fig)


def _interp_at_x(x: np.ndarray, y: np.ndarray, x0: float) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(mask)) == 0:
        return float("nan")
    xx = x[mask]
    yy = y[mask]
    order = np.argsort(xx)
    xx = xx[order]
    yy = yy[order]
    if xx.size == 1:
        return float(yy[0])
    if float(x0) < float(xx[0]) or float(x0) > float(xx[-1]):
        return float("nan")
    return float(np.interp(float(x0), xx, yy))


def _interp_or_extrap_at_x(x: np.ndarray, y: np.ndarray, x0: float) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(mask)) == 0:
        return float("nan")
    xx = np.asarray(x[mask], dtype=float)
    yy = np.asarray(y[mask], dtype=float)
    order = np.argsort(xx)
    xx = xx[order]
    yy = yy[order]
    if xx.size == 1:
        return float(yy[0])
    x0f = float(x0)
    if float(xx[0]) <= x0f <= float(xx[-1]):
        return float(np.interp(x0f, xx, yy))
    if x0f < float(xx[0]):
        x1, x2 = float(xx[0]), float(xx[1])
        y1, y2 = float(yy[0]), float(yy[1])
    else:
        x1, x2 = float(xx[-2]), float(xx[-1])
        y1, y2 = float(yy[-2]), float(yy[-1])
    if abs(x2 - x1) < 1e-12:
        return float(y2)
    return float(y1 + (y2 - y1) * (x0f - x1) / (x2 - x1))


def _smooth_xy_curve(
    x: np.ndarray,
    y: np.ndarray,
    points: int = 360,
    y_min: float | None = None,
    y_max: float | None = None,
    trend: str = "none",
    pre_smooth_window: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(mask)) == 0:
        return np.zeros(0, dtype=float), np.zeros(0, dtype=float)

    xx = np.asarray(x[mask], dtype=float)
    yy = np.asarray(y[mask], dtype=float)
    order = np.argsort(xx)
    xx = xx[order]
    yy = yy[order]

    # Keep last sample for duplicated x, preserving ascending order.
    if xx.size > 1:
        keep = np.ones(xx.size, dtype=bool)
        keep[:-1] = np.diff(xx) != 0.0
        xx = xx[keep]
        yy = yy[keep]
    if xx.size > 1 and float(xx[0]) > float(xx[-1]):
        xx = xx[::-1]
        yy = yy[::-1]
    yy_edge = yy.copy()
    if int(pre_smooth_window) >= 3 and yy.size >= 3:
        win = int(pre_smooth_window)
        if win % 2 == 0:
            win -= 1
        win = max(3, min(win, int(yy.size if yy.size % 2 == 1 else yy.size - 1)))
        if win >= 3:
            pad = win // 2
            y_pad = np.pad(yy, (pad, pad), mode="edge")
            kernel = np.ones(win, dtype=float) / float(win)
            yy = np.convolve(y_pad, kernel, mode="valid")
            yy[0] = float(yy_edge[0])
            yy[-1] = float(yy_edge[-1])
    if str(trend).lower() == "non_increasing":
        yy = np.minimum.accumulate(yy)
    elif str(trend).lower() == "non_decreasing":
        yy = np.maximum.accumulate(yy)
    if xx.size <= 2:
        ys = yy
        if y_min is not None or y_max is not None:
            ys = np.clip(
                ys,
                -np.inf if y_min is None else float(y_min),
                np.inf if y_max is None else float(y_max),
            )
        return xx, ys

    x_dense = np.linspace(float(xx[0]), float(xx[-1]), int(max(points, xx.size)))
    y_dense = np.interp(x_dense, xx, yy)
    try:
        from scipy.interpolate import PchipInterpolator

        spline = PchipInterpolator(xx, yy, extrapolate=False)
        y_spl = np.asarray(spline(x_dense), dtype=float)
        if np.any(np.isfinite(y_spl)):
            y_dense = y_spl
    except Exception:
        pass

    if y_min is not None or y_max is not None:
        y_dense = np.clip(
            y_dense,
            -np.inf if y_min is None else float(y_min),
            np.inf if y_max is None else float(y_max),
        )
    return x_dense, y_dense


def _plot_air56_mech_journal(
    out_path: Path,
    foc: List[Dict[str, float]],
    mic: List[Dict[str, float]],
    common_p2_kw: float = 0.25,
    char_load_factors: Tuple[float, float] = (0.5, 1.0),
) -> None:
    plt = apply_vak_style(ensure_matplotlib())
    fig, axes = plt.subplots(2, 1, figsize=(13.6, 10.8), sharex=True)

    colors = {
        "M2": "#1f4e79",
        "I1": "#8f5a2a",
        "n2": "#2f6b3f",
        "eta": "#7a2f2f",
        "cosphi": "#5b4b8a",
    }

    x_f = np.asarray([float(r["p2_kw"]) for r in foc], dtype=float)
    x_m = np.asarray([float(r["p2_kw"]) for r in mic], dtype=float)
    x_all = np.concatenate([x_f, x_m]) if x_f.size and x_m.size else (x_f if x_f.size else x_m)
    if x_all.size == 0:
        return
    x_min = float(np.nanmin(x_all))
    x_max = float(np.nanmax(x_all))
    x_span_raw = max(1e-6, x_max - x_min)
    x_pad = max(0.002, 0.05 * x_span_raw)
    x_left = x_min - x_pad
    x_right = x_max + x_pad
    if np.isfinite(common_p2_kw):
        # Keep the common power marker visible even when it is slightly
        # outside sampled P2 points (e.g. target 0.250 kW near right edge).
        right_extra = max(0.004, 0.04 * x_span_raw)
        x_right = max(x_right, float(common_p2_kw) + right_extra)

    def _characteristic_rows(rows: List[Dict[str, float]]) -> List[Dict[str, float]]:
        if not rows:
            return []
        out: List[Dict[str, float]] = []
        for lf in char_load_factors:
            idx = int(
                np.nanargmin(
                    np.asarray([abs(float(r.get("load_factor", float("nan"))) - float(lf)) for r in rows], dtype=float)
                )
            )
            out.append(rows[idx])
        return out

    def _draw_panel(ax_m2, rows: List[Dict[str, float]], panel_title: str) -> None:
        import matplotlib.ticker as mticker

        x = np.asarray([float(r["p2_kw"]) for r in rows], dtype=float)
        m2 = np.asarray([float(r["m2"]) for r in rows], dtype=float)
        i1 = np.asarray([float(r["i_rms"]) for r in rows], dtype=float)
        n2 = np.asarray([float(r["n2_rpm"]) for r in rows], dtype=float)
        eta = np.asarray([float(r["eta_pct"]) for r in rows], dtype=float)
        cosphi = np.asarray([float(r["cos_phi"]) for r in rows], dtype=float)
        ax_i1 = ax_m2.twinx()
        ax_n2 = ax_m2.twinx()
        ax_eta = ax_m2.twinx()
        ax_cosphi = ax_m2.twinx()
        for ax_extra in (ax_i1, ax_n2, ax_eta, ax_cosphi):
            ax_extra.set_frame_on(True)
            ax_extra.patch.set_visible(False)

        # Left axes.
        ax_i1.spines["right"].set_visible(False)
        ax_i1.spines["left"].set_visible(True)
        ax_i1.spines["left"].set_position(("axes", -0.14))
        ax_i1.yaxis.set_label_position("left")
        ax_i1.yaxis.set_ticks_position("left")

        ax_n2.spines["right"].set_visible(False)
        ax_n2.spines["left"].set_visible(True)
        ax_n2.spines["left"].set_position(("axes", -0.27))
        ax_n2.yaxis.set_label_position("left")
        ax_n2.yaxis.set_ticks_position("left")

        # Right axes.
        ax_eta.spines["right"].set_visible(True)
        ax_eta.spines["right"].set_position(("axes", 1.03))
        ax_eta.yaxis.set_label_position("right")
        ax_eta.yaxis.set_ticks_position("right")
        ax_cosphi.spines["right"].set_visible(True)
        ax_cosphi.spines["right"].set_position(("axes", 1.14))
        ax_cosphi.yaxis.set_label_position("right")
        ax_cosphi.yaxis.set_ticks_position("right")

        lw = 1.9
        ms = 2.4
        x_m2, y_m2 = _smooth_xy_curve(x, m2, points=420, y_min=0.0)
        x_i1, y_i1 = _smooth_xy_curve(x, i1, points=420, y_min=0.0)
        x_n2, y_n2 = _smooth_xy_curve(
            x,
            n2,
            points=420,
            trend="non_increasing",
            pre_smooth_window=3,
        )
        x_eta, y_eta = _smooth_xy_curve(x, eta, points=420, y_min=0.0, y_max=100.0)
        x_cos, y_cos = _smooth_xy_curve(x, cosphi, points=420, y_min=0.0, y_max=1.0)
        ax_m2.plot(x_m2, y_m2, color=colors["M2"], linewidth=lw)
        ax_i1.plot(x_i1, y_i1, color=colors["I1"], linewidth=lw)
        ax_n2.plot(x_n2, y_n2, color=colors["n2"], linewidth=lw)
        ax_eta.plot(x_eta, y_eta, color=colors["eta"], linewidth=lw)
        ax_cosphi.plot(x_cos, y_cos, color=colors["cosphi"], linewidth=lw)
        # Keep raw simulation points visible on top of smooth curves.
        for ax_obj, y_raw, c in (
            (ax_m2, m2, colors["M2"]),
            (ax_i1, i1, colors["I1"]),
            (ax_n2, n2, colors["n2"]),
            (ax_eta, eta, colors["eta"]),
            (ax_cosphi, cosphi, colors["cosphi"]),
        ):
            ax_obj.plot(
                x,
                y_raw,
                linestyle="None",
                marker="o",
                markersize=ms,
                markerfacecolor="white",
                markeredgecolor=c,
                markeredgewidth=0.9,
                alpha=0.95,
            )

        char_rows = _characteristic_rows(rows)
        if char_rows:
            xc = np.asarray([float(r.get("p2_kw", float("nan"))) for r in char_rows], dtype=float)
            if np.any(np.isfinite(xc)):
                m2c = np.asarray([float(r.get("m2", float("nan"))) for r in char_rows], dtype=float)
                i1c = np.asarray([float(r.get("i_rms", float("nan"))) for r in char_rows], dtype=float)
                n2c = np.asarray([float(r.get("n2_rpm", float("nan"))) for r in char_rows], dtype=float)
                etac = np.asarray([float(r.get("eta_pct", float("nan"))) for r in char_rows], dtype=float)
                cosc = np.asarray([float(r.get("cos_phi", float("nan"))) for r in char_rows], dtype=float)
                for ax_obj, yc in (
                    (ax_m2, m2c),
                    (ax_i1, i1c),
                    (ax_n2, n2c),
                    (ax_eta, etac),
                    (ax_cosphi, cosc),
                ):
                    ax_obj.scatter(
                        xc,
                        yc,
                        marker="D",
                        s=24,
                        facecolors="none",
                        edgecolors="black",
                        linewidths=0.8,
                        zorder=7,
                    )

        # Common P2 marker and eta projection.
        has_common = bool(np.isfinite(common_p2_kw) and float(x_left) <= float(common_p2_kw) <= float(x_right))
        if has_common:
            ax_m2.axvline(float(common_p2_kw), color="0.55", linestyle="--", linewidth=1.0, zorder=0)
        from matplotlib.transforms import blended_transform_factory

        x_span = max(1e-9, x_right - x_left)
        eta_at_common = float("nan")
        if has_common:
            p2_label = f"P2={common_p2_kw:.3f} кВт".replace(".", ",")
            ax_m2.text(
                float(common_p2_kw) - 0.012 * x_span,
                0.72,
                p2_label,
                rotation=90,
                ha="right",
                va="center",
                color="0.45",
                fontsize=9,
                transform=blended_transform_factory(ax_m2.transData, ax_m2.transAxes),
            )
            eta_at_common = _interp_or_extrap_at_x(x_eta, y_eta, float(common_p2_kw))
            if np.isfinite(eta_at_common):
                ax_eta.plot([float(common_p2_kw), x_right], [eta_at_common, eta_at_common], linestyle="--", color=colors["eta"], linewidth=1.0)
                ax_eta.scatter([float(common_p2_kw)], [eta_at_common], s=42, facecolors="white", edgecolors=colors["eta"], linewidths=1.2, zorder=7)

        ax_m2.set_title(panel_title, loc="left", fontweight="bold")
        ax_m2.set_xlim(x_left, x_right)
        for ax_no_grid in (ax_m2, ax_i1, ax_n2, ax_eta, ax_cosphi):
            ax_no_grid.grid(False)

        # Axis ranges and plain-number formatting.
        ax_m2.set_ylim(0.0, float(max(np.nanmax(m2) * 1.06, 1e-3)))
        i1_lo = float(np.nanmin(i1))
        i1_hi = float(np.nanmax(i1))
        i1_pad = max(0.02, 0.08 * max(i1_hi - i1_lo, 1e-6))
        ax_i1.set_ylim(max(0.0, i1_lo - i1_pad), i1_hi + i1_pad)
        n2_lo = float(np.nanmin(n2))
        n2_hi = float(np.nanmax(n2))
        n2_pad = max(8.0, 0.12 * max(n2_hi - n2_lo, 1.0))
        ax_n2.set_ylim(n2_lo - n2_pad, n2_hi + n2_pad)
        ax_n2.ticklabel_format(style="plain", axis="y", useOffset=False)
        ax_n2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
        eta_lo = float(np.nanmin(eta))
        eta_hi = float(np.nanmax(eta))
        eta_pad = max(1.0, 0.10 * max(eta_hi - eta_lo, 1.0))
        ax_eta.set_ylim(max(0.0, eta_lo - eta_pad), eta_hi + eta_pad)
        cos_lo = float(np.nanmin(cosphi))
        cos_hi = float(np.nanmax(cosphi))
        cos_pad = max(0.01, 0.08 * max(cos_hi - cos_lo, 1e-6))
        ax_cosphi.set_ylim(max(0.0, cos_lo - cos_pad), min(1.0, cos_hi + cos_pad))
        if np.isfinite(eta_at_common):
            eta_label = f"η={eta_at_common:.1f}%".replace(".", ",")
            ax_eta.text(
                1.055,
                eta_at_common,
                eta_label,
                color=colors["eta"],
                fontsize=9,
                ha="left",
                va="center",
                transform=blended_transform_factory(ax_eta.transAxes, ax_eta.transData),
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.70, "pad": 0.20},
                clip_on=False,
            )

        ax_m2.set_ylabel("M2, Н·м", color=colors["M2"])
        ax_i1.set_ylabel("I1, A", color=colors["I1"])
        ax_n2.set_ylabel("n2, об/мин", color=colors["n2"])
        ax_eta.set_ylabel("η, %", color=colors["eta"])
        ax_cosphi.set_ylabel("cosφ, о.е.", color=colors["cosphi"])

        ax_m2.tick_params(axis="y", colors=colors["M2"])
        ax_i1.tick_params(axis="y", colors=colors["I1"])
        ax_n2.tick_params(axis="y", colors=colors["n2"])
        ax_eta.tick_params(axis="y", colors=colors["eta"])
        ax_cosphi.tick_params(axis="y", colors=colors["cosphi"])

        ax_m2.spines["left"].set_color(colors["M2"])
        ax_i1.spines["left"].set_color(colors["I1"])
        ax_n2.spines["left"].set_color(colors["n2"])
        ax_eta.spines["right"].set_color(colors["eta"])
        ax_cosphi.spines["right"].set_color(colors["cosphi"])
        ax_eta.spines["right"].set_linewidth(1.0)
        ax_cosphi.spines["right"].set_linewidth(1.0)
        ax_i1.spines["left"].set_linewidth(1.0)
        ax_n2.spines["left"].set_linewidth(1.0)
        ax_m2.spines["left"].set_linewidth(1.0)

    _draw_panel(axes[0], foc, "а) FOC")
    _draw_panel(axes[1], mic, "б) MIC")
    axes[1].set_xlabel("P2, кВт")
    fig.suptitle("Рабочие характеристики AIR56: раздельно для FOC (а) и MIC (б)", y=0.985)

    from matplotlib.lines import Line2D

    handles = [
        Line2D([0], [0], color=colors["M2"], linestyle="-", label="M2"),
        Line2D([0], [0], color=colors["I1"], linestyle="-", label="I1"),
        Line2D([0], [0], color=colors["n2"], linestyle="-", label="n2"),
        Line2D([0], [0], color=colors["eta"], linestyle="-", label="η"),
        Line2D([0], [0], color=colors["cosphi"], linestyle="-", label="cosφ"),
        Line2D([0], [0], color="black", marker="D", linestyle="None", markerfacecolor="none", label="Характерные точки (0,5Mном; 1,0Mном)"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.955))
    fig.subplots_adjust(left=0.20, right=0.84, top=0.88, bottom=0.08, hspace=0.12)
    save_figure(fig, out_path)
    plt.close(fig)


def _select_journal_rows(
    load_rows: List[Dict[str, object]],
    drop_zero_load: bool,
    max_speed_err_rel: float | None,
    max_n2_step_rpm: float | None,
) -> tuple[List[Dict[str, float]], List[Dict[str, float]]]:
    foc_map: Dict[float, Dict[str, float]] = {}
    mic_map: Dict[float, Dict[str, float]] = {}
    err_lim = None
    if max_speed_err_rel is not None and float(max_speed_err_rel) > 0.0:
        err_lim = float(max_speed_err_rel)

    for row in load_rows:
        try:
            policy = str(row.get("policy", ""))
            lf = float(row.get("load_factor", float("nan")))
            err_rel = float(row.get("speed_err_rel", 0.0))
        except Exception:
            continue
        if not np.isfinite(lf):
            continue
        if drop_zero_load and lf <= 1e-9:
            continue
        if err_lim is not None and err_rel > err_lim:
            continue
        key = round(lf, 8)
        row_copy = dict(row)
        if policy == "FOC":
            foc_map[key] = row_copy
        elif policy == "MIC_AI":
            mic_map[key] = row_copy

    keys = sorted(set(foc_map.keys()) & set(mic_map.keys()))
    foc = [foc_map[k] for k in keys]
    mic = [mic_map[k] for k in keys]
    step_lim = None
    if max_n2_step_rpm is not None and float(max_n2_step_rpm) > 0.0:
        step_lim = float(max_n2_step_rpm)
    if step_lim is not None and len(keys) >= 3:
        n2_f = np.asarray([float(r.get("n2_rpm", float("nan"))) for r in foc], dtype=float)
        n2_m = np.asarray([float(r.get("n2_rpm", float("nan"))) for r in mic], dtype=float)
        dn = np.maximum(np.abs(np.diff(n2_f)), np.abs(np.diff(n2_m)))
        bad = np.where(dn > step_lim)[0]
        if bad.size:
            cut = int(max(2, bad[0] + 1))
            foc = foc[:cut]
            mic = mic[:cut]
    return foc, mic


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
    parser.add_argument("--load-factors", default=None, help="Explicit load factors (M/Mnom), comma-separated.")
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
    parser.add_argument("--foc-kp-speed", type=float, default=None, help="Override FOC speed-loop Kp for all comparisons.")
    parser.add_argument("--foc-ki-speed", type=float, default=None, help="Override FOC speed-loop Ki for all comparisons.")
    parser.add_argument("--ai-id-relative", action="store_true", help="Use relative id_ref around base when ai_id_ref.")
    parser.add_argument("--delta-id-max", type=float, default=0.3, help="Relative id_ref delta scale.")
    parser.add_argument("--speed-tol", type=float, default=0.05, help="Relative speed tolerance for valid points.")
    parser.add_argument("--speed-tol-abs", type=float, default=None, help="Absolute speed tolerance for valid points, rad/s.")
    parser.add_argument("--plot-air56-journal", action="store_true", help="Build split FOC/MIC journal figure with multi-axes.")
    parser.add_argument("--journal-common-p2-kw", type=float, default=0.25, help="Common P2 marker (kW) for eta projection.")
    parser.add_argument("--journal-out-base", default=None, help="Output base path for journal figure (without extension).")
    parser.add_argument("--journal-drop-zero-load", action="store_true", help="Drop load_factor=0 points for publication chart.")
    parser.add_argument(
        "--journal-max-speed-err-rel",
        type=float,
        default=0.05,
        help="Use only stable points for journal figure: keep points with speed_err_rel <= this value (<=0 disables).",
    )
    parser.add_argument(
        "--journal-max-n2-step-rpm",
        type=float,
        default=25.0,
        help="For publication chart, clip tail after first abrupt n2 jump larger than this threshold (<=0 disables).",
    )
    parser.add_argument("--cosphi-calibrate-nameplate", action="store_true", help="Scale cosphi so nominal FOC matches nameplate cos_phi_n.")
    parser.add_argument("--export-abc-traces", action="store_true", help="Export per-load traces with v_abc/i_abc to CSV.")
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
        env_cfg = _replace_cfg_keep_extras(env_cfg, foc=replace(env_cfg.foc, iq_limit=float(args.i_max)))
    if args.foc_kp_speed is not None or args.foc_ki_speed is not None:
        foc_now = env_cfg.foc
        kp_speed = float(args.foc_kp_speed) if args.foc_kp_speed is not None else float(getattr(foc_now, "kp_speed"))
        ki_speed = float(args.foc_ki_speed) if args.foc_ki_speed is not None else float(getattr(foc_now, "ki_speed"))
        env_cfg = _replace_cfg_keep_extras(env_cfg, foc=replace(foc_now, kp_speed=kp_speed, ki_speed=ki_speed))
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
    env_cfg = _replace_cfg_keep_extras(env_cfg, sim=replace(env_cfg.sim, dt=dt, t_end=t_end))

    omega_nom = _omega_nominal(env_cfg, args.omega_nom_source)
    m_nom = float(args.m_nom) if args.m_nom is not None else _rated_torque(env_cfg)
    nameplate = _extract_nameplate(env_cfg)
    cos_phi_nom = None
    if nameplate is not None and "cos_phi_n" in nameplate:
        try:
            cos_phi_nom = float(nameplate.get("cos_phi_n"))
        except Exception:
            cos_phi_nom = None
    omega_ref = float(args.omega_ref_pu) * omega_nom

    load_values = None
    load_factors = None
    if args.load_factors is not None:
        load_factors = [float(x) for x in str(args.load_factors).split(",") if str(x).strip()]
        if not load_factors:
            raise ValueError("--load-factors provided but parsed list is empty")
        load_values = [float(lf) * float(m_nom) for lf in load_factors]
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
    traces_dir = out_dir / "load_traces"
    if bool(args.export_abc_traces):
        traces_dir.mkdir(parents=True, exist_ok=True)
    for load in loads:
        foc_vals = _simulate_foc_case(env_cfg, omega_ref, float(load), dt, t_end)
        lf = float(load / max(m_nom, 1e-9))
        foc_stats = dict(_summarize_window(foc_vals, args.window_frac, debug_name=f"FOC|lf={lf:.3f}"))
        foc_stats["load_factor"] = float(load / max(m_nom, 1e-9))
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
            mic_vals = _simulate_foc_case(_replace_cfg_keep_extras(env_cfg, foc=foc_mic), omega_ref, float(load), dt, t_end)
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
        mic_stats = dict(_summarize_window(mic_vals, args.window_frac, debug_name=f"MIC|lf={lf:.3f}"))
        mic_stats["load_factor"] = float(load / max(m_nom, 1e-9))
        if bool(args.export_abc_traces):
            tag = f"lf_{lf:.3f}".replace(".", "p")
            _save_trace_csv(
                traces_dir / f"{tag}_foc.csv",
                foc_vals,
                omega_ref=omega_ref,
                load_torque=float(load),
                policy="FOC",
            )
            _save_trace_csv(
                traces_dir / f"{tag}_mic.csv",
                mic_vals,
                omega_ref=omega_ref,
                load_torque=float(load),
                policy="MIC_AI",
            )
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
    _sanity_check_curves("FOC", foc_load_stats)
    _sanity_check_curves("MIC_AI", mic_load_stats)

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
            foc_stats = _summarize_window(
                foc_vals,
                args.window_frac,
                debug_name=f"FOC|work|lf={float(load / max(m_nom, 1e-9)):.3f}|wref={float(speed):.2f}",
            )
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
                mic_vals = _simulate_foc_case(_replace_cfg_keep_extras(env_cfg, foc=foc_mic), float(speed), float(load), dt, t_end)
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
            mic_stats = _summarize_window(
                mic_vals,
                args.window_frac,
                debug_name=f"MIC|work|lf={float(load / max(m_nom, 1e-9)):.3f}|wref={float(speed):.2f}",
            )
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

    if bool(args.plot_air56_journal):
        foc_j, mic_j = _select_journal_rows(
            load_rows=load_rows,
            drop_zero_load=bool(args.journal_drop_zero_load),
            max_speed_err_rel=float(args.journal_max_speed_err_rel),
            max_n2_step_rpm=float(args.journal_max_n2_step_rpm),
        )
        if bool(args.cosphi_calibrate_nameplate):
            _calibrate_cosphi_to_nameplate(foc_j, mic_j, cos_phi_nom=cos_phi_nom)
        if len(foc_j) >= 2 and len(mic_j) >= 2:
            out_base = Path(args.journal_out_base) if args.journal_out_base else (out_dir / "fig_air56_mech_journal")
            _plot_air56_mech_journal(
                out_path=out_base,
                foc=foc_j,
                mic=mic_j,
                common_p2_kw=float(args.journal_common_p2_kw),
            )

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
        "foc_kp_speed": None if args.foc_kp_speed is None else float(args.foc_kp_speed),
        "foc_ki_speed": None if args.foc_ki_speed is None else float(args.foc_ki_speed),
        "speed_tol_rel": float(args.speed_tol),
        "speed_tol_abs": None if args.speed_tol_abs is None else float(args.speed_tol_abs),
        "ai_id_relative": bool(args.ai_id_relative),
        "delta_id_max": float(args.delta_id_max),
        "export_abc_traces": bool(args.export_abc_traces),
        "plot_style": "vak_ru",
        "plot_formats": ["png", "pdf", "svg"],
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"Saved outputs to {out_dir}")


if __name__ == "__main__":
    main()
