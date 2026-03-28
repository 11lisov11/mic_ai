from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import copy
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mic_ai.ai.agents.ppo_voltage import PPOVoltageAgent
from mic_ai.ai.ai_env import AiEnvConfig, MicAiAIEnv
from mic_ai.ai.id_ref_supervisor import AiIdRefSupervisor, AiIdRefSupervisorConfig
from mic_ai.ai.train_ai_id_ref import FEATURE_KEYS as ID_FEATURE_KEYS, build_feature_keys
from mic_ai.analysis.metrics import calc_i_rms, calc_p_el, calc_p_mech
from mic_ai.core.env import make_env_from_config
from mic_ai.tools.checkpoint_adaptation import adapt_checkpoint_state_dict_for_model
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


def _infer_action_dim(state: Dict[str, torch.Tensor]) -> int:
    for key in ("actor_mu.weight", "actor_head.weight", "log_std", "actor_mu.bias", "actor_head.bias"):
        value = state.get(key)
        if value is None:
            continue
        try:
            if getattr(value, "ndim", 0) >= 2:
                dim = int(value.shape[0])
            elif getattr(value, "ndim", 0) == 1:
                dim = int(value.shape[0])
            else:
                continue
        except Exception:
            continue
        if dim > 0:
            return dim
    return 1


def _parse_feature_keys_arg(feature_keys: object | None) -> List[str]:
    if feature_keys is None:
        return []
    if isinstance(feature_keys, str):
        return [s.strip() for s in feature_keys.split(",") if s.strip()]
    if isinstance(feature_keys, (list, tuple)):
        out: List[str] = []
        for item in feature_keys:
            text = str(item).strip()
            if text:
                out.append(text)
        return out
    return []


def _resolve_feature_keys(feature_keys: object | None, state: Dict[str, torch.Tensor] | None = None) -> List[str]:
    """
    Resolve observation feature keys for checkpoint loading.

    Compatibility note:
    `tools/step27_pipeline.py` imports this helper. Keep it stable and tolerant
    to older checkpoints trained with a reduced feature set.
    """
    explicit = _parse_feature_keys_arg(feature_keys)
    if explicit:
        return explicit

    defaults = list(ID_FEATURE_KEYS)
    defaults_with_episode_eta = list(build_feature_keys(include_energy_obs=True, include_episode_eta_obs=True))
    if not defaults:
        return []
    energy_obs = {"p_in_norm", "p_el_filt", "p_shaft_norm", "eta_norm"}
    base = [k for k in defaults if k not in energy_obs]

    if state is None:
        return defaults
    w0 = state.get("actor_body.0.weight")
    if w0 is None or len(getattr(w0, "shape", ())) < 2:
        return defaults
    try:
        in_dim = int(w0.shape[1])
    except Exception:
        return defaults

    if in_dim == len(defaults_with_episode_eta):
        return defaults_with_episode_eta
    if in_dim == len(defaults):
        return defaults
    if in_dim == len(base):
        return base
    if 0 < in_dim < len(base):
        return list(base[:in_dim])
    if len(base) < in_dim < len(defaults):
        energy_count = max(0, in_dim - len(base))
        energy_keys = [k for k in defaults if k in energy_obs]
        return list(base + energy_keys[:energy_count])
    if len(defaults) < in_dim < len(defaults_with_episode_eta):
        extra = defaults_with_episode_eta[len(defaults) : in_dim]
        return list(defaults + extra)
    return defaults


def _sanitize_name(name: str) -> str:
    return str(name).replace(":", "_").replace("/", "_").replace(".", "p")


def _steady_slice(n: int, window_frac: float) -> slice:
    if n <= 0:
        return slice(0, 0)
    window_frac = float(max(min(window_frac, 0.95), 0.05))
    start = int(max(0, n * (1.0 - window_frac)))
    return slice(start, n)


def _save_csv(path: Path, series: Dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = ["t", "omega", "omega_ref", "i_rms", "p_el", "p_mech"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for k in range(int(series["t"].size)):
            writer.writerow(
                [
                    float(series["t"][k]),
                    float(series["omega"][k]),
                    float(series["omega_ref"][k]),
                    float(series["i_rms"][k]),
                    float(series["p_el"][k]),
                    float(series["p_mech"][k]),
                ]
            )


def _plot_power(out_path: Path, foc: Dict[str, np.ndarray], mic: Dict[str, np.ndarray], clip_negative: bool) -> None:
    plt = apply_vak_style(ensure_matplotlib())
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    p_foc = np.maximum(foc["p_el"], 0.0) if clip_negative else foc["p_el"]
    p_mic = np.maximum(mic["p_el"], 0.0) if clip_negative else mic["p_el"]
    ax.plot(foc["t"], p_foc, color="black", label="FOC")
    ax.plot(mic["t"], p_mic, color="0.35", linestyle="--", label="MIC AI")
    ax.set_xlabel("t, s")
    ax.set_ylabel("P_el, W" if not clip_negative else "P_el^+, W")
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)


def _plot_speed_error(out_path: Path, foc: Dict[str, np.ndarray], mic: Dict[str, np.ndarray]) -> None:
    plt = apply_vak_style(ensure_matplotlib())
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    err_foc = np.abs(foc["omega_ref"] - foc["omega"])
    err_mic = np.abs(mic["omega_ref"] - mic["omega"])
    ax.plot(foc["t"], err_foc, color="black", label="FOC")
    ax.plot(mic["t"], err_mic, color="0.35", linestyle="--", label="MIC AI")
    ax.set_xlabel("t, s")
    ax.set_ylabel("|omega_ref - omega|, rad/s")
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)


def _err_limit(foc_err: float, rel_tol: float, abs_tol: float) -> float:
    base = float(foc_err) * (1.0 + float(rel_tol))
    return max(base, float(abs_tol))


def _summarize(series: Dict[str, np.ndarray], window_frac: float) -> Dict[str, float]:
    n = int(series["t"].size)
    sl = _steady_slice(n, window_frac)
    p_el = series["p_el"][sl]
    p_mech = series["p_mech"][sl]
    i_rms = series["i_rms"][sl]
    err = np.abs(series["omega_ref"][sl] - series["omega"][sl])
    p_el_mean = float(np.mean(p_el)) if p_el.size else 0.0
    p_el_pos_mean = float(np.mean(np.maximum(p_el, 0.0))) if p_el.size else 0.0
    p_mech_mean = float(np.mean(p_mech)) if p_mech.size else 0.0
    eta = float(p_mech_mean / p_el_mean) if p_el_mean > 1e-9 else 0.0
    return {
        "omega_ss": float(np.mean(series["omega"][sl])) if n else 0.0,
        "mean_abs_speed_err": float(np.mean(err)) if err.size else 0.0,
        "mean_p_el": p_el_mean,
        "mean_p_el_pos": p_el_pos_mean,
        "p_mech": p_mech_mean,
        "eta": eta,
        "mean_i_rms": float(np.mean(i_rms)) if i_rms.size else 0.0,
        "peak_i_rms": float(np.max(i_rms)) if i_rms.size else 0.0,
    }


def _simulate_controller(
    env_cfg: object,
    dt: float,
    t_end: float,
    mode: str,
    use_total_power: bool,
    hybrid_opts: Dict[str, float] | None = None,
) -> Dict[str, np.ndarray]:
    sim_cfg = replace(env_cfg.sim, dt=dt, t_end=t_end, mode=str(mode).lower())
    env_cfg_s = _clone_with_sim(env_cfg, sim_cfg)
    if str(mode).lower() == "hybrid" and hybrid_opts:
        for key, value in hybrid_opts.items():
            _set_attr(env_cfg_s, key, float(value))
    env = InductionMotorEnv(env_cfg_s)
    env.reset()
    steps = int(max(t_end / dt, 1))
    t = np.zeros(steps, dtype=float)
    omega = np.zeros(steps, dtype=float)
    omega_ref = np.zeros(steps, dtype=float)
    i_rms = np.zeros(steps, dtype=float)
    p_el = np.zeros(steps, dtype=float)
    p_mech = np.zeros(steps, dtype=float)

    for k in range(steps):
        obs, _r, done, info = env.step(None)
        t[k] = float(env.t)
        omega[k] = float(obs[0])
        omega_ref[k] = float(obs[1])
        i_abc = np.asarray(info.get("i_abc", (0.0, 0.0, 0.0)), dtype=float)
        v_abc = np.asarray(info.get("v_abc", (0.0, 0.0, 0.0)), dtype=float)
        torque = float(info.get("torque_e", obs[2]))
        i_rms[k] = calc_i_rms(i_abc)
        p_el_val = calc_p_el(v_abc, i_abc)
        if use_total_power:
            p_el_val = float(info.get("p_in_total", p_el_val))
        p_el[k] = p_el_val
        p_mech[k] = calc_p_mech(omega[k], torque)
        if done:
            t = t[: k + 1]
            omega = omega[: k + 1]
            omega_ref = omega_ref[: k + 1]
            i_rms = i_rms[: k + 1]
            p_el = p_el[: k + 1]
            p_mech = p_mech[: k + 1]
            break
    return {
        "t": t,
        "omega": omega,
        "omega_ref": omega_ref,
        "i_rms": i_rms,
        "p_el": p_el,
        "p_mech": p_mech,
    }


def _simulate_foc(env_cfg: object, dt: float, t_end: float, use_total_power: bool) -> Dict[str, np.ndarray]:
    return _simulate_controller(env_cfg, dt, t_end, mode="foc", use_total_power=use_total_power)


def _simulate_v3(env_cfg: object, dt: float, t_end: float, use_total_power: bool) -> Dict[str, np.ndarray]:
    return _simulate_controller(env_cfg, dt, t_end, mode="v3", use_total_power=use_total_power)


def _clone_env_cfg(env_cfg: object) -> object:
    try:
        return copy.copy(env_cfg)
    except Exception:
        return env_cfg


def _set_attr(obj: object, name: str, value: object) -> bool:
    try:
        object.__setattr__(obj, name, value)
        return True
    except Exception:
        try:
            setattr(obj, name, value)
            return True
        except Exception:
            return False


def _clone_with_sim(env_cfg: object, sim_cfg: object) -> object:
    env_cfg_s = _clone_env_cfg(env_cfg)
    if not _set_attr(env_cfg_s, "sim", sim_cfg):
        env_cfg_s = replace(env_cfg, sim=sim_cfg)
    return env_cfg_s


def _clone_with_foc(env_cfg: object, foc_cfg: object) -> object:
    env_cfg_s = _clone_env_cfg(env_cfg)
    if not _set_attr(env_cfg_s, "foc", foc_cfg):
        env_cfg_s = replace(env_cfg, foc=foc_cfg)
    return env_cfg_s


def _build_ai_env(
    env_cfg: object,
    dt: float,
    t_end: float,
    ai_control_mode: str,
    id_ref_alpha: float,
    id_ref_rate_limit: float | None,
    id_ref_gate_speed_tol: float | None,
    id_ref_gate_speed_tol_rel: float | None,
    id_ref_gate_min_scale: float,
    id_ref_gate_exponent: float,
    ai_id_relative: bool,
    delta_id_max: float,
) -> MicAiAIEnv:
    i_base = float(getattr(env_cfg.motor, "I_n", 1.0))
    iq_limit = float(getattr(getattr(env_cfg, "foc", None), "iq_limit", i_base * 8.0))
    i_limit = max(iq_limit, i_base)
    control_mode = str(ai_control_mode).lower().strip()
    if control_mode in {"ai_current", "ai_speed"}:
        i_base = float(i_limit)
    id_ref_base = float(getattr(getattr(env_cfg, "foc", None), "id_ref", 0.0) or 0.0)
    # Allow a wider id_ref range for small motors where id_ref_base can be > I_n
    # (dq vs line-current scaling). This avoids artificially capping MIC's flux search.
    id_ref_max = max(i_base * 1.5, id_ref_base, id_ref_base * 1.6)
    omega_ref_nom = float(2.0 * math.pi * 10.0 / max(env_cfg.motor.p, 1))
    steps = int(max(t_end / dt, 1))
    ai_cfg = AiEnvConfig(
        episode_steps=steps,
        dt=dt,
        omega_ref=omega_ref_nom,
        omega_ref_max=max(abs(omega_ref_nom) * 1.2, 1e-3),
        w_speed_error=0.0,
        w_current_rms=0.0,
        control_mode=control_mode,
        i_base=i_base,
        i_max=i_limit,
        # NOTE: For benchmarking, we must keep FOC and MIC horizons identical.
        # With the current Clarke/Park scaling and separate (id_ref, iq_limit),
        # the phase current can transiently be far above iq_limit during start/stop
        # even for the FOC baseline. A too-tight hard limit would terminate MIC
        # rollouts early and bias metrics. Here we effectively disable hard current
        # termination by using a very large margin; current peaks should be tracked
        # separately as a metric.
        i_hard_limit=float(i_limit * 50.0),
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
        id_ref_max=float(id_ref_max),
        curriculum_omega_pu=(1.0,),
        curriculum_stage_episodes=(),
        omega_piecewise_steps=(),
        omega_piecewise_multipliers=(1.0,),
        override_load_torque=False,
        override_omega_ref=False,
        drift_every_episodes=0,
        enable_id_control=bool(control_mode in {"ai_current", "foc_assist", "ai_speed"}),
    )
    base_env = InductionMotorEnv(env_cfg)
    return MicAiAIEnv(base_env, ai_cfg, curiosity=None, world_model=None, world_input_keys=ID_FEATURE_KEYS, world_target_keys=["omega_norm"])


def _simulate_ai(
    agent: PPOVoltageAgent,
    env_cfg: object,
    dt: float,
    t_end: float,
    ai_control_mode: str,
    id_ref_alpha: float,
    id_ref_rate_limit: float | None,
    id_ref_gate_speed_tol: float | None,
    id_ref_gate_speed_tol_rel: float | None,
    id_ref_gate_min_scale: float,
    id_ref_gate_exponent: float,
    ai_id_relative: bool,
    delta_id_max: float,
    use_total_power: bool,
    supervisor_cfg: AiIdRefSupervisorConfig | None = None,
    ai_id_allow_positive_delta: bool = True,
) -> Dict[str, np.ndarray]:
    env = _build_ai_env(
        env_cfg,
        dt,
        t_end,
        ai_control_mode,
        id_ref_alpha,
        id_ref_rate_limit,
        id_ref_gate_speed_tol,
        id_ref_gate_speed_tol_rel,
        id_ref_gate_min_scale,
        id_ref_gate_exponent,
        ai_id_relative,
        delta_id_max,
    )
    obs = env.reset()

    omega_nom = float(2.0 * math.pi * env_cfg.scalar_vf.f_max / max(env_cfg.motor.p, 1))
    supervisor: AiIdRefSupervisor | None = None
    if supervisor_cfg is not None and bool(supervisor_cfg.enabled) and str(ai_control_mode).lower().strip() == "ai_id_ref":
        supervisor = AiIdRefSupervisor(supervisor_cfg, omega_nominal=omega_nom)
        supervisor.reset()

    steps = int(max(t_end / dt, 1))
    t = np.zeros(steps, dtype=float)
    omega = np.zeros(steps, dtype=float)
    omega_ref = np.zeros(steps, dtype=float)
    i_rms = np.zeros(steps, dtype=float)
    p_el = np.zeros(steps, dtype=float)
    p_mech = np.zeros(steps, dtype=float)

    for k in range(steps):
        # Deterministic policy rollout for reproducible benchmarking.
        with torch.no_grad():
            state_t = agent._to_tensor(obs).unsqueeze(0)
            mu, _std, _value = agent.net(state_t)
        action = torch.clamp(mu, -1.0, 1.0).squeeze(0).cpu().numpy().astype(np.float32)
        if (
            (not bool(ai_id_allow_positive_delta))
            and str(ai_control_mode).lower().strip() == "ai_id_ref"
            and bool(ai_id_relative)
            and action.size >= 1
        ):
            action[0] = np.float32(min(float(action[0]), 0.0))
        gate_open = False
        if supervisor is not None:
            omega_obs = float(obs.get("omega", 0.0))
            omega_ref_obs = float(obs.get("omega_ref", 0.0))
            action0, gate_open = supervisor.adjust_action(float(action[0]), omega_ref=omega_ref_obs, omega=float(omega_obs))
            action[0] = np.float32(action0)
        obs, _r, done, info = env.step(action)
        t[k] = float(getattr(env.base_env, "t", k * dt))
        omega[k] = float(info.get("omega_meas", obs.get("omega", 0.0)))
        omega_ref[k] = float(info.get("omega_ref", obs.get("omega_ref", 0.0)))
        i_abc = np.asarray(info.get("i_abc", (0.0, 0.0, 0.0)), dtype=float)
        v_abc = np.asarray(info.get("v_abc", (0.0, 0.0, 0.0)), dtype=float)
        torque = float(info.get("torque_e", getattr(env.base_env, "last_torque", 0.0)))
        i_rms[k] = calc_i_rms(i_abc)
        p_el_val = calc_p_el(v_abc, i_abc)
        if use_total_power:
            p_el_val = float(info.get("p_in_total", p_el_val))
        p_el[k] = p_el_val
        p_mech[k] = calc_p_mech(omega[k], torque)
        if supervisor is not None:
            supervisor.update(
                p_in_pos=max(0.0, float(p_el_val)),
                p_shaft_pos=max(0.0, float(p_mech[k])),
                gate_open=gate_open,
            )
        if done:
            t = t[: k + 1]
            omega = omega[: k + 1]
            omega_ref = omega_ref[: k + 1]
            i_rms = i_rms[: k + 1]
            p_el = p_el[: k + 1]
            p_mech = p_mech[: k + 1]
            break
    return {
        "t": t,
        "omega": omega,
        "omega_ref": omega_ref,
        "i_rms": i_rms,
        "p_el": p_el,
        "p_mech": p_mech,
    }


def _simulate_mic_rule(
    env_cfg: object,
    dt: float,
    t_end: float,
    id_ref_low: float,
    id_ref_high: float,
    speed_tol_rel: float,
    omega_min_pu: float,
    use_total_power: bool,
) -> Dict[str, np.ndarray]:
    env = InductionMotorEnv(env_cfg)
    env.reset()
    omega_nom = 2.0 * math.pi * env_cfg.scalar_vf.f_max / env_cfg.motor.p
    steps = int(max(t_end / dt, 1))
    t = np.zeros(steps, dtype=float)
    omega = np.zeros(steps, dtype=float)
    omega_ref = np.zeros(steps, dtype=float)
    i_rms = np.zeros(steps, dtype=float)
    p_el = np.zeros(steps, dtype=float)
    p_mech = np.zeros(steps, dtype=float)

    for k in range(steps):
        t_now = float(env.t)
        omega_ref_k = float(env.omega_ref_func(t_now))
        omega_meas = float(getattr(getattr(env.motor, "state", None), "omega_m", 0.0))
        omega_ref_scale = max(abs(omega_ref_k), 1e-6)
        err = abs(omega_ref_k - omega_meas)
        id_ref_target = float(id_ref_high)
        if abs(omega_ref_k) >= float(omega_min_pu) * omega_nom and err <= float(speed_tol_rel) * omega_ref_scale:
            id_ref_target = float(id_ref_low)
        env.controller.params = replace(env.controller.params, id_ref=id_ref_target)

        obs, _r, done, info = env.step(None)
        t[k] = float(env.t)
        omega[k] = float(obs[0])
        omega_ref[k] = float(obs[1])
        i_abc = np.asarray(info.get("i_abc", (0.0, 0.0, 0.0)), dtype=float)
        v_abc = np.asarray(info.get("v_abc", (0.0, 0.0, 0.0)), dtype=float)
        torque = float(info.get("torque_e", obs[2]))
        i_rms[k] = calc_i_rms(i_abc)
        p_el_val = calc_p_el(v_abc, i_abc)
        if use_total_power:
            p_el_val = float(info.get("p_in_total", p_el_val))
        p_el[k] = p_el_val
        p_mech[k] = calc_p_mech(omega[k], torque)
        if done:
            t = t[: k + 1]
            omega = omega[: k + 1]
            omega_ref = omega_ref[: k + 1]
            i_rms = i_rms[: k + 1]
            p_el = p_el[: k + 1]
            p_mech = p_mech[: k + 1]
            break
    return {
        "t": t,
        "omega": omega,
        "omega_ref": omega_ref,
        "i_rms": i_rms,
        "p_el": p_el,
        "p_mech": p_mech,
    }


def _simulate_mic_search(
    env_cfg: object,
    dt: float,
    t_end: float,
    id_ref_step: float,
    id_ref_min: float,
    id_ref_max: float | None,
    update_every: int,
    ema_tau: float,
    speed_tol_rel: float,
    omega_min_pu: float,
    use_total_power: bool,
) -> Dict[str, np.ndarray]:
    env = InductionMotorEnv(env_cfg)
    env.reset()
    omega_nom = 2.0 * math.pi * env_cfg.scalar_vf.f_max / env_cfg.motor.p
    i_base = float(getattr(env_cfg.motor, "I_n", 1.0))
    id_ref_max_val = float(id_ref_max) if id_ref_max is not None else max(i_base * 1.5, id_ref_min + id_ref_step)
    id_ref = float(getattr(getattr(env_cfg, "foc", None), "id_ref", 0.0) or 0.0)
    id_ref = float(np.clip(id_ref, id_ref_min, id_ref_max_val))
    direction = -1.0
    p_filt = 0.0
    p_prev = None

    steps = int(max(t_end / dt, 1))
    t = np.zeros(steps, dtype=float)
    omega = np.zeros(steps, dtype=float)
    omega_ref = np.zeros(steps, dtype=float)
    i_rms = np.zeros(steps, dtype=float)
    p_el = np.zeros(steps, dtype=float)
    p_mech = np.zeros(steps, dtype=float)

    for k in range(steps):
        t_now = float(env.t)
        omega_ref_k = float(env.omega_ref_func(t_now))
        omega_meas = float(getattr(getattr(env.motor, "state", None), "omega_m", 0.0))
        omega_ref_scale = max(abs(omega_ref_k), 1e-6)
        err = abs(omega_ref_k - omega_meas)
        if abs(omega_ref_k) >= float(omega_min_pu) * omega_nom and err <= float(speed_tol_rel) * omega_ref_scale:
            if update_every > 0 and (k % max(update_every, 1) == 0):
                if p_prev is not None and p_filt > p_prev:
                    direction *= -1.0
                id_ref = float(np.clip(id_ref + direction * float(id_ref_step), id_ref_min, id_ref_max_val))
                p_prev = p_filt
        env.controller.params = replace(env.controller.params, id_ref=float(id_ref))

        obs, _r, done, info = env.step(None)
        t[k] = float(env.t)
        omega[k] = float(obs[0])
        omega_ref[k] = float(obs[1])
        i_abc = np.asarray(info.get("i_abc", (0.0, 0.0, 0.0)), dtype=float)
        v_abc = np.asarray(info.get("v_abc", (0.0, 0.0, 0.0)), dtype=float)
        torque = float(info.get("torque_e", obs[2]))
        i_rms[k] = calc_i_rms(i_abc)
        p_el_val = calc_p_el(v_abc, i_abc)
        if use_total_power:
            p_el_val = float(info.get("p_in_total", p_el_val))
        p_el[k] = p_el_val
        p_mech[k] = calc_p_mech(omega[k], torque)

        p_in_pos = max(0.0, float(p_el_val))
        if ema_tau > 0.0:
            alpha = math.exp(-dt / max(ema_tau, 1e-6))
            p_filt = alpha * p_filt + (1.0 - alpha) * p_in_pos
        else:
            p_filt = p_in_pos

        if done:
            t = t[: k + 1]
            omega = omega[: k + 1]
            omega_ref = omega_ref[: k + 1]
            i_rms = i_rms[: k + 1]
            p_el = p_el[: k + 1]
            p_mech = p_mech[: k + 1]
            break
    return {
        "t": t,
        "omega": omega,
        "omega_ref": omega_ref,
        "i_rms": i_rms,
        "p_el": p_el,
        "p_mech": p_mech,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare FOC vs MIC AI across scenarios.")
    parser.add_argument("--env-config", default="config/env_demo_true_motor1.py")
    parser.add_argument("--ai-checkpoint", default=None)
    parser.add_argument("--mic-id-ref", type=float, default=None, help="Use fixed id_ref for MIC curve.")
    parser.add_argument("--mic-id-ref-low", type=float, default=None, help="Low id_ref for MIC rule.")
    parser.add_argument("--mic-id-ref-high", type=float, default=None, help="High id_ref for MIC rule.")
    parser.add_argument("--mic-id-ref-search", action="store_true", help="Enable online id_ref search (hill-climb).")
    parser.add_argument("--mic-id-ref-step", type=float, default=0.02)
    parser.add_argument("--mic-id-ref-min", type=float, default=0.0)
    parser.add_argument("--mic-id-ref-max", type=float, default=None)
    parser.add_argument("--mic-id-ref-update", type=int, default=10, help="Steps between id_ref updates.")
    parser.add_argument("--mic-id-ref-ema-tau", type=float, default=0.02, help="EMA time constant for power filtering.")
    parser.add_argument("--mic-id-ref-speed-tol-rel", type=float, default=0.05, help="Speed error tol (rel).")
    parser.add_argument("--mic-id-ref-omega-min", type=float, default=0.1, help="Min omega_ref pu for low id_ref.")
    parser.add_argument("--scenarios", default="speed_step,ramp,load_step,start_stop")
    parser.add_argument("--t-end", type=float, default=None)
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--load-torque", type=float, default=None, help="Override constant load torque, N*m.")
    parser.add_argument("--window-frac", type=float, default=0.25)
    parser.add_argument("--error-tol-rel", type=float, default=0.0, help="Allowed error increase vs FOC.")
    parser.add_argument("--error-tol-abs", type=float, default=0.0, help="Absolute speed error tolerance (rad/s).")
    parser.add_argument(
        "--mic-controller-mode",
        default=None,
        choices=["foc", "v3", "hybrid"],
        help="Use built-in controller mode as MIC curve instead of AI/rule/search/fixed id_ref.",
    )
    parser.add_argument("--mic-hybrid-load-ratio", type=float, default=0.6, help="Hybrid: low-load ratio threshold.")
    parser.add_argument("--mic-hybrid-err-tol-rel", type=float, default=0.02, help="Hybrid: speed error tolerance (relative).")
    parser.add_argument("--mic-hybrid-err-tol-abs", type=float, default=0.0, help="Hybrid: speed error tolerance (absolute).")
    parser.add_argument("--mic-hybrid-min-omega-pu", type=float, default=0.1, help="Hybrid: minimum speed, pu.")
    parser.add_argument("--include-v3", action="store_true", help="Also simulate V3 controller.")
    parser.add_argument("--use-total-power", action="store_true", help="Use p_in_total if available.")
    parser.add_argument("--foc-disable-lut", action="store_true", help="Disable id_ref LUT for FOC baseline.")
    parser.add_argument("--ai-control-mode", type=str, default="ai_id_ref", choices=["ai_id_ref", "ai_current", "ai_voltage", "foc_assist", "ai_speed"])
    parser.add_argument("--ai-id-relative", action="store_true", help="Use relative id_ref around base.")
    parser.add_argument("--delta-id-max", type=float, default=0.1)
    parser.add_argument("--ai-feature-keys", default=None, help="Comma-separated feature keys for AI checkpoint.")
    parser.add_argument("--id-ref-alpha", type=float, default=1.0)
    parser.add_argument("--id-ref-rate-limit", type=float, default=None)
    parser.add_argument("--id-ref-gate-speed-tol", type=float, default=None)
    parser.add_argument("--id-ref-gate-speed-tol-rel", type=float, default=0.05)
    parser.add_argument("--id-ref-gate-min-scale", type=float, default=0.0)
    parser.add_argument("--id-ref-gate-exponent", type=float, default=1.0)
    parser.add_argument("--ai-supervisor", action="store_true", help="Enable online extremum supervisor over AI action.")
    parser.add_argument(
        "--ai-sup-objective",
        default="specific_power",
        choices=["specific_power", "p_in", "eta_inv"],
        help="Supervisor objective.",
    )
    parser.add_argument("--ai-sup-update", type=int, default=20, help="Supervisor update window, steps.")
    parser.add_argument("--ai-sup-dither", type=float, default=0.04, help="Supervisor dither amplitude (action units).")
    parser.add_argument("--ai-sup-step", type=float, default=0.01, help="Bias update step (action units).")
    parser.add_argument("--ai-sup-bias-max", type=float, default=0.25, help="Max absolute supervisor bias.")
    parser.add_argument("--ai-sup-speed-tol-rel", type=float, default=0.05, help="Speed gate tolerance (relative).")
    parser.add_argument("--ai-sup-speed-tol-abs", type=float, default=0.0, help="Speed gate tolerance (absolute, rad/s).")
    parser.add_argument("--ai-sup-omega-min", type=float, default=0.1, help="Min omega_ref, pu, for supervisor activity.")
    parser.add_argument("--ai-sup-shaft-eps", type=float, default=10.0, help="Shaft power floor for specific-power objective.")
    parser.add_argument("--ai-sup-reset-decay", type=float, default=0.98, help="Bias decay on transient gate close.")
    parser.add_argument("--ai-sup-idle-enable", action="store_true", help="Enable idle demagnetization at low speed reference.")
    parser.add_argument("--ai-sup-idle-omega-min", type=float, default=0.05, help="Idle threshold in pu of nominal omega_ref.")
    parser.add_argument("--ai-sup-idle-action", type=float, default=-1.0, help="Action clamp used in idle mode.")
    parser.add_argument("--ai-sup-idle-exit-boost", type=int, default=0, help="Steps to force high-flux action after idle exit.")
    parser.add_argument("--ai-sup-idle-exit-action", type=float, default=1.0, help="Action floor used during idle-exit boost.")
    parser.add_argument("--ai-sup-idle-bias-decay", type=float, default=0.95, help="Bias decay multiplier while idle mode active.")
    parser.add_argument("--plots", action="store_true")
    parser.add_argument("--clip-negative", action="store_true")
    parser.add_argument("--seed", type=int, default=None, help="Seed for reproducible AI sampling/eval.")
    parser.add_argument("--out-dir", default="outputs/scenario_compare")
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(int(args.seed))
        torch.manual_seed(int(args.seed))

    env_path = _resolve_config_path(args.env_config)
    env_cfg = make_env_from_config(str(env_path)).env_config
    dt = float(args.dt) if args.dt is not None else float(env_cfg.sim.dt)
    t_end = float(args.t_end) if args.t_end is not None else float(env_cfg.sim.t_end)

    mic_id_ref = None if args.mic_id_ref is None else float(args.mic_id_ref)
    mic_ctrl_mode = None if args.mic_controller_mode is None else str(args.mic_controller_mode).lower().strip()
    mic_rule = False
    mic_search = bool(args.mic_id_ref_search)
    mic_id_ref_low = None if args.mic_id_ref_low is None else float(args.mic_id_ref_low)
    mic_id_ref_high = None if args.mic_id_ref_high is None else float(args.mic_id_ref_high)
    if mic_id_ref_low is not None or mic_id_ref_high is not None:
        if mic_id_ref_low is None or mic_id_ref_high is None:
            raise ValueError("Provide both --mic-id-ref-low and --mic-id-ref-high.")
        mic_rule = True
    if mic_search and mic_rule:
        raise ValueError("Choose either mic-id-ref rule or mic-id-ref-search.")
    if mic_ctrl_mode is not None and (mic_id_ref is not None or mic_rule or mic_search):
        raise ValueError("mic-controller-mode cannot be combined with fixed/rule/search id_ref options.")
    hybrid_opts: Dict[str, float] | None = None
    if mic_ctrl_mode == "hybrid":
        hybrid_opts = {
            "hybrid_load_low_ratio": float(args.mic_hybrid_load_ratio),
            "hybrid_err_tol_rel": float(args.mic_hybrid_err_tol_rel),
            "hybrid_err_tol_abs": float(args.mic_hybrid_err_tol_abs),
            "hybrid_min_omega_pu": float(args.mic_hybrid_min_omega_pu),
        }
    agent = None
    if mic_ctrl_mode is None and mic_id_ref is None and not mic_rule and not mic_search:
        if args.ai_checkpoint is None:
            raise ValueError("Provide --ai-checkpoint or --mic-id-ref.")
        ckpt = Path(args.ai_checkpoint)
        state = torch.load(ckpt, map_location="cpu")
        hidden = _infer_hidden_sizes(state) or (128, 128)
        action_dim = _infer_action_dim(state)
        if str(args.ai_control_mode).lower().strip() in {"ai_current", "ai_voltage", "foc_assist", "ai_speed"}:
            action_dim = max(action_dim, 2)
        feature_keys = _resolve_feature_keys(args.ai_feature_keys, state)
        agent = PPOVoltageAgent(feature_keys=feature_keys, action_dim=action_dim, device="cpu", hidden_sizes=hidden)
        adapted_state, _ = adapt_checkpoint_state_dict_for_model(
            state,
            agent.net.state_dict(),
            target_control_mode=str(args.ai_control_mode).lower(),
        )
        agent.net.load_state_dict(adapted_state, strict=False)
        agent.set_action_std(1e-6)

    supervisor_cfg: AiIdRefSupervisorConfig | None = None
    if bool(args.ai_supervisor):
        supervisor_cfg = AiIdRefSupervisorConfig(
            enabled=True,
            speed_tol_rel=float(args.ai_sup_speed_tol_rel),
            speed_tol_abs=float(args.ai_sup_speed_tol_abs),
            omega_min_pu=float(args.ai_sup_omega_min),
            update_steps=int(args.ai_sup_update),
            dither_amp=float(args.ai_sup_dither),
            bias_step=float(args.ai_sup_step),
            bias_max=float(args.ai_sup_bias_max),
            objective=str(args.ai_sup_objective),
            shaft_eps=float(args.ai_sup_shaft_eps),
            reset_decay=float(args.ai_sup_reset_decay),
            idle_enable=bool(args.ai_sup_idle_enable),
            idle_omega_pu=float(args.ai_sup_idle_omega_min),
            idle_action=float(args.ai_sup_idle_action),
            idle_exit_boost_steps=int(args.ai_sup_idle_exit_boost),
            idle_exit_action=float(args.ai_sup_idle_exit_action),
            idle_bias_decay=float(args.ai_sup_idle_bias_decay),
        )

    scenario_list = [s.strip() for s in str(args.scenarios).split(",") if s.strip()]
    load_torque = float(args.load_torque) if args.load_torque is not None else float(env_cfg.sim.load_torque)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, float | str | bool]] = []
    summary_rows_v3: List[Dict[str, float | str | bool]] = []
    for scenario in scenario_list:
        file_tag = _sanitize_name(scenario)
        sim_cfg = replace(env_cfg.sim, scenario_name=str(scenario), dt=dt, t_end=t_end, load_torque=load_torque)
        env_cfg_s = _clone_with_sim(env_cfg, sim_cfg)

        use_total_power = bool(args.use_total_power)
        env_cfg_foc = env_cfg_s
        if bool(args.foc_disable_lut):
            env_cfg_foc = _clone_env_cfg(env_cfg_s)
            if getattr(env_cfg_foc, "id_ref_lut_path", None) is not None:
                _set_attr(env_cfg_foc, "id_ref_lut_path", None)
        foc = _simulate_foc(env_cfg_foc, dt, t_end, use_total_power)
        if mic_ctrl_mode is not None:
            mic = _simulate_controller(
                env_cfg_s,
                dt,
                t_end,
                mode=mic_ctrl_mode,
                use_total_power=use_total_power,
                hybrid_opts=hybrid_opts,
            )
        elif mic_rule:
            mic = _simulate_mic_rule(
                env_cfg_s,
                dt,
                t_end,
                mic_id_ref_low,
                mic_id_ref_high,
                float(args.mic_id_ref_speed_tol_rel),
                float(args.mic_id_ref_omega_min),
                use_total_power,
            )
        elif mic_search:
            mic = _simulate_mic_search(
                env_cfg_s,
                dt,
                t_end,
                id_ref_step=float(args.mic_id_ref_step),
                id_ref_min=float(args.mic_id_ref_min),
                id_ref_max=None if args.mic_id_ref_max is None else float(args.mic_id_ref_max),
                update_every=int(args.mic_id_ref_update),
                ema_tau=float(args.mic_id_ref_ema_tau),
                speed_tol_rel=float(args.mic_id_ref_speed_tol_rel),
                omega_min_pu=float(args.mic_id_ref_omega_min),
                use_total_power=use_total_power,
            )
        elif mic_id_ref is not None:
            foc_mic = replace(env_cfg_s.foc, id_ref=mic_id_ref)
            mic_cfg = _clone_with_foc(env_cfg_s, foc_mic)
            mic = _simulate_foc(mic_cfg, dt, t_end, use_total_power)
        else:
            mic = _simulate_ai(
                agent,
                env_cfg_s,
                dt,
                t_end,
                str(args.ai_control_mode),
                float(args.id_ref_alpha),
                None if args.id_ref_rate_limit is None else float(args.id_ref_rate_limit),
                None if args.id_ref_gate_speed_tol is None else float(args.id_ref_gate_speed_tol),
                None if args.id_ref_gate_speed_tol_rel is None else float(args.id_ref_gate_speed_tol_rel),
                float(args.id_ref_gate_min_scale),
                float(args.id_ref_gate_exponent),
                bool(args.ai_id_relative),
                float(args.delta_id_max),
                use_total_power,
                supervisor_cfg=supervisor_cfg,
            )

        _save_csv(out_dir / f"{file_tag}_foc.csv", foc)
        _save_csv(out_dir / f"{file_tag}_mic_ai.csv", mic)

        foc_sum = _summarize(foc, float(args.window_frac))
        mic_sum = _summarize(mic, float(args.window_frac))
        err_tol = float(args.error_tol_rel)
        err_tol_abs = float(args.error_tol_abs)
        err_limit = _err_limit(foc_sum["mean_abs_speed_err"], err_tol, err_tol_abs)
        err_ok = mic_sum["mean_abs_speed_err"] <= err_limit
        power_saving_pct = 0.0
        if foc_sum["mean_p_el_pos"] > 1e-9:
            power_saving_pct = 100.0 * (1.0 - mic_sum["mean_p_el_pos"] / foc_sum["mean_p_el_pos"])
        eta_gain_pct = 0.0
        if foc_sum["eta"] > 1e-9:
            eta_gain_pct = 100.0 * (mic_sum["eta"] / foc_sum["eta"] - 1.0)
        if mic_ctrl_mode is not None:
            mic_variant = f"mode:{mic_ctrl_mode}"
        elif supervisor_cfg is not None:
            mic_variant = "ai_supervisor"
        elif mic_rule:
            mic_variant = "mic_rule"
        elif mic_search:
            mic_variant = "mic_search"
        elif mic_id_ref is not None:
            mic_variant = "mic_fixed"
        else:
            mic_variant = "mic_ai"

        summary_rows.append(
            {
                "scenario": scenario,
                "file_tag": file_tag,
                "mic_variant": mic_variant,
                "foc_mean_err": foc_sum["mean_abs_speed_err"],
                "mic_mean_err": mic_sum["mean_abs_speed_err"],
                "err_ok": bool(err_ok),
                "err_limit": err_limit,
                "foc_p_el_pos": foc_sum["mean_p_el_pos"],
                "mic_p_el_pos": mic_sum["mean_p_el_pos"],
                "power_saving_pct": power_saving_pct,
                "foc_eta": foc_sum["eta"],
                "mic_eta": mic_sum["eta"],
                "eta_gain_pct": eta_gain_pct,
            }
        )

        if bool(args.plots):
            _plot_power(out_dir / f"{file_tag}_power", foc, mic, bool(args.clip_negative))
            _plot_speed_error(out_dir / f"{file_tag}_speed_error", foc, mic)

        if bool(args.include_v3):
            v3 = _simulate_v3(env_cfg_s, dt, t_end, use_total_power)
            _save_csv(out_dir / f"{file_tag}_v3.csv", v3)
            v3_sum = _summarize(v3, float(args.window_frac))
            v3_err_limit = _err_limit(foc_sum["mean_abs_speed_err"], err_tol, err_tol_abs)
            v3_err_ok = v3_sum["mean_abs_speed_err"] <= v3_err_limit
            v3_power_saving_pct = 0.0
            if foc_sum["mean_p_el_pos"] > 1e-9:
                v3_power_saving_pct = 100.0 * (1.0 - v3_sum["mean_p_el_pos"] / foc_sum["mean_p_el_pos"])
            v3_eta_gain_pct = 0.0
            if foc_sum["eta"] > 1e-9:
                v3_eta_gain_pct = 100.0 * (v3_sum["eta"] / foc_sum["eta"] - 1.0)

            summary_rows_v3.append(
                {
                    "scenario": scenario,
                    "file_tag": file_tag,
                    "foc_mean_err": foc_sum["mean_abs_speed_err"],
                    "v3_mean_err": v3_sum["mean_abs_speed_err"],
                    "err_ok": bool(v3_err_ok),
                    "err_limit": v3_err_limit,
                    "foc_p_el_pos": foc_sum["mean_p_el_pos"],
                    "v3_p_el_pos": v3_sum["mean_p_el_pos"],
                    "power_saving_pct": v3_power_saving_pct,
                    "foc_eta": foc_sum["eta"],
                    "v3_eta": v3_sum["eta"],
                    "eta_gain_pct": v3_eta_gain_pct,
                }
            )

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
    if summary_rows_v3:
        summary_v3_path = out_dir / "summary_v3.json"
        summary_v3_path.write_text(json.dumps(summary_rows_v3, indent=2), encoding="utf-8")
    print(f"Saved outputs to {out_dir}")


if __name__ == "__main__":
    main()
