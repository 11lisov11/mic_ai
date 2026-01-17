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

from mic_ai.ai.agents.ppo_voltage import PPOVoltageAgent
from mic_ai.ai.ai_env import AiEnvConfig, MicAiAIEnv
from mic_ai.ai.train_ai_id_ref import FEATURE_KEYS as ID_FEATURE_KEYS
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


def _summarize(series: Dict[str, np.ndarray], window_frac: float) -> Dict[str, float]:
    n = int(series["t"].size)
    sl = _steady_slice(n, window_frac)
    p_el = series["p_el"][sl]
    p_mech = series["p_mech"][sl]
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
    }


def _simulate_foc(env_cfg: object, dt: float, t_end: float) -> Dict[str, np.ndarray]:
    env = InductionMotorEnv(env_cfg)
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
        p_el[k] = calc_p_el(v_abc, i_abc)
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


def _build_ai_env(
    env_cfg: object,
    dt: float,
    t_end: float,
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
    id_ref_base = float(getattr(getattr(env_cfg, "foc", None), "id_ref", 0.0) or 0.0)
    id_ref_max = max(i_base * 1.5, id_ref_base, id_ref_base * 1.2)
    omega_ref_nom = float(2.0 * math.pi * 10.0 / max(env_cfg.motor.p, 1))
    steps = int(max(t_end / dt, 1))
    ai_cfg = AiEnvConfig(
        episode_steps=steps,
        dt=dt,
        omega_ref=omega_ref_nom,
        omega_ref_max=max(abs(omega_ref_nom) * 1.2, 1e-3),
        w_speed_error=0.0,
        w_current_rms=0.0,
        control_mode="ai_id_ref",
        i_base=i_base,
        i_max=i_limit,
        i_hard_limit=float(i_limit * 2.0),
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
    )
    base_env = InductionMotorEnv(env_cfg)
    return MicAiAIEnv(base_env, ai_cfg, curiosity=None, world_model=None, world_input_keys=ID_FEATURE_KEYS, world_target_keys=["omega_norm"])


def _simulate_ai(
    agent: PPOVoltageAgent,
    env_cfg: object,
    dt: float,
    t_end: float,
    id_ref_alpha: float,
    id_ref_rate_limit: float | None,
    id_ref_gate_speed_tol: float | None,
    id_ref_gate_speed_tol_rel: float | None,
    id_ref_gate_min_scale: float,
    id_ref_gate_exponent: float,
    ai_id_relative: bool,
    delta_id_max: float,
) -> Dict[str, np.ndarray]:
    env = _build_ai_env(
        env_cfg,
        dt,
        t_end,
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

    steps = int(max(t_end / dt, 1))
    t = np.zeros(steps, dtype=float)
    omega = np.zeros(steps, dtype=float)
    omega_ref = np.zeros(steps, dtype=float)
    i_rms = np.zeros(steps, dtype=float)
    p_el = np.zeros(steps, dtype=float)
    p_mech = np.zeros(steps, dtype=float)

    for k in range(steps):
        action, _lp, _v = agent.act(obs)
        obs, _r, done, info = env.step(action)
        t[k] = float(getattr(env.base_env, "t", k * dt))
        omega[k] = float(info.get("omega_meas", obs.get("omega", 0.0)))
        omega_ref[k] = float(info.get("omega_ref", obs.get("omega_ref", 0.0)))
        i_abc = np.asarray(info.get("i_abc", (0.0, 0.0, 0.0)), dtype=float)
        v_abc = np.asarray(info.get("v_abc", (0.0, 0.0, 0.0)), dtype=float)
        torque = float(info.get("torque_e", getattr(env.base_env, "last_torque", 0.0)))
        i_rms[k] = calc_i_rms(i_abc)
        p_el[k] = calc_p_el(v_abc, i_abc)
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


def _simulate_mic_rule(
    env_cfg: object,
    dt: float,
    t_end: float,
    id_ref_low: float,
    id_ref_high: float,
    speed_tol_rel: float,
    omega_min_pu: float,
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
        p_el[k] = calc_p_el(v_abc, i_abc)
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare FOC vs MIC AI across scenarios.")
    parser.add_argument("--env-config", default="config/env_demo_true_motor1.py")
    parser.add_argument("--ai-checkpoint", default=None)
    parser.add_argument("--mic-id-ref", type=float, default=None, help="Use fixed id_ref for MIC curve.")
    parser.add_argument("--mic-id-ref-low", type=float, default=None, help="Low id_ref for MIC rule.")
    parser.add_argument("--mic-id-ref-high", type=float, default=None, help="High id_ref for MIC rule.")
    parser.add_argument("--mic-id-ref-speed-tol-rel", type=float, default=0.05, help="Speed error tol (rel).")
    parser.add_argument("--mic-id-ref-omega-min", type=float, default=0.1, help="Min omega_ref pu for low id_ref.")
    parser.add_argument("--scenarios", default="speed_step,ramp,load_step,start_stop")
    parser.add_argument("--t-end", type=float, default=None)
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--load-torque", type=float, default=None, help="Override constant load torque, N*m.")
    parser.add_argument("--window-frac", type=float, default=0.25)
    parser.add_argument("--error-tol-rel", type=float, default=0.0, help="Allowed error increase vs FOC.")
    parser.add_argument("--ai-id-relative", action="store_true", help="Use relative id_ref around base.")
    parser.add_argument("--delta-id-max", type=float, default=0.1)
    parser.add_argument("--id-ref-alpha", type=float, default=1.0)
    parser.add_argument("--id-ref-rate-limit", type=float, default=None)
    parser.add_argument("--id-ref-gate-speed-tol", type=float, default=None)
    parser.add_argument("--id-ref-gate-speed-tol-rel", type=float, default=0.05)
    parser.add_argument("--id-ref-gate-min-scale", type=float, default=0.0)
    parser.add_argument("--id-ref-gate-exponent", type=float, default=1.0)
    parser.add_argument("--plots", action="store_true")
    parser.add_argument("--clip-negative", action="store_true")
    parser.add_argument("--out-dir", default="outputs/scenario_compare")
    args = parser.parse_args()

    env_path = _resolve_config_path(args.env_config)
    env_cfg = make_env_from_config(str(env_path)).env_config
    dt = float(args.dt) if args.dt is not None else float(env_cfg.sim.dt)
    t_end = float(args.t_end) if args.t_end is not None else float(env_cfg.sim.t_end)

    mic_id_ref = None if args.mic_id_ref is None else float(args.mic_id_ref)
    mic_rule = False
    mic_id_ref_low = None if args.mic_id_ref_low is None else float(args.mic_id_ref_low)
    mic_id_ref_high = None if args.mic_id_ref_high is None else float(args.mic_id_ref_high)
    if mic_id_ref_low is not None or mic_id_ref_high is not None:
        if mic_id_ref_low is None or mic_id_ref_high is None:
            raise ValueError("Provide both --mic-id-ref-low and --mic-id-ref-high.")
        mic_rule = True
    agent = None
    if mic_id_ref is None and not mic_rule:
        if args.ai_checkpoint is None:
            raise ValueError("Provide --ai-checkpoint or --mic-id-ref.")
        ckpt = Path(args.ai_checkpoint)
        state = torch.load(ckpt, map_location="cpu")
        hidden = _infer_hidden_sizes(state) or (128, 128)
        agent = PPOVoltageAgent(feature_keys=ID_FEATURE_KEYS, action_dim=1, device="cpu", hidden_sizes=hidden)
        agent.net.load_state_dict(state)
        agent.set_action_std(1e-6)

    scenario_list = [s.strip() for s in str(args.scenarios).split(",") if s.strip()]
    load_torque = float(args.load_torque) if args.load_torque is not None else float(env_cfg.sim.load_torque)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, float | str | bool]] = []
    for scenario in scenario_list:
        file_tag = _sanitize_name(scenario)
        sim_cfg = replace(env_cfg.sim, scenario_name=str(scenario), dt=dt, t_end=t_end, load_torque=load_torque)
        env_cfg_s = replace(env_cfg, sim=sim_cfg)

        foc = _simulate_foc(env_cfg_s, dt, t_end)
        if mic_rule:
            mic = _simulate_mic_rule(
                env_cfg_s,
                dt,
                t_end,
                mic_id_ref_low,
                mic_id_ref_high,
                float(args.mic_id_ref_speed_tol_rel),
                float(args.mic_id_ref_omega_min),
            )
        elif mic_id_ref is not None:
            foc_mic = replace(env_cfg_s.foc, id_ref=mic_id_ref)
            mic = _simulate_foc(replace(env_cfg_s, foc=foc_mic), dt, t_end)
        else:
            mic = _simulate_ai(
                agent,
                env_cfg_s,
                dt,
                t_end,
                float(args.id_ref_alpha),
                None if args.id_ref_rate_limit is None else float(args.id_ref_rate_limit),
                None if args.id_ref_gate_speed_tol is None else float(args.id_ref_gate_speed_tol),
                None if args.id_ref_gate_speed_tol_rel is None else float(args.id_ref_gate_speed_tol_rel),
                float(args.id_ref_gate_min_scale),
                float(args.id_ref_gate_exponent),
                bool(args.ai_id_relative),
                float(args.delta_id_max),
            )

        _save_csv(out_dir / f"{file_tag}_foc.csv", foc)
        _save_csv(out_dir / f"{file_tag}_mic_ai.csv", mic)

        foc_sum = _summarize(foc, float(args.window_frac))
        mic_sum = _summarize(mic, float(args.window_frac))
        err_tol = float(args.error_tol_rel)
        err_ok = mic_sum["mean_abs_speed_err"] <= foc_sum["mean_abs_speed_err"] * (1.0 + err_tol)
        power_saving_pct = 0.0
        if foc_sum["mean_p_el_pos"] > 1e-9:
            power_saving_pct = 100.0 * (1.0 - mic_sum["mean_p_el_pos"] / foc_sum["mean_p_el_pos"])

        summary_rows.append(
            {
                "scenario": scenario,
                "file_tag": file_tag,
                "foc_mean_err": foc_sum["mean_abs_speed_err"],
                "mic_mean_err": mic_sum["mean_abs_speed_err"],
                "err_ok": bool(err_ok),
                "foc_p_el_pos": foc_sum["mean_p_el_pos"],
                "mic_p_el_pos": mic_sum["mean_p_el_pos"],
                "power_saving_pct": power_saving_pct,
                "foc_eta": foc_sum["eta"],
                "mic_eta": mic_sum["eta"],
            }
        )

        if bool(args.plots):
            _plot_power(out_dir / f"{file_tag}_power", foc, mic, bool(args.clip_negative))
            _plot_speed_error(out_dir / f"{file_tag}_speed_error", foc, mic)

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
    print(f"Saved outputs to {out_dir}")


if __name__ == "__main__":
    main()
