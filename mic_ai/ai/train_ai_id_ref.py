from __future__ import annotations

import sys
import argparse
import json
import os
import random
import shutil
import time
import subprocess
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mic_ai.ai.agents.ppo_voltage import PPOVoltageAgent
from mic_ai.ai.ai_env import AiEnvConfig, MicAiAIEnv
from mic_ai.ai.ai_voltage_config import get_curriculum_config, load_ai_voltage_config
from mic_ai.core.env import make_env_from_config
from simulation.gym_env import InductionMotorEnv


BASE_FEATURE_KEYS = [
    "omega_norm",
    "omega_ref_norm",
    "err_norm",
    "id_norm",
    "iq_norm",
    "slip_norm",
    "load_torque_norm",
]


def build_feature_keys(include_energy_obs: bool) -> List[str]:
    keys = list(BASE_FEATURE_KEYS)
    if include_energy_obs:
        keys += ["p_in_norm", "p_el_filt", "p_shaft_norm", "eta_norm"]
    # de-dup preserving order
    seen = set()
    out: List[str] = []
    for key in keys:
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


# Default feature set for id_ref policies.
# NOTE: Paper checkpoints were trained with energy-related observations enabled.
# Keep this in sync so evaluation tools (e.g. scenario_compare) load those checkpoints by default.
FEATURE_KEYS = build_feature_keys(include_energy_obs=True)

OUTPUT_DIR = Path(os.environ.get("MIC_AI_ID_REF_OUTPUT_DIR", "outputs/ai_id_ref"))
EPISODE_LOG_DIR = OUTPUT_DIR / "episode_logs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
RESULTS_ROOT = Path(os.environ.get("MIC_AI_RESULTS_ROOT", "results_run"))


def _parse_scenarios(text: str) -> List[str]:
    names = [item.strip() for item in str(text).split(",") if item.strip()]
    return names


def _parse_range(text: str | None) -> tuple[float, float] | None:
    if not text:
        return None
    raw = str(text).strip().replace(":", ",")
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if len(parts) != 2:
        return None
    try:
        lo = float(parts[0])
        hi = float(parts[1])
    except ValueError:
        return None
    if hi < lo:
        lo, hi = hi, lo
    return lo, hi


def _normalize_range(value: object | None) -> tuple[float, float] | None:
    if value is None:
        return None
    if isinstance(value, (tuple, list)) and len(value) == 2:
        try:
            lo = float(value[0])
            hi = float(value[1])
        except Exception:
            return None
        if hi < lo:
            lo, hi = hi, lo
        return lo, hi
    if isinstance(value, str):
        return _parse_range(value)
    return None


def _prepare_output_file(path: Path) -> Path:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = path.with_name(f"{path.stem}_backup_{ts}{path.suffix}")
        path.rename(backup)
        print(f"[log] existing {path.name} -> backup {backup.name}")
    return path


def _run_eval(
    env_config: str,
    checkpoint_path: Path,
    out_dir: Path,
    scenarios: str,
    dt: float | None,
    t_end: float | None,
    window_frac: float,
    error_tol_rel: float,
    error_tol_abs: float,
    use_total_power: bool,
    ai_id_relative: bool,
    delta_id_max: float,
    id_ref_alpha: float,
    id_ref_rate_limit: float | None,
    id_ref_gate_speed_tol: float | None,
    id_ref_gate_speed_tol_rel: float | None,
    id_ref_gate_min_scale: float,
    id_ref_gate_exponent: float,
    feature_keys: List[str],
) -> None:
    cmd = [
        sys.executable,
        "-m",
        "mic_ai.tools.scenario_compare",
        "--env-config",
        str(env_config),
        "--ai-checkpoint",
        str(checkpoint_path),
        "--out-dir",
        str(out_dir),
        "--scenarios",
        str(scenarios),
        "--window-frac",
        str(window_frac),
        "--error-tol-rel",
        str(error_tol_rel),
        "--error-tol-abs",
        str(error_tol_abs),
        "--id-ref-alpha",
        str(id_ref_alpha),
        "--id-ref-gate-min-scale",
        str(id_ref_gate_min_scale),
        "--id-ref-gate-exponent",
        str(id_ref_gate_exponent),
    ]
    if feature_keys:
        cmd += ["--ai-feature-keys", ",".join(feature_keys)]
    if dt is not None:
        cmd += ["--dt", str(dt)]
    if t_end is not None:
        cmd += ["--t-end", str(t_end)]
    if use_total_power:
        cmd += ["--use-total-power"]
    if ai_id_relative:
        cmd += ["--ai-id-relative", "--delta-id-max", str(delta_id_max)]
    if id_ref_rate_limit is not None:
        cmd += ["--id-ref-rate-limit", str(id_ref_rate_limit)]
    if id_ref_gate_speed_tol is not None:
        cmd += ["--id-ref-gate-speed-tol", str(id_ref_gate_speed_tol)]
    if id_ref_gate_speed_tol_rel is not None:
        cmd += ["--id-ref-gate-speed-tol-rel", str(id_ref_gate_speed_tol_rel)]
    subprocess.run(cmd, check=False)


def build_env(
    env_config_path: str,
    episode_steps: int,
    control_mode: str,
    w_speed: float,
    w_power: float,
    w_current: float | None,
    w_smooth: float,
    w_mag: float,
    w_shaft: float,
    w_eta: float,
    eta_clip: float,
    override_load_torque: bool,
    override_omega_ref: bool,
    ai_id_ref_relative: bool,
    delta_id_max: float,
    id_ref_alpha: float,
    id_ref_rate_limit: float | None,
    ai_id_speed_tol: float,
    ai_id_speed_tol_rel: float | None,
    id_ref_gate_speed_tol: float | None,
    id_ref_gate_speed_tol_rel: float | None,
    id_ref_gate_min_scale: float,
    id_ref_gate_exponent: float,
    load_torque: float | None,
    omega_ref_override: float | None,
    feature_keys: List[str],
) -> MicAiAIEnv:
    env_sim = make_env_from_config(env_config_path)
    env_cfg = env_sim.env_config

    if omega_ref_override is None:
        omega_ref = float(2.0 * np.pi * 10.0 / max(env_cfg.motor.p, 1))
    else:
        omega_ref = float(omega_ref_override)
    i_base_nom = float(getattr(env_cfg.motor, "I_n", 1.0))
    foc_cfg = getattr(env_cfg, "foc", None)
    iq_limit_cfg = getattr(foc_cfg, "iq_limit", None)
    if iq_limit_cfg is None:
        iq_limit_cfg = i_base_nom * 8.0
    iq_limit = float(iq_limit_cfg)
    id_ref_base = float(getattr(foc_cfg, "id_ref", 0.0) or 0.0)
    mode = str(control_mode).lower()
    if mode == "ai_current":
        # Current-control mode: keep a wide normalization range because the agent can command iq/id directly.
        i_limit = float(max(iq_limit, i_base_nom * 8.0, 5.0))
        i_base = float(i_limit)
    else:
        # id_ref-supervision mode: normalize power and current to the realistic (iq,id) vector range.
        # Using i_base_nom*8 here makes p_in_norm almost zero for bigger motors and hurts learning,
        # while evaluation (scenario_compare) uses a much smaller i_max. Keep train/eval consistent.
        # Wider id_ref range improves learning for small motors where id_ref_base can be > I_n
        # due to dq vs line-current scaling.
        id_ref_max_est = float(max(i_base_nom * 1.5, id_ref_base, id_ref_base * 1.6))
        i_limit = float(max(math.hypot(iq_limit, id_ref_max_est), i_base_nom, 5.0))
        i_base = float(i_base_nom)
    id_ref_max = max(i_base * 1.5, id_ref_base, id_ref_base * 1.6)

    cfg = load_ai_voltage_config()
    curriculum_cfg = get_curriculum_config(cfg)
    piecewise_steps = curriculum_cfg.get("piecewise_steps", (150, 300))
    piecewise_multipliers = curriculum_cfg.get("piecewise_multipliers", (1.0, 0.8, 1.0))
    curriculum_stages = curriculum_cfg.get("omega_pu_stages", (0.3, 0.5))
    stage_boundaries = curriculum_cfg.get("stage_episode_boundaries", (150, 300))

    w_current_cfg = w_current
    if w_current_cfg is None:
        w_current_cfg = float(getattr(env_cfg, "ai_w_id_current", 0.0))

    ai_cfg = AiEnvConfig(
        episode_steps=int(episode_steps),
        dt=float(env_cfg.sim.dt),
        omega_ref=omega_ref,
        omega_ref_max=max(abs(omega_ref) * 1.2, 1e-3),
        w_speed_error=0.0,
        w_current_rms=0.0,
        i_base=i_base,
        i_max=i_limit,
        control_mode=str(control_mode).lower(),
        reward_min=-10.0,
        reward_max=1.0,
        w_ai_id_speed=float(w_speed),
        w_ai_id_power=float(w_power),
        w_ai_id_current=float(w_current_cfg),
        w_ai_id_smooth=float(w_smooth),
        w_ai_id_mag=float(w_mag),
        w_ai_id_shaft=float(w_shaft),
        w_ai_id_eta=float(w_eta),
        ai_id_eta_clip=float(eta_clip),
        sigma_omega=float(getattr(env_cfg, "ai_sigma_omega", 0.05)),
        sigma_id=float(getattr(env_cfg, "ai_sigma_id", 0.03)),
        sigma_iq=float(getattr(env_cfg, "ai_sigma_iq", 0.03)),
        drift_every_episodes=int(getattr(env_cfg, "ai_drift_every_episodes", 5)),
        drift_scale=float(getattr(env_cfg, "ai_drift_scale", 0.04)),
        w_ext_scale=float(getattr(env_cfg, "ai_w_ext_scale", 1.0)),
        w_int_scale=float(getattr(env_cfg, "ai_w_int_scale", 0.0)),
        wm_lr=float(getattr(env_cfg, "ai_wm_lr", 1e-4)),
        curiosity_beta=float(getattr(env_cfg, "ai_curiosity_beta", 0.0)),
        id_ref_alpha=float(id_ref_alpha),
        id_ref_rate_limit=None if id_ref_rate_limit is None else float(id_ref_rate_limit),
        id_ref_gate_speed_tol=None if id_ref_gate_speed_tol is None else float(id_ref_gate_speed_tol),
        id_ref_gate_speed_tol_rel=None if id_ref_gate_speed_tol_rel is None else float(id_ref_gate_speed_tol_rel),
        id_ref_gate_min_scale=float(id_ref_gate_min_scale),
        id_ref_gate_exponent=float(id_ref_gate_exponent),
        delta_id_max=float(delta_id_max),
        ai_id_speed_tol=float(ai_id_speed_tol),
        ai_id_speed_tol_rel=None if ai_id_speed_tol_rel is None else float(ai_id_speed_tol_rel),
        curriculum_omega_pu=tuple(float(x) for x in curriculum_stages),
        curriculum_stage_episodes=tuple(int(x) for x in stage_boundaries),
        omega_piecewise_steps=tuple(int(x) for x in piecewise_steps),
        omega_piecewise_multipliers=tuple(float(x) for x in piecewise_multipliers),
        id_ref_min=0.0,
        id_ref_max=float(id_ref_max),
        ai_id_ref_relative=bool(ai_id_ref_relative),
        # Keep some safety margin: phase current can exceed iq_limit because id_ref and iq add
        # in the current vector. Too-tight limit would truncate episodes near the end of start/stop.
        i_hard_limit=float(i_limit * 4.0),
        load_torque_override=None if load_torque is None else float(load_torque),
        override_load_torque=bool(override_load_torque),
        override_omega_ref=bool(override_omega_ref),
        enable_id_control=bool(str(control_mode).lower() == "ai_current"),
    )

    base_env = InductionMotorEnv(env_cfg)
    base_env.omega_ref_func = lambda _t, ref=omega_ref: ref
    if load_torque is None:
        base_env.load_torque_func = lambda _t: getattr(env_cfg.sim, "load_torque", 0.0)
    else:
        base_env.load_torque_func = lambda _t, load=load_torque: float(load)

    return MicAiAIEnv(base_env, ai_cfg, curiosity=None, world_model=None, world_input_keys=feature_keys, world_target_keys=["omega_norm"])


def train(
    env_config: str,
    episodes: int,
    episode_steps: int,
    control_mode: str,
    w_speed: float,
    w_power: float,
    w_current: float | None,
    w_smooth: float,
    w_mag: float,
    w_shaft: float,
    w_eta: float,
    eta_clip: float,
    id_ref_alpha: float,
    id_ref_rate_limit: float | None,
    ai_id_speed_tol: float,
    ai_id_speed_tol_rel: float | None,
    id_ref_gate_speed_tol: float | None,
    id_ref_gate_speed_tol_rel: float | None,
    id_ref_gate_min_scale: float,
    id_ref_gate_exponent: float,
    fast: bool,
    time_budget_min: float | None,
    override_load_torque: bool,
    override_omega_ref: bool,
    ai_id_ref_relative: bool,
    delta_id_max: float,
    load_torque: float | None,
    omega_ref_override: float | None,
    scenarios: List[str] | None,
    scenario_sample: str,
    omega_ref_range: tuple[float, float] | None,
    load_torque_range: tuple[float, float] | None,
    seed: int | None,
    sigma_start: float,
    sigma_end: float,
    sigma_decay_episodes: int,
    power_warmup_episodes: int,
    power_ramp_episodes: int,
    eval_interval: int,
    eval_scenarios: str,
    eval_dt: float | None,
    eval_t_end: float | None,
    eval_window_frac: float,
    eval_error_tol_rel: float,
    eval_error_tol_abs: float,
    eval_use_total_power: bool,
    include_energy_obs: bool,
    update_every_episodes: int,
    init_checkpoint: str | None = None,
    output_dir: str | None = None,
    results_root: str | None = None,
) -> Dict[str, str]:
    feature_keys = build_feature_keys(include_energy_obs)
    if seed is not None:
        random.seed(int(seed))
        np.random.seed(int(seed))
        torch.manual_seed(int(seed))

    env = build_env(
        env_config,
        episode_steps=episode_steps,
        control_mode=str(control_mode),
        w_speed=w_speed,
        w_power=w_power,
        w_current=w_current,
        w_smooth=w_smooth,
        w_mag=w_mag,
        w_shaft=w_shaft,
        w_eta=w_eta,
        eta_clip=eta_clip,
        override_load_torque=override_load_torque,
        override_omega_ref=override_omega_ref,
        ai_id_ref_relative=ai_id_ref_relative,
        delta_id_max=delta_id_max,
        id_ref_alpha=id_ref_alpha,
        id_ref_rate_limit=id_ref_rate_limit,
        ai_id_speed_tol=ai_id_speed_tol,
        ai_id_speed_tol_rel=ai_id_speed_tol_rel,
        id_ref_gate_speed_tol=id_ref_gate_speed_tol,
        id_ref_gate_speed_tol_rel=id_ref_gate_speed_tol_rel,
        id_ref_gate_min_scale=id_ref_gate_min_scale,
        id_ref_gate_exponent=id_ref_gate_exponent,
        load_torque=load_torque,
        omega_ref_override=omega_ref_override,
        feature_keys=feature_keys,
    )

    scenarios = [s for s in (scenarios or []) if s]
    scenario_sample = str(scenario_sample or "random").lower()
    rng = np.random.default_rng(seed)

    hidden_sizes = (64, 64) if fast else (128, 128)
    train_epochs = 3 if fast else 5
    minibatch_frac = 0.5 if fast else 0.25
    action_dim = 2 if str(control_mode).lower() == "ai_current" else 1
    agent = PPOVoltageAgent(
        feature_keys=feature_keys,
        action_dim=action_dim,
        device="cpu",
        hidden_sizes=hidden_sizes,
        lr=5e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        entropy_coef=0.005,
        value_coef=0.3,
        max_grad_norm=0.5,
        train_epochs=train_epochs,
        minibatch_frac=minibatch_frac,
    )
    if init_checkpoint:
        init_path = Path(str(init_checkpoint)).resolve()
        if not init_path.exists():
            raise FileNotFoundError(f"Init checkpoint not found: {init_path}")
        state = torch.load(init_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state and isinstance(state.get("state_dict"), dict):
            state = state["state_dict"]
        if not isinstance(state, dict):
            raise ValueError(f"Unsupported checkpoint format: {init_path}")
        missing_keys, unexpected_keys = agent.net.load_state_dict(state, strict=False)
        print(
            "[train_ai_id_ref] warm-start checkpoint={} missing_keys={} unexpected_keys={}".format(
                init_path,
                len(missing_keys),
                len(unexpected_keys),
            )
        )

    output_root_path = OUTPUT_DIR if output_dir is None else Path(str(output_dir)).expanduser()
    if not output_root_path.is_absolute():
        output_root_path = (Path.cwd() / output_root_path).resolve()
    episode_log_dir = output_root_path / "episode_logs"
    checkpoint_root = output_root_path / "checkpoints"

    results_root_path = RESULTS_ROOT if results_root is None else Path(str(results_root)).expanduser()
    if not results_root_path.is_absolute():
        results_root_path = (Path.cwd() / results_root_path).resolve()

    env_name = Path(env_config).stem
    ckpt_dir = (checkpoint_root / env_name).resolve()
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    mode_tag = "ai_current" if str(control_mode).lower() == "ai_current" else "ai_id_ref"
    run_dir = results_root_path / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{env_name}_{mode_tag}"
    run_dir.mkdir(parents=True, exist_ok=True)

    episodes_log: List[Dict[str, float]] = []
    best_score = float("inf")
    best_ckpt: Path | None = None
    t0 = time.perf_counter()
    max_seconds = None if time_budget_min is None else float(time_budget_min) * 60.0

    update_every = max(int(update_every_episodes), 1)
    for ep in range(int(episodes)):
        if max_seconds is not None and (time.perf_counter() - t0) >= max_seconds:
            print(f"[{env_name}] time budget reached at ep {ep}")
            break

        scenario_name = ""
        if scenarios:
            if scenario_sample == "cycle":
                scenario_name = scenarios[ep % len(scenarios)]
            else:
                scenario_name = str(rng.choice(scenarios))
            env.set_scenario(scenario_name)
        else:
            if omega_ref_range is not None:
                env.cfg.override_omega_ref = False
                omega_ref_val = float(rng.uniform(omega_ref_range[0], omega_ref_range[1]))
                env.cfg.omega_ref = omega_ref_val
                env.base_env.omega_ref_func = lambda _t, ref=omega_ref_val: ref
            if load_torque_range is not None:
                env.cfg.override_load_torque = False
                load_val = float(rng.uniform(load_torque_range[0], load_torque_range[1]))
                env.base_env.load_torque_func = lambda _t, load=load_val: load

        obs = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        if sigma_decay_episodes <= 0:
            sigma = float(sigma_end)
        else:
            frac = min(1.0, ep / max(sigma_decay_episodes, 1))
            sigma = float(sigma_start + (sigma_end - sigma_start) * frac)
        agent.set_action_std(sigma)

        power_scale = 1.0
        if power_warmup_episodes > 0 or power_ramp_episodes > 0:
            if ep < power_warmup_episodes:
                power_scale = 0.0
            elif ep < power_warmup_episodes + max(power_ramp_episodes, 1):
                power_scale = (ep - power_warmup_episodes) / max(power_ramp_episodes, 1)
            else:
                power_scale = 1.0
        env.cfg.w_ai_id_power = float(w_power) * float(power_scale)

        while not done and steps < int(episode_steps):
            action, logp, value = agent.act(obs)
            obs_next, reward, done, info = env.step(action)
            agent.store(obs, action, logp, reward, done, value)
            total_reward += float(reward)
            obs = obs_next
            steps += 1

        losses = {"actor_loss": agent.last_actor_loss, "value_loss": agent.last_value_loss}
        if (ep + 1) % update_every == 0 or ep == episodes - 1:
            with torch.no_grad():
                last_value = float(agent.net(agent._to_tensor(obs).unsqueeze(0))[2].item())
            losses = agent.update(last_value=last_value)
        m = env.episode_metrics()

        omega_ref_logged = float(getattr(env.cfg, "omega_ref", 0.0))
        load_logged = float(getattr(env.base_env, "load_torque_func", lambda _t: 0.0)(0.0))
        entry = {
            "episode": float(ep),
            "steps": float(m.get("steps", steps)),
            "mean_speed_error": float(m.get("mean_speed_error", 0.0)),
            "mean_p_in_pos": float(m.get("mean_p_in_pos", 0.0)),
            "mean_p_shaft_pos": float(m.get("mean_p_shaft_pos", 0.0)),
            "mean_p_shaft_target_pos": float(m.get("mean_p_shaft_target_pos", 0.0)),
            "mean_eta_inst": float(m.get("mean_eta_inst", 0.0)),
            "eta_energy": float(m.get("eta_energy", 0.0)),
            "mean_current_rms": float(m.get("mean_current_rms", 0.0)),
            "mean_action_norm": float(m.get("action_norm", 0.0)),
            "mean_reward": float(total_reward / max(steps, 1)),
            "actor_loss": float(losses.get("actor_loss", 0.0)),
            "value_loss": float(losses.get("value_loss", 0.0)),
            "scenario": scenario_name,
            "omega_ref": omega_ref_logged,
            "load_torque": load_logged,
            "w_power_eff": float(getattr(env.cfg, "w_ai_id_power", w_power)),
            "exploration_sigma": float(sigma),
        }
        episodes_log.append(entry)

        # Score: minimize electric input, keep tracking quality, and avoid shaft-power deficit.
        shaft_deficit = max(0.0, entry["mean_p_shaft_target_pos"] - entry["mean_p_shaft_pos"])
        score = (
            entry["mean_p_in_pos"]
            + 50.0 * entry["mean_speed_error"]
            + 3.0 * shaft_deficit
            - 5.0 * entry["eta_energy"]
        )
        if score < best_score:
            best_score = score
            best_ckpt = ckpt_dir / "best_actor.pth"
            torch.save(agent.net.state_dict(), best_ckpt)

        # Always save per-episode checkpoints for offline, reproducible evaluation.
        eval_root = run_dir / "eval"
        eval_root.mkdir(parents=True, exist_ok=True)
        eval_ckpt = eval_root / f"actor_ep{ep:03d}.pth"
        torch.save(agent.net.state_dict(), eval_ckpt)
        if eval_interval > 0 and (ep % eval_interval == 0):
            _run_eval(
                env_config=env_config,
                checkpoint_path=eval_ckpt,
                out_dir=eval_root / f"ep_{ep:03d}",
                scenarios=eval_scenarios,
                dt=eval_dt,
                t_end=eval_t_end,
                window_frac=eval_window_frac,
                error_tol_rel=eval_error_tol_rel,
                error_tol_abs=eval_error_tol_abs,
                use_total_power=eval_use_total_power,
                ai_id_relative=bool(ai_id_ref_relative),
                delta_id_max=float(delta_id_max),
                id_ref_alpha=float(id_ref_alpha),
                id_ref_rate_limit=id_ref_rate_limit,
                id_ref_gate_speed_tol=id_ref_gate_speed_tol,
                id_ref_gate_speed_tol_rel=id_ref_gate_speed_tol_rel,
                id_ref_gate_min_scale=float(id_ref_gate_min_scale),
                id_ref_gate_exponent=float(id_ref_gate_exponent),
                feature_keys=feature_keys,
            )

        if ep % 10 == 0 or ep == episodes - 1:
            print(
                f"[{env_name}] ep {ep:03d} | mean_p_in_pos {entry['mean_p_in_pos']:.3f} | "
                f"mean_p_shaft_pos {entry['mean_p_shaft_pos']:.3f} | "
                f"eta {entry['eta_energy']:.3f} | mean|e_w| {entry['mean_speed_error']:.3f} | "
                f"act_norm {entry['mean_action_norm']:.3f}"
            )

    last_ckpt = ckpt_dir / "last_actor.pth"
    torch.save(agent.net.state_dict(), last_ckpt)

    episodes_path = _prepare_output_file(episode_log_dir / f"ai_id_ref_{env_name}_episodes.json")
    with episodes_path.open("w", encoding="utf-8") as f:
        json.dump(episodes_log, f, indent=2)

    with (run_dir / "training_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(episodes_log, f, indent=2)
    torch.save(agent.net.state_dict(), run_dir / "actor_critic.pth")
    run_config = {
        "env_config": str(env_config),
        "control_mode": str(control_mode).lower(),
        "episodes": int(episodes),
        "episode_steps": int(episode_steps),
        "weights": {
            "w_speed": float(w_speed),
            "w_power": float(w_power),
            "w_current": None if w_current is None else float(w_current),
            "w_smooth": float(w_smooth),
            "w_mag": float(w_mag),
            "w_shaft": float(w_shaft),
            "w_eta": float(w_eta),
            "eta_clip": float(eta_clip),
        },
        "id_ref_alpha": float(id_ref_alpha),
        "id_ref_rate_limit": None if id_ref_rate_limit is None else float(id_ref_rate_limit),
        "ai_id_speed_tol": float(ai_id_speed_tol),
        "ai_id_speed_tol_rel": None if ai_id_speed_tol_rel is None else float(ai_id_speed_tol_rel),
        "id_ref_gate_speed_tol": None if id_ref_gate_speed_tol is None else float(id_ref_gate_speed_tol),
        "id_ref_gate_speed_tol_rel": None if id_ref_gate_speed_tol_rel is None else float(id_ref_gate_speed_tol_rel),
        "id_ref_gate_min_scale": float(id_ref_gate_min_scale),
        "id_ref_gate_exponent": float(id_ref_gate_exponent),
        "ai_id_ref_relative": bool(ai_id_ref_relative),
        "delta_id_max": float(delta_id_max),
        "load_torque_override": None if load_torque is None else float(load_torque),
        "omega_ref_override": None if omega_ref_override is None else float(omega_ref_override),
        "omega_ref_range": omega_ref_range,
        "load_torque_range": load_torque_range,
        "scenarios": scenarios or [],
        "scenario_sample": str(scenario_sample),
        "seed": None if seed is None else int(seed),
        "sigma_start": float(sigma_start),
        "sigma_end": float(sigma_end),
        "sigma_decay_episodes": int(sigma_decay_episodes),
        "power_warmup_episodes": int(power_warmup_episodes),
        "power_ramp_episodes": int(power_ramp_episodes),
        "eval_interval": int(eval_interval),
        "eval_scenarios": str(eval_scenarios),
        "eval_dt": None if eval_dt is None else float(eval_dt),
        "eval_t_end": None if eval_t_end is None else float(eval_t_end),
        "eval_window_frac": float(eval_window_frac),
        "eval_error_tol_rel": float(eval_error_tol_rel),
        "eval_error_tol_abs": float(eval_error_tol_abs),
        "eval_use_total_power": bool(eval_use_total_power),
        "include_energy_obs": bool(include_energy_obs),
        "update_every_episodes": int(update_every),
        "feature_keys": feature_keys,
        "init_checkpoint": None if init_checkpoint is None else str(Path(init_checkpoint).resolve()),
        "output_dir": str(output_root_path),
        "results_root": str(results_root_path),
    }
    (run_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    if best_ckpt is None:
        best_ckpt = ckpt_dir / "best_actor.pth"
        shutil.copyfile(last_ckpt, best_ckpt)

    print(f"Saved checkpoints: {best_ckpt} | {last_ckpt}")
    return {"episodes": str(episodes_path), "best": str(best_ckpt), "last": str(last_ckpt), "run_dir": str(run_dir)}


def main() -> None:
    p = argparse.ArgumentParser(description="Train AI to adapt FOC id_ref for efficiency (minimize P_in).")
    p.add_argument("config", help="Env config path (.py)")
    p.add_argument("--control-mode", type=str, default="ai_id_ref", choices=["ai_id_ref", "ai_current"])
    p.add_argument("--episodes", type=int, default=400)
    p.add_argument("--episode-steps", type=int, default=200)
    p.add_argument("--w-speed", type=float, default=1.0)
    p.add_argument("--w-power", type=float, default=6.0)
    p.add_argument("--w-current", type=float, default=None, help="Penalty for current magnitude (defaults to config).")
    p.add_argument("--w-smooth", type=float, default=0.05)
    p.add_argument("--w-mag", type=float, default=0.0)
    p.add_argument("--w-shaft", type=float, default=2.0, help="Penalty for shaft-power deficit vs omega_ref*load.")
    p.add_argument("--w-eta", type=float, default=1.0, help="Penalty for low instantaneous efficiency.")
    p.add_argument("--eta-clip", type=float, default=1.2, help="Upper clip for eta term in reward.")
    p.add_argument("--ai-id-speed-tol", type=float, default=0.5)
    p.add_argument("--ai-id-speed-tol-rel", type=float, default=None, help="Relative speed tol (e.g., 0.05).")
    p.add_argument("--id-ref-alpha", type=float, default=1.0)
    p.add_argument("--id-ref-rate-limit", type=float, default=None, help="Max d(id_ref)/dt, A/s.")
    p.add_argument("--id-ref-gate-speed-tol", type=float, default=None, help="Gate id_ref when |e_omega| exceeds tol.")
    p.add_argument("--id-ref-gate-speed-tol-rel", type=float, default=None, help="Relative gate tol (e.g., 0.05).")
    p.add_argument("--id-ref-gate-min-scale", type=float, default=0.0)
    p.add_argument("--id-ref-gate-exponent", type=float, default=1.0)
    p.add_argument("--fast", action="store_true")
    p.add_argument("--time-budget-min", type=float, default=None)
    p.add_argument("--override-load-torque", action="store_true", help="Force zero load during training.")
    p.add_argument("--no-override-omega-ref", dest="override_omega_ref", action="store_false", help="Use scenario omega_ref.")
    p.add_argument("--relative", action="store_true", help="Interpret action as delta around base id_ref.")
    p.add_argument("--delta-id-max", type=float, default=0.3, help="Relative id_ref delta scale.")
    p.add_argument("--load-torque", type=float, default=None, help="Override constant load torque, N*m.")
    p.add_argument("--omega-ref", type=float, default=None, help="Override omega_ref, rad/s.")
    p.add_argument("--omega-ref-pu", type=float, default=0.8, help="Omega_ref as pu of base omega (2*pi*10/p).")
    p.add_argument("--omega-ref-range", type=str, default=None, help="Random omega_ref range, e.g., 20,120 (rad/s).")
    p.add_argument("--omega-ref-pu-range", type=str, default=None, help="Random omega_ref range in pu, e.g., 0.2,1.2.")
    p.add_argument("--scenarios", type=str, default="", help="Comma-separated scenario list (e.g., speed_step,ramp,load_step,start_stop).")
    p.add_argument("--scenario-sample", type=str, default="random", choices=["random", "cycle"])
    p.add_argument("--load-torque-range", type=str, default=None, help="Random load torque range, N*m (min,max).")
    p.add_argument("--load-mult-range", type=str, default=None, help="Random load multiplier of env load (min,max).")
    p.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    p.add_argument("--sigma-start", type=float, default=0.2, help="Exploration sigma at episode 0.")
    p.add_argument("--sigma-end", type=float, default=0.05, help="Final exploration sigma.")
    p.add_argument("--sigma-decay-episodes", type=int, default=100, help="Episodes to decay sigma.")
    p.add_argument("--power-warmup-episodes", type=int, default=0, help="Episodes before enabling power penalty.")
    p.add_argument("--power-ramp-episodes", type=int, default=50, help="Episodes to ramp power penalty.")
    p.add_argument(
        "--include-energy-obs",
        action="store_true",
        help="Add p_in_norm, p_el_filt, p_shaft_norm and eta_norm to observations.",
    )
    p.add_argument("--update-every-episodes", type=int, default=1, help="PPO update frequency in episodes.")
    p.add_argument("--eval-interval", type=int, default=0, help="Run scenario_compare every N episodes (0 disables).")
    p.add_argument("--eval-scenarios", type=str, default="speed_step,ramp,load_step", help="Scenarios for eval.")
    p.add_argument("--eval-dt", type=float, default=None, help="Override dt for eval.")
    p.add_argument("--eval-t-end", type=float, default=None, help="Override t_end for eval.")
    p.add_argument("--eval-window-frac", type=float, default=0.25)
    p.add_argument("--eval-error-tol-rel", type=float, default=0.05)
    p.add_argument("--eval-error-tol-abs", type=float, default=0.0)
    p.add_argument("--eval-use-total-power", action="store_true")
    p.add_argument("--init-checkpoint", type=str, default=None, help="Optional actor checkpoint to warm-start training.")
    p.set_defaults(override_omega_ref=True)
    args = p.parse_args()
    omega_ref_override = None
    if args.omega_ref is not None:
        omega_ref_override = float(args.omega_ref)
    elif args.omega_ref_pu is not None:
        env_cfg = make_env_from_config(args.config).env_config
        omega_base = float(2.0 * np.pi * 10.0 / max(env_cfg.motor.p, 1))
        omega_ref_override = float(args.omega_ref_pu) * omega_base

    scenarios = _parse_scenarios(args.scenarios)
    omega_ref_range = _parse_range(args.omega_ref_range)
    omega_ref_pu_range = _parse_range(args.omega_ref_pu_range)
    load_range = _parse_range(args.load_torque_range)
    load_mult_range = _parse_range(args.load_mult_range)
    override_omega_ref = bool(args.override_omega_ref)
    override_load_torque = bool(args.override_load_torque)
    if scenarios:
        override_omega_ref = False
        override_load_torque = False

    env_cfg = make_env_from_config(args.config).env_config
    if omega_ref_range is None and omega_ref_pu_range is not None:
        omega_base = float(2.0 * np.pi * 10.0 / max(env_cfg.motor.p, 1))
        omega_ref_range = (omega_ref_pu_range[0] * omega_base, omega_ref_pu_range[1] * omega_base)
    if load_range is None and load_mult_range is not None:
        base_load = float(getattr(env_cfg.sim, "load_torque", 0.0))
        load_range = (load_mult_range[0] * base_load, load_mult_range[1] * base_load)

    cfg_omega_range = _normalize_range(getattr(env_cfg, "ai_omega_ref_range", None))
    cfg_omega_pu_range = _normalize_range(getattr(env_cfg, "ai_omega_ref_pu_range", None))
    cfg_load_range = _normalize_range(getattr(env_cfg, "ai_load_torque_range", None))
    cfg_load_mult = _normalize_range(getattr(env_cfg, "ai_load_mult_range", None))
    if omega_ref_range is None and cfg_omega_range is not None:
        omega_ref_range = cfg_omega_range
    if omega_ref_range is None and cfg_omega_pu_range is not None:
        omega_base = float(2.0 * np.pi * 10.0 / max(env_cfg.motor.p, 1))
        omega_ref_range = (cfg_omega_pu_range[0] * omega_base, cfg_omega_pu_range[1] * omega_base)
    if load_range is None and cfg_load_range is not None:
        load_range = cfg_load_range
    if load_range is None and cfg_load_mult is not None:
        base_load = float(getattr(env_cfg.sim, "load_torque", 0.0))
        load_range = (cfg_load_mult[0] * base_load, cfg_load_mult[1] * base_load)

    train(
        env_config=args.config,
        episodes=args.episodes,
        episode_steps=args.episode_steps,
        control_mode=str(args.control_mode),
        w_speed=args.w_speed,
        w_power=args.w_power,
        w_current=args.w_current,
        w_smooth=args.w_smooth,
        w_mag=args.w_mag,
        w_shaft=args.w_shaft,
        w_eta=args.w_eta,
        eta_clip=args.eta_clip,
        id_ref_alpha=float(args.id_ref_alpha),
        id_ref_rate_limit=None if args.id_ref_rate_limit is None else float(args.id_ref_rate_limit),
        ai_id_speed_tol=float(args.ai_id_speed_tol),
        ai_id_speed_tol_rel=None if args.ai_id_speed_tol_rel is None else float(args.ai_id_speed_tol_rel),
        id_ref_gate_speed_tol=None if args.id_ref_gate_speed_tol is None else float(args.id_ref_gate_speed_tol),
        id_ref_gate_speed_tol_rel=None if args.id_ref_gate_speed_tol_rel is None else float(args.id_ref_gate_speed_tol_rel),
        id_ref_gate_min_scale=float(args.id_ref_gate_min_scale),
        id_ref_gate_exponent=float(args.id_ref_gate_exponent),
        fast=bool(args.fast),
        time_budget_min=args.time_budget_min,
        override_load_torque=override_load_torque,
        override_omega_ref=override_omega_ref,
        ai_id_ref_relative=bool(args.relative),
        delta_id_max=float(args.delta_id_max),
        load_torque=None if args.load_torque is None else float(args.load_torque),
        omega_ref_override=omega_ref_override,
        scenarios=scenarios,
        scenario_sample=str(args.scenario_sample),
        omega_ref_range=omega_ref_range,
        load_torque_range=load_range,
        seed=args.seed,
        sigma_start=float(args.sigma_start),
        sigma_end=float(args.sigma_end),
        sigma_decay_episodes=int(args.sigma_decay_episodes),
        power_warmup_episodes=int(args.power_warmup_episodes),
        power_ramp_episodes=int(args.power_ramp_episodes),
        eval_interval=int(args.eval_interval),
        eval_scenarios=str(args.eval_scenarios),
        eval_dt=None if args.eval_dt is None else float(args.eval_dt),
        eval_t_end=None if args.eval_t_end is None else float(args.eval_t_end),
        eval_window_frac=float(args.eval_window_frac),
        eval_error_tol_rel=float(args.eval_error_tol_rel),
        eval_error_tol_abs=float(args.eval_error_tol_abs),
        eval_use_total_power=bool(args.eval_use_total_power),
        include_energy_obs=bool(args.include_energy_obs),
        update_every_episodes=int(args.update_every_episodes),
        init_checkpoint=args.init_checkpoint,
    )


if __name__ == "__main__":
    main()
