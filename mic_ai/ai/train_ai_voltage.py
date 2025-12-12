from __future__ import annotations

import sys
from pathlib import Path
import time

# Ensure project root on path for direct script execution (python mic_ai/ai/train_ai_voltage.py ...)
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import shutil
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import torch

from mic_ai.ai.ai_env import AiEnvConfig, MicAiAIEnv
from mic_ai.ai.ai_voltage_config import (
    get_curriculum_config,
    get_exploration_config,
    get_reward_weights,
    get_success_config,
    get_voltage_scale,
    load_ai_voltage_config,
)
from mic_ai.ai.agents.ppo_voltage import PPOVoltageAgent
from mic_ai.ai.foc_baseline import save_foc_baseline
from mic_ai.core.env import make_env_from_config
from mic_ai.ident.apply import load_and_apply_ident
from mic_ai.tools.ai_voltage_report import build_ai_voltage_report, print_summary
from simulation.gym_env import InductionMotorEnv

FEATURE_KEYS = [
    "omega_norm",
    "omega_ref_norm",
    "err_norm",
    "id_norm",
    "iq_norm",
    "slip_norm",
    "last_action_vd",
    "last_action_vq",
]
WORLD_INPUT_KEYS: List[str] = []
WORLD_TARGET_KEYS: List[str] = []
OUTPUT_DIR = Path("outputs/demo_ai")
EPISODE_LOG_DIR = OUTPUT_DIR / "episode_logs"
PLOTS_DIR = OUTPUT_DIR / "plots"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
RESULTS_ROOT = Path("results_run")
AI_VOLTAGE_CFG = load_ai_voltage_config()


def _motor_key_from_config(config_name: str) -> str:
    """Infer motor key (motor1/motor2) from config filename."""
    stem = Path(config_name).stem.lower()
    if "motor1" in stem:
        return "motor1"
    if "motor2" in stem:
        return "motor2"
    # Allow arbitrary config names; fall back to a generic key.
    return "custom"


def resolve_config_path(config_name: str) -> Path:
    path = Path(config_name)
    if path.is_file():
        return path
    candidate = Path("config") / f"{config_name}.py"
    if candidate.is_file():
        return candidate.resolve()
    raise FileNotFoundError(f"Cannot find config file for {config_name}")


def _prepare_output_file(path: Path) -> Path:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = path.with_name(f"{path.stem}_backup_{ts}{path.suffix}")
        path.rename(backup)
        print(f"[log] existing {path.name} -> backup {backup.name}")
    return path


def build_env(
    env_config_path: Path,
    episode_steps: int = 400,
    reward_weights: Dict[str, float] | None = None,
    curriculum_cfg: Dict[str, object] | None = None,
    voltage_scale: float | None = None,
    motor_key: str | None = None,
    ident_path: str | None = None,
) -> MicAiAIEnv:
    env_sim = make_env_from_config(str(env_config_path))
    env_cfg = env_sim.env_config
    if ident_path:
        env_cfg = load_and_apply_ident(env_cfg, ident_path)
        env_sim.env_config = env_cfg
    omega_ref = float(2.0 * np.pi * 10.0 / max(env_cfg.motor.p, 1))
    i_base = float(getattr(env_cfg.motor, "I_n", 1.0))
    steps = int(max(episode_steps, 1))
    omega_ref_max = max(abs(omega_ref) * 1.2, 1e-3)
    i_limit = float(max(getattr(getattr(env_cfg, "foc", None), "iq_limit", i_base * 8.0), i_base * 8.0, 5.0))
    weights = reward_weights or {}
    curriculum_cfg = curriculum_cfg or {}
    piecewise_steps = curriculum_cfg.get("piecewise_steps", (150, 300))
    piecewise_multipliers = curriculum_cfg.get("piecewise_multipliers", (1.0, 0.5, 1.2))
    curriculum_stages = curriculum_cfg.get("omega_pu_stages", (0.3, 0.5, 0.7))
    stage_boundaries = curriculum_cfg.get("stage_episode_boundaries", (150, 300, 450))

    motor_key = motor_key or _motor_key_from_config(str(env_config_path))
    success_cfg = get_success_config(AI_VOLTAGE_CFG)
    i_nom_cfg = success_cfg.get("I_nom", {}) if isinstance(success_cfg, dict) else {}
    i_nom_mul = float(i_nom_cfg.get(motor_key, 1.0)) if isinstance(i_nom_cfg, dict) else 1.0
    current_tol = float(success_cfg.get("current_tol", 0.2)) if isinstance(success_cfg, dict) else 0.2
    speed_tol = float(success_cfg.get("speed_tol", 0.5)) if isinstance(success_cfg, dict) else 0.5
    i_soft_limit = i_nom_mul * i_base * (1.0 + current_tol)
    i_soft_limit = float(np.clip(i_soft_limit, 0.1 * i_limit, 0.9 * i_limit))

    ai_cfg = AiEnvConfig(
        episode_steps=steps,
        dt=float(env_cfg.sim.dt),
        omega_ref=omega_ref,
        omega_ref_max=omega_ref_max,
        w_speed_error=0.0,
        w_current_rms=0.0,
        i_base=i_base,
        control_mode="ai_voltage",
        v_max=float(voltage_scale) if voltage_scale is not None else 0.0,
        i_max=i_limit,
        reward_clip=5.0,
        w_int_scale=0.0,
        curiosity_beta=0.0,
        wm_lr=1e-4,
        w_action_delta=0.0,
        w_action_activity=0.0,
        w_ai_voltage_speed=float(weights.get("w_speed", 0.3)),
        w_ai_voltage_current=float(weights.get("w_current", 0.05)),
        w_ai_voltage_action=float(weights.get("w_action", 0.02)),
        ai_voltage_speed_tol=speed_tol,
        curriculum_omega_pu=tuple(float(x) for x in curriculum_stages),
        curriculum_stage_episodes=tuple(int(x) for x in stage_boundaries),
        omega_piecewise_steps=tuple(int(x) for x in piecewise_steps),
        omega_piecewise_multipliers=tuple(float(x) for x in piecewise_multipliers),
        i_soft_limit=i_soft_limit,
        i_soft_penalty=0.0,
        i_hard_limit=i_limit,
        reward_min=-10.0,
        reward_max=1.0,
        reward_min_td3=-10.0,
        reward_max_td3=1.0,
    )

    base_env = InductionMotorEnv(env_cfg)
    base_env.omega_ref_func = lambda _t, ref=omega_ref: ref
    base_env.load_torque_func = lambda _t: getattr(env_cfg.sim, "load_torque", 0.0)

    env = MicAiAIEnv(
        base_env,
        ai_cfg,
        curiosity=None,
        world_model=None,
        world_input_keys=WORLD_INPUT_KEYS,
        world_target_keys=WORLD_TARGET_KEYS,
    )
    return env


def plot_curves(path: Path, episodes: np.ndarray, curves: Dict[str, np.ndarray]) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:  # pragma: no cover
        print(f"Plot skipped ({exc}) for {path}")
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    items = list(curves.items())
    for ax, (name, values) in zip(axes.ravel(), items):
        ax.plot(episodes, values)
        ax.set_title(name)
        ax.set_xlabel("Episode")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _run_eval_episodes(
    agent: PPOVoltageAgent, env: MicAiAIEnv, episodes: int, episode_steps: int, log_path: Path
) -> Tuple[Path, List[Dict[str, float]]]:
    """
    Run evaluation episodes with deterministic policy and persist metrics.
    """
    logs: List[Dict[str, float]] = []
    for ep in range(episodes):
        obs = env.reset()
        done = False
        total_reward = 0.0
        step_counter = 0

        while not done and step_counter < episode_steps:
            action, _logp, _value = agent.act(obs)
            obs, reward, done, info = env.step(action)
            reward = float(np.clip(reward, env.cfg.reward_min_td3, env.cfg.reward_max_td3))
            total_reward += reward
            step_counter += 1

        metrics_env = env.episode_metrics()
        steps = int(metrics_env.get("steps", step_counter))
        mean_reward = total_reward / max(steps, 1)
        entry = {
            "episode": int(ep),
            "total_reward": float(metrics_env.get("total_reward", total_reward)),
            "total_reward_raw": float(metrics_env.get("total_reward_raw", total_reward)),
            "mean_reward": float(mean_reward),
            "mean_speed_error": float(metrics_env.get("mean_speed_error", 0.0)),
            "mean_current_rms": float(metrics_env.get("mean_current_rms", 0.0)),
            "mean_action_norm": float(metrics_env.get("action_norm", 0.0)),
            "steps": steps,
            "hard_terminated": int(metrics_env.get("hard_terminated", 0)),
            "wm_loss_mean": float(metrics_env.get("wm_loss_mean", 0.0)),
        }
        logs.append(entry)

    eval_path = _prepare_output_file(Path(log_path))
    with eval_path.open("w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2)
    return eval_path, logs


def train_ai_voltage(
    config_name: str,
    num_episodes: int = 600,
    episode_steps: int = 400,
    episodes_log_path: str | Path = EPISODE_LOG_DIR / "ai_voltage_env_demo_true_motor1_episodes.json",
    learning_log_path: str | Path = PLOTS_DIR / "ai_voltage_learning_env_demo_true_motor1.npz",
    results_dir: str | Path | None = None,
    device: str = "cpu",
    ident_path: str | None = None,
    max_seconds: float | None = None,
    target_mean_speed_error: float | None = None,
    target_patience: int = 5,
    fast: bool = False,
) -> Dict[str, object]:
    env_path = resolve_config_path(config_name)
    motor_key = _motor_key_from_config(config_name)
    reward_weights = get_reward_weights(AI_VOLTAGE_CFG, motor_key)
    curriculum_cfg = get_curriculum_config(AI_VOLTAGE_CFG)
    exploration_cfg = get_exploration_config(AI_VOLTAGE_CFG)
    voltage_scale = get_voltage_scale(AI_VOLTAGE_CFG, motor_key)
    env = build_env(
        env_path,
        episode_steps=episode_steps,
        reward_weights=reward_weights,
        curriculum_cfg=curriculum_cfg,
        voltage_scale=voltage_scale,
        motor_key=motor_key,
        ident_path=ident_path,
    )

    hidden_sizes = (64, 64) if fast else (128, 128)
    train_epochs = 3 if fast else 5
    minibatch_frac = 0.5 if fast else 0.25
    agent = PPOVoltageAgent(
        feature_keys=FEATURE_KEYS,
        action_dim=2,
        device=device,
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

    motor_ckpt_dir = CHECKPOINT_DIR / motor_key
    motor_ckpt_dir.mkdir(parents=True, exist_ok=True)

    episodes_log: List[Dict[str, float]] = []
    epi_idx: List[int] = []
    arr_total_reward: List[float] = []
    arr_mean_reward: List[float] = []
    arr_speed_err: List[float] = []
    arr_current_rms: List[float] = []
    arr_action_norm: List[float] = []
    arr_actor_loss: List[float] = []
    arr_value_loss: List[float] = []
    arr_steps: List[int] = []
    arr_hard: List[int] = []
    arr_sigma: List[float] = []
    arr_omega_mean: List[float] = []
    arr_delta_speed: List[float] = []
    success_cfg = get_success_config(AI_VOLTAGE_CFG)
    score_w_speed = float(success_cfg.get("w_speed", 1.0)) if isinstance(success_cfg, dict) else 1.0
    score_w_current = float(success_cfg.get("w_current", 5.0)) if isinstance(success_cfg, dict) else 5.0
    score_speed_tol = float(success_cfg.get("speed_tol", 0.5)) if isinstance(success_cfg, dict) else 0.5
    score_i_soft = float(getattr(env.cfg, "i_soft_limit", 0.4))

    best_episode = -1
    best_speed_err = float("inf")
    best_current = float("inf")
    best_score = float("inf")
    best_ckpt_path: Path | None = None
    sigma_start = float(exploration_cfg.get("sigma_start", 0.3))
    sigma_end = float(exploration_cfg.get("sigma_end", 0.05))
    sigma_decay = max(int(exploration_cfg.get("sigma_decay_episodes", 100)), 1)

    # Efficiency curriculum: first learn to track speed, then gradually tighten current.
    w_current_base = float(getattr(env.cfg, "w_ai_voltage_current", 0.0))
    i_soft_final = float(getattr(env.cfg, "i_soft_limit", 0.0))
    i_hard = float(getattr(env.cfg, "i_hard_limit", max(i_soft_final, 1.0)))
    i_soft_start = max(i_soft_final, 0.6 * i_hard)

    speed_tol_final = float(getattr(env.cfg, "ai_voltage_speed_tol", 0.5))
    speed_tol_start = max(speed_tol_final, 3.0)

    eff_start = max(1, int(0.25 * num_episodes))
    eff_end = max(eff_start + 1, int(0.55 * num_episodes))

    t0 = time.perf_counter()
    consecutive_success = 0

    for ep in range(num_episodes):
        if max_seconds is not None and (time.perf_counter() - t0) >= float(max_seconds):
            print(f"[{env_path.stem}] time budget reached: {float(max_seconds):.1f}s at ep {ep}")
            break

        if num_episodes <= 1:
            eff_scale = 1.0
        elif ep < eff_start:
            eff_scale = 0.0
        elif ep >= eff_end:
            eff_scale = 1.0
        else:
            eff_scale = (ep - eff_start) / max(eff_end - eff_start, 1)

        env.cfg.w_ai_voltage_current = w_current_base * eff_scale
        env.cfg.i_soft_limit = i_soft_start + (i_soft_final - i_soft_start) * eff_scale
        env.cfg.ai_voltage_speed_tol = speed_tol_start + (speed_tol_final - speed_tol_start) * eff_scale

        obs = env.reset()
        done = False
        total_reward = 0.0
        step_counter = 0
        omega_ref_series: List[float] = []
        omega_series: List[float] = []
        last_info: Dict[str, object] = {}

        if ep < 50:
            sigma = sigma_start
        else:
            frac = min(1.0, (ep - 50) / sigma_decay)
            sigma = sigma_start + (sigma_end - sigma_start) * frac
        agent.set_action_std(sigma)

        while not done and step_counter < episode_steps:
            action, logprob, value = agent.act(obs)
            obs_next, reward, done, info = env.step(action)
            last_info = info
            omega_ref_series.append(float(info.get("omega_ref", env.cfg.omega_ref)))
            omega_series.append(float(info.get("omega_meas", 0.0)))

            agent.store(obs, action, logprob, reward, done, value)

            total_reward += reward
            obs = obs_next
            step_counter += 1

        with torch.no_grad():
            last_value = float(agent.net(agent._to_tensor(obs).unsqueeze(0))[2].item())
        losses = agent.update(last_value=last_value)

        metrics_env = env.episode_metrics()
        steps = int(metrics_env.get("steps", step_counter))
        mean_reward = total_reward / max(steps, 1)

        entry = {
            "episode": int(ep),
            "total_reward": float(total_reward),
            "total_reward_raw": float(total_reward),
            "mean_reward": float(mean_reward),
            "mean_speed_error": float(metrics_env.get("mean_speed_error", 0.0)),
            "mean_current_rms": float(metrics_env.get("mean_current_rms", 0.0)),
            "mean_action_norm": float(metrics_env.get("action_norm", 0.0)),
            "i_soft_limit": float(getattr(env.cfg, "i_soft_limit", 0.0)),
            "speed_tol": float(getattr(env.cfg, "ai_voltage_speed_tol", 0.0)),
            "w_current_eff": float(getattr(env.cfg, "w_ai_voltage_current", 0.0)),
            "steps": steps,
            "hard_terminated": int(metrics_env.get("hard_terminated", 0)),
            "wm_loss_mean": float(metrics_env.get("wm_loss_mean", 0.0)),
            "actor_loss": float(losses.get("actor_loss", 0.0)),
            "value_loss": float(losses.get("value_loss", 0.0)),
            "omega_mean": float(metrics_env.get("omega_mean", 0.0)),
            "delta_speed": float(metrics_env.get("delta_speed", 0.0)),
            "momentum_mean": float(metrics_env.get("momentum_mean", 0.0)),
            "omega_ref_trace": omega_ref_series,
            "exploration_sigma": sigma,
            "done_reason": str(last_info.get("done_reason", "")) if last_info else "",
        }

        episodes_log.append(entry)
        epi_idx.append(ep)
        arr_total_reward.append(entry["total_reward"])
        arr_mean_reward.append(entry["mean_reward"])
        arr_speed_err.append(entry["mean_speed_error"])
        arr_current_rms.append(entry["mean_current_rms"])
        arr_action_norm.append(entry["mean_action_norm"])
        arr_actor_loss.append(entry["actor_loss"])
        arr_value_loss.append(entry["value_loss"])
        arr_steps.append(entry["steps"])
        arr_hard.append(entry["hard_terminated"])
        arr_sigma.append(entry["exploration_sigma"])
        arr_omega_mean.append(entry["omega_mean"])
        arr_delta_speed.append(entry["delta_speed"])

        if entry["mean_speed_error"] > score_speed_tol:
            score = score_w_speed * entry["mean_speed_error"]
        else:
            score = score_w_speed * entry["mean_speed_error"] + score_w_current * max(0.0, entry["mean_current_rms"] - score_i_soft)
        if score < best_score:
            best_score = score
            best_episode = ep
            best_speed_err = entry["mean_speed_error"]
            best_current = entry["mean_current_rms"]
            best_ckpt_path = motor_ckpt_dir / f"best_actor_ep{ep:03d}.pth"
            torch.save(agent.net.state_dict(), best_ckpt_path)

        if target_mean_speed_error is not None:
            if entry["mean_speed_error"] <= float(target_mean_speed_error):
                consecutive_success += 1
            else:
                consecutive_success = 0
            if consecutive_success >= max(int(target_patience), 1):
                print(
                    f"[{env_path.stem}] early-stop: mean_speed_error <= {float(target_mean_speed_error):.4f} "
                    f"for {consecutive_success} episodes (ep={ep})"
                )
                break

        print(
            f"[{env_path.stem}] ep {ep:03d} | steps {steps:3d} | hard {entry['hard_terminated']} | "
            f"mean_reward {entry['mean_reward']:.3f} | mean|e_w| {entry['mean_speed_error']:.4f} | "
            f"mean_i_rms {entry['mean_current_rms']:.4f} | act_norm {entry['mean_action_norm']:.3f} | "
            f"loss_pi {entry['actor_loss']:.4f} loss_v {entry['value_loss']:.4f} | sigma {sigma:.3f}"
        )

    # Persist episode logs
    episodes_path = _prepare_output_file(Path(episodes_log_path))
    learning_path = _prepare_output_file(Path(learning_log_path))

    with episodes_path.open("w", encoding="utf-8") as f:
        json.dump(episodes_log, f, indent=2)

    np.savez(
        learning_path,
        episodes=np.array(epi_idx, dtype=np.int32),
        total_reward=np.array(arr_total_reward, dtype=np.float32),
        total_reward_raw=np.array(arr_total_reward, dtype=np.float32),
        mean_reward=np.array(arr_mean_reward, dtype=np.float32),
        mean_speed_error=np.array(arr_speed_err, dtype=np.float32),
        mean_current_rms=np.array(arr_current_rms, dtype=np.float32),
        mean_action_norm=np.array(arr_action_norm, dtype=np.float32),
        actor_loss=np.array(arr_actor_loss, dtype=np.float32),
        value_loss=np.array(arr_value_loss, dtype=np.float32),
        steps=np.array(arr_steps, dtype=np.int32),
        hard_terminated=np.array(arr_hard, dtype=np.int32),
        exploration_sigma=np.array(arr_sigma, dtype=np.float32),
        omega_mean=np.array(arr_omega_mean, dtype=np.float32),
        delta_speed=np.array(arr_delta_speed, dtype=np.float32),
    )

    # results_run artifacts
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RESULTS_ROOT / f"{ts}_{Path(config_name).stem}" if results_dir is None else Path(results_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    with (run_dir / "training_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(episodes_log, f, indent=2)
    np.savez(
        run_dir / "metrics.npz",
        episodes=np.array(epi_idx, dtype=np.int32),
        total_reward=np.array(arr_total_reward, dtype=np.float32),
        total_reward_raw=np.array(arr_total_reward, dtype=np.float32),
        mean_reward=np.array(arr_mean_reward, dtype=np.float32),
        mean_speed_error=np.array(arr_speed_err, dtype=np.float32),
        mean_current_rms=np.array(arr_current_rms, dtype=np.float32),
        mean_action_norm=np.array(arr_action_norm, dtype=np.float32),
        actor_loss=np.array(arr_actor_loss, dtype=np.float32),
        value_loss=np.array(arr_value_loss, dtype=np.float32),
        steps=np.array(arr_steps, dtype=np.int32),
        hard_terminated=np.array(arr_hard, dtype=np.int32),
        exploration_sigma=np.array(arr_sigma, dtype=np.float32),
        omega_mean=np.array(arr_omega_mean, dtype=np.float32),
        delta_speed=np.array(arr_delta_speed, dtype=np.float32),
    )
    torch.save(agent.net.state_dict(), run_dir / "ppo_actor_critic.pth")

    last_actor_path = motor_ckpt_dir / "last_actor.pth"
    torch.save(agent.net.state_dict(), last_actor_path)
    best_actor_path = motor_ckpt_dir / "best_actor.pth"
    if best_ckpt_path is not None and best_ckpt_path.exists():
        shutil.copyfile(best_ckpt_path, best_actor_path)
    else:
        best_ckpt_path = best_actor_path
        torch.save(agent.net.state_dict(), best_actor_path)

    plot_curves(run_dir / "plot_speed_error.png", np.array(epi_idx), {"mean_speed_error": np.array(arr_speed_err)})
    plot_curves(run_dir / "plot_reward.png", np.array(epi_idx), {"mean_reward": np.array(arr_mean_reward)})
    plot_curves(run_dir / "plot_current_rms.png", np.array(epi_idx), {"mean_current_rms": np.array(arr_current_rms)})
    plot_curves(run_dir / "plot_action_norm.png", np.array(epi_idx), {"mean_action_norm": np.array(arr_action_norm)})

    if best_episode < 0 and arr_speed_err:
        best_episode = int(np.argmin(arr_speed_err))
        best_speed_err = float(np.min(arr_speed_err))
        best_current = float(arr_current_rms[best_episode]) if best_episode < len(arr_current_rms) else best_current
        if best_speed_err > score_speed_tol:
            best_score = float(score_w_speed * best_speed_err)
        else:
            best_score = float(score_w_speed * best_speed_err + score_w_current * max(0.0, best_current - score_i_soft))

    print(
        f"SUCCESS: training finished\n"
        f"best_episode = {best_episode}\n"
        f"best_mean_speed_error = {best_speed_err:.4f}"
    )

    return {
        "episodes": str(episodes_path),
        "learning": str(learning_path),
        "run_dir": str(run_dir),
        "reward_weights": reward_weights,
        "motor_key": motor_key,
        "best_episode": best_episode,
        "best_mean_speed_error": best_speed_err,
        "best_mean_current_rms": best_current,
        "best_score": best_score,
        "best_checkpoint": str(best_ckpt_path) if best_ckpt_path is not None else "",
        "last_checkpoint": str(last_actor_path),
        "sigma_schedule": exploration_cfg,
    }


def main() -> None:
    n_eval = int(AI_VOLTAGE_CFG.get("eval", {}).get("episodes", 5))
    curriculum_cfg = get_curriculum_config(AI_VOLTAGE_CFG)

    # motor1
    motor1_result = train_ai_voltage(
        config_name="env_demo_true_motor1",
        num_episodes=600,
        episode_steps=400,
        episodes_log_path=EPISODE_LOG_DIR / "ai_voltage_env_demo_true_motor1_episodes.json",
        learning_log_path=PLOTS_DIR / "ai_voltage_learning_env_demo_true_motor1.npz",
        results_dir=None,
        device="cpu",
    )

    # motor2
    motor2_result = train_ai_voltage(
        config_name="env_demo_true_motor2",
        num_episodes=600,
        episode_steps=400,
        episodes_log_path=EPISODE_LOG_DIR / "ai_voltage_env_demo_true_motor2_episodes.json",
        learning_log_path=PLOTS_DIR / "ai_voltage_learning_env_demo_true_motor2.npz",
        results_dir=None,
        device="cpu",
    )

    motor_runs = {"motor1": motor1_result, "motor2": motor2_result}

    baseline_paths = {
        "motor1": _prepare_output_file(EPISODE_LOG_DIR / "foc_motor1_episodes.json"),
        "motor2": _prepare_output_file(EPISODE_LOG_DIR / "foc_motor2_episodes.json"),
    }
    save_foc_baseline(
        resolve_config_path("env_demo_true_motor1"),
        curriculum_cfg,
        log_path=baseline_paths["motor1"],
        n_episodes_eval=n_eval,
        episode_steps=400,
    )
    save_foc_baseline(
        resolve_config_path("env_demo_true_motor2"),
        curriculum_cfg,
        log_path=baseline_paths["motor2"],
        n_episodes_eval=n_eval,
        episode_steps=400,
    )

    distillation_block = {
        "prepared": True,
        "teacher_actor_paths": {
            "motor1": str(CHECKPOINT_DIR / "motor1" / "best_actor.pth"),
            "motor2": str(CHECKPOINT_DIR / "motor2" / "best_actor.pth"),
        },
    }

    report_path = OUTPUT_DIR / "final_report.json"
    build_ai_voltage_report(
        motor_runs=motor_runs,
        eval_runs=None,
        baseline_runs=baseline_paths,
        config=AI_VOLTAGE_CFG,
        output_path=report_path,
        distillation=distillation_block,
    )
    print_summary(report_path=report_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train PPO AI-voltage agent.")
    parser.add_argument("config", nargs="?", help="Env config path or name (e.g., env_demo_true_motor1)")
    parser.add_argument("--ident", default=None, help="Identification JSON to apply before training")
    parser.add_argument("--episodes", type=int, default=600)
    parser.add_argument("--episode-steps", type=int, default=400)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--time-budget-min", type=float, default=None, help="Stop training after N minutes")
    parser.add_argument("--target-mean-speed-error", type=float, default=None, help="Early-stop threshold for mean|e_w|")
    parser.add_argument("--target-patience", type=int, default=5, help="Consecutive episodes to confirm target")
    parser.add_argument("--fast", action="store_true", help="Faster training settings (smaller net, fewer epochs)")
    parser.add_argument(
        "--demo-all",
        action="store_true",
        help="Run full two-motor demo + baseline + report (default when no config given).",
    )
    args = parser.parse_args()

    if args.demo_all or args.config is None:
        main()
    else:
        max_seconds = None if args.time_budget_min is None else float(args.time_budget_min) * 60.0
        train_ai_voltage(
            args.config,
            num_episodes=args.episodes,
            episode_steps=args.episode_steps,
            device=args.device,
            ident_path=args.ident,
            max_seconds=max_seconds,
            target_mean_speed_error=args.target_mean_speed_error,
            target_patience=args.target_patience,
            fast=bool(args.fast),
        )
