from __future__ import annotations

import argparse
import sys
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np

# Ensure project root on path for direct script execution
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mic_ai.ai.ai_env import AiEnvConfig, MicAiAIEnv
from mic_ai.ai.curiosity import WorldModelCuriosity
from mic_ai.ai.world_model import SimpleWorldModel
from mic_ai.ai.simple_agent import SimpleAdaptiveAgent
from mic_ai.core.env import make_env_from_config
from mic_ai.ident.auto_id import run_full_identification
from mic_ai.ident.io import save_ident_result
from mic_ai.ident.apply import apply_estimated_params_to_env_config
from outputs.styles import apply_style
from simulation.gym_env import InductionMotorEnv


def _ident_duration_from_env(env: Any) -> float:
    total = 0.0
    for name in ("ident_total_time", "ident_locked_total_time", "ident_runup_time", "ident_coast_time"):
        if hasattr(env, name):
            total += float(getattr(env, name))
    return total if total > 0 else 1.0


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def run_demo_for_motor(env_config_path: str, output_prefix: str, num_episodes: int = 10) -> Dict[str, Any]:
    base_dir = Path(output_prefix).expanduser().resolve()
    base_dir.mkdir(parents=True, exist_ok=True)
    prefix_name = base_dir.name
    env_sim = make_env_from_config(env_config_path)
    motor_name = Path(env_config_path).stem
    ident_result = run_full_identification(env_sim, motor_name=motor_name, source="simulation")

    ident_out = base_dir / f"{prefix_name}_{motor_name}_ident.json"
    _ensure_dir(ident_out)
    save_ident_result(ident_result, str(ident_out))

    env_cfg = getattr(env_sim, "env_config", None)
    if env_cfg is None:
        raise ValueError("env_config is required inside config module for AI demo")
    # Use identified parameters for training env to mirror real workflow.
    env_cfg = apply_estimated_params_to_env_config(env_cfg, ident_result.estimated)
    env_sim.env_config = env_cfg

    omega_ref = float(2.0 * np.pi * 15.0 / max(env_cfg.motor.p, 1))
    i_base = getattr(env_cfg.motor, "I_n", 1.0)

    def _run_baseline(steps: int) -> tuple[float, float]:
        env_base = InductionMotorEnv(env_cfg)
        env_base.omega_ref_func = lambda _t, ref=omega_ref: ref
        env_base.load_torque_func = lambda _t: getattr(env_cfg.sim, "load_torque", 0.0)
        cum_err = 0.0
        cum_current = 0.0
        obs = env_base.reset()
        for _ in range(steps):
            obs, _, done, info = env_base.step(None)
            omega_meas = float(obs[0]) if hasattr(obs, "__len__") and len(obs) > 0 else float(info.get("omega_meas", 0.0))
            omega_ref_step = float(info.get("omega_ref", omega_ref))
            err = abs(omega_ref_step - omega_meas)
            i_abc = info.get("i_abc", (0.0, 0.0, 0.0))
            i_rms = float(np.sqrt(np.mean(np.square(i_abc))))
            cum_err += err
            cum_current += i_rms
            if done:
                break
        return cum_err, cum_current

    ai_env_cfg = AiEnvConfig(
        episode_steps=int(max(0.2 / env_cfg.sim.dt, 1)),
        dt=float(env_cfg.sim.dt),
        omega_ref=omega_ref,
        w_speed_error=1.0,
        w_current_rms=0.1,
        i_base=i_base,
        lambda_int=0.05,
        control_mode="foc_assist",
        delta_iq_max=getattr(env_cfg, "ai_delta_iq_max", 0.5),
        wm_lr=float(getattr(env_cfg, "ai_wm_lr", 1e-4)),
        curiosity_beta=float(getattr(env_cfg, "ai_curiosity_beta", 0.0)),
        wm_batch_size=32,
        w_ext_scale=float(getattr(env_cfg, "ai_w_ext_scale", 1.0)),
        w_int_scale=float(getattr(env_cfg, "ai_w_int_scale", 0.0)),
        sigma_omega=float(getattr(env_cfg, "ai_sigma_omega", 0.05)),
        sigma_iq=float(getattr(env_cfg, "ai_sigma_iq", 0.03)),
        sigma_id=float(getattr(env_cfg, "ai_sigma_id", 0.03)),
        drift_every_episodes=int(getattr(env_cfg, "ai_drift_every_episodes", 0)),
        drift_scale=float(getattr(env_cfg, "ai_drift_scale", 0.0)),
        enable_id_control=True,
        delta_id_max=getattr(env_cfg, "ai_delta_id_max", 0.5),
    )
    base_err = float(getattr(env_cfg, "baseline_speed_err", 0.0))
    base_curr = float(getattr(env_cfg, "baseline_current_rms", 0.0))
    if base_err <= 0.0 or base_curr <= 0.0:
        base_err, base_curr = _run_baseline(ai_env_cfg.episode_steps)
    ai_env_cfg.baseline_speed_err = base_err
    ai_env_cfg.baseline_current_rms = base_curr
    ai_env_cfg.ext_scale = float(getattr(env_cfg, "ext_scale", max((base_err + base_curr) / max(ai_env_cfg.episode_steps, 1), 1.0)))
    training_env = InductionMotorEnv(env_cfg)
    world_input_keys = [
        "omega_norm",
        "omega_ref_norm",
        "err_norm",
        "id_norm",
        "iq_norm",
        "u_dc_norm",
        "load_torque_norm",
        "prev_delta_norm",
        "prev_delta_id_norm",
        "action_iq_norm",
        "action_id_norm",
    ]
    world_target_keys = ["omega_norm", "id_norm", "iq_norm"]
    world_model = SimpleWorldModel(
        len(world_input_keys), len(world_target_keys), hidden_sizes=(32, 32), lr=ai_env_cfg.wm_lr
    )
    curiosity = WorldModelCuriosity(world_model, beta=ai_env_cfg.curiosity_beta)
    ai_env = MicAiAIEnv(
        training_env,
        ai_env_cfg,
        curiosity=curiosity,
        world_model=world_model,
        world_input_keys=world_input_keys,
        world_target_keys=world_target_keys,
    )
    agent = SimpleAdaptiveAgent(
        feature_keys=["omega_norm", "omega_ref_norm", "err_norm", "id_norm", "iq_norm", "u_dc_norm", "load_torque_norm"],
        lr=7e-4,
        gamma=0.99,
        hidden_sizes=(64, 64),
        sigma=0.15,
        action_dim=2,
    )

    episodes: List[Dict[str, Any]] = []
    best_ext = -float("inf")
    no_improve = 0
    phase_switch = max(1, num_episodes // 2)
    policy_resets = 0
    for ep_idx in range(num_episodes):
        ai_env.phase = "explore" if ep_idx < phase_switch else "improve"
        if ai_env.phase == "improve":
            ai_env.cfg.w_int_scale = 0.0
            ai_env.cfg.curiosity_beta = 0.0
            ai_env.cfg.lambda_int = 0.0
            if ai_env.curiosity is not None:
                ai_env.curiosity.beta = 0.0
        if ep_idx == phase_switch:
            agent.reset_parameters()
        obs = ai_env.reset()
        agent.start_episode()
        t_series: List[float] = []
        omega_series: List[float] = []
        omega_ref_series: List[float] = []
        reward_series: List[float] = []
        delta_series: List[float] = []
        speed_err_series: List[float] = []
        current_norm_series: List[float] = []
        ext_series: List[float] = []
        int_series: List[float] = []
        wm_series: List[float] = []

        done = False
        while not done:
            delta_rel = agent.act(obs)
            obs_next, reward, done, info = ai_env.step(delta_rel)
            agent.record_reward(reward, next_obs=obs_next)
            t_series.append(info.get("t", len(t_series) * ai_env_cfg.dt))
            omega_series.append(info.get("omega_meas", 0.0))
            omega_ref_series.append(info.get("omega_ref", omega_ref))
            reward_series.append(float(reward))
            delta_series.append(float(info.get("delta_iq_rel", delta_rel)))
            speed_err_series.append(float(info.get("speed_err", 0.0)))
            current_norm_series.append(float(info.get("current_norm", 0.0)))
            ext_series.append(float(info.get("r_ext", 0.0)))
            int_series.append(float(info.get("r_int", 0.0)))
            wm_series.append(float(info.get("wm_loss", 0.0)))
            obs = obs_next

        losses = agent.update_after_episode()
        loss = float(losses.get("actor_loss", 0.0)) if isinstance(losses, dict) else float(losses)
        mean_err = float(np.mean(np.abs(np.array(speed_err_series)))) if speed_err_series else 0.0
        omega_final = float(omega_series[-1]) if omega_series else 0.0
        total_ext = float(np.sum(ext_series)) if ext_series else 0.0
        total_int = float(np.sum(int_series)) if int_series else 0.0
        wm_loss_mean = float(np.mean(wm_series)) if wm_series else 0.0

        if total_ext > best_ext + 1e-3:
            best_ext = total_ext
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= 5:
            agent.reset_parameters()
            policy_resets += 1
            no_improve = 0

        episodes.append(
            {
                "t": t_series,
                "omega": omega_series,
                "omega_ref": omega_ref_series,
                "reward": reward_series,
                "delta_iq_rel": delta_series,
                "speed_err": speed_err_series,
                "current_norm": current_norm_series,
                "total_reward": float(np.sum(reward_series)),
                "total_ext_reward": total_ext,
                "total_int_reward": total_int,
                "wm_loss_mean": wm_loss_mean,
                "episode_idx": ep_idx,
                "mean_speed_error": mean_err,
                "omega_final": omega_final,
                "loss": loss,
                "mean_reward": float(np.mean(reward_series)) if reward_series else 0.0,
            }
        )

    episodes_out = base_dir / f"{prefix_name}_{motor_name}_episodes.json"
    _ensure_dir(episodes_out)
    with open(episodes_out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "motor_name": motor_name,
                "config_path": env_config_path,
                "ai_env_cfg": ai_env_cfg.__dict__,
                "episodes": episodes,
            },
            f,
            indent=2,
        )

    learning_out = base_dir / f"{prefix_name}_learning_{motor_name}.json"
    _ensure_dir(learning_out)
    with open(learning_out, "w", encoding="utf-8") as f:
        json.dump(
            [
                {
                    "episode_idx": ep["episode_idx"],
                    "total_reward": ep["total_reward"],
                    "mean_speed_error": ep["mean_speed_error"],
                    "omega_final": ep.get("omega_final", 0.0),
                    "loss": ep.get("loss", 0.0),
                }
                for ep in episodes
            ],
            f,
            indent=2,
        )

    def _print_diagnostics():
        print(f"\nLearning summary for motor {motor_name}")
        print("Episode | total_reward | total_ext | mean_speed_error | omega_final | loss")
        for ep in episodes:
            print(
                f"{ep['episode_idx']:7d} | {ep['total_reward']:12.4f} | "
                f"{ep.get('total_ext_reward',0.0):10.4f} | "
                f"{ep['mean_speed_error']:16.4f} | {ep.get('omega_final', 0.0):11.3f} | {ep.get('loss', 0.0):8.4f}"
            )
        if episodes:
            imp_r = episodes[-1]["total_reward"] - episodes[0]["total_reward"]
            imp_e = episodes[0]["mean_speed_error"] - episodes[-1]["mean_speed_error"]
            if imp_r <= 0 or imp_e <= 0:
                print(f"WARNING: learning did not improve for motor {motor_name} (reward/error stagnation).")
            else:
                print(
                    f"Improvement: reward {episodes[0]['total_reward']:.3f}->{episodes[-1]['total_reward']:.3f}, "
                    f"error {episodes[0]['mean_speed_error']:.4f}->{episodes[-1]['mean_speed_error']:.4f}"
                )
        print(f"Policy resets: {policy_resets}")

    _print_diagnostics()

    improvement_reward = 0.0
    improvement_error = 0.0
    if episodes:
        improvement_reward = episodes[-1]["total_reward"] - episodes[0]["total_reward"]
        improvement_error = episodes[0]["mean_speed_error"] - episodes[-1]["mean_speed_error"]

    return {
        "motor_name": motor_name,
        "config_path": env_config_path,
        "ident_result": ident_result,
        "ident_save_path": str(ident_out),
        "episodes": episodes,
        "ai_env_cfg": ai_env_cfg,
        "ident_duration": _ident_duration_from_env(env_sim),
        "episodes_save_path": str(episodes_out),
        "base_dir": str(base_dir),
        "prefix_name": prefix_name,
        "learning_save_path": str(learning_out),
        "improvement_reward": improvement_reward,
        "improvement_error": improvement_error,
        "policy_resets": policy_resets,
    }


def plot_demo_results(all_results: List[Dict[str, Any]], output_path: str) -> None:
    if not all_results:
        return
    apply_style()
    fig, (ax_time, ax_curve) = plt.subplots(2, 1, figsize=(12, 8))
    ax_curve_reward = ax_curve.twinx()
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(all_results), 1)))

    for idx, res in enumerate(all_results):
        color = colors[idx % len(colors)]
        ident_duration = float(res.get("ident_duration", 1.0))
        motor_name = res.get("motor_name", f"motor_{idx}")
        episodes = res.get("episodes", [])
        if not episodes:
            continue

        # Time-domain view: identification phase + first/last episodes.
        ax_time.axvspan(0.0, ident_duration, color=color, alpha=0.12, label=f"{motor_name} identification")
        ident_res = res.get("ident_result")
        if ident_res is not None and getattr(ident_res, "rel_error", None):
            rel = ident_res.rel_error or {}
            parts = [f"{k}:{rel[k]:.1f}%" for k in ("Rs", "Lm", "J") if k in rel]
            if not parts and rel:
                parts.append(f"max:{max(rel.values()):.1f}%")
            if parts:
                ax_time.text(
                    0.02,
                    0.9 - 0.08 * idx,
                    f"{motor_name} ID {'/'.join(parts)}",
                    transform=ax_time.transAxes,
                    fontsize=8,
                    bbox={"facecolor": "white", "alpha": 0.6, "edgecolor": "none"},
                )
        episode_dt = float(getattr(res.get("ai_env_cfg"), "dt", 0.001))
        first_ep = episodes[0]
        last_ep = episodes[-1]

        def _plot_episode(ep: Dict[str, Any], offset: float, label_suffix: str, alpha: float) -> float:
            t = np.asarray(ep.get("t", []), dtype=float)
            omega = np.asarray(ep.get("omega", []), dtype=float)
            omega_ref = np.asarray(ep.get("omega_ref", []), dtype=float)
            if t.size == 0:
                return offset
            t_offset = t + offset
            ax_time.plot(t_offset, omega, color=color, alpha=alpha, label=f"{motor_name} {label_suffix}")
            ax_time.plot(t_offset, omega_ref, color=color, linestyle="--", alpha=alpha * 0.7)
            return float(t_offset[-1])

        end_time = _plot_episode(first_ep, ident_duration + episode_dt, "early", 0.5)
        end_time = _plot_episode(last_ep, end_time + episode_dt, "stable", 0.9)
        ax_time.axvspan(end_time, end_time + ident_duration * 0.2, color=color, alpha=0.08, label=None)

        # Learning curve
        episode_idx = [ep.get("episode_idx", i) for i, ep in enumerate(episodes)]
        mean_err = [ep.get("mean_speed_error", 0.0) for ep in episodes]
        total_r = [ep.get("total_reward", 0.0) for ep in episodes]
        ax_curve.plot(episode_idx, mean_err, marker="o", color=color, label=f"{motor_name} |e_w|")
        ax_curve_reward.plot(
            episode_idx,
            total_r,
            linestyle="--",
            color=color,
            alpha=0.6,
            label=f"{motor_name} reward",
        )

    ax_time.set_title("Identification -> AI training timeline")
    ax_time.set_xlabel("Time, s")
    ax_time.set_ylabel("Speed, rad/s")
    ax_time.legend(loc="upper right")

    ax_curve.set_title("Episode metrics")
    ax_curve.set_xlabel("Episode")
    ax_curve.set_ylabel("Mean |omega-omega_ref|")
    ax_curve_reward.set_ylabel("Cumulative reward")
    ax_curve.grid(True)

    handles1, labels1 = ax_curve.get_legend_handles_labels()
    handles2, labels2 = ax_curve_reward.get_legend_handles_labels()
    ax_curve.legend(handles1 + handles2, labels1 + labels2, loc="upper right")

    fig.tight_layout()
    out_path = Path(output_path)
    _ensure_dir(out_path)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def parse_args(argv: List[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env-configs",
        nargs="+",
        required=True,
        help="Paths to motor config modules (e.g., config/env_demo_true_motor1.py).",
    )
    parser.add_argument("--output-prefix", default="outputs/demo_ai", help="Prefix for JSON/PNG outputs.")
    parser.add_argument("--episodes", type=int, default=10, help="Number of training episodes per motor.")
    return parser.parse_args(argv)


def main(argv: List[str] | None = None):
    args = parse_args(argv)
    all_results = []
    for cfg in args.env_configs:
        result = run_demo_for_motor(cfg, output_prefix=args.output_prefix, num_episodes=args.episodes)
        all_results.append(result)
    base_dir = Path(args.output_prefix).expanduser().resolve()
    base_dir.mkdir(parents=True, exist_ok=True)
    prefix_name = base_dir.name
    plot_demo_results(all_results, output_path=str(base_dir / f"{prefix_name}_plot.png"))


if __name__ == "__main__":
    main()
