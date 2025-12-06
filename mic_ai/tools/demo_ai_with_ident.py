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
from mic_ai.ai.curiosity import SimpleCuriosityModule
from mic_ai.ai.simple_agent import SimpleAdaptiveAgent
from mic_ai.core.env import make_env_from_config
from mic_ai.ident.auto_id import run_full_identification
from mic_ai.ident.io import save_ident_result
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

    omega_ref = float(2.0 * np.pi * 15.0 / max(env_cfg.motor.p, 1))
    i_base = getattr(env_cfg.motor, "I_n", 1.0)
    ai_env_cfg = AiEnvConfig(
        episode_steps=int(max(150, 0.5 / env_cfg.sim.dt)),
        dt=float(env_cfg.sim.dt),
        omega_ref=omega_ref,
        w_speed_error=1.0,
        w_current_rms=0.1,
        i_base=i_base,
        lambda_int=0.05,
        control_mode="foc_assist",
        delta_iq_max=0.2,
    )
    training_env = InductionMotorEnv(env_cfg)
    curiosity = SimpleCuriosityModule(weight=0.05)
    ai_env = MicAiAIEnv(training_env, ai_env_cfg, curiosity=curiosity)
    agent = SimpleAdaptiveAgent(
        feature_keys=["omega_norm", "omega_ref_norm", "err_norm", "id_norm", "iq_norm", "prev_delta_norm"],
        lr=1e-3,
        action_scale=1.0,
    )

    episodes: List[Dict[str, Any]] = []
    for ep_idx in range(num_episodes):
        obs = ai_env.reset()
        agent.start_episode()
        t_series: List[float] = []
        omega_series: List[float] = []
        omega_ref_series: List[float] = []
        reward_series: List[float] = []
        delta_series: List[float] = []
        speed_err_series: List[float] = []
        current_norm_series: List[float] = []

        done = False
        while not done:
            delta_rel = agent.act(obs)
            obs_next, reward, done, info = ai_env.step(delta_rel)
            agent.record_reward(reward)
            t_series.append(info.get("t", len(t_series) * ai_env_cfg.dt))
            omega_series.append(info.get("omega_meas", 0.0))
            omega_ref_series.append(info.get("omega_ref", omega_ref))
            reward_series.append(float(reward))
            delta_series.append(float(info.get("delta_iq_rel", delta_rel)))
            speed_err_series.append(float(info.get("speed_err", 0.0)))
            current_norm_series.append(float(info.get("current_norm", 0.0)))
            obs = obs_next

        loss = float(agent.update_after_episode())
        mean_err = float(np.mean(np.abs(np.array(speed_err_series)))) if speed_err_series else 0.0
        omega_final = float(omega_series[-1]) if omega_series else 0.0
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
        print("Episode | total_reward | mean_speed_error | omega_final | loss")
        for ep in episodes:
            print(
                f"{ep['episode_idx']:7d} | {ep['total_reward']:12.4f} | "
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
