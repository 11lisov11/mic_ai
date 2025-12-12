from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root on path for direct script execution (python mic_ai/ai/train_agent.py ...)
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
from typing import Dict, List, Tuple

import numpy as np

from mic_ai.ai.ai_env import AiEnvConfig, MicAiAIEnv
from mic_ai.ai.curiosity import WorldModelCuriosity
from mic_ai.ai.simple_agent import ActorCriticAgent
from mic_ai.ai.world_model import SimpleWorldModel
from mic_ai.core.env import make_env_from_config
from mic_ai.ident.apply import load_and_apply_ident
from simulation.gym_env import InductionMotorEnv


FEATURE_KEYS = [
    "omega_norm",
    "omega_ref_norm",
    "err_norm",
    "id_norm",
    "iq_norm",
    "u_dc_norm",
    "load_torque_norm",
    "prev_delta_norm",
    "prev_delta_id_norm",
]
WORLD_INPUT_KEYS = FEATURE_KEYS + ["action_iq_norm", "action_id_norm"]
WORLD_TARGET_KEYS = ["omega_norm", "id_norm", "iq_norm"]


def _default_omega_ref(env_cfg) -> float:
    """Use a mid-range speed reference (15 Hz mechanical) unless overridden."""
    return float(2.0 * np.pi * 15.0 / max(getattr(env_cfg.motor, "p", 1), 1))


def build_ai_env(
    env_config: str,
    control_mode: str,
    enable_id_control: bool,
    ident_path: str | None = None,
    curiosity_beta: float = 0.0,
    w_int_scale: float | None = None,
    w_ext_scale: float | None = None,
) -> Tuple[MicAiAIEnv, AiEnvConfig]:
    env_sim = make_env_from_config(env_config)
    env_cfg = env_sim.env_config
    if ident_path:
        env_cfg = load_and_apply_ident(env_cfg, ident_path)
        env_sim.env_config = env_cfg

    omega_ref = _default_omega_ref(env_cfg)
    i_base = getattr(env_cfg.motor, "I_n", 1.0)
    train_horizon = min(env_cfg.sim.t_end, 0.2)
    episode_steps = int(max(train_horizon / env_cfg.sim.dt, 1))

    baseline_speed_err = float(getattr(env_cfg, "baseline_speed_err", 0.0))
    baseline_current_rms = float(getattr(env_cfg, "baseline_current_rms", 0.0))

    def _run_baseline(steps: int) -> tuple[float, float]:
        env_base = InductionMotorEnv(env_cfg)
        env_base.omega_ref_func = lambda _t, ref=omega_ref: ref
        env_base.load_torque_func = lambda _t: getattr(env_cfg.sim, "load_torque", 0.0)
        cum_err = 0.0
        cum_current = 0.0
        obs = env_base.reset()
        for _ in range(steps):
            obs, _, done, info = env_base.step(None)
            omega_meas = float(obs[0])
            omega_ref_step = float(info.get("omega_ref", omega_ref))
            err = abs(omega_ref_step - omega_meas)
            i_abc = info.get("i_abc", (0.0, 0.0, 0.0))
            i_rms = float(np.sqrt(np.mean(np.square(i_abc))))
            cum_err += err
            cum_current += i_rms
            if done:
                break
        return cum_err, cum_current

    if baseline_speed_err <= 0.0 or baseline_current_rms <= 0.0:
        baseline_speed_err, baseline_current_rms = _run_baseline(episode_steps)

    ext_scale = float(
        getattr(
            env_cfg,
            "ext_scale",
            max((baseline_speed_err + baseline_current_rms) / max(episode_steps, 1), 1.0),
        )
    )
    w_int = w_int_scale if w_int_scale is not None else float(getattr(env_cfg, "ai_w_int_scale", 0.0))
    w_ext = w_ext_scale if w_ext_scale is not None else float(getattr(env_cfg, "ai_w_ext_scale", 1.0))

    ai_env_cfg = AiEnvConfig(
        episode_steps=episode_steps,
        dt=float(env_cfg.sim.dt),
        omega_ref=omega_ref,
        w_speed_error=1.0,
        w_current_rms=0.1,
        i_base=i_base,
        control_mode=control_mode,
        delta_iq_max=float(getattr(env_cfg, "ai_delta_iq_max", 0.5)),
        delta_id_max=float(getattr(env_cfg, "ai_delta_id_max", 0.5)),
        enable_id_control=enable_id_control,
        wm_lr=float(getattr(env_cfg, "ai_wm_lr", 1e-4)),
        curiosity_beta=curiosity_beta,
        wm_batch_size=32,
        wm_update_interval=10,
        w_ext_scale=w_ext,
        w_int_scale=w_int,
        sigma_omega=float(getattr(env_cfg, "ai_sigma_omega", 0.05)),
        sigma_iq=float(getattr(env_cfg, "ai_sigma_iq", 0.03)),
        sigma_id=float(getattr(env_cfg, "ai_sigma_id", 0.03)),
        drift_every_episodes=int(getattr(env_cfg, "ai_drift_every_episodes", 0)),
        drift_scale=float(getattr(env_cfg, "ai_drift_scale", 0.0)),
        baseline_speed_err=baseline_speed_err,
        baseline_current_rms=baseline_current_rms,
        ext_scale=ext_scale,
        phase="improve" if control_mode == "ai_speed" else "explore",
        action_penalty=float(getattr(env_cfg, "ai_action_penalty", 0.01)),
        i_max=float(getattr(env_cfg, "ai_i_max", i_base)),
    )

    world_model = SimpleWorldModel(len(WORLD_INPUT_KEYS), len(WORLD_TARGET_KEYS), hidden_sizes=(64, 64), lr=ai_env_cfg.wm_lr)
    curiosity = WorldModelCuriosity(world_model, beta=ai_env_cfg.curiosity_beta)

    env = MicAiAIEnv(
        InductionMotorEnv(env_cfg),
        ai_env_cfg,
        curiosity=curiosity,
        world_model=world_model,
        world_input_keys=WORLD_INPUT_KEYS,
        world_target_keys=WORLD_TARGET_KEYS,
    )
    return env, ai_env_cfg


def _episode_loop(env: MicAiAIEnv, agent: ActorCriticAgent) -> Tuple[float, float, float, Dict[str, float]]:
    obs = env.reset()
    agent.start_episode()
    done = False
    total_reward = 0.0
    total_r_ext = 0.0
    total_r_int = 0.0
    while not done:
        action = agent.act(obs)
        obs_next, reward, done, info = env.step(action)
        agent.record_reward(reward, next_obs=obs_next)
        total_reward += float(reward)
        total_r_ext += float(info.get("r_ext", 0.0))
        total_r_int += float(info.get("r_int", 0.0))
        obs = obs_next
    metrics = env.episode_metrics()
    metrics.update({"total_r_ext": total_r_ext, "total_r_int": total_r_int, "total_reward": total_reward})
    losses = agent.update_after_episode()
    metrics.update({"actor_loss": float(losses.get("actor_loss", 0.0)), "critic_loss": float(losses.get("critic_loss", 0.0))})
    return total_reward, total_r_ext, total_r_int, metrics


def train(
    env_config: str,
    episodes: int = 50,
    control_mode: str = "ai_speed",
    enable_id_control: bool = False,
    curiosity_beta: float = 0.0,
) -> List[dict]:
    control_mode = control_mode.lower()
    w_int_override = 0.0 if control_mode == "ai_speed" else None
    env, _ = build_ai_env(
        env_config,
        control_mode=control_mode,
        enable_id_control=enable_id_control,
        ident_path=None,
        curiosity_beta=curiosity_beta if control_mode != "ai_speed" else 0.0,
        w_int_scale=w_int_override,
        w_ext_scale=None,
    )

    action_dim = 2 if enable_id_control else 1
    agent = ActorCriticAgent(
        feature_keys=FEATURE_KEYS,
        action_dim=action_dim,
        lr_actor=7e-4,
        lr_critic=7e-4,
        gamma=0.99,
        sigma=0.15,
        max_grad_norm=5.0,
    )

    history: List[dict] = []
    phase_switch = episodes // 2 if control_mode == "foc_assist" else episodes
    for ep in range(episodes):
        env.phase = "explore" if control_mode == "foc_assist" and ep < phase_switch else "improve"
        if env.phase == "improve":
            env.cfg.w_int_scale = 0.0
            env.cfg.curiosity_beta = 0.0
            env.cfg.lambda_int = 0.0
            if env.curiosity is not None:
                env.curiosity.beta = 0.0
        if control_mode == "foc_assist" and ep == phase_switch:
            agent.reset_parameters()

        total_reward, total_r_ext, total_r_int, metrics = _episode_loop(env, agent)
        metrics.update({"episode": ep})
        history.append(metrics)

        print(
            f"ep {ep:03d} | phase {env.phase:7s} | reward {total_reward:7.3f} | "
            f"r_ext {total_r_ext:7.3f} | mean|e_w| {metrics['mean_speed_error']:.5f} | "
            f"mean_i_rms {metrics['mean_current_rms']:.5f}"
        )
    return history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-config", required=True, help="Path to env config module")
    parser.add_argument("--ident", default=None, help="Path to identification JSON to apply before training")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--control-mode", choices=["ai_speed", "foc_assist"], default="ai_speed")
    parser.add_argument("--enable-id-control", action="store_true", help="Allow agent to control d-axis current")
    parser.add_argument("--curiosity-beta", type=float, default=0.0, help="Curiosity weight for explore phase")
    args = parser.parse_args()

    # Rebuild env with ident if provided.
    def _train_with_ident() -> List[dict]:
        control_mode = args.control_mode.lower()
        w_int_override = 0.0 if control_mode == "ai_speed" else None
        env, _ = build_ai_env(
            args.env_config,
            control_mode=control_mode,
            enable_id_control=bool(args.enable_id_control),
            ident_path=args.ident,
            curiosity_beta=args.curiosity_beta if control_mode != "ai_speed" else 0.0,
            w_int_scale=w_int_override,
            w_ext_scale=None,
        )
        action_dim = 2 if args.enable_id_control else 1
        agent = ActorCriticAgent(
            feature_keys=FEATURE_KEYS,
            action_dim=action_dim,
            lr_actor=7e-4,
            lr_critic=7e-4,
            gamma=0.99,
            sigma=0.15,
            max_grad_norm=5.0,
        )
        history: List[dict] = []
        phase_switch = args.episodes // 2 if control_mode == "foc_assist" else args.episodes
        for ep in range(args.episodes):
            env.phase = "explore" if control_mode == "foc_assist" and ep < phase_switch else "improve"
            if env.phase == "improve":
                env.cfg.w_int_scale = 0.0
                env.cfg.curiosity_beta = 0.0
                env.cfg.lambda_int = 0.0
                if env.curiosity is not None:
                    env.curiosity.beta = 0.0
            if control_mode == "foc_assist" and ep == phase_switch:
                agent.reset_parameters()

            total_reward, total_r_ext, total_r_int, metrics = _episode_loop(env, agent)
            metrics.update({"episode": ep})
            history.append(metrics)
            print(
                f"ep {ep:03d} | phase {env.phase:7s} | reward {total_reward:7.3f} | "
                f"r_ext {total_r_ext:7.3f} | mean|e_w| {metrics['mean_speed_error']:.5f} | "
                f"mean_i_rms {metrics['mean_current_rms']:.5f}"
            )
        return history

    hist = _train_with_ident()
    if hist:
        last = hist[-1]
        print(
            f"Done. Last episode reward={last.get('total_reward', 0.0):.3f}, "
            f"mean|e_w|={last.get('mean_speed_error', 0.0):.5f}, "
            f"mean_i_rms={last.get('mean_current_rms', 0.0):.5f}"
        )


if __name__ == "__main__":
    main()
