from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

from mic_ai.ai.agents.ppo_voltage import PPOVoltageAgent
from mic_ai.ai.ai_voltage_config import get_curriculum_config, get_reward_weights, get_voltage_scale, load_ai_voltage_config
from mic_ai.ai.foc_baseline import save_foc_baseline
from mic_ai.ai.train_ai_voltage import FEATURE_KEYS, build_env, resolve_config_path, _motor_key_from_config
from mic_ai.core.env import make_env_from_config


@dataclass(frozen=True)
class Series:
    episodes: np.ndarray
    i_rms: np.ndarray
    p_in_pos: np.ndarray
    speed_err: np.ndarray


def _ensure_matplotlib():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _rolling_mean(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or y.size == 0:
        return y
    window = min(int(window), int(y.size))
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(y, kernel, mode="same")


def _infer_hidden_sizes(state: Dict[str, torch.Tensor]) -> tuple[int, ...] | None:
    w0 = state.get("actor_body.0.weight")
    w2 = state.get("actor_body.2.weight")
    if w0 is None or w2 is None:
        return None
    try:
        return (int(w0.shape[0]), int(w2.shape[0]))
    except Exception:
        return None


def _load_episode_list(path: Path) -> List[Dict[str, float]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("episodes", [])
    if not isinstance(data, list):
        raise ValueError(f"Unsupported JSON format: {path}")
    return [e for e in data if isinstance(e, dict)]


def _extract_series(episodes: List[Dict[str, float]], prefer_i_rms_abc: bool) -> Series:
    xs: List[int] = []
    i: List[float] = []
    p: List[float] = []
    s: List[float] = []

    for idx, ep in enumerate(episodes):
        ep_idx = ep.get("episode", idx)
        try:
            x = int(ep_idx)
        except Exception:
            x = idx
        xs.append(x)

        if prefer_i_rms_abc and "mean_i_rms_abc" in ep:
            i.append(float(ep.get("mean_i_rms_abc", 0.0)))
        else:
            i.append(float(ep.get("mean_current_rms", 0.0)))
        p.append(float(ep.get("mean_p_in_pos", 0.0)))
        s.append(float(ep.get("mean_speed_error", 0.0)))

    order = np.argsort(np.asarray(xs, dtype=int))
    return Series(
        episodes=np.asarray(xs, dtype=int)[order],
        i_rms=np.asarray(i, dtype=float)[order],
        p_in_pos=np.asarray(p, dtype=float)[order],
        speed_err=np.asarray(s, dtype=float)[order],
    )


def eval_ai_checkpoint(
    env_config: Path,
    checkpoint: Path,
    out_json: Path,
    episodes: int,
    episode_steps: int,
    voltage_scale: float | None,
    disable_noise: bool = True,
) -> Path:
    cfg = load_ai_voltage_config()
    motor_key = _motor_key_from_config(str(env_config))
    reward_weights = get_reward_weights(cfg, motor_key)
    curriculum_cfg = get_curriculum_config(cfg)
    v_scale = get_voltage_scale(cfg, motor_key) if voltage_scale is None else float(voltage_scale)

    # Enforce identical current limit to FOC (iq_limit) for reproducible comparison.
    env_cfg = make_env_from_config(str(env_config)).env_config
    foc_iq_limit = float(getattr(getattr(env_cfg, "foc", None), "iq_limit", 0.0) or 0.0)
    override_i_max = foc_iq_limit if foc_iq_limit > 0 else None

    env = build_env(
        env_config,
        episode_steps=int(episode_steps),
        reward_weights=reward_weights,
        curriculum_cfg=curriculum_cfg,
        voltage_scale=v_scale,
        motor_key=motor_key,
        ident_path=None,
        override_i_max=override_i_max,
    )
    if disable_noise:
        env.cfg.sigma_omega = 0.0
        env.cfg.sigma_id = 0.0
        env.cfg.sigma_iq = 0.0

    state = torch.load(checkpoint, map_location="cpu")
    hidden = _infer_hidden_sizes(state) or (128, 128)
    agent = PPOVoltageAgent(feature_keys=FEATURE_KEYS, action_dim=2, device="cpu", hidden_sizes=hidden)
    agent.net.load_state_dict(state)
    agent.set_action_std(1e-6)

    logs: List[Dict[str, float]] = []
    for ep in range(int(episodes)):
        obs = env.reset()
        done = False
        steps = 0
        while not done and steps < int(episode_steps):
            action, _lp, _v = agent.act(obs)
            obs, _r, done, _info = env.step(action)
            steps += 1
        m = env.episode_metrics()
        logs.append(
            {
                "episode": int(ep),
                "steps": int(m.get("steps", steps)),
                "mean_speed_error": float(m.get("mean_speed_error", 0.0)),
                "mean_current_rms": float(m.get("mean_current_rms", 0.0)),
                "mean_i_rms_abc": float(m.get("mean_i_rms_abc", m.get("mean_current_rms", 0.0))),
                "mean_p_in_pos": float(m.get("mean_p_in_pos", 0.0)),
                "hard_terminated": int(m.get("hard_terminated", 0)),
            }
        )

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2)
    return out_json


def _set_ieee_style():
    plt = _ensure_matplotlib()
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 10,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "lines.linewidth": 1.8,
            "figure.dpi": 200,
            "savefig.dpi": 300,
        }
    )
    return plt


def _save_fig(fig, path_base: Path) -> None:
    path_base.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path_base.with_suffix(".png"))
    fig.savefig(path_base.with_suffix(".pdf"))


def _axis_limits(ai: np.ndarray, foc_mean: float) -> Tuple[float, float]:
    vals = np.asarray(list(ai) + [float(foc_mean)], dtype=float)
    vmin = float(np.nanmin(vals))
    vmax = float(np.nanmax(vals))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        return (0.0, 1.0)
    pad = 0.05 * (vmax - vmin)
    return (vmin - pad, vmax + pad)


def _pick_best_ai_for_pareto(ai: Series, foc_speed_mean: float) -> int:
    # Prefer minimal P_in_pos among episodes that are not worse than FOC in speed error.
    feasible = np.where(ai.speed_err <= float(foc_speed_mean))[0]
    if feasible.size:
        return int(feasible[np.argmin(ai.p_in_pos[feasible])])
    return int(np.argmin(ai.p_in_pos))


def make_plots(
    out_dir: Path,
    ai: Series,
    foc: Series,
    window: int,
) -> Path:
    plt = _set_ieee_style()

    foc_i_mean = float(np.mean(foc.i_rms))
    foc_p_mean = float(np.mean(foc.p_in_pos))
    foc_s_mean = float(np.mean(foc.speed_err))

    ai_i_roll = _rolling_mean(ai.i_rms, window)
    ai_p_roll = _rolling_mean(ai.p_in_pos, window)
    ai_s_roll = _rolling_mean(ai.speed_err, window)

    ai_i_best = float(np.min(ai.i_rms))
    ai_p_best = float(np.min(ai.p_in_pos))
    ai_s_best = float(np.min(ai.speed_err))

    # Figure 1: Irms
    fig1, ax1 = plt.subplots(figsize=(6.6, 3.2))
    ax1.plot(ai.episodes, ai_i_roll, color="black", linestyle="-", label=f"AI (скользящее среднее, окно {window})")
    ax1.axhline(ai_i_best, color="0.35", linestyle="-", linewidth=1.4, label="AI (лучшая политика)")
    ax1.axhline(foc_i_mean, color="black", linestyle="--", label="FOC (среднее)")
    ax1.set_title("Сравнение среднеквадратичного тока статора при AI- и FOC-управлении")
    ax1.set_xlabel("Номер эпизода")
    ax1.set_ylabel(r"$I_{\mathrm{rms}},\,\mathrm{A}$")
    ax1.set_xlim(int(ai.episodes.min()), int(ai.episodes.max()))
    ax1.set_ylim(*_axis_limits(ai.i_rms, foc_i_mean))
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="best", frameon=True)
    _save_fig(fig1, out_dir / "fig1_Irms")
    plt.close(fig1)

    # Figure 2: Pin+
    fig2, ax2 = plt.subplots(figsize=(6.6, 3.2))
    ax2.plot(ai.episodes, ai_p_roll, color="black", linestyle="-", label=f"AI (скользящее среднее, окно {window})")
    ax2.axhline(ai_p_best, color="0.35", linestyle="-", linewidth=1.4, label="AI (лучшая политика)")
    ax2.axhline(foc_p_mean, color="black", linestyle="--", label="FOC (среднее)")
    ax2.set_title("Сравнение потребляемой входной мощности при AI- и FOC-управлении")
    ax2.set_xlabel("Номер эпизода")
    ax2.set_ylabel(r"$P_{\mathrm{in}}^{+},\,\mathrm{Вт}$")
    ax2.set_xlim(int(ai.episodes.min()), int(ai.episodes.max()))
    ax2.set_ylim(*_axis_limits(ai.p_in_pos, foc_p_mean))
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="best", frameon=True)
    _save_fig(fig2, out_dir / "fig2_Pin_pos")
    plt.close(fig2)

    # Figure 3: speed error
    fig3, ax3 = plt.subplots(figsize=(6.6, 3.2))
    ax3.plot(ai.episodes, ai_s_roll, color="black", linestyle="-", label=f"AI (скользящее среднее, окно {window})")
    ax3.axhline(ai_s_best, color="0.35", linestyle="-", linewidth=1.4, label="AI (лучшая политика)")
    ax3.axhline(foc_s_mean, color="black", linestyle="--", label="FOC (среднее)")
    ax3.set_title("Сравнение средней ошибки регулирования скорости при AI- и FOC-управлении")
    ax3.set_xlabel("Номер эпизода")
    ax3.set_ylabel(r"$|\omega_{\mathrm{ref}}-\omega|,\,\mathrm{рад/с}$")
    ax3.set_xlim(int(ai.episodes.min()), int(ai.episodes.max()))
    ax3.set_ylim(*_axis_limits(ai.speed_err, foc_s_mean))
    ax3.grid(True, alpha=0.25)
    ax3.legend(loc="best", frameon=True)
    _save_fig(fig3, out_dir / "fig3_speed_error")
    plt.close(fig3)

    # Figure 4: Pareto
    best_idx = _pick_best_ai_for_pareto(ai, foc_s_mean)
    fig4, ax4 = plt.subplots(figsize=(5.6, 4.2))
    ax4.scatter(ai.speed_err, ai.p_in_pos, s=22, c="black", alpha=0.65, label="AI (эпизоды)")
    ax4.scatter([foc_s_mean], [foc_p_mean], s=70, marker="D", c="none", edgecolors="black", linewidths=1.6, label="FOC (среднее)")
    ax4.scatter([ai.speed_err[best_idx]], [ai.p_in_pos[best_idx]], s=85, marker="*", c="black", label="AI (выделенная политика)")
    ax4.set_title("Парето-сравнение качества управления: ошибка скорости и входная мощность")
    ax4.set_xlabel(r"$|\omega_{\mathrm{ref}}-\omega|,\,\mathrm{рад/с}$")
    ax4.set_ylabel(r"$P_{\mathrm{in}}^{+},\,\mathrm{Вт}$")
    ax4.grid(True, alpha=0.25)
    ax4.legend(loc="best", frameon=True)
    _save_fig(fig4, out_dir / "fig4_pareto")
    plt.close(fig4)

    captions = [
        "Рис. 1. Сравнение среднеквадратичного значения тока статора при управлении асинхронным двигателем с использованием AI-контроллера и классического FOC.",
        "Рис. 2. Сравнение потребляемой входной мощности (учтена только положительная составляющая) при AI- и FOC-управлении.",
        "Рис. 3. Сравнение средней ошибки регулирования скорости |omega_ref - omega| при AI- и FOC-управлении.",
        "Рис. 4. Парето-диаграмма, демонстрирующая соотношение между точностью регулирования скорости и потребляемой входной мощностью для AI- и FOC-управления.",
    ]
    captions_path = out_dir / "captions_ru.txt"
    # Use UTF-8 with BOM for predictable Cyrillic rendering on Windows.
    captions_path.write_text("\n".join(captions) + "\n", encoding="utf-8-sig")
    return captions_path


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate IEEE-style scientific plots comparing AI vs FOC (Russian labels).")
    p.add_argument("--env-config", default="config/env_demo_true_motor1.py")
    p.add_argument("--ai-checkpoint", default="outputs/demo_ai/checkpoints/motor1/last_actor.pth")
    p.add_argument("--episode-steps", type=int, default=200)
    p.add_argument("--episodes-per-stage", type=int, default=25)
    p.add_argument("--window", type=int, default=5, help="Rolling mean window for AI curves.")
    p.add_argument("--out-dir", default="outputs/paper_figures")
    p.add_argument("--voltage-scale", type=float, default=1.25, help="Per-unit voltage_scale; 1.25 ~ full Vdc/sqrt(3) if base is 0.8.")
    p.add_argument("--disable-noise", action="store_true", help="Disable measurement noise in AI env for eval.")
    return p.parse_args(argv)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    env_config = resolve_config_path(str(args.env_config))
    motor_key = _motor_key_from_config(str(env_config))

    cfg = load_ai_voltage_config()
    curriculum = get_curriculum_config(cfg)
    n_stages = len(curriculum.get("omega_pu_stages", [0.3, 0.5]))
    episodes = int(args.episodes_per_stage) * int(n_stages)

    # Match AI voltage limit: MicAiAIEnv uses base_v_limit = 0.8*Vdc/sqrt(3) and scales it by voltage_scale (<=5 treated as pu).
    env_cfg = make_env_from_config(str(env_config)).env_config
    vdc = float(getattr(getattr(env_cfg, "inverter", None), "Vdc", 0.0) or 0.0)
    v_limit_ai = float(args.voltage_scale) * (0.8 * vdc / np.sqrt(3.0)) if vdc > 0 else None

    foc_json = out_dir / f"foc_baseline_{motor_key}.json"
    save_foc_baseline(
        config_name=str(env_config),
        curriculum_config=curriculum,
        log_path=foc_json,
        n_episodes_eval=int(args.episodes_per_stage),
        episode_steps=int(args.episode_steps),
        v_limit=v_limit_ai,
    )

    ai_json = out_dir / f"ai_eval_{motor_key}.json"
    eval_ai_checkpoint(
        env_config=env_config,
        checkpoint=Path(args.ai_checkpoint),
        out_json=ai_json,
        episodes=episodes,
        episode_steps=int(args.episode_steps),
        voltage_scale=float(args.voltage_scale),
        disable_noise=bool(args.disable_noise),
    )

    ai_eps = _load_episode_list(ai_json)
    foc_eps = _load_episode_list(foc_json)

    ai_series = _extract_series(ai_eps, prefer_i_rms_abc=True)
    foc_series = _extract_series(foc_eps, prefer_i_rms_abc=False)

    captions_path = make_plots(out_dir, ai_series, foc_series, window=int(args.window))
    print(f"Saved figures to {out_dir}")
    print(f"Saved captions to {captions_path}")


if __name__ == "__main__":
    main()
