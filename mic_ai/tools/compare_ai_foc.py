from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from mic_ai.ai.ai_voltage_config import get_curriculum_config, load_ai_voltage_config
from mic_ai.ai.foc_baseline import save_foc_baseline


def _load_episode_list(path: Path) -> List[Dict[str, float]]:
    if not path.is_file():
        raise FileNotFoundError(f"Episode log not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("episodes", [])
    if not isinstance(data, list):
        raise ValueError(f"Unsupported episode log format: {path}")
    return [e for e in data if isinstance(e, dict)]


def _motor_key_from_env_config(env_config: str) -> str:
    name = Path(env_config).stem.lower()
    if "motor2" in name:
        return "motor2"
    return "motor1"


def _extract_xy(episodes: List[Dict[str, float]], key: str) -> Tuple[np.ndarray, np.ndarray]:
    xs: List[int] = []
    ys: List[float] = []
    for idx, ep in enumerate(episodes):
        ep_idx = ep.get("episode", idx)
        try:
            ep_int = int(ep_idx)
        except Exception:
            ep_int = idx
        try:
            val = float(ep.get(key, np.nan))
        except Exception:
            val = float("nan")
        if np.isfinite(val):
            xs.append(ep_int)
            ys.append(val)
    if not xs:
        return np.asarray([], dtype=int), np.asarray([], dtype=float)
    order = np.argsort(xs)
    return np.asarray(xs, dtype=int)[order], np.asarray(ys, dtype=float)[order]


def _rolling_mean(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or y.size == 0:
        return y
    window = min(window, int(y.size))
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(y, kernel, mode="same")


def _ensure_matplotlib():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _plot_metric(
    out_path: Path,
    title: str,
    y_label: str,
    ai_x: np.ndarray,
    ai_y: np.ndarray,
    foc_y: np.ndarray,
    rolling_window: int,
) -> None:
    plt = _ensure_matplotlib()

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.set_title(title)
    ax.set_xlabel("Episode")
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.25)

    if ai_x.size:
        ax.plot(ai_x, _rolling_mean(ai_y, rolling_window), label=f"AI (rolling mean, w={rolling_window})", linewidth=2.0)
        ax.plot(ai_x, np.minimum.accumulate(ai_y), label="AI (best so far)", linewidth=1.5, alpha=0.9)

    if foc_y.size:
        foc_mean = float(np.mean(foc_y))
        ax.axhline(foc_mean, linestyle="--", linewidth=2.0, label=f"FOC baseline mean = {foc_mean:.3f}")

    ax.legend(loc="best")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate AI vs FOC comparison plots (speed error + current RMS).")
    p.add_argument("--env-config", default="config/env_demo_true_motor1.py", help="Env config path for the baseline.")
    p.add_argument(
        "--ai-episodes",
        default="outputs/demo_ai/episode_logs/ai_voltage_env_demo_true_motor1_episodes.json",
        help="Path to AI episode log JSON (list of episodes).",
    )
    p.add_argument("--output-dir", default="outputs/compare_ai_foc", help="Where to save comparison PNGs/JSON.")
    p.add_argument("--episode-steps", type=int, default=400, help="Episode steps for baseline episodes.")
    p.add_argument("--foc-eval-episodes", type=int, default=5, help="FOC evaluation episodes per curriculum stage.")
    p.add_argument("--rolling-window", type=int, default=20, help="Rolling mean window for AI curves.")
    return p.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)

    env_config = str(args.env_config)
    motor_key = _motor_key_from_env_config(env_config)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ai_episodes_path = Path(args.ai_episodes)
    ai_eps = _load_episode_list(ai_episodes_path)

    cfg = load_ai_voltage_config()
    curriculum = get_curriculum_config(cfg)
    foc_log_path = output_dir / f"foc_baseline_{motor_key}.json"
    save_foc_baseline(
        config_name=env_config,
        curriculum_config=curriculum,
        log_path=foc_log_path,
        n_episodes_eval=int(args.foc_eval_episodes),
        episode_steps=int(args.episode_steps),
    )
    foc_eps = _load_episode_list(foc_log_path)

    ai_x_s, ai_y_s = _extract_xy(ai_eps, "mean_speed_error")
    _, foc_y_s = _extract_xy(foc_eps, "mean_speed_error")
    ai_x_i, ai_y_i = _extract_xy(ai_eps, "mean_current_rms")
    _, foc_y_i = _extract_xy(foc_eps, "mean_current_rms")
    ai_x_p, ai_y_p = _extract_xy(ai_eps, "mean_p_in_pos")
    _, foc_y_p = _extract_xy(foc_eps, "mean_p_in_pos")

    _plot_metric(
        out_path=output_dir / "ai_vs_foc_speed_error.png",
        title=f"AI vs FOC: speed error ({motor_key})",
        y_label="mean |ω_ref - ω|",
        ai_x=ai_x_s,
        ai_y=ai_y_s,
        foc_y=foc_y_s,
        rolling_window=int(args.rolling_window),
    )
    _plot_metric(
        out_path=output_dir / "ai_vs_foc_current_rms.png",
        title=f"AI vs FOC: current RMS ({motor_key})",
        y_label="mean i_rms",
        ai_x=ai_x_i,
        ai_y=ai_y_i,
        foc_y=foc_y_i,
        rolling_window=int(args.rolling_window),
    )
    _plot_metric(
        out_path=output_dir / "ai_vs_foc_power_in_pos.png",
        title=f"AI vs FOC: input power (+) ({motor_key})",
        y_label="mean P_in (positive part)",
        ai_x=ai_x_p,
        ai_y=ai_y_p,
        foc_y=foc_y_p,
        rolling_window=int(args.rolling_window),
    )

    print("Saved comparison plots to", str(output_dir.resolve()))


if __name__ == "__main__":
    main()
