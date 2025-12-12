from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from mic_ai.ai.ai_voltage_config import get_reward_weights, get_success_config, load_ai_voltage_config

OUTPUT_DIR = Path("outputs/demo_ai")


def _load_episodes(path: Path) -> List[Dict[str, float]]:
    """Load a list of episode dictionaries from JSON."""
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def summarize_training(episodes: List[Dict[str, float]]) -> Dict[str, object]:
    """Aggregate basic stats from training episodes."""
    count = len(episodes)
    hard_terminated_count = sum(int(ep.get("hard_terminated", 0)) for ep in episodes)
    last_episode = episodes[-1] if episodes else {}
    first_episode = episodes[0] if episodes else {}
    high_current_eps = sum(1 for ep in episodes if ep.get("mean_current_rms", 0.0) > 0.7)
    saturation_eps = sum(1 for ep in episodes if ep.get("mean_action_norm", 0.0) >= 0.95)
    best_speed = min(episodes, key=lambda ep: ep.get("mean_speed_error", float("inf")), default={})
    best_reward = max(episodes, key=lambda ep: ep.get("mean_reward", -float("inf")), default={})
    avg_speed = float(np.mean([ep.get("mean_speed_error", 0.0) for ep in episodes])) if episodes else 0.0
    avg_current = float(np.mean([ep.get("mean_current_rms", 0.0) for ep in episodes])) if episodes else 0.0
    return {
        "count": count,
        "hard_terminated_count": hard_terminated_count,
        "last_episode": last_episode,
        "first_episode": first_episode,
        "episodes_high_current_only": high_current_eps,
        "episodes_with_saturation": saturation_eps,
        "best_mean_speed_error": {
            "episode": int(best_speed.get("episode", -1)),
            "mean_speed_error": float(best_speed.get("mean_speed_error", 0.0)),
            "mean_current_rms": float(best_speed.get("mean_current_rms", 0.0)),
            "mean_action_norm": float(best_speed.get("mean_action_norm", 0.0)),
            "mean_reward": float(best_speed.get("mean_reward", 0.0)),
        },
        "best_mean_reward": {
            "episode": int(best_reward.get("episode", -1)),
            "mean_reward": float(best_reward.get("mean_reward", 0.0)),
            "mean_speed_error": float(best_reward.get("mean_speed_error", 0.0)),
            "mean_current_rms": float(best_reward.get("mean_current_rms", 0.0)),
            "mean_action_norm": float(best_reward.get("mean_action_norm", 0.0)),
        },
        "avg_mean_speed_error": avg_speed,
        "avg_mean_current_rms": avg_current,
    }


def summarize_eval(episodes: List[Dict[str, float]]) -> Dict[str, float | int]:
    """Aggregate evaluation episode metrics."""
    count = len(episodes)
    avg_speed = float(np.mean([ep.get("mean_speed_error", 0.0) for ep in episodes])) if episodes else 0.0
    avg_current = float(np.mean([ep.get("mean_current_rms", 0.0) for ep in episodes])) if episodes else 0.0
    avg_reward = float(np.mean([ep.get("mean_reward", 0.0) for ep in episodes])) if episodes else 0.0
    return {
        "count": count,
        "avg_mean_speed_error": avg_speed,
        "avg_mean_current_rms": avg_current,
        "avg_mean_reward": avg_reward,
    }


def summarize_baseline(episodes: List[Dict[str, float]]) -> Dict[str, float | int]:
    """Aggregate baseline FOC metrics."""
    count = len(episodes)
    avg_speed = float(np.mean([ep.get("mean_speed_error", 0.0) for ep in episodes])) if episodes else 0.0
    avg_current = float(np.mean([ep.get("mean_current_rms", 0.0) for ep in episodes])) if episodes else 0.0
    return {"count": count, "avg_mean_speed_error": avg_speed, "avg_mean_current_rms": avg_current}


def _success_score(
    summary: Dict[str, object], motor_key: str, success_cfg: Dict[str, object]
) -> Tuple[bool, float, str]:
    """Compute success flag, score, and human-readable reason."""
    if int(summary.get("count", 0)) <= 0:
        return False, 0.0, "no episodes"
    avg_speed = float(summary.get("avg_mean_speed_error", 0.0))
    avg_current = float(summary.get("avg_mean_current_rms", 0.0))

    best_block = summary.get("best_mean_speed_error", {}) if isinstance(summary.get("best_mean_speed_error", {}), dict) else {}
    best_speed = float(best_block.get("mean_speed_error", avg_speed))
    best_current = float(best_block.get("mean_current_rms", avg_current))

    w_speed = float(success_cfg.get("w_speed", 1.0))
    w_current = float(success_cfg.get("w_current", 5.0))
    speed_tol = float(success_cfg.get("speed_tol", 0.5))

    # Prefer the per-run current threshold if training logged it (i_soft_limit).
    first_ep = summary.get("first_episode", {}) if isinstance(summary.get("first_episode", {}), dict) else {}
    current_threshold = float(first_ep.get("i_soft_limit", 0.0)) if first_ep else 0.0
    if current_threshold <= 0.0:
        current_threshold = 0.4

    if best_speed > speed_tol:
        score = w_speed * best_speed
    else:
        score = w_speed * best_speed + w_current * max(0.0, best_current - current_threshold)

    reasons = []
    if best_speed > speed_tol:
        reasons.append(f"best_mean_speed_error > {speed_tol}")
    if best_speed <= speed_tol and best_current > current_threshold:
        reasons.append(f"best_mean_current_rms > {current_threshold}")

    success = not reasons
    reason_text = " and ".join(reasons)
    return success, float(score), reason_text


def run_foc_baseline(motor_config: dict, n_episodes: int) -> Dict[str, object]:
    """
    Placeholder interface for future FOC baseline evaluation.
    Returns a stub dict to keep report structure stable.
    """
    return {"implemented": False, "config": motor_config, "n_episodes": n_episodes}


def build_ai_voltage_report(
    motor_runs: Dict[str, Dict[str, object]],
    eval_runs: Dict[str, str | Path] | None = None,
    baseline_runs: Dict[str, str | Path] | None = None,
    config: Dict | None = None,
    output_path: str | Path = OUTPUT_DIR / "ai_voltage_report.json",
    distillation: Dict[str, object] | None = None,
) -> Dict[str, object]:
    """
    Build ai_voltage training report and persist it to JSON.
    """
    cfg = load_ai_voltage_config() if config is None else config
    success_cfg = get_success_config(cfg)
    reward_cfg = cfg.get("reward", {})

    report: Dict[str, object] = {}
    baseline_block: Dict[str, object] = {}
    comparison: Dict[str, object] = {}

    # Baseline aggregation
    if baseline_runs:
        for motor_key in ("motor1", "motor2"):
            path = Path(baseline_runs.get(motor_key, "")) if isinstance(baseline_runs, dict) else None
            eps = _load_episodes(path) if path else []
            baseline_block[motor_key] = summarize_baseline(eps) | {"episodes_file": str(path) if path else ""}
    report["baseline"] = baseline_block if baseline_block else {"implemented": False}

    for motor_key in ("motor1", "motor2"):
        motor_info = motor_runs.get(motor_key, {})
        episodes_file = Path(motor_info.get("episodes", OUTPUT_DIR / f"ai_voltage_env_demo_true_{motor_key}_episodes.json"))
        learning_file = Path(motor_info.get("learning", OUTPUT_DIR / f"ai_voltage_learning_env_demo_true_{motor_key}.npz"))
        episodes = _load_episodes(episodes_file)
        summary = summarize_training(episodes)
        success, score, reason = _success_score(summary, motor_key, success_cfg)
        reward_weights = get_reward_weights(cfg, motor_key)
        best_block = {
            "episode": motor_info.get("best_episode", summary.get("best_mean_speed_error", {}).get("episode", -1)),
            "score": motor_info.get("best_score", 0.0),
            "mean_speed_error": motor_info.get("best_mean_speed_error", summary.get("best_mean_speed_error", {}).get("mean_speed_error", 0.0)),
            "mean_current_rms": motor_info.get("best_mean_current_rms", summary.get("best_mean_speed_error", {}).get("mean_current_rms", 0.0)),
        }

        motor_block = {
            **summary,
            "train_avg_mean_speed_error": summary.get("avg_mean_speed_error", 0.0),
            "train_avg_mean_current_rms": summary.get("avg_mean_current_rms", 0.0),
            "best_episode_mean_speed_error": best_block["mean_speed_error"],
            "best_episode_mean_current_rms": best_block["mean_current_rms"],
            "episodes_file": str(episodes_file),
            "learning_file": str(learning_file),
            "success": success,
            "score": score,
            "reason": reason,
            "reward_weights": reward_weights,
            "best": best_block,
        }
        report[motor_key] = motor_block

        # Comparison RL vs baseline
        baseline_stats = baseline_block.get(motor_key, {}) if baseline_block else {}
        foc_speed = baseline_stats.get("avg_mean_speed_error", 0.0)
        foc_current = baseline_stats.get("avg_mean_current_rms", 0.0)
        rl_under = bool((foc_speed and foc_speed < best_block["mean_speed_error"]) or (foc_current and foc_current < best_block["mean_current_rms"]))
        comparison[motor_key] = {
            "rl_best_speed_error": best_block["mean_speed_error"],
            "rl_best_current_rms": best_block["mean_current_rms"],
            "foc_avg_speed_error": foc_speed,
            "foc_avg_current_rms": foc_current,
            "rl_underperforms_foc": rl_under,
        }
        report[f"{motor_key}_best"] = best_block

    if eval_runs:
        for motor_key in ("motor1", "motor2"):
            eval_path = Path(eval_runs.get(motor_key, "")) if isinstance(eval_runs, dict) else None
            eval_eps = _load_episodes(eval_path) if eval_path else []
            eval_summary = summarize_eval(eval_eps)
            eval_summary["episodes_file"] = str(eval_path) if eval_path else ""
            report[f"{motor_key}_eval"] = eval_summary

    report["comparison"] = comparison
    report["config"] = {
        "reward": reward_cfg,
        "success": success_cfg,
    }
    if distillation is not None:
        report["distillation"] = distillation

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report


def print_summary(report_path: str | Path = OUTPUT_DIR / "final_report.json") -> None:
    """
    Print a compact RL vs FOC summary for motor1 and motor2.
    """
    path = Path(report_path)
    if not path.is_file():
        print(f"[summary] report not found at {path}")
        return
    with path.open("r", encoding="utf-8") as f:
        report = json.load(f)

    comparison = report.get("comparison", {}) if isinstance(report, dict) else {}
    for motor_key in ("motor1", "motor2"):
        comp = comparison.get(motor_key, {}) if isinstance(comparison, dict) else {}
        rl_spd = comp.get("rl_best_speed_error", 0.0)
        rl_cur = comp.get("rl_best_current_rms", 0.0)
        foc_spd = comp.get("foc_avg_speed_error", 0.0)
        foc_cur = comp.get("foc_avg_current_rms", 0.0)
        under = bool(comp.get("rl_underperforms_foc", False))
        conclusion = "RL worse than FOC" if under else "RL >= FOC"
        print(
            f"{motor_key}: speed_error RL/FOC = {rl_spd:.4f} / {foc_spd:.4f}, "
            f"current RL/FOC = {rl_cur:.4f} / {foc_cur:.4f} -> {conclusion}"
        )


def main() -> None:
    """CLI entry-point to regenerate ai_voltage report with defaults."""
    motor_runs = {
        "motor1": {
            "episodes": OUTPUT_DIR / "ai_voltage_env_demo_true_motor1_episodes.json",
            "learning": OUTPUT_DIR / "ai_voltage_learning_env_demo_true_motor1.npz",
        },
        "motor2": {
            "episodes": OUTPUT_DIR / "ai_voltage_env_demo_true_motor2_episodes.json",
            "learning": OUTPUT_DIR / "ai_voltage_learning_env_demo_true_motor2.npz",
        },
    }
    eval_runs = {
        "motor1": OUTPUT_DIR / "ai_voltage_eval_demo_true_motor1_episodes.json",
        "motor2": OUTPUT_DIR / "ai_voltage_eval_demo_true_motor2_episodes.json",
    }
    build_ai_voltage_report(motor_runs=motor_runs, eval_runs=eval_runs, config=load_ai_voltage_config())


if __name__ == "__main__":
    main()
