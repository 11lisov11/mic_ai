from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from scipy import stats
import torch

from mic_ai.tools.paper_plots_ai_vs_foc import eval_ai_checkpoint, _extract_series, _load_episode_list
from mic_ai.ai.ai_voltage_config import get_curriculum_config, load_ai_voltage_config
from mic_ai.ai.foc_baseline import save_foc_baseline
from mic_ai.ai.train_ai_voltage import _motor_key_from_config, resolve_config_path
from mic_ai.core.env import make_env_from_config


@dataclass(frozen=True)
class RunStats:
    seed: int
    overall: Dict[str, float]
    by_stage: Dict[int, Dict[str, float]]


def _ensure_matplotlib():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _set_seed(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _mean_std(x: Iterable[float]) -> Tuple[float, float]:
    arr = np.asarray(list(x), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0, 0.0
    if arr.size == 1:
        return float(arr[0]), 0.0
    return float(np.mean(arr)), float(np.std(arr, ddof=1))


def _ci95_from_mean_std(mean: float, std: float, n: int) -> float:
    if n <= 1 or not np.isfinite(std):
        return 0.0
    df = max(int(n) - 1, 1)
    tcrit = float(stats.t.ppf(0.975, df=df))
    if not np.isfinite(tcrit) or tcrit <= 0:
        tcrit = 1.96
    return float(tcrit * float(std) / np.sqrt(float(n)))


def _summarize_series(episodes: List[Dict[str, float]]) -> RunStats:
    series = _extract_series(episodes, prefer_i_rms_abc=True)
    overall = {
        "i_rms": float(np.mean(series.i_rms)) if series.i_rms.size else 0.0,
        "p_in_pos": float(np.mean(series.p_in_pos)) if series.p_in_pos.size else 0.0,
        "speed_err": float(np.mean(series.speed_err)) if series.speed_err.size else 0.0,
    }
    by_stage: Dict[int, Dict[str, float]] = {}
    for st in sorted(set(int(x) for x in series.stage.tolist() if int(x) >= 0)):
        mask = series.stage == st
        by_stage[int(st)] = {
            "i_rms": float(np.mean(series.i_rms[mask])) if np.any(mask) else 0.0,
            "p_in_pos": float(np.mean(series.p_in_pos[mask])) if np.any(mask) else 0.0,
            "speed_err": float(np.mean(series.speed_err[mask])) if np.any(mask) else 0.0,
        }
    return RunStats(seed=-1, overall=overall, by_stage=by_stage)


def _aggregate_runs(runs: List[RunStats], n_stages: int) -> Tuple[Dict[str, float], Dict[int, Dict[str, float]]]:
    overall: Dict[str, float] = {}
    by_stage: Dict[int, Dict[str, float]] = {}
    n = int(len(runs))

    for key in ("i_rms", "p_in_pos", "speed_err"):
        m, s = _mean_std(r.overall.get(key, 0.0) for r in runs)
        overall[f"{key}_mean"] = m
        overall[f"{key}_std"] = s
        overall[f"{key}_ci95"] = _ci95_from_mean_std(m, s, n)

    for st in range(int(n_stages)):
        by_stage[st] = {}
        for key in ("i_rms", "p_in_pos", "speed_err"):
            m, s = _mean_std(r.by_stage.get(st, {}).get(key, 0.0) for r in runs)
            by_stage[st][f"{key}_mean"] = m
            by_stage[st][f"{key}_std"] = s
            by_stage[st][f"{key}_ci95"] = _ci95_from_mean_std(m, s, n)
    return overall, by_stage


def _summarize_foc_single(foc_eps: List[Dict[str, float]], n_stages: int) -> Tuple[Dict[str, float], Dict[int, Dict[str, float]]]:
    series = _extract_series(foc_eps, prefer_i_rms_abc=False)
    overall = {
        "i_rms_mean": float(np.mean(series.i_rms)) if series.i_rms.size else 0.0,
        "i_rms_std": 0.0,
        "i_rms_ci95": 0.0,
        "p_in_pos_mean": float(np.mean(series.p_in_pos)) if series.p_in_pos.size else 0.0,
        "p_in_pos_std": 0.0,
        "p_in_pos_ci95": 0.0,
        "speed_err_mean": float(np.mean(series.speed_err)) if series.speed_err.size else 0.0,
        "speed_err_std": 0.0,
        "speed_err_ci95": 0.0,
    }
    by_stage: Dict[int, Dict[str, float]] = {}
    for st in range(int(n_stages)):
        mask = series.stage == st
        by_stage[st] = {
            "i_rms_mean": float(np.mean(series.i_rms[mask])) if np.any(mask) else 0.0,
            "i_rms_std": 0.0,
            "i_rms_ci95": 0.0,
            "p_in_pos_mean": float(np.mean(series.p_in_pos[mask])) if np.any(mask) else 0.0,
            "p_in_pos_std": 0.0,
            "p_in_pos_ci95": 0.0,
            "speed_err_mean": float(np.mean(series.speed_err[mask])) if np.any(mask) else 0.0,
            "speed_err_std": 0.0,
            "speed_err_ci95": 0.0,
        }
    return overall, by_stage


def _write_csv_overall(path: Path, ai: Dict[str, float], foc: Dict[str, float]) -> None:
    rows = []
    for metric in ("i_rms", "p_in_pos", "speed_err"):
        rows.append(
            {
                "metric": metric,
                "ai_mean": ai.get(f"{metric}_mean", 0.0),
                "ai_std": ai.get(f"{metric}_std", 0.0),
                "ai_ci95": ai.get(f"{metric}_ci95", 0.0),
                "foc_mean": foc.get(f"{metric}_mean", 0.0),
                "foc_std": foc.get(f"{metric}_std", 0.0),
                "foc_ci95": foc.get(f"{metric}_ci95", 0.0),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _write_csv_by_stage(path: Path, ai: Dict[int, Dict[str, float]], foc: Dict[int, Dict[str, float]], n_stages: int) -> None:
    rows = []
    for st in range(int(n_stages)):
        for metric in ("i_rms", "p_in_pos", "speed_err"):
            rows.append(
                {
                    "stage": st,
                    "metric": metric,
                    "ai_mean": ai.get(st, {}).get(f"{metric}_mean", 0.0),
                    "ai_std": ai.get(st, {}).get(f"{metric}_std", 0.0),
                    "ai_ci95": ai.get(st, {}).get(f"{metric}_ci95", 0.0),
                    "foc_mean": foc.get(st, {}).get(f"{metric}_mean", 0.0),
                    "foc_std": foc.get(st, {}).get(f"{metric}_std", 0.0),
                    "foc_ci95": foc.get(st, {}).get(f"{metric}_ci95", 0.0),
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _plot_metric_by_stage(
    out_base: Path,
    title: str,
    ylabel: str,
    ai_means: List[float],
    ai_ci95: List[float],
    foc_means: List[float],
    n_stages: int,
) -> None:
    plt = _ensure_matplotlib()
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "figure.dpi": 200,
            "savefig.dpi": 300,
        }
    )
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    xs = np.arange(int(n_stages), dtype=float)
    ax.errorbar(xs, ai_means, yerr=ai_ci95, color="black", linestyle="-", linewidth=2.2, marker="o", capsize=3, label="AI (95% ДИ)")
    ax.plot(xs, foc_means, color="black", linestyle="--", linewidth=2.0, marker="s", label="FOC")
    ax.set_title(title)
    ax.set_xlabel("Стадия")
    ax.set_ylabel(ylabel)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(i) for i in range(int(n_stages))])
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", frameon=True, framealpha=0.9)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-seed evaluation suite for paper (tables + CI plots).")
    p.add_argument("--env-config", default="config/env_demo_true_motor1.py")
    p.add_argument("--ai-checkpoint", default="outputs/demo_ai/checkpoints/motor1/last_actor.pth")
    p.add_argument("--episode-steps", type=int, default=200)
    p.add_argument("--episodes-per-stage", type=int, default=25)
    p.add_argument("--voltage-scale", type=float, default=1.25)
    p.add_argument("--seeds", type=int, default=10, help="Number of evaluation runs (different RNG seeds).")
    p.add_argument("--seed0", type=int, default=0, help="First seed value; runs use seed0..seed0+seeds-1.")
    p.add_argument("--disable-noise", action="store_true", help="Disable measurement noise in AI env during eval.")
    p.add_argument("--sigma-omega", type=float, default=None, help="Std of omega measurement noise (rad/s).")
    p.add_argument("--sigma-i", type=float, default=None, help="Std of current measurement noise (A).")
    p.add_argument("--out-dir", default="outputs/paper_eval_suite")
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
    episodes_per_stage = int(args.episodes_per_stage)
    episodes = int(episodes_per_stage) * int(n_stages)

    env_cfg = make_env_from_config(str(env_config)).env_config
    vdc = float(getattr(getattr(env_cfg, "inverter", None), "Vdc", 0.0) or 0.0)
    v_limit_ai = float(args.voltage_scale) * (0.8 * vdc / np.sqrt(3.0)) if vdc > 0 else None
    if bool(args.disable_noise):
        sigma_omega = 0.0
        sigma_i = 0.0
    else:
        sigma_omega = float(args.sigma_omega) if args.sigma_omega is not None else 0.05
        sigma_i = float(args.sigma_i) if args.sigma_i is not None else 0.03

    runs: List[RunStats] = []
    foc_runs: List[RunStats] = []
    runs_dir = out_dir / "runs"
    noise_enabled = (not bool(args.disable_noise)) and (sigma_omega > 0.0 or sigma_i > 0.0)
    for k in range(int(args.seeds)):
        seed = int(args.seed0) + k
        _set_seed(seed)

        run_dir = runs_dir / f"seed_{seed:04d}"
        run_dir.mkdir(parents=True, exist_ok=True)

        if noise_enabled:
            foc_json_seed = run_dir / f"foc_baseline_{motor_key}.json"
            save_foc_baseline(
                config_name=str(env_config),
                curriculum_config=curriculum,
                log_path=foc_json_seed,
                n_episodes_eval=int(episodes_per_stage),
                episode_steps=int(args.episode_steps),
                v_limit=v_limit_ai,
                sigma_omega=float(sigma_omega),
                sigma_i_abc=float(sigma_i),
            )
            foc_eps_seed = _load_episode_list(foc_json_seed)
            foc_stat = _summarize_series(foc_eps_seed)
            foc_runs.append(RunStats(seed=seed, overall=foc_stat.overall, by_stage=foc_stat.by_stage))

        ai_json = run_dir / f"ai_eval_{motor_key}.json"
        eval_ai_checkpoint(
            env_config=env_config,
            checkpoint=Path(args.ai_checkpoint),
            out_json=ai_json,
            episodes=episodes,
            episodes_per_stage=episodes_per_stage,
            episode_steps=int(args.episode_steps),
            voltage_scale=float(args.voltage_scale),
            disable_noise=bool(args.disable_noise),
            sigma_omega=float(sigma_omega) if not bool(args.disable_noise) else None,
            sigma_id=float(sigma_i) if not bool(args.disable_noise) else None,
            sigma_iq=float(sigma_i) if not bool(args.disable_noise) else None,
        )
        ai_eps = _load_episode_list(ai_json)
        stat = _summarize_series(ai_eps)
        runs.append(RunStats(seed=seed, overall=stat.overall, by_stage=stat.by_stage))

    if noise_enabled and foc_runs:
        foc_overall, foc_by_stage = _aggregate_runs(foc_runs, n_stages=n_stages)
    else:
        foc_json = out_dir / f"foc_baseline_{motor_key}.json"
        save_foc_baseline(
            config_name=str(env_config),
            curriculum_config=curriculum,
            log_path=foc_json,
            n_episodes_eval=int(episodes_per_stage),
            episode_steps=int(args.episode_steps),
            v_limit=v_limit_ai,
            sigma_omega=float(sigma_omega),
            sigma_i_abc=float(sigma_i),
        )
        foc_eps = _load_episode_list(foc_json)
        foc_overall, foc_by_stage = _summarize_foc_single(foc_eps, n_stages=n_stages)

    ai_overall, ai_by_stage = _aggregate_runs(runs, n_stages=n_stages)

    _write_csv_overall(out_dir / "summary_overall.csv", ai=ai_overall, foc=foc_overall)
    _write_csv_by_stage(out_dir / "summary_by_stage.csv", ai=ai_by_stage, foc=foc_by_stage, n_stages=n_stages)

    # CI/STD figures by stage
    ai_i = [ai_by_stage[s].get("i_rms_mean", 0.0) for s in range(n_stages)]
    ai_i_ci = [ai_by_stage[s].get("i_rms_ci95", 0.0) for s in range(n_stages)]
    foc_i = [foc_by_stage[s].get("i_rms_mean", 0.0) for s in range(n_stages)]
    _plot_metric_by_stage(
        out_dir / "fig_stage_Irms",
        title="Среднеквадратичное значение тока статора по стадиям (среднее и 95% ДИ по сид‑прогонам)",
        ylabel=r"$I_{\mathrm{rms}}$, А",
        ai_means=ai_i,
        ai_ci95=ai_i_ci,
        foc_means=foc_i,
        n_stages=n_stages,
    )

    ai_p = [ai_by_stage[s].get("p_in_pos_mean", 0.0) for s in range(n_stages)]
    ai_p_ci = [ai_by_stage[s].get("p_in_pos_ci95", 0.0) for s in range(n_stages)]
    foc_p = [foc_by_stage[s].get("p_in_pos_mean", 0.0) for s in range(n_stages)]
    _plot_metric_by_stage(
        out_dir / "fig_stage_Pin_pos",
        title="Положительная составляющая входной мощности по стадиям (среднее и 95% ДИ по сид‑прогонам)",
        ylabel=r"$P_{\mathrm{in}}^{+}$, Вт",
        ai_means=ai_p,
        ai_ci95=ai_p_ci,
        foc_means=foc_p,
        n_stages=n_stages,
    )

    ai_e = [ai_by_stage[s].get("speed_err_mean", 0.0) for s in range(n_stages)]
    ai_e_ci = [ai_by_stage[s].get("speed_err_ci95", 0.0) for s in range(n_stages)]
    foc_e = [foc_by_stage[s].get("speed_err_mean", 0.0) for s in range(n_stages)]
    _plot_metric_by_stage(
        out_dir / "fig_stage_speed_error",
        title="Средняя ошибка регулирования скорости по стадиям (среднее и 95% ДИ по сид‑прогонам)",
        ylabel=r"$|\omega_{\mathrm{ref}}-\omega|$, рад/с",
        ai_means=ai_e,
        ai_ci95=ai_e_ci,
        foc_means=foc_e,
        n_stages=n_stages,
    )

    meta = {
        "env_config": str(env_config),
        "ai_checkpoint": str(Path(args.ai_checkpoint).resolve()),
        "episode_steps": int(args.episode_steps),
        "episodes_per_stage": int(episodes_per_stage),
        "stages": int(n_stages),
        "episodes_total": int(episodes),
        "voltage_scale": float(args.voltage_scale),
        "disable_noise": bool(args.disable_noise),
        "sigma_omega": float(sigma_omega),
        "sigma_i": float(sigma_i),
        "seeds": [int(args.seed0) + k for k in range(int(args.seeds))],
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8-sig")
    print(f"Saved suite outputs to {out_dir}")


if __name__ == "__main__":
    main()
