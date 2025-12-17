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
from mic_ai.ai.train_ai_voltage import FEATURE_KEYS, _motor_key_from_config, build_env, resolve_config_path
from mic_ai.core.env import make_env_from_config


@dataclass(frozen=True)
class Series:
    episodes: np.ndarray
    stage: np.ndarray
    i_rms: np.ndarray
    p_in_pos: np.ndarray
    speed_err: np.ndarray


@dataclass(frozen=True)
class PreparedData:
    out_dir: Path
    env_config: Path
    motor_key: str
    n_stages: int
    episodes_per_stage: int
    episode_steps: int
    voltage_scale: float
    disable_noise: bool
    foc_json: Path
    ai_json: Path
    ai: Series
    foc: Series
    stage_omega_ref_rad_s: List[float]


def _ensure_matplotlib():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _rolling_mean_std(y: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=float)
    if window <= 1 or y.size == 0:
        return y, np.zeros_like(y)
    window = min(int(window), int(y.size))
    kernel = np.ones(window, dtype=float) / float(window)
    mean = np.convolve(y, kernel, mode="same")
    mean2 = np.convolve(y * y, kernel, mode="same")
    var = np.maximum(0.0, mean2 - mean * mean)
    return mean, np.sqrt(var)


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
    st: List[int] = []
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

        try:
            st.append(int(ep.get("stage", -1)))
        except Exception:
            st.append(-1)

        if prefer_i_rms_abc and "mean_i_rms_abc" in ep:
            i.append(float(ep.get("mean_i_rms_abc", 0.0)))
        else:
            i.append(float(ep.get("mean_current_rms", 0.0)))
        p.append(float(ep.get("mean_p_in_pos", 0.0)))
        s.append(float(ep.get("mean_speed_error", 0.0)))

    order = np.argsort(np.asarray(xs, dtype=int))
    return Series(
        episodes=np.asarray(xs, dtype=int)[order],
        stage=np.asarray(st, dtype=int)[order],
        i_rms=np.asarray(i, dtype=float)[order],
        p_in_pos=np.asarray(p, dtype=float)[order],
        speed_err=np.asarray(s, dtype=float)[order],
    )


def eval_ai_checkpoint(
    env_config: Path,
    checkpoint: Path,
    out_json: Path,
    episodes: int,
    episodes_per_stage: int,
    episode_steps: int,
    voltage_scale: float | None,
    disable_noise: bool = True,
    sigma_omega: float | None = None,
    sigma_id: float | None = None,
    sigma_iq: float | None = None,
    env_cfg_override: object | None = None,
) -> Path:
    cfg = load_ai_voltage_config()
    motor_key = _motor_key_from_config(str(env_config))
    reward_weights = get_reward_weights(cfg, motor_key)
    curriculum_cfg = get_curriculum_config(cfg)
    v_scale = get_voltage_scale(cfg, motor_key) if voltage_scale is None else float(voltage_scale)

    env_cfg = make_env_from_config(str(env_config)).env_config
    if env_cfg_override is not None:
        env_cfg = env_cfg_override  # type: ignore[assignment]
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
        env_cfg_override=env_cfg_override,
    )
    if disable_noise:
        env.cfg.sigma_omega = 0.0
        env.cfg.sigma_id = 0.0
        env.cfg.sigma_iq = 0.0
    else:
        if sigma_omega is not None:
            env.cfg.sigma_omega = float(sigma_omega)
        if sigma_id is not None:
            env.cfg.sigma_id = float(sigma_id)
        if sigma_iq is not None:
            env.cfg.sigma_iq = float(sigma_iq)

    stages = tuple(getattr(env.cfg, "curriculum_omega_pu", (0.3, 0.5)))
    if episodes_per_stage > 0 and len(stages) > 1:
        env.cfg.curriculum_stage_episodes = tuple((k + 1) * int(episodes_per_stage) for k in range(len(stages) - 1))

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
        stage_idx = int(getattr(env, "_curriculum_stage_idx", -1))
        logs.append(
            {
                "episode": int(ep),
                "stage": stage_idx,
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
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.titlesize": 15,
            "axes.labelsize": 13,
            "legend.fontsize": 11,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "lines.linewidth": 2.0,
            "figure.dpi": 200,
            "savefig.dpi": 300,
            "axes.unicode_minus": True,
        }
    )
    return plt


def _save_fig(fig, path_base: Path) -> None:
    path_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_base.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(path_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path_base.with_suffix(".svg"), bbox_inches="tight")


def _finite_or_default(x: float, default: float) -> float:
    return float(x) if np.isfinite(x) else float(default)


def _zoom_limits(y: np.ndarray, q_low: float = 0.02, q_high: float = 0.98, pad_frac: float = 0.08) -> tuple[float, float]:
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    if y.size == 0:
        return (0.0, 1.0)
    lo = float(np.quantile(y, q_low))
    hi = float(np.quantile(y, q_high))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo = float(np.min(y))
        hi = float(np.max(y))
    pad = pad_frac * max(hi - lo, 1e-9)
    return (lo - pad, hi + pad)


def _full_limits(ai: np.ndarray, foc_stage_means: Dict[int, float], pad_frac: float = 0.06) -> tuple[float, float]:
    vals: List[float] = []
    for v in np.asarray(ai, dtype=float).tolist():
        if np.isfinite(v):
            vals.append(float(v))
    for v in foc_stage_means.values():
        if np.isfinite(v):
            vals.append(float(v))
    if not vals:
        return (0.0, 1.0)
    lo = float(min(vals))
    hi = float(max(vals))
    if lo == hi:
        return (lo - 1.0, hi + 1.0)
    pad = pad_frac * (hi - lo)
    return (lo - pad, hi + pad)


def _stage_from_episode(ep: int, episodes_per_stage: int, n_stages: int) -> int:
    if episodes_per_stage <= 0 or n_stages <= 1:
        return 0
    st = int(ep) // int(episodes_per_stage)
    return int(np.clip(st, 0, max(n_stages - 1, 0)))


def _stage_means(series: Series, episodes_per_stage: int, n_stages: int, y: np.ndarray) -> Dict[int, float]:
    stage = np.asarray(series.stage, dtype=int)
    if stage.size != series.episodes.size or np.any(stage < 0):
        stage = np.asarray([_stage_from_episode(ep, episodes_per_stage, n_stages) for ep in series.episodes], dtype=int)
    out: Dict[int, float] = {}
    for st in range(int(n_stages)):
        mask = stage == st
        if np.any(mask):
            out[st] = float(np.mean(np.asarray(y, dtype=float)[mask]))
    if not out:
        out[0] = float(np.mean(np.asarray(y, dtype=float)))
    return out


def _plot_foc_stage_means(
    ax,
    foc_means: Dict[int, float],
    episodes_per_stage: int,
    n_stages: int,
    x_min: int,
    x_max: int,
    label: str,
):
    handle = None
    for st in range(int(n_stages)):
        y = float(foc_means.get(st, foc_means.get(0, 0.0)))
        xs = int(st * episodes_per_stage)
        xe = int((st + 1) * episodes_per_stage - 1) if episodes_per_stage > 0 else x_max
        xs = max(xs, x_min)
        xe = min(xe, x_max)
        if xe < xs:
            continue
        (h,) = ax.plot(
            [xs, xe],
            [y, y],
            color="0.35",
            linestyle="-",
            linewidth=1.1,
            label=label if handle is None else None,
            zorder=1,
        )
        handle = h if handle is None else handle
    return handle


def _format_stage_group_ru(stages: List[int]) -> str:
    if not stages:
        return ""
    stages = sorted({int(s) for s in stages})
    if len(stages) == 1:
        return f"стадия {stages[0]}"
    if stages == list(range(stages[0], stages[-1] + 1)):
        return f"стадии {stages[0]}–{stages[-1]}"
    return "стадии " + ", ".join(str(s) for s in stages)


def _annotate_foc_stage_means(
    ax,
    foc_means: Dict[int, float],
    x_anchor: int,
    decimals: int,
    unit_ru: str,
    y0_axes: float = 0.90,
    dy_axes: float = 0.10,
    fontsize: int = 8,
) -> None:
    groups: Dict[str, Dict[str, List[float] | List[int]]] = {}
    for st, mean_val in sorted(foc_means.items()):
        key = f"{_finite_or_default(mean_val, 0.0):.{int(decimals)}f}"
        if key not in groups:
            groups[key] = {"stages": [], "values": []}
        groups[key]["stages"].append(int(st))  # type: ignore[call-arg]
        groups[key]["values"].append(float(mean_val))  # type: ignore[call-arg]

    items: List[Tuple[float, str, List[int]]] = []
    for key, v in groups.items():
        stages = list(v["stages"])  # type: ignore[arg-type]
        values = np.asarray(v["values"], dtype=float)  # type: ignore[arg-type]
        rep = float(np.mean(values)) if values.size else 0.0
        items.append((rep, key, stages))
    items.sort(key=lambda t: t[0], reverse=True)

    for idx, (rep_y, key, stages) in enumerate(items):
        stage_str = _format_stage_group_ru(stages)
        label = f"FOC ({stage_str}): {key} {unit_ru}"
        ax.annotate(
            label,
            xy=(int(x_anchor), float(rep_y)),
            xycoords="data",
            xytext=(0.98, float(y0_axes) - float(dy_axes) * float(idx)),
            textcoords="axes fraction",
            ha="right",
            va="top",
            fontsize=int(fontsize),
            bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": "none", "alpha": 0.75},
            arrowprops={"arrowstyle": "-", "lw": 0.6, "color": "0.35"},
        )


def _add_stage_boundaries(ax, episodes_per_stage: int, n_stages: int, x_min: int, x_max: int) -> None:
    if episodes_per_stage <= 0:
        return
    for st in range(1, int(n_stages)):
        x = int(st * episodes_per_stage)
        if x_min <= x <= x_max:
            ax.axvline(x, color="0.90", linestyle=":", linewidth=0.6, alpha=0.6, zorder=0)


def _style_axes(ax, spine_lw: float = 0.8, spine_color: str = "0.25") -> None:
    for side in ("left", "right", "top", "bottom"):
        try:
            ax.spines[side].set_linewidth(spine_lw)
            ax.spines[side].set_color(spine_color)
        except Exception:
            pass
    try:
        ax.tick_params(width=spine_lw, colors="black")
    except Exception:
        pass


def _legend_compact(ax, ncol: int = 2) -> None:
    ax.legend(
        loc="upper right",
        frameon=True,
        framealpha=0.90,
        ncol=int(ncol),
        borderpad=0.30,
        handlelength=2.0,
        handletextpad=0.6,
        columnspacing=1.0,
        fontsize=9,
    )


def _legend_topbar(ax, ncol: int = 4) -> None:
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        frameon=True,
        framealpha=0.92,
        ncol=int(ncol),
        borderpad=0.35,
        handlelength=2.0,
        handletextpad=0.6,
        columnspacing=1.0,
        fontsize=10,
    )


def _clamp(v: float, lo: float, hi: float) -> float:
    return float(min(max(float(v), float(lo)), float(hi)))


def _format_stage_omega_text(stage_omega_ref_rad_s: List[float]) -> str:
    parts: List[str] = []
    for idx, w in enumerate(stage_omega_ref_rad_s):
        if not np.isfinite(w):
            continue
        parts.append(rf"Стадия {idx}: $\omega_{{\mathrm{{ref}}}}$ = {float(w):.2f} рад/с")
    return "; ".join(parts)


def _add_stage_omega_labels(ax, stage_omega_ref_rad_s: List[float]) -> None:
    text = _format_stage_omega_text(stage_omega_ref_rad_s)
    if not text:
        return
    ax.text(
        0.01,
        1.01,
        text,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        color="0.25",
        clip_on=False,
    )


def _legend_ordered(ax, order: List[str], **kwargs) -> None:
    handles, labels = ax.get_legend_handles_labels()
    label_to_handle: Dict[str, object] = {}
    for h, l in zip(handles, labels):
        if not l:
            continue
        if l not in label_to_handle:
            label_to_handle[l] = h
    ordered_labels = [l for l in order if l in label_to_handle]
    ordered_handles = [label_to_handle[l] for l in ordered_labels]
    leg = ax.legend(ordered_handles, ordered_labels, **kwargs)
    try:
        leg.set_zorder(10)
    except Exception:
        pass


def _pick_best_ai_for_pareto(
    ai: Series,
    foc_speed_means: Dict[int, float],
    episodes_per_stage: int,
    n_stages: int,
) -> int:
    stage = np.asarray(ai.stage, dtype=int)
    if stage.size != ai.episodes.size or np.any(stage < 0):
        stage = np.asarray([_stage_from_episode(ep, episodes_per_stage, n_stages) for ep in ai.episodes], dtype=int)
    thr = np.asarray([float(foc_speed_means.get(int(st), foc_speed_means.get(0, 0.0))) for st in stage], dtype=float)
    feasible = np.where(np.asarray(ai.speed_err, dtype=float) <= thr)[0]
    if feasible.size:
        return int(feasible[np.argmin(np.asarray(ai.p_in_pos, dtype=float)[feasible])])
    return int(np.argmin(np.asarray(ai.p_in_pos, dtype=float)))


def prepare_data(
    *,
    env_config: str | Path,
    ai_checkpoint: str | Path,
    out_dir: str | Path,
    episode_steps: int,
    episodes_per_stage: int,
    window: int,
    voltage_scale: float,
    disable_noise: bool,
    force_eval: bool,
) -> PreparedData:
    out_dir_path = Path(out_dir).resolve()
    out_dir_path.mkdir(parents=True, exist_ok=True)

    env_config_path = resolve_config_path(str(env_config))
    motor_key = _motor_key_from_config(str(env_config_path))

    cfg = load_ai_voltage_config()
    curriculum = get_curriculum_config(cfg)
    n_stages = int(len(curriculum.get("omega_pu_stages", [0.3, 0.5])))
    episodes_per_stage = int(episodes_per_stage)
    episodes = int(episodes_per_stage) * int(n_stages)

    env_cfg = make_env_from_config(str(env_config_path)).env_config
    vdc = float(getattr(getattr(env_cfg, "inverter", None), "Vdc", 0.0) or 0.0)
    v_limit_ai = float(voltage_scale) * (0.8 * vdc / np.sqrt(3.0)) if vdc > 0 else None

    foc_json = out_dir_path / f"foc_baseline_{motor_key}.json"
    ai_json = out_dir_path / f"ai_eval_{motor_key}.json"

    if force_eval or (not foc_json.exists()):
        save_foc_baseline(
            config_name=str(env_config_path),
            curriculum_config=curriculum,
            log_path=foc_json,
            n_episodes_eval=int(episodes_per_stage),
            episode_steps=int(episode_steps),
            v_limit=v_limit_ai,
        )

    if force_eval or (not ai_json.exists()):
        eval_ai_checkpoint(
            env_config=env_config_path,
            checkpoint=Path(ai_checkpoint),
            out_json=ai_json,
            episodes=episodes,
            episodes_per_stage=episodes_per_stage,
            episode_steps=int(episode_steps),
            voltage_scale=float(voltage_scale),
            disable_noise=bool(disable_noise),
        )

    ai_eps = _load_episode_list(ai_json)
    foc_eps = _load_episode_list(foc_json)

    ai_series = _extract_series(ai_eps, prefer_i_rms_abc=True)
    foc_series = _extract_series(foc_eps, prefer_i_rms_abc=False)

    omega_pu_stages = list(curriculum.get("omega_pu_stages", [0.3, 0.5]))
    pole_pairs = int(getattr(getattr(env_cfg, "motor", None), "p", 1) or 1)
    omega_nominal = float(2.0 * np.pi * 10.0 / max(pole_pairs, 1))
    stage_omega_ref_rad_s = [float(pu) * float(omega_nominal) for pu in omega_pu_stages]

    _ = int(window)  # validate early
    return PreparedData(
        out_dir=out_dir_path,
        env_config=env_config_path,
        motor_key=motor_key,
        n_stages=int(n_stages),
        episodes_per_stage=int(episodes_per_stage),
        episode_steps=int(episode_steps),
        voltage_scale=float(voltage_scale),
        disable_noise=bool(disable_noise),
        foc_json=foc_json,
        ai_json=ai_json,
        ai=ai_series,
        foc=foc_series,
        stage_omega_ref_rad_s=stage_omega_ref_rad_s,
    )


def make_plots(
    out_dir: Path,
    ai: Series,
    foc: Series,
    window: int,
    episodes_per_stage: int,
    n_stages: int,
    stage_omega_ref_rad_s: List[float] | None = None,
    figures: List[int] | None = None,
    write_captions: bool = True,
) -> Path | None:
    plt = _set_ieee_style()

    stage_omega_ref_rad_s = list(stage_omega_ref_rad_s or [])
    figs = set(int(f) for f in (figures or [1, 2, 3, 4]))

    x_min = int(min(np.min(ai.episodes), np.min(foc.episodes)))
    x_max = int(max(np.max(ai.episodes), np.max(foc.episodes)))
    x_min_plot = float(x_min) - 1.0
    x_max_plot = float(x_max) + 1.0

    foc_i_means = _stage_means(foc, episodes_per_stage, n_stages, foc.i_rms)
    foc_p_means = _stage_means(foc, episodes_per_stage, n_stages, foc.p_in_pos)
    foc_s_means = _stage_means(foc, episodes_per_stage, n_stages, foc.speed_err)

    foc_i_mean = float(np.mean(foc.i_rms))
    foc_p_mean = float(np.mean(foc.p_in_pos))
    foc_s_mean = float(np.mean(foc.speed_err))

    best_idx = _pick_best_ai_for_pareto(ai, foc_s_means, episodes_per_stage, n_stages)
    best_ep = int(ai.episodes[best_idx])

    label_ai_eps = "AI (эксперименты)"
    label_ai_roll = f"AI (скользящее среднее, окно {int(window)})"
    label_foc = "FOC (среднее по стадиям)"
    label_ai_selected = "AI selected (Парето)"

    ai_i_roll, _ = _rolling_mean_std(ai.i_rms, window)
    ai_p_roll, _ = _rolling_mean_std(ai.p_in_pos, window)
    ai_s_roll, ai_s_std = _rolling_mean_std(ai.speed_err, window)

    stage_text = _format_stage_omega_text(stage_omega_ref_rad_s)

    if 1 in figs:
        # --- Figure 1: Irms (two panels) ---
        fig1, (ax1a, ax1b) = plt.subplots(
            2,
            1,
            sharex=True,
            figsize=(7.6, 5.0),
            gridspec_kw={"height_ratios": [1.0, 1.0]},
        )
        for ax in (ax1a, ax1b):
            _style_axes(ax, spine_lw=0.7, spine_color="0.30")
            ax.plot(ai.episodes, ai.i_rms, color="0.55", linewidth=1.0, alpha=0.35, label="_nolegend_")
            ax.plot(
                ai.episodes,
                ai_i_roll,
                color="black",
                linestyle="-",
                linewidth=2.9,
                label=label_ai_roll if ax is ax1a else None,
            )
            _plot_foc_stage_means(
                ax,
                foc_i_means,
                episodes_per_stage=episodes_per_stage,
                n_stages=n_stages,
                x_min=x_min,
                x_max=x_max,
                label="",
            )
            _add_stage_boundaries(ax, episodes_per_stage, n_stages, x_min, x_max)
            ax.scatter(
                [best_ep],
                [_finite_or_default(float(ai.i_rms[best_idx]), 0.0)],
                s=110 if ax is ax1b else 95,
                marker="D",
                color="black",
                edgecolors="white",
                linewidths=1.2,
                label=label_ai_selected if ax is ax1a else None,
                zorder=5,
            )
            ax.grid(True, alpha=0.25)

        ax1a.set_title(
            "Сравнение среднеквадратичного тока статора при AI- и FOC-управлении",
            pad=22,
        )
        _add_stage_omega_labels(ax1a, stage_omega_ref_rad_s)
        ax1a.set_ylabel(r"$I_{\mathrm{rms}}$, А")
        ax1a.set_xlim(x_min_plot, x_max_plot)
        ax1a.set_ylim(*_full_limits(ai.i_rms, foc_i_means, pad_frac=0.10))
        _legend_ordered(
            ax1a,
            [label_ai_roll, label_ai_selected],
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            frameon=True,
            framealpha=0.90,
            ncol=1,
            borderpad=0.28,
            fontsize=8,
            handlelength=2.0,
        )
        _annotate_foc_stage_means(ax1a, foc_i_means, x_anchor=x_max, decimals=2, unit_ru="А")

        ax1b.set_ylabel("")
        ax1b.set_xlabel("Номер эксперимента")
        ax1b.set_ylim(*_zoom_limits(ai.i_rms, q_low=0.02, q_high=0.98, pad_frac=0.22))
        _save_fig(fig1, out_dir / "fig1_Irms")
        plt.close(fig1)

    if 2 in figs:
        # --- Figure 2: Pin+ (two panels) ---
        fig2, (ax2a, ax2b) = plt.subplots(
            2,
            1,
            sharex=True,
            figsize=(7.6, 5.0),
            gridspec_kw={"height_ratios": [1.0, 1.0]},
        )
        for ax in (ax2a, ax2b):
            _style_axes(ax, spine_lw=0.7, spine_color="0.30")
            ax.plot(
                ai.episodes,
                ai.p_in_pos,
                color="0.55",
                linewidth=1.0,
                alpha=0.35,
                label="_nolegend_",
            )
            ax.plot(
                ai.episodes,
                ai_p_roll,
                color="black",
                linestyle="-",
                linewidth=2.9,
                label=label_ai_roll if ax is ax2a else None,
            )
            _plot_foc_stage_means(
                ax,
                foc_p_means,
                episodes_per_stage=episodes_per_stage,
                n_stages=n_stages,
                x_min=x_min,
                x_max=x_max,
                label="",
            )
            _add_stage_boundaries(ax, episodes_per_stage, n_stages, x_min, x_max)
            ax.scatter(
                [best_ep],
                [_finite_or_default(float(ai.p_in_pos[best_idx]), 0.0)],
                s=95,
                marker="D",
                color="black",
                edgecolors="white",
                linewidths=1.2,
                label=label_ai_selected if ax is ax2a else None,
                zorder=5,
            )
            ax.grid(True, alpha=0.25)

        ax2a.set_title(
            "Сравнение потребляемой входной мощности при AI- и FOC-управлении",
            pad=22,
        )
        _add_stage_omega_labels(ax2a, stage_omega_ref_rad_s)
        ax2a.set_ylabel(r"$P_{\mathrm{in}}^{+}$, Вт")
        ax2a.set_xlim(x_min_plot, x_max_plot)
        ax2a.set_ylim(*_full_limits(ai.p_in_pos, foc_p_means, pad_frac=0.10))
        _legend_ordered(
            ax2a,
            [label_ai_roll, label_ai_selected],
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            frameon=True,
            framealpha=0.90,
            ncol=1,
            borderpad=0.28,
            fontsize=8,
            handlelength=2.0,
        )

        _annotate_foc_stage_means(ax2a, foc_p_means, x_anchor=x_max, decimals=1, unit_ru="Вт")

        ax2b.set_ylabel("")
        ax2b.set_xlabel("Номер эксперимента")
        ylo_z, yhi_z = _zoom_limits(ai.p_in_pos, q_low=0.02, q_high=0.98, pad_frac=0.22)
        ax2b.set_ylim(ylo_z - 6.0, yhi_z + 2.0)
        _save_fig(fig2, out_dir / "fig2_Pin_pos")
        plt.close(fig2)

    if 3 in figs:
        # --- Figure 3: speed error (single panel + band) ---
        fig3, ax3 = plt.subplots(figsize=(7.6, 4.0))
        _style_axes(ax3, spine_lw=0.7, spine_color="0.30")
        ax3.plot(ai.episodes, ai.speed_err, color="0.55", linewidth=1.0, alpha=0.35, label="_nolegend_")
        ax3.fill_between(ai.episodes, ai_s_roll - ai_s_std, ai_s_roll + ai_s_std, color="black", alpha=0.15, linewidth=0, zorder=1)
        ax3.plot(
            ai.episodes,
            ai_s_roll,
            color="black",
            linestyle="-",
            linewidth=3.0,
            label=label_ai_roll,
            zorder=2,
        )
        _plot_foc_stage_means(
            ax3,
            foc_s_means,
            episodes_per_stage=episodes_per_stage,
            n_stages=n_stages,
            x_min=x_min,
            x_max=x_max,
            label="",
        )
        _add_stage_boundaries(ax3, episodes_per_stage, n_stages, x_min, x_max)
        ax3.scatter(
            [best_ep],
            [_finite_or_default(float(ai.speed_err[best_idx]), 0.0)],
            s=65,
            marker="D",
            color="0.25",
            edgecolors="white",
            linewidths=1.0,
            alpha=0.65,
            label=label_ai_selected,
            zorder=3,
        )
        ax3.set_title(
            "Сравнение средней ошибки регулирования скорости при AI- и FOC-управлении",
            pad=22,
        )
        _add_stage_omega_labels(ax3, stage_omega_ref_rad_s)
        ax3.set_xlabel("Номер эксперимента")
        ax3.set_ylabel(r"$|\omega_{\mathrm{ref}}-\omega|$, рад/с")
        ax3.set_xlim(x_min_plot, x_max_plot)
        ax3.set_ylim(*_full_limits(ai.speed_err, foc_s_means, pad_frac=0.08))
        ax3.grid(True, alpha=0.25)
        _legend_ordered(
            ax3,
            [label_ai_roll, label_ai_selected],
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            frameon=True,
            framealpha=0.90,
            ncol=1,
            borderpad=0.28,
            fontsize=8,
            handlelength=2.0,
        )
        _annotate_foc_stage_means(ax3, foc_s_means, x_anchor=x_max, decimals=2, unit_ru="рад/с")
        _save_fig(fig3, out_dir / "fig3_speed_error")
        plt.close(fig3)

    if 4 in figs:
        # --- Figure 4: Pareto ---
        fig4, ax4 = plt.subplots(figsize=(6.2, 4.8))
        _style_axes(ax4, spine_lw=0.7, spine_color="0.30")
        ax4.scatter(ai.speed_err, ai.p_in_pos, s=18, c="black", alpha=0.30, label=label_ai_eps)
        foc_points: List[tuple[float, float]] = []
        for st in range(int(n_stages)):
            sx = float(foc_s_means.get(st, foc_s_mean))
            py = float(foc_p_means.get(st, foc_p_mean))
            ax4.scatter(
                [sx],
                [py],
                s=85,
                marker="D",
                c="none",
                edgecolors="black",
                linewidths=1.8,
                label=label_foc if st == 0 else None,
            )
            foc_points.append((sx, py))
        ax4.scatter(
            [ai.speed_err[best_idx]],
            [ai.p_in_pos[best_idx]],
            s=105,
            marker="D",
            c="black",
            edgecolors="white",
            linewidths=1.2,
            label=label_ai_selected,
        )
        for st, (sx, py) in enumerate(foc_points):
            ax4.annotate(
                f"FOC (стадия {int(st)})",
                xy=(float(sx), float(py)),
                xytext=(10, 8),
                textcoords="offset points",
                ha="left",
                va="bottom",
                fontsize=10,
                arrowprops={"arrowstyle": "-", "lw": 0.8, "color": "0.25"},
            )
        ax4.annotate(
            "AI selected",
            xy=(float(ai.speed_err[best_idx]), float(ai.p_in_pos[best_idx])),
            xytext=(10, -12),
            textcoords="offset points",
            ha="left",
            va="top",
            fontsize=10,
            arrowprops={"arrowstyle": "-", "lw": 0.8, "color": "0.25"},
        )
        ax4.set_title("Парето-сравнение качества управления: ошибка скорости и входная мощность", pad=22)
        ax4.set_xlabel(r"$|\omega_{\mathrm{ref}}-\omega|$, рад/с")
        ax4.set_ylabel(r"$P_{\mathrm{in}}^{+}$, Вт")
        ax4.grid(True, alpha=0.25)
        _legend_ordered(
            ax4,
            [label_ai_eps, label_ai_selected],
            loc="upper right",
            frameon=False,
            ncol=1,
            borderpad=0.28,
            fontsize=8,
            handlelength=2.0,
        )
        _save_fig(fig4, out_dir / "fig4_pareto")
        plt.close(fig4)

    stage_text_en_parts: List[str] = []
    for idx, w in enumerate(stage_omega_ref_rad_s):
        if np.isfinite(w):
            stage_text_en_parts.append(rf"Stage {idx}: $\omega_{{\mathrm{{ref}}}}$ = {float(w):.2f} rad/s")
    stage_text_en = "; ".join(stage_text_en_parts)

    stage_prefix_ru = f"{stage_text}. " if stage_text else ""
    stage_prefix_en = f"{stage_text_en}. " if stage_text_en else ""

    captions_ru = [
        (
            "Рис. 1. Сравнение среднеквадратичного значения тока статора "
            r"$I_{\mathrm{rms}}$ при AI- и FOC-управлении (средние значения по эксперименту). "
            f"{stage_prefix_ru}"
            f"Серые линии — значения AI по экспериментам; чёрная линия — скользящее среднее (окно {int(window)}). "
            "Тонкая линия FOC — среднее значение по стадиям; вариативность FOC по экспериментам мала и не показана. "
            "Вертикальные линии обозначают границы стадий; верхняя панель показывает полный диапазон значений, "
            "нижняя — увеличенный фрагмент области AI."
        ),
        (
            "Рис. 2. Сравнение положительной составляющей входной мощности "
            r"$P_{\mathrm{in}}^{+}$ при AI- и FOC-управлении (учитывается только положительная часть мощности). "
            f"{stage_prefix_ru}"
            f"Серые линии — значения AI по экспериментам; чёрная линия — скользящее среднее (окно {int(window)}). "
            "Тонкая линия FOC — среднее значение по стадиям; вариативность FOC по экспериментам мала и не показана. "
            "Вертикальные линии обозначают границы стадий; верхняя панель показывает полный диапазон значений, "
            "нижняя — увеличенный фрагмент области AI."
        ),
        (
            "Рис. 3. Сравнение средней по эксперименту ошибки регулирования скорости "
            r"$|e_\omega|=|\omega_{\mathrm{ref}}-\omega|$ при AI- и FOC-управлении. "
            f"{stage_prefix_ru}"
            f"Серые линии — значения AI по экспериментам; чёрная линия — скользящее среднее (окно {int(window)}), "
            r"затенение — $\pm\sigma$ в окне. "
            "Тонкая линия FOC — среднее значение по стадиям; вариативность FOC по экспериментам мала и не показана. "
            r"Маркер «AI selected (Парето)» выбран по критерию $\arg\min P_{\mathrm{in}}^{+}$ при ограничении "
            r"$|e_\omega|\leq |e_\omega|_{\mathrm{FOC}}$ (по стадиям), поэтому не обязан минимизировать ошибку скорости."
        ),
        (
            "Рис. 4. Парето-сравнение качества управления по критериям "
            r"$|e_\omega|=|\omega_{\mathrm{ref}}-\omega|$ и $P_{\mathrm{in}}^{+}$ для AI- и FOC-управления "
            "(средние значения по эксперименту). "
            "Точки — эксперименты AI; полые ромбы — средние значения FOC по стадиям. "
            r"Точка «AI selected (Парето)» выбрана как решение $\arg\min P_{\mathrm{in}}^{+}$ при условии "
            r"$|e_\omega|\leq |e_\omega|_{\mathrm{FOC}}$ в соответствующей стадии. "
            "Врезка показывает увеличенную область, содержащую эксперименты AI."
        ),
    ]

    captions_en = [
        (
            "Fig. 1. Comparison of stator RMS current "
            r"$I_{\mathrm{rms}}$ under AI-based control and classical field-oriented control (FOC) "
            "(run-averaged values). "
            f"{stage_prefix_en}"
            f"Gray curves show AI run values; the solid black curve is the rolling mean (window {int(window)}). "
            "The thin FOC curve is the stage-wise mean; FOC run-to-run variability is small and omitted for readability. "
            "Vertical lines indicate stage boundaries; the top panel shows the full range, "
            "the bottom panel shows a zoomed view of the AI region."
        ),
        (
            "Fig. 2. Comparison of positive input power component "
            r"$P_{\mathrm{in}}^{+}$ under AI-based control and FOC (only the positive power component is included). "
            f"{stage_prefix_en}"
            f"Gray curves show AI run values; the solid black curve is the rolling mean (window {int(window)}). "
            "The thin FOC curve is the stage-wise mean; FOC run-to-run variability is small and omitted for readability. "
            "Vertical lines indicate stage boundaries; the top panel shows the full range, "
            "the bottom panel shows a zoomed view of the AI region."
        ),
        (
            "Fig. 3. Comparison of run-averaged speed regulation error "
            r"$|e_\omega|=|\omega_{\mathrm{ref}}-\omega|$ under AI-based control and FOC. "
            f"{stage_prefix_en}"
            f"Gray curves show AI run values; the solid black curve is the rolling mean (window {int(window)}), "
            r"and the shaded band indicates $\pm\sigma$ within the window. "
            "The thin FOC curve is the stage-wise mean; FOC run-to-run variability is small and omitted for readability. "
            r"The “AI selected (Pareto)” marker is selected as $\arg\min P_{\mathrm{in}}^{+}$ subject to "
            r"$|e_\omega|\leq |e_\omega|_{\mathrm{FOC}}$ (stage-wise), therefore it is not expected to minimize the speed error."
        ),
        (
            "Fig. 4. Pareto comparison of control quality using "
            r"$|e_\omega|=|\omega_{\mathrm{ref}}-\omega|$ and $P_{\mathrm{in}}^{+}$ for AI-based control and FOC "
            "(run-averaged values). "
            "Dots show AI runs; open diamonds show stage-wise FOC means. "
            r"The “AI selected (Pareto)” point is selected as $\arg\min P_{\mathrm{in}}^{+}$ subject to "
            r"$|e_\omega|\leq |e_\omega|_{\mathrm{FOC}}$ for the corresponding stage. "
            "The inset shows a zoomed region containing AI runs."
        ),
    ]

    if not write_captions:
        return None

    captions_ru = [captions_ru[i - 1] for i in sorted(figs) if 1 <= i <= len(captions_ru)]
    captions_en = [captions_en[i - 1] for i in sorted(figs) if 1 <= i <= len(captions_en)]

    captions_path_ru = out_dir / "captions_ru.txt"
    captions_path_en = out_dir / "captions_en.txt"
    captions_path_ru.write_text("\n".join(captions_ru) + "\n", encoding="utf-8-sig")
    captions_path_en.write_text("\n".join(captions_en) + "\n", encoding="utf-8-sig")
    return captions_path_ru


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
    p.add_argument(
        "--figures",
        type=int,
        nargs="+",
        choices=[1, 2, 3, 4],
        default=None,
        help="Generate only selected figures (1..4). Omit to generate all.",
    )
    p.add_argument("--force-eval", action="store_true", help="Recompute baseline and AI eval logs even if JSON files exist.")
    p.add_argument("--no-captions", action="store_true", help="Do not write captions_ru.txt / captions_en.txt.")
    return p.parse_args(argv)


def main() -> None:
    args = parse_args()
    prepared = prepare_data(
        env_config=args.env_config,
        ai_checkpoint=args.ai_checkpoint,
        out_dir=args.out_dir,
        episode_steps=int(args.episode_steps),
        episodes_per_stage=int(args.episodes_per_stage),
        window=int(args.window),
        voltage_scale=float(args.voltage_scale),
        disable_noise=bool(args.disable_noise),
        force_eval=bool(args.force_eval),
    )
    captions_path = make_plots(
        prepared.out_dir,
        prepared.ai,
        prepared.foc,
        window=int(args.window),
        episodes_per_stage=int(prepared.episodes_per_stage),
        n_stages=int(prepared.n_stages),
        stage_omega_ref_rad_s=prepared.stage_omega_ref_rad_s,
        figures=args.figures,
        write_captions=not bool(args.no_captions),
    )
    print(f"Saved figures to {prepared.out_dir}")
    if captions_path is not None:
        print(f"Saved captions to {captions_path}")


if __name__ == "__main__":
    main()
