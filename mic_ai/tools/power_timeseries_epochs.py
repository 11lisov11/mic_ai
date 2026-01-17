from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mic_ai.ai.agents.ppo_voltage import PPOVoltageAgent
from mic_ai.ai.ai_voltage_config import get_voltage_scale, load_ai_voltage_config
from mic_ai.ai.train_ai_foc_assist import FEATURE_KEYS as FOC_FEATURE_KEYS
from mic_ai.ai.train_ai_id_ref import FEATURE_KEYS as ID_FEATURE_KEYS
from mic_ai.ai.train_ai_voltage import FEATURE_KEYS as VOLT_FEATURE_KEYS, _motor_key_from_config, resolve_config_path
from mic_ai.core.env import make_env_from_config
from mic_ai.tools.plot_style import apply_vak_style, ensure_matplotlib, save_figure
from mic_ai.tools.timeseries_compare import _make_load_func, _simulate_ai, _simulate_foc


def _parse_list(value: str | None) -> List[str]:
    if value is None:
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _default_epoch_labels(paths: Sequence[Path]) -> List[str]:
    labels: List[str] = []
    for idx, path in enumerate(paths):
        match = re.search(r"ep(\d+)", path.stem, flags=re.IGNORECASE)
        if match:
            labels.append(str(int(match.group(1))))
        else:
            labels.append(str(idx))
    return labels


def _infer_hidden_sizes(state: Dict[str, torch.Tensor]) -> Tuple[int, ...] | None:
    w0 = state.get("actor_body.0.weight")
    w2 = state.get("actor_body.2.weight")
    if w0 is None or w2 is None:
        return None
    try:
        return int(w0.shape[0]), int(w2.shape[0])
    except Exception:
        return None


def _build_agent(checkpoint: Path, ai_mode: str) -> PPOVoltageAgent:
    state = torch.load(checkpoint, map_location="cpu")
    hidden = _infer_hidden_sizes(state) or (128, 128)
    mode = str(ai_mode).lower()
    if mode == "ai_id_ref":
        feature_keys = ID_FEATURE_KEYS
        action_dim = 1
    elif mode == "foc_assist":
        feature_keys = FOC_FEATURE_KEYS
        action_dim = 2
    else:
        feature_keys = VOLT_FEATURE_KEYS
        action_dim = 2
    agent = PPOVoltageAgent(feature_keys=feature_keys, action_dim=action_dim, device="cpu", hidden_sizes=hidden)
    agent.net.load_state_dict(state)
    agent.set_action_std(1e-6)
    return agent


def _resolve_omega_ref(env_cfg: object, args: argparse.Namespace) -> float:
    if args.omega_ref is not None:
        return float(args.omega_ref)
    omega_base = float(2.0 * math.pi * 10.0 / max(env_cfg.motor.p, 1))
    return float(args.omega_ref_pu) * omega_base


def _plot_power_epochs(
    out_path: Path,
    foc: Dict[str, np.ndarray],
    mic_series: Sequence[Dict[str, np.ndarray]],
    epoch_labels: Sequence[str],
    epoch_normalized: bool,
    mark_every: int | None,
) -> None:
    plt = apply_vak_style(ensure_matplotlib(), font_family="sans")
    fig, ax = plt.subplots(figsize=(9.2, 4.6))

    n_series = max(len(mic_series), 1)
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i / max(n_series - 1, 1)) for i in range(n_series)]

    if mark_every is None:
        mark_every = max(int(foc["t"].size / 12), 1)

    for color, mic in zip(colors, mic_series):
        ax.plot(
            foc["t"],
            foc["p_el"],
            color=color,
            linestyle="--",
            marker="o",
            markevery=mark_every,
            linewidth=2.0,
        )
        ax.plot(
            mic["t"],
            mic["p_el"],
            color=color,
            linestyle="-",
            marker="s",
            markevery=mark_every,
            linewidth=2.0,
        )

    ax.set_xlabel("Время, с")
    ax.set_ylabel("Потребляемая активная мощность, Вт")

    from matplotlib.lines import Line2D

    method_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            linestyle="--",
            marker="o",
            label="Классическое векторное управление (FOC)",
        ),
        Line2D(
            [0],
            [0],
            color="black",
            linestyle="-",
            marker="s",
            label="Нейросетевое управление MIC_AI",
        ),
    ]
    method_legend = ax.legend(handles=method_handles, loc="upper right", frameon=False)
    ax.add_artist(method_legend)

    epoch_handles = [
        Line2D([0], [0], color=colors[idx], linestyle="-", marker="s", label=label)
        for idx, label in enumerate(epoch_labels)
    ]
    epoch_title = "Эпоха обучения RL-агента"
    if epoch_normalized:
        epoch_title += " (нормированная шкала)"

    fig.legend(
        handles=epoch_handles,
        labels=[h.get_label() for h in epoch_handles],
        loc="lower center",
        ncol=3,
        frameon=False,
        title=epoch_title,
    )
    fig.tight_layout(rect=[0.0, 0.12, 1.0, 1.0])
    save_figure(fig, out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Active power vs time for multiple RL epochs (FOC vs MIC_AI)."
    )
    parser.add_argument("--env-config", default="config/env_demo_true_motor1.py")
    parser.add_argument(
        "--ai-checkpoints",
        required=True,
        help="Comma-separated list of MIC AI checkpoint paths.",
    )
    parser.add_argument(
        "--epoch-labels",
        default=None,
        help="Comma-separated epoch labels (same length as checkpoints).",
    )
    parser.add_argument(
        "--epoch-normalized",
        action="store_true",
        help="Add 'нормированная шкала' to the epoch legend title.",
    )
    parser.add_argument("--ai-mode", choices=["ai_voltage", "ai_id_ref", "foc_assist"], default="ai_id_ref")
    parser.add_argument("--ai-id-relative", action="store_true", help="Use relative id_ref around base for ai_id_ref.")
    parser.add_argument("--delta-id-max", type=float, default=0.1)
    parser.add_argument("--omega-ref", type=float, default=None, help="Absolute omega_ref, rad/s.")
    parser.add_argument("--omega-ref-pu", type=float, default=0.8)
    parser.add_argument("--t-end", type=float, default=None)
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--out-dir", default="outputs/power_timeseries_epochs")
    parser.add_argument("--out-name", default="power_timeseries_epochs")
    parser.add_argument("--mark-every", type=int, default=None, help="Marker interval in points.")
    parser.add_argument("--load-profile", choices=["step", "ramp", "sine", "csv"], default="step")
    parser.add_argument("--load-steps", default="0:0.0,0.6:0.05,1.2:0.1")
    parser.add_argument("--load-t0", type=float, default=0.0)
    parser.add_argument("--load-t1", type=float, default=1.0)
    parser.add_argument("--load-start", type=float, default=0.0)
    parser.add_argument("--load-end", type=float, default=0.2)
    parser.add_argument("--load-amp", type=float, default=0.05)
    parser.add_argument("--load-offset", type=float, default=0.05)
    parser.add_argument("--load-freq", type=float, default=0.5)
    parser.add_argument("--load-csv", default=None)
    args = parser.parse_args()

    ckpt_list = [Path(p) for p in _parse_list(args.ai_checkpoints)]
    if not ckpt_list:
        raise ValueError("Provide at least one checkpoint in --ai-checkpoints.")
    for ckpt in ckpt_list:
        if not ckpt.exists():
            raise FileNotFoundError(f"AI checkpoint not found: {ckpt}")

    epoch_labels = _parse_list(args.epoch_labels)
    if not epoch_labels:
        epoch_labels = _default_epoch_labels(ckpt_list)
    if len(epoch_labels) != len(ckpt_list):
        raise ValueError("Length of --epoch-labels must match --ai-checkpoints.")

    env_path = resolve_config_path(args.env_config)
    env_cfg = make_env_from_config(str(env_path)).env_config

    dt = float(args.dt) if args.dt is not None else float(env_cfg.sim.dt)
    t_end = float(args.t_end) if args.t_end is not None else float(env_cfg.sim.t_end)
    env_cfg = replace(env_cfg, sim=replace(env_cfg.sim, dt=dt, t_end=t_end))

    omega_ref = _resolve_omega_ref(env_cfg, args)
    load_func = _make_load_func(args)

    v_scale = None
    if str(args.ai_mode).lower() == "ai_voltage":
        motor_key = _motor_key_from_config(str(env_path))
        ai_cfg = load_ai_voltage_config()
        v_scale = float(get_voltage_scale(ai_cfg, motor_key))

    foc = _simulate_foc(env_cfg, omega_ref, load_func, dt, t_end)

    mic_series: List[Dict[str, np.ndarray]] = []
    for ckpt in ckpt_list:
        agent = _build_agent(ckpt, args.ai_mode)
        mic = _simulate_ai(
            agent,
            env_cfg,
            omega_ref,
            load_func,
            dt,
            t_end,
            args.ai_mode,
            v_scale,
            bool(args.ai_id_relative),
            float(args.delta_id_max),
        )
        mic_series.append(mic)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.out_name}.png"
    _plot_power_epochs(out_path, foc, mic_series, epoch_labels, bool(args.epoch_normalized), args.mark_every)

    meta = {
        "env_config": str(env_path),
        "ai_mode": str(args.ai_mode),
        "ai_checkpoints": [str(p) for p in ckpt_list],
        "epoch_labels": list(epoch_labels),
        "epoch_normalized": bool(args.epoch_normalized),
        "omega_ref": omega_ref,
        "dt": dt,
        "t_end": t_end,
        "load_profile": str(args.load_profile),
        "load_steps": str(args.load_steps),
        "ai_id_relative": bool(args.ai_id_relative),
        "delta_id_max": float(args.delta_id_max),
        "plot_style": "vak_ru_sans",
        "plot_formats": ["png", "pdf", "svg"],
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved outputs to {out_dir}")


if __name__ == "__main__":
    main()
