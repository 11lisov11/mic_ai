from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from mic_ai.analysis.metrics import calc_i_rms, calc_p_el, calc_p_mech
from mic_ai.core.env import make_env_from_config
from mic_ai.tools.plot_style import apply_vak_style, ensure_matplotlib, save_figure
from simulation.gym_env import InductionMotorEnv


def _steady_slice(n: int, window_frac: float) -> slice:
    if n <= 0:
        return slice(0, 0)
    window_frac = float(max(min(window_frac, 0.95), 0.05))
    start = int(max(0, n * (1.0 - window_frac)))
    return slice(start, n)


def _attach_extras(src: object, dst: object) -> None:
    fields = set(getattr(src, "__dataclass_fields__", {}).keys())
    extras = {k: v for k, v in getattr(src, "__dict__", {}).items() if k not in fields}
    for key, value in extras.items():
        try:
            object.__setattr__(dst, key, value)
        except Exception:
            setattr(dst, key, value)


def _simulate(env_cfg: object, dt: float, t_end: float, scenario: str, load_torque: float, id_ref: float) -> Dict[str, np.ndarray]:
    sim_cfg = replace(env_cfg.sim, dt=dt, t_end=t_end, scenario_name=str(scenario), load_torque=float(load_torque))
    foc_cfg = replace(env_cfg.foc, id_ref=float(id_ref))
    env_cfg_s = replace(env_cfg, sim=sim_cfg, foc=foc_cfg)
    _attach_extras(env_cfg, env_cfg_s)
    env = InductionMotorEnv(env_cfg_s)
    env.reset()

    steps = int(max(t_end / dt, 1))
    series = {
        "t": np.zeros(steps, dtype=float),
        "omega": np.zeros(steps, dtype=float),
        "omega_ref": np.zeros(steps, dtype=float),
        "i_rms": np.zeros(steps, dtype=float),
        "p_in_total": np.zeros(steps, dtype=float),
        "p_mech": np.zeros(steps, dtype=float),
        "p_inv_loss": np.zeros(steps, dtype=float),
        "p_core_loss": np.zeros(steps, dtype=float),
        "p_mech_loss": np.zeros(steps, dtype=float),
    }

    for k in range(steps):
        obs, _r, done, info = env.step(None)
        series["t"][k] = float(env.t)
        series["omega"][k] = float(obs[0])
        series["omega_ref"][k] = float(obs[1])
        i_abc = np.asarray(info.get("i_abc", (0.0, 0.0, 0.0)), dtype=float)
        v_abc = np.asarray(info.get("v_abc", (0.0, 0.0, 0.0)), dtype=float)
        series["i_rms"][k] = calc_i_rms(i_abc)
        p_el_val = calc_p_el(v_abc, i_abc)
        series["p_in_total"][k] = float(info.get("p_in_total", p_el_val))
        torque = float(info.get("torque_e", obs[2]))
        series["p_mech"][k] = calc_p_mech(float(obs[0]), torque)
        series["p_inv_loss"][k] = float(info.get("p_inv_loss", 0.0))
        series["p_core_loss"][k] = float(info.get("p_core_loss", 0.0))
        series["p_mech_loss"][k] = float(info.get("p_mech_loss", 0.0))

        if done:
            for key in series:
                series[key] = series[key][: k + 1]
            break

    return series


def _summarize(series: Dict[str, np.ndarray], window_frac: float) -> Dict[str, float]:
    n = int(series["t"].size)
    sl = _steady_slice(n, window_frac)
    omega = series["omega"][sl]
    omega_ref = series["omega_ref"][sl]
    p_in = series["p_in_total"][sl]
    p_mech = series["p_mech"][sl]
    p_inv = series["p_inv_loss"][sl]
    p_core = series["p_core_loss"][sl]
    p_mech_loss = series["p_mech_loss"][sl]
    i_rms = series["i_rms"][sl]

    mean_p_in = float(np.mean(np.maximum(p_in, 0.0))) if p_in.size else 0.0
    mean_p_mech = float(np.mean(p_mech)) if p_mech.size else 0.0
    mean_losses = mean_p_in - mean_p_mech
    mean_err = float(np.mean(np.abs(omega_ref - omega))) if omega.size else 0.0
    eta = mean_p_mech / mean_p_in if mean_p_in > 1e-9 else 0.0

    return {
        "omega_ss": float(np.mean(omega)) if omega.size else 0.0,
        "omega_ref": float(np.mean(omega_ref)) if omega_ref.size else 0.0,
        "mean_abs_speed_err": mean_err,
        "mean_p_in_total_pos": mean_p_in,
        "mean_p_mech": mean_p_mech,
        "mean_loss_total": float(mean_losses),
        "mean_p_inv_loss": float(np.mean(p_inv)) if p_inv.size else 0.0,
        "mean_p_core_loss": float(np.mean(p_core)) if p_core.size else 0.0,
        "mean_p_mech_loss": float(np.mean(p_mech_loss)) if p_mech_loss.size else 0.0,
        "mean_i_rms": float(np.mean(i_rms)) if i_rms.size else 0.0,
        "eta": float(eta),
    }


def _save_csv(path: Path, series: Dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "t",
        "omega",
        "omega_ref",
        "i_rms",
        "p_in_total",
        "p_mech",
        "p_inv_loss",
        "p_core_loss",
        "p_mech_loss",
    ]
    rows = np.column_stack([series[h] for h in header])
    np.savetxt(path, rows, delimiter=",", header=",".join(header), comments="")


def _plot_timeseries(out_path: Path, foc: Dict[str, np.ndarray], mic: Dict[str, np.ndarray]) -> None:
    plt = apply_vak_style(ensure_matplotlib())
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.4), sharex=True)

    axes[0].plot(foc["t"], foc["omega"], color="black", label="FOC")
    axes[0].plot(mic["t"], mic["omega"], color="0.35", linestyle="--", label="MIC")
    axes[0].plot(foc["t"], foc["omega_ref"], color="tab:blue", linewidth=1.4, alpha=0.7, label="omega_ref")
    axes[0].set_ylabel("omega, rad/s")
    axes[0].legend(frameon=False)

    axes[1].plot(foc["t"], np.maximum(foc["p_in_total"], 0.0), color="black", label="FOC")
    axes[1].plot(mic["t"], np.maximum(mic["p_in_total"], 0.0), color="0.35", linestyle="--", label="MIC")
    axes[1].set_xlabel("t, s")
    axes[1].set_ylabel("P_in_total^+, W")
    axes[1].legend(frameon=False)

    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)


def _plot_loss_bars(out_path: Path, foc_sum: Dict[str, float], mic_sum: Dict[str, float]) -> None:
    plt = apply_vak_style(ensure_matplotlib())
    labels = ["Total", "Inverter", "Core", "Mech"]
    foc_vals = [
        foc_sum["mean_loss_total"],
        foc_sum["mean_p_inv_loss"],
        foc_sum["mean_p_core_loss"],
        foc_sum["mean_p_mech_loss"],
    ]
    mic_vals = [
        mic_sum["mean_loss_total"],
        mic_sum["mean_p_inv_loss"],
        mic_sum["mean_p_core_loss"],
        mic_sum["mean_p_mech_loss"],
    ]

    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.bar(x - width / 2, foc_vals, width, color="black", label="FOC")
    ax.bar(x + width / 2, mic_vals, width, color="0.45", label="MIC")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Losses, W")
    ax.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, out_path)
    plt.close(fig)


def _delta(foc: Dict[str, float], mic: Dict[str, float], key: str) -> float:
    base = float(foc.get(key, 0.0))
    if abs(base) < 1e-9:
        return 0.0
    return 100.0 * (float(mic.get(key, 0.0)) / base - 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare FOC vs MIC (id_ref shift) with loss breakdown.")
    parser.add_argument("--env-config", required=True)
    parser.add_argument("--scenario", default="hold:0.6")
    parser.add_argument("--load-torque", type=float, default=0.05)
    parser.add_argument("--dt", type=float, default=1e-4)
    parser.add_argument("--t-end", type=float, default=1.2)
    parser.add_argument("--id-ref-foc", type=float, default=0.4)
    parser.add_argument("--id-ref-mic", type=float, default=0.34)
    parser.add_argument("--window-frac", type=float, default=0.25)
    parser.add_argument("--out-dir", default="outputs/study_loss_compare")
    args = parser.parse_args()

    env_cfg = make_env_from_config(args.env_config).env_config
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    foc = _simulate(env_cfg, args.dt, args.t_end, args.scenario, args.load_torque, args.id_ref_foc)
    mic = _simulate(env_cfg, args.dt, args.t_end, args.scenario, args.load_torque, args.id_ref_mic)

    foc_sum = _summarize(foc, args.window_frac)
    mic_sum = _summarize(mic, args.window_frac)

    report = {
        "env_config": str(args.env_config),
        "scenario": str(args.scenario),
        "load_torque": float(args.load_torque),
        "dt": float(args.dt),
        "t_end": float(args.t_end),
        "id_ref_foc": float(args.id_ref_foc),
        "id_ref_mic": float(args.id_ref_mic),
        "window_frac": float(args.window_frac),
        "foc": foc_sum,
        "mic": mic_sum,
        "delta_pct": {
            "mean_abs_speed_err": _delta(foc_sum, mic_sum, "mean_abs_speed_err"),
            "mean_p_in_total_pos": _delta(foc_sum, mic_sum, "mean_p_in_total_pos"),
            "mean_loss_total": _delta(foc_sum, mic_sum, "mean_loss_total"),
            "mean_p_inv_loss": _delta(foc_sum, mic_sum, "mean_p_inv_loss"),
            "mean_p_core_loss": _delta(foc_sum, mic_sum, "mean_p_core_loss"),
            "mean_p_mech_loss": _delta(foc_sum, mic_sum, "mean_p_mech_loss"),
            "mean_i_rms": _delta(foc_sum, mic_sum, "mean_i_rms"),
        },
    }

    _save_csv(out_dir / "timeseries_foc.csv", foc)
    _save_csv(out_dir / "timeseries_mic.csv", mic)
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    _plot_timeseries(out_dir / "timeseries_compare", foc, mic)
    _plot_loss_bars(out_dir / "loss_breakdown", foc_sum, mic_sum)


if __name__ == "__main__":
    main()
