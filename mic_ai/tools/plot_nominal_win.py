# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np

from mic_ai.tools.plot_style import apply_vak_style, ensure_matplotlib, save_figure


def _load_csv(path: Path) -> Dict[str, np.ndarray]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"CSV is empty: {path}")
    header = lines[0].split(",")
    cols: Dict[str, list[float]] = {name: [] for name in header}
    for line in lines[1:]:
        parts = line.split(",")
        if len(parts) != len(header):
            continue
        for key, value in zip(header, parts):
            cols[key].append(float(value))
    return {key: np.asarray(values, dtype=float) for key, values in cols.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper-ready nominal win figure (FOC vs MIC rule).")
    parser.add_argument("--source-dir", default="outputs/scenario_compare_nominal_rule_id1p0")
    parser.add_argument("--case", default="speed_step_0p2")
    parser.add_argument("--out-dir", default="outputs/paper_win_nominal_speed_step")
    parser.add_argument("--window-frac", type=float, default=0.25)
    args = parser.parse_args()

    base = Path(args.source_dir)
    foc = _load_csv(base / f"{args.case}_foc.csv")
    mic = _load_csv(base / f"{args.case}_mic_ai.csv")

    n = int(min(foc["t"].size, mic["t"].size))
    for key in foc:
        foc[key] = foc[key][:n]
    for key in mic:
        mic[key] = mic[key][:n]

    start = int(n * (1.0 - float(args.window_frac)))
    start = max(0, min(start, max(n - 1, 0)))

    p_foc = np.maximum(foc["p_el"], 0.0)
    p_mic = np.maximum(mic["p_el"], 0.0)

    p_foc_mean = float(np.mean(p_foc[start:])) if n else 0.0
    p_mic_mean = float(np.mean(p_mic[start:])) if n else 0.0
    saving_pct = 0.0
    if p_foc_mean > 1e-9:
        saving_pct = 100.0 * (1.0 - p_mic_mean / p_foc_mean)

    err = np.abs(mic["omega_ref"] - mic["omega"])
    omega_ref_ss = float(np.mean(mic["omega_ref"][start:])) if n else 0.0
    err_rel_pct = 0.0
    if abs(omega_ref_ss) > 1e-9:
        err_rel_pct = 100.0 * float(np.mean(err[start:]) / abs(omega_ref_ss))

    plt = apply_vak_style(ensure_matplotlib(), font_family="serif")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 5.2), sharex=True)

    ax1.plot(foc["t"], p_foc, color="black", label="FOC")
    ax1.plot(mic["t"], p_mic, color="0.35", linestyle="--", label="MIC")
    ax1.axvspan(foc["t"][start], foc["t"][-1], color="0.9", alpha=0.4)
    ax1.text(
        0.98,
        0.93,
        f"Экономия в установившемся режиме:\n{saving_pct:.1f}%",
        fontsize=9,
        ha="right",
        va="top",
        transform=ax1.transAxes,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )
    ax1.set_ylabel("Pэл, Вт")
    ax1.legend(frameon=False)

    err_foc = np.abs(foc["omega_ref"] - foc["omega"])
    err_mic = np.abs(mic["omega_ref"] - mic["omega"])
    ax2.plot(foc["t"], err_foc, color="black", label="FOC")
    ax2.plot(mic["t"], err_mic, color="0.35", linestyle="--", label="MIC")
    ax2.axvspan(foc["t"][start], foc["t"][-1], color="0.9", alpha=0.4)
    ax2.text(
        0.98,
        0.93,
        f"Средняя относит. ошибка MIC:\n{err_rel_pct:.2f}%",
        fontsize=9,
        ha="right",
        va="top",
        transform=ax2.transAxes,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )
    ax2.set_ylabel("|ω_зад - ω|, рад/с")
    ax2.set_xlabel("t, с")

    fig.tight_layout()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_dir / "fig_nominal_win")
    plt.close(fig)

    meta = {
        "source_dir": str(base),
        "case": str(args.case),
        "window_frac": float(args.window_frac),
        "power_saving_pct": saving_pct,
        "mic_err_rel_pct": err_rel_pct,
    }
    (out_dir / "fig_nominal_win_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
