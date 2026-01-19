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
    parser = argparse.ArgumentParser(description="Paper-ready power-only figure (FOC vs MIC).")
    parser.add_argument("--source-dir", default="outputs/scenario_compare_nominal_rule_id1p0")
    parser.add_argument("--case", default="speed_step_0p2")
    parser.add_argument("--out-dir", default="outputs/paper_win_nominal_power")
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

    plt = apply_vak_style(ensure_matplotlib(), font_family="serif")
    fig, ax = plt.subplots(figsize=(7.2, 3.3))

    ax.plot(foc["t"], p_foc, color="black", label="FOC")
    ax.plot(mic["t"], p_mic, color="0.35", linestyle="--", label="MIC")
    ax.set_ylabel("P_эл, Вт")
    ax.set_xlabel("t, с")
    ax.legend(frameon=False)

    fig.tight_layout()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_dir / "fig_nominal_power")
    plt.close(fig)

    meta = {
        "source_dir": str(base),
        "case": str(args.case),
        "window_frac": float(args.window_frac),
        "power_saving_pct": saving_pct,
    }
    (out_dir / "fig_nominal_power_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
