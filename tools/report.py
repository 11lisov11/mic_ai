from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from mic_ai.tools.plot_style import apply_vak_style, ensure_matplotlib, save_figure


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(path)
    return {k: data[k] for k in data.files}


def _plot_series(ax, t: np.ndarray, y: np.ndarray, label: str, color: str, style: str = "-") -> None:
    ax.plot(t, y, style, color=color, label=label)


def build_report(
    foc_npz: Path,
    mic_npz: Path,
    out_dir: Path,
    title: Optional[str] = None,
    metrics_foc: Optional[Dict[str, float]] = None,
    metrics_mic: Optional[Dict[str, float]] = None,
) -> Path:
    foc = _load_npz(foc_npz)
    mic = _load_npz(mic_npz)

    plt = apply_vak_style(ensure_matplotlib())
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.6), sharex=True)

    _plot_series(axes[0, 0], foc["t"], foc["omega"], "FOC", "black")
    _plot_series(axes[0, 0], mic["t"], mic["omega"], "MIC", "0.35", "--")
    _plot_series(axes[0, 0], foc["t"], foc["omega_ref"], "ref", "0.6", ":")
    axes[0, 0].set_ylabel("omega, rad/s")

    _plot_series(axes[0, 1], foc["t"], foc["i_rms"], "FOC", "black")
    _plot_series(axes[0, 1], mic["t"], mic["i_rms"], "MIC", "0.35", "--")
    axes[0, 1].set_ylabel("i_rms, A")

    _plot_series(axes[1, 0], foc["t"], foc["p_el"], "FOC", "black")
    _plot_series(axes[1, 0], mic["t"], mic["p_el"], "MIC", "0.35", "--")
    axes[1, 0].set_ylabel("p_in, W")

    _plot_series(axes[1, 1], foc["t"], foc["p_mech"], "FOC", "black")
    _plot_series(axes[1, 1], mic["t"], mic["p_mech"], "MIC", "0.35", "--")
    axes[1, 1].set_ylabel("p_mech, W")

    for ax in axes[1, :]:
        ax.set_xlabel("t, s")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)

    if title:
        fig.suptitle(title, y=0.98)

    if metrics_foc or metrics_mic:
        lines = []
        if metrics_foc:
            lines.append(
                f"FOC score={metrics_foc.get('score', 0.0):.3g}, "
                f"err={metrics_foc.get('mean_abs_speed_error', 0.0):.3g}, "
                f"i_rms={metrics_foc.get('mean_i_rms', 0.0):.3g}"
            )
        if metrics_mic:
            lines.append(
                f"MIC score={metrics_mic.get('score', 0.0):.3g}, "
                f"err={metrics_mic.get('mean_abs_speed_error', 0.0):.3g}, "
                f"i_rms={metrics_mic.get('mean_i_rms', 0.0):.3g}"
            )
        fig.text(0.02, 0.02, "\n".join(lines), fontsize=9)

    fig.tight_layout(rect=[0.0, 0.05, 1.0, 0.95])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"report_{foc_npz.stem}_vs_{mic_npz.stem}.png"
    save_figure(fig, out_path)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build report from FOC/MIC NPZ logs.")
    parser.add_argument("--foc-npz", required=True, help="Path to FOC NPZ log.")
    parser.add_argument("--mic-npz", required=True, help="Path to MIC NPZ log.")
    parser.add_argument("--out-dir", default="outputs/reports", help="Output directory.")
    parser.add_argument("--title", default=None, help="Report title.")
    args = parser.parse_args()

    out_path = build_report(Path(args.foc_npz), Path(args.mic_npz), Path(args.out_dir), title=args.title)
    print(f"[report] saved: {out_path}")


if __name__ == "__main__":
    main()
