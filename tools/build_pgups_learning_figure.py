from __future__ import annotations

"""
Build Fig. "fig_learning_vs_foc_ru" for the PGUPS paper.

This script intentionally depends ONLY on the committed paper tables in
`paper/pgups_2026/data/` (no training logs, no RL checkpoints).
"""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd


def _setup_plot() -> None:
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": ["Times New Roman", "DejaVu Serif", "STIXGeneral"],
            "mathtext.fontset": "stix",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "axes.grid": True,
            "grid.alpha": 0.30,
            "grid.linestyle": "-",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def _save_all(fig, out_base: Path, stem: str) -> None:
    out_base.mkdir(parents=True, exist_ok=True)
    for ext in (".png", ".pdf", ".svg"):
        fig.savefig(out_base / f"{stem}{ext}", bbox_inches="tight", dpi=320)


def main() -> None:
    _setup_plot()
    import matplotlib.pyplot as plt

    paper_dir = Path("paper/pgups_2026")
    data_dir = paper_dir / "data"
    fig_dir = paper_dir / "fig"

    ms = pd.read_csv(data_dir / "motor_summary_multi_motor.csv")
    ts = pd.read_csv(data_dir / "time_to_foc_summary_ru.csv")

    df = ms.merge(ts, on=["motor_key", "motor_label"], how="left")

    # Ensure numeric columns.
    for col in (
        "avg_saving_full_pct",
        "avg_saving_steady_pct",
        "t_equal_foc_wall_s",
        "t_better_foc_wall_s",
    ):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    x = np.arange(len(df))
    w = 0.34
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 6.6), sharex=True)

    axes[0].bar(
        x - w / 2,
        df["avg_saving_full_pct"].to_numpy(float),
        width=w,
        color="0.25",
        label="Экономия к FOC (полный интервал)",
    )
    axes[0].bar(
        x + w / 2,
        df["avg_saving_steady_pct"].to_numpy(float),
        width=w,
        color="0.62",
        label="Экономия к FOC (установившееся окно)",
    )
    axes[0].axhline(0.0, color="black", lw=0.9)
    axes[0].set_ylabel("Экономия Pвх+, %")
    axes[0].set_title("Обучаемость MIC AI: итоговая экономия и время выхода на уровень FOC")
    axes[0].legend(frameon=False, loc="upper right")

    t_eq = df["t_equal_foc_wall_s"].to_numpy(float)
    t_better = df["t_better_foc_wall_s"].to_numpy(float)
    axes[1].bar(
        x - w / 2,
        t_eq,
        width=w,
        color="0.35",
        label="Время до уровня «не хуже FOC»",
    )
    axes[1].bar(
        x + w / 2,
        t_better,
        width=w,
        color="0.72",
        label="Время до уровня «лучше FOC»",
    )
    axes[1].set_ylabel("Время, с")
    axes[1].set_xlabel("Двигатель")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(df["motor_label"].tolist(), rotation=8)
    ymax = float(np.nanmax(np.r_[t_eq, t_better]))
    axes[1].set_ylim(0.0, max(60.0, ymax * 1.15 + 1.0))
    axes[1].legend(frameon=False, loc="upper right")

    fig.tight_layout()
    _save_all(fig, fig_dir, "fig_learning_vs_foc_ru")
    plt.close(fig)

    print(f"OK: wrote {fig_dir / 'fig_learning_vs_foc_ru.png'}")


if __name__ == "__main__":
    main()

