from __future__ import annotations

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
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
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


def _save_all(fig, name: str, out_dirs: list[Path]) -> None:
    for out in out_dirs:
        out.mkdir(parents=True, exist_ok=True)
        for ext in (".png", ".pdf", ".svg"):
            fig.savefig(out / f"{name}{ext}", bbox_inches="tight", dpi=320)


def regen_learning_vs_foc(out_dirs: list[Path]) -> None:
    import matplotlib.pyplot as plt

    # Use the corrected multi-motor study summary as the single source of truth for
    # savings/eta metrics, and join it with the (separately computed) learning-time table.
    src = Path("outputs/research20260212/study_final/motor_summary_multi_motor.csv")
    df = pd.read_csv(src)
    time_src = Path("outputs/research20260212/study_final/time_to_foc_summary_ru.csv")
    tdf = pd.read_csv(time_src)

    tdf = tdf.set_index("motor_key")
    df = df.set_index("motor_key").join(
        tdf[["t_equal_foc_wall_s", "t_better_foc_wall_s", "not_reached_until_wall_s"]],
        how="left",
    ).reset_index()
    for col in ("t_equal_foc_wall_s", "t_better_foc_wall_s", "not_reached_until_wall_s"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    x = np.arange(len(df))
    fig, axes = plt.subplots(2, 1, figsize=(8.6, 6.4), sharex=True)
    w = 0.34

    axes[0].bar(
        x - w / 2,
        df["avg_saving_full_pct"].to_numpy(float),
        width=w,
        color="0.25",
        label="\u042d\u043a\u043e\u043d\u043e\u043c\u0438\u044f \u043a FOC (\u043f\u043e\u043b\u043d\u044b\u0439 \u0446\u0438\u043a\u043b)",
    )
    axes[0].bar(
        x + w / 2,
        df["avg_saving_steady_pct"].to_numpy(float),
        width=w,
        color="0.62",
        label="\u042d\u043a\u043e\u043d\u043e\u043c\u0438\u044f \u043a FOC (\u0443\u0441\u0442\u0430\u043d\u043e\u0432\u0438\u0432\u0448\u0435\u0435\u0441\u044f \u043e\u043a\u043d\u043e)",
    )
    axes[0].axhline(0, color="black", lw=0.9)
    axes[0].set_ylabel("\u042d\u043a\u043e\u043d\u043e\u043c\u0438\u044f P\u0432\u0445+, %")
    axes[0].set_title(
        "\u041e\u0431\u0443\u0447\u0430\u0435\u043c\u043e\u0441\u0442\u044c MIC AI: \u0438\u0442\u043e\u0433\u043e\u0432\u0430\u044f \u044d\u043a\u043e\u043d\u043e\u043c\u0438\u044f \u0438 \u0432\u0440\u0435\u043c\u044f \u0432\u044b\u0445\u043e\u0434\u0430 \u043d\u0430 \u0443\u0440\u043e\u0432\u0435\u043d\u044c FOC"
    )
    axes[0].legend(frameon=False, loc="upper right")

    t_eq = df["t_equal_foc_wall_s"].to_numpy(float)
    t_better = df["t_better_foc_wall_s"].to_numpy(float)
    t_not = df["not_reached_until_wall_s"].to_numpy(float)
    axes[1].bar(
        x - w / 2,
        t_eq,
        width=w,
        color="0.35",
        label="\u0412\u0440\u0435\u043c\u044f \u0434\u043e \u0443\u0440\u043e\u0432\u043d\u044f \u00ab\u043d\u0435 \u0445\u0443\u0436\u0435 FOC\u00bb",
    )
    axes[1].bar(
        x + w / 2,
        t_better,
        width=w,
        color="0.72",
        label="\u0412\u0440\u0435\u043c\u044f \u0434\u043e \u0443\u0440\u043e\u0432\u043d\u044f \u00ab\u043b\u0443\u0447\u0448\u0435 FOC\u00bb",
    )
    shown_not_reached = False
    for i, tn in enumerate(t_not):
        if np.isfinite(tn):
            axes[1].bar(
                x[i] - w / 2,
                tn,
                width=w,
                facecolor="white",
                edgecolor="0.25",
                linewidth=1.6,
                label="\u041d\u0435 \u0434\u043e\u0441\u0442\u0438\u0433\u043d\u0443\u0442\u043e (\u043d\u0438\u0436\u043d\u044f\u044f \u0433\u0440\u0430\u043d\u0438\u0446\u0430 \u043f\u043e \u0432\u0440\u0435\u043c\u0435\u043d\u0438)" if not shown_not_reached else None,
            )
            axes[1].bar(
                x[i] + w / 2,
                tn,
                width=w,
                facecolor="white",
                edgecolor="0.25",
                linewidth=1.6,
            )
            axes[1].text(x[i], tn + 8.0, "\u043d/\u0434", ha="center", va="bottom", fontsize=10)
            shown_not_reached = True

    axes[1].set_ylabel("\u0412\u0440\u0435\u043c\u044f, \u0441")
    axes[1].set_xlabel("\u0414\u0432\u0438\u0433\u0430\u0442\u0435\u043b\u044c")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(df["motor_label"].tolist(), rotation=8)
    if np.isfinite(t_not).any():
        ymax = float(np.nanmax(np.where(np.isfinite(t_not), t_not, 0.0)))
    else:
        ymax = float(np.nanmax(np.where(np.isfinite(t_better), t_better, 0.0)))
    axes[1].set_ylim(0.0, max(60.0, ymax * 1.15 + 1.0))
    axes[1].legend(frameon=False, loc="upper right")

    fig.tight_layout()
    _save_all(fig, "fig_learning_vs_foc_ru", out_dirs)
    plt.close(fig)


def regen_scenario_heatmap(out_dirs: list[Path]) -> None:
    import matplotlib.pyplot as plt

    src = Path("outputs/research20260212/study_final/scenario_metrics_multi_motor.csv")
    df = pd.read_csv(src)

    motor_map = {
        "AIR56 0.25 kW": "\u0410\u0418\u042056 0,25 \u043a\u0412\u0442",
        "AL-31-4 0.6 kW": "\u0410\u041b-31-4 0,6 \u043a\u0412\u0442",
        "AO2-32-4 3.0 kW": "\u0410\u041e2-32-4 3,0 \u043a\u0412\u0442",
    }
    df["motor_label"] = df["motor_label"].replace(motor_map)

    scenario_order = ["hold:0.8", "speed_step", "ramp", "load_profile", "start_stop"]
    scenario_ru = {
        "hold:0.8": "\u0423\u0441\u0442\u0430\u043d. \u0440\u0435\u0436\u0438\u043c",
        "speed_step": "\u0421\u0442\u0443\u043f\u0435\u043d\u044c \u0441\u043a\u043e\u0440\u043e\u0441\u0442\u0438",
        "ramp": "\u0420\u0430\u0437\u0433\u043e\u043d/\u0442\u043e\u0440\u043c.",
        "load_profile": "\u041f\u0440\u043e\u0444\u0438\u043b\u044c \u043d\u0430\u0433\u0440\u0443\u0437\u043a\u0438",
        "start_stop": "\u041f\u0443\u0441\u043a-\u0441\u0442\u043e\u043f",
    }

    pivot = df.pivot(index="motor_label", columns="scenario", values="saving_full_pct")
    pivot = pivot.reindex(columns=[c for c in scenario_order if c in pivot.columns])
    data = pivot.to_numpy(dtype=float)
    vmax = float(max(np.max(np.abs(data)), 1.0))

    fig, ax = plt.subplots(figsize=(8.8, 3.9))
    im = ax.imshow(data, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels([scenario_ru.get(c, c) for c in pivot.columns], rotation=18, ha="right")
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels(pivot.index.tolist())
    ax.set_title(
        "\u042d\u043a\u043e\u043d\u043e\u043c\u0438\u044f P\u0432\u0445+ \u043f\u043e \u0441\u0446\u0435\u043d\u0430\u0440\u0438\u044f\u043c, % (MIC \u043e\u0442\u043d\u043e\u0441\u0438\u0442\u0435\u043b\u044c\u043d\u043e FOC)"
    )

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{data[i, j]:.1f}", ha="center", va="center", fontsize=9)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("\u042d\u043a\u043e\u043d\u043e\u043c\u0438\u044f, %")
    fig.tight_layout()
    _save_all(fig, "fig_multi_motor_scenario_heatmap_ru", out_dirs)
    plt.close(fig)


def main() -> None:
    _setup_plot()
    out_dirs = [
        Path("outputs/research20260212/study_final"),
        Path("outputs/research20260213/multi_motor_study"),
    ]
    regen_learning_vs_foc(out_dirs)
    regen_scenario_heatmap(out_dirs)
    print("OK: refreshed fig_learning_vs_foc_ru and fig_multi_motor_scenario_heatmap_ru")


if __name__ == "__main__":
    main()
