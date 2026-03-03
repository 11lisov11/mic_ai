from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


MODE_DIRS = {
    "mode1_foc_encoder_vs_mic_sensorless": "mode1",
    "mode2_foc_sensorless_vs_mic_sensorless": "mode2",
}

CONTROLLERS = ("PI", "FOC", "MIC")
COLORS = {"PI": "#666666", "FOC": "#1f77b4", "MIC": "#2ca02c"}


def _load_per_seed(step28_dir: Path) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for mode_dir, mode_name in MODE_DIRS.items():
        csv_path = step28_dir / mode_dir / "step27_per_seed_metrics.csv"
        if not csv_path.exists():
            continue
        part = pd.read_csv(csv_path)
        part["mode"] = mode_name
        rows.append(part)
    if not rows:
        raise FileNotFoundError(f"No step27_per_seed_metrics.csv found under {step28_dir}")
    df = pd.concat(rows, ignore_index=True)
    df["controller"] = (
        df["controller"]
        .astype(str)
        .str.upper()
        .replace({"MIC_RULE": "MIC", "MIC_AI": "MIC"})
    )
    for col in (
        "avg_power_saving_pct",
        "avg_eta_gain_pct",
        "err_failures",
        "worst_current_peak_ratio",
        "worst_current_mean_ratio",
        "avg_controller_speed_err",
    ):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _build_stats(df: pd.DataFrame) -> pd.DataFrame:
    grp = df.groupby(["mode", "motor", "controller"], dropna=False)
    stats = grp.agg(
        n=("seed", "nunique"),
        power_mean=("avg_power_saving_pct", "mean"),
        power_std=("avg_power_saving_pct", "std"),
        power_min=("avg_power_saving_pct", "min"),
        power_max=("avg_power_saving_pct", "max"),
        eta_mean=("avg_eta_gain_pct", "mean"),
        eta_std=("avg_eta_gain_pct", "std"),
        eta_min=("avg_eta_gain_pct", "min"),
        eta_max=("avg_eta_gain_pct", "max"),
        err_fail_max=("err_failures", "max"),
        current_peak_ratio_max=("worst_current_peak_ratio", "max"),
        current_mean_ratio_max=("worst_current_mean_ratio", "max"),
        speed_err_mean=("avg_controller_speed_err", "mean"),
    ).reset_index()
    stats["power_worst_case"] = stats["power_min"]
    stats["eta_worst_case"] = stats["eta_min"]
    return stats.sort_values(["mode", "motor", "controller"]).reset_index(drop=True)


def _to_markdown(df: pd.DataFrame) -> str:
    lines: List[str] = []
    lines.append("# IEEE PI/FOC/MIC Stats (3 motors)")
    lines.append("")
    lines.append(
        "| mode | motor | controller | n | power mean | power std | power min | power max | eta mean | eta std | eta min | eta max | worst err_fail |"
    )
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in df.iterrows():
        lines.append(
            "| {mode} | {motor} | {controller} | {n:.0f} | {power_mean:+.3f} | {power_std:.3f} | {power_min:+.3f} | {power_max:+.3f} | {eta_mean:+.3f} | {eta_std:.3f} | {eta_min:+.3f} | {eta_max:+.3f} | {err_fail_max:.0f} |".format(
                **r.to_dict()
            )
        )
    lines.append("")
    return "\n".join(lines)


def _plot_power_bars(stats: pd.DataFrame, out_base: Path) -> None:
    motors = sorted(stats["motor"].astype(str).unique().tolist())
    modes = ("mode1", "mode2")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    width = 0.24
    x = np.arange(len(motors))
    offsets: Dict[str, float] = {"PI": -width, "FOC": 0.0, "MIC": width}

    for ax, mode in zip(axes, modes):
        part = stats[stats["mode"] == mode].copy()
        for ctrl in CONTROLLERS:
            vals = []
            errs = []
            worsts = []
            for motor in motors:
                row = part[(part["motor"] == motor) & (part["controller"] == ctrl)]
                if row.empty:
                    vals.append(np.nan)
                    errs.append(0.0)
                    worsts.append(np.nan)
                else:
                    vals.append(float(row.iloc[0]["power_mean"]))
                    std_val = row.iloc[0]["power_std"]
                    errs.append(0.0 if pd.isna(std_val) else float(std_val))
                    worsts.append(float(row.iloc[0]["power_worst_case"]))
            xpos = x + offsets[ctrl]
            ax.bar(
                xpos,
                vals,
                width=width,
                yerr=errs,
                capsize=3,
                color=COLORS[ctrl],
                alpha=0.9,
                label=ctrl,
                edgecolor="black",
                linewidth=0.5,
            )
            ax.scatter(xpos, worsts, marker="v", color=COLORS[ctrl], s=26, zorder=5)
        ax.axhline(0.0, color="black", linewidth=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(motors)
        ax.set_title(mode)
        ax.set_xlabel("Motor")
        ax.grid(axis="y", linestyle="--", alpha=0.25)
    axes[0].set_ylabel("Power saving, % (mean ± std)\nworst-case = triangle marker")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncols=3, loc="upper center", frameon=False)
    fig.suptitle("PI vs FOC vs MIC: 3 motors, mode1/mode2")
    fig.tight_layout(rect=(0, 0, 1, 0.9))

    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_base.with_suffix(".png")), dpi=300, bbox_inches="tight")
    fig.savefig(str(out_base.with_suffix(".pdf")), bbox_inches="tight")
    fig.savefig(str(out_base.with_suffix(".svg")), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build IEEE tables/figures (mean/std/min/max + worst-case) from step28 package."
    )
    parser.add_argument("--step28-dir", required=True, help="Path to step28 package directory (contains mode1/mode2).")
    parser.add_argument("--out-dir", default="", help="Output directory. Default: <step28-dir>/derived_ieee")
    args = parser.parse_args()

    step28_dir = Path(args.step28_dir).expanduser().resolve()
    if not step28_dir.exists():
        raise FileNotFoundError(step28_dir)
    out_dir = Path(args.out_dir).expanduser().resolve() if str(args.out_dir).strip() else (step28_dir / "derived_ieee")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _load_per_seed(step28_dir)
    stats = _build_stats(df)

    out_csv = out_dir / "ieee_pi_foc_mic_stats.csv"
    out_md = out_dir / "ieee_pi_foc_mic_stats.md"
    out_fig_base = out_dir / "fig_ieee_pi_foc_mic_power"

    stats.to_csv(out_csv, index=False)
    out_md.write_text(_to_markdown(stats), encoding="utf-8")
    _plot_power_bars(stats, out_fig_base)

    print(f"saved: {out_csv}")
    print(f"saved: {out_md}")
    print(f"saved: {out_fig_base.with_suffix('.png')}")
    print(f"saved: {out_fig_base.with_suffix('.pdf')}")
    print(f"saved: {out_fig_base.with_suffix('.svg')}")


if __name__ == "__main__":
    main()
