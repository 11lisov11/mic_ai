from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path("outputs/research20260212/study_final")
FINAL_BEST = ROOT / "final_best"


def _ensure_plt():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": ["Times New Roman", "DejaVu Serif", "STIXGeneral"],
            "mathtext.fontset": "stix",
            "font.size": 12,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "lines.linewidth": 2.0,
            "figure.dpi": 220,
            "savefig.dpi": 320,
            "axes.unicode_minus": True,
            "axes.grid": True,
            "grid.alpha": 0.35,
            "grid.linestyle": "-",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "text.usetex": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    return plt


def _save_figure(fig, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".png", ".pdf", ".svg"):
        fig.savefig(out_base.with_suffix(ext), bbox_inches="tight")


def _read_series(tag: str, method: str) -> pd.DataFrame:
    suffix = "foc" if method.lower() == "foc" else "mic_ai"
    path = FINAL_BEST / f"{tag}_{suffix}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _load_all_method(method: str) -> pd.DataFrame:
    suffix = "foc" if method.lower() == "foc" else "mic_ai"
    parts: list[pd.DataFrame] = []
    for fp in sorted(FINAL_BEST.glob(f"*_{suffix}.csv")):
        parts.append(pd.read_csv(fp))
    if not parts:
        raise FileNotFoundError(f"No series for method={method} in {FINAL_BEST}")
    out = pd.concat(parts, ignore_index=True)
    out["P_вх+"] = np.maximum(out["p_el"].to_numpy(dtype=float), 0.0)
    out["P_2"] = np.maximum(out["p_mech"].to_numpy(dtype=float), 0.0)
    omega_abs = np.maximum(np.abs(out["omega"].to_numpy(dtype=float)), 1e-3)
    out["M_2"] = out["P_2"] / omega_abs
    out["n"] = omega_abs * 60.0 / (2.0 * np.pi)
    out["I_1"] = out["i_rms"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        eta = np.where(out["P_вх+"].to_numpy() > 1.0, out["P_2"].to_numpy() / out["P_вх+"].to_numpy(), np.nan)
    out["η"] = np.clip(eta, 0.0, 1.2)
    out["P_2_кВт"] = out["P_2"] / 1000.0
    return out


def _bin_curve(df: pd.DataFrame, p2_max: float, bins_count: int = 24) -> pd.DataFrame:
    clean = df[(df["P_2"] > 5.0) & (np.abs(df["omega"]) > 2.0) & (df["P_2_кВт"] <= p2_max)].copy()
    if clean.empty:
        raise ValueError("No valid points for mechanical curve")
    bins = np.linspace(0.0, p2_max, bins_count)
    clean["bin"] = pd.cut(clean["P_2_кВт"], bins, include_lowest=True)
    grp = clean.groupby("bin", observed=False).agg(
        P2=("P_2_кВт", "mean"),
        M2=("M_2", "mean"),
        n=("n", "mean"),
        I1=("I_1", "mean"),
        eta=("η", "mean"),
    )
    grp = grp.dropna().reset_index(drop=True)
    return grp


def _plot_mechanical_characteristics_ru(out_base: Path) -> None:
    plt = _ensure_plt()
    foc = _load_all_method("foc")
    mic = _load_all_method("mic")

    p2_max = float(min(foc["P_2_кВт"].quantile(0.98), mic["P_2_кВт"].quantile(0.98)))
    p2_max = max(p2_max, 0.05)
    f = _bin_curve(foc, p2_max=p2_max)
    m = _bin_curve(mic, p2_max=p2_max)

    max_m = float(max(f["M2"].max(), m["M2"].max(), 1e-9))
    max_n = float(max(f["n"].max(), m["n"].max(), 1e-9))
    max_i = float(max(f["I1"].max(), m["I1"].max(), 1e-9))

    fig, ax1 = plt.subplots(figsize=(8.3, 5.4))
    ax2 = ax1.twinx()

    var_colors = {
        "M2": "#204a87",  # темно-синий
        "n": "#2e5c3a",  # темно-зеленый
        "I1": "#8f5a2a",  # коричневый
        "eta": "#7a2f2f",  # темно-красный
    }
    style = {"FOC": "-", "MIC": "--"}

    x_f = f["P2"].to_numpy()
    x_m = m["P2"].to_numpy()

    ax1.plot(x_f, f["M2"] / max_m, style["FOC"], color=var_colors["M2"])
    ax1.plot(x_m, m["M2"] / max_m, style["MIC"], color=var_colors["M2"])
    ax1.plot(x_f, f["n"] / max_n, style["FOC"], color=var_colors["n"])
    ax1.plot(x_m, m["n"] / max_n, style["MIC"], color=var_colors["n"])
    ax1.plot(x_f, f["I1"] / max_i, style["FOC"], color=var_colors["I1"])
    ax1.plot(x_m, m["I1"] / max_i, style["MIC"], color=var_colors["I1"])

    ax2.plot(x_f, f["eta"], style["FOC"], color=var_colors["eta"])
    ax2.plot(x_m, m["eta"], style["MIC"], color=var_colors["eta"])

    ax1.set_xlabel("P2, кВт")
    ax1.set_ylabel("M2*, n*, I1*, отн. ед.")
    ax2.set_ylabel("η, отн. ед.")
    ax1.set_xlim(0.0, p2_max)
    ax1.set_ylim(0.0, 1.2)
    ax2.set_ylim(0.0, 1.05)
    ax1.set_title("Сравнение механических характеристик: FOC и MIC")

    # Аннотации кривых в духе классических графиков.
    if len(x_f) > 4:
        ax1.text(x_f[min(len(x_f) - 1, 16)], float((f["M2"] / max_m).iloc[min(len(f) - 1, 16)]) + 0.02, "M2", color=var_colors["M2"])
        ax1.text(x_f[min(len(x_f) - 1, 12)], float((f["n"] / max_n).iloc[min(len(f) - 1, 12)]) + 0.02, "n", color=var_colors["n"])
        ax1.text(x_f[min(len(x_f) - 1, 10)], float((f["I1"] / max_i).iloc[min(len(f) - 1, 10)]) + 0.02, "I1", color=var_colors["I1"])
        ax2.text(x_f[min(len(x_f) - 1, 18)], float(f["eta"].iloc[min(len(f) - 1, 18)]) - 0.03, "η", color=var_colors["eta"])

    method_handles = [
        Line2D([0], [0], color="black", linestyle="-", label="FOC"),
        Line2D([0], [0], color="black", linestyle="--", label="MIC"),
    ]
    var_handles = [
        Line2D([0], [0], color=var_colors["M2"], linestyle="-", label="M2"),
        Line2D([0], [0], color=var_colors["n"], linestyle="-", label="n"),
        Line2D([0], [0], color=var_colors["I1"], linestyle="-", label="I1"),
        Line2D([0], [0], color=var_colors["eta"], linestyle="-", label="η"),
    ]
    leg1 = ax1.legend(handles=method_handles, loc="lower right", frameon=False, title="Метод")
    ax1.add_artist(leg1)
    ax1.legend(handles=var_handles, loc="upper left", frameon=False, title="Параметр")

    fig.tight_layout()
    _save_figure(fig, out_base)
    plt.close(fig)


def _cumtrapz(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    if y.size == 0:
        return np.zeros(0, dtype=float)
    out = np.zeros_like(y, dtype=float)
    if y.size == 1:
        return out
    dx = np.diff(x)
    area = 0.5 * (y[1:] + y[:-1]) * dx
    out[1:] = np.cumsum(area)
    return out


def _plot_power_eta_time_ru(out_base: Path, tag: str = "start_stop") -> None:
    plt = _ensure_plt()
    foc = _read_series(tag, "foc")
    mic = _read_series(tag, "mic")

    t_f = foc["t"].to_numpy(dtype=float)
    t_m = mic["t"].to_numpy(dtype=float)

    p_in_f = np.maximum(foc["p_el"].to_numpy(dtype=float), 0.0)
    p_in_m = np.maximum(mic["p_el"].to_numpy(dtype=float), 0.0)
    p_sh_f = np.maximum(foc["p_mech"].to_numpy(dtype=float), 0.0)
    p_sh_m = np.maximum(mic["p_mech"].to_numpy(dtype=float), 0.0)

    e_in_f = _cumtrapz(p_in_f, t_f)
    e_in_m = _cumtrapz(p_in_m, t_m)
    e_sh_f = _cumtrapz(p_sh_f, t_f)
    e_sh_m = _cumtrapz(p_sh_m, t_m)

    eta_f = np.divide(e_sh_f, e_in_f, out=np.zeros_like(e_sh_f), where=e_in_f > 1e-9)
    eta_m = np.divide(e_sh_m, e_in_m, out=np.zeros_like(e_sh_m), where=e_in_m > 1e-9)

    fig, axes = plt.subplots(2, 1, figsize=(8.3, 6.5), sharex=True)

    axes[0].plot(t_f, p_in_f, color="black", linestyle="-", label="FOC")
    axes[0].plot(t_m, p_in_m, color="black", linestyle="--", label="MIC")
    axes[0].set_ylabel("Pвх+, Вт")
    axes[0].set_title("Сравнение потребления и КПД во времени (сценарий start_stop)")
    axes[0].legend(frameon=False, loc="upper right")

    axes[1].plot(t_f, eta_f, color="black", linestyle="-", label="FOC")
    axes[1].plot(t_m, eta_m, color="black", linestyle="--", label="MIC")
    axes[1].set_ylabel("ηнак, отн. ед.")
    axes[1].set_xlabel("t, c")
    axes[1].set_ylim(0.0, 1.0)

    fig.tight_layout()
    _save_figure(fig, out_base)
    plt.close(fig)


def _plot_power_saving_ru(out_base: Path) -> None:
    plt = _ensure_plt()
    rows = pd.read_csv(ROOT / "scenario_metrics.csv")
    x = np.arange(rows.shape[0])

    full = rows["p_in_saving_full_pct"].to_numpy(dtype=float)
    steady = rows["p_in_saving_steady_pct"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    w = 0.38
    ax.bar(x - w / 2, full, width=w, color="0.25", label="Полный цикл")
    ax.bar(x + w / 2, steady, width=w, color="0.65", label="Установившееся окно")
    ax.axhline(0.0, color="black", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(rows["scenario"].tolist(), rotation=25, ha="right")
    ax.set_ylabel("Экономия Pвх+, %")
    ax.set_title("Экономия входной мощности по сценариям")
    ax.legend(frameon=False)
    fig.tight_layout()
    _save_figure(fig, out_base)
    plt.close(fig)


def _plot_ablation_ru(out_base: Path) -> None:
    plt = _ensure_plt()
    rows = pd.read_csv(ROOT / "ablation_methods_summary.csv")

    ru_names = {
        "FOC baseline": "FOC (база)",
        "MIC fixed (id_ref=1.3)": "MIC фикс. (id_ref=1.3)",
        "MIC rule (1.30/1.35)": "MIC rule (1.30/1.35)",
        "MIC search (hill-climb)": "MIC search",
        "MIC AI eta-aware PPO": "MIC AI (PPO, η-aware)",
    }
    labels = [ru_names.get(x, x) for x in rows["method"].tolist()]
    x = np.arange(len(labels))
    avg = rows["avg_p_in_saving_full_pct"].to_numpy(dtype=float)
    lo = rows["ci95_p_in_saving_lo"].to_numpy(dtype=float)
    hi = rows["ci95_p_in_saving_hi"].to_numpy(dtype=float)
    yerr = np.vstack([avg - lo, hi - avg])

    fig, ax = plt.subplots(figsize=(8.2, 4.5))
    bars = ax.bar(x, avg, color=["0.15", "0.45", "0.65", "0.80"], edgecolor="black")
    ax.errorbar(x, avg, yerr=yerr, fmt="none", ecolor="black", capsize=4, linewidth=1.0)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylabel("Экономия Pвх+, %")
    ax.set_title("Абляционное сравнение методов (с 95% ДИ)")

    for b, v in zip(bars, avg):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.12, f"{v:.2f}", ha="center", va="bottom", fontsize=10)

    fig.tight_layout()
    _save_figure(fig, out_base)
    plt.close(fig)


def _plot_algorithm_block_ru(out_base: Path) -> None:
    plt = _ensure_plt()
    from matplotlib.patches import FancyArrowPatch, Rectangle

    # ГОСТ-стиль: строгая ч/б схема, крупный шрифт, прямоугольные блоки.
    fig, ax = plt.subplots(figsize=(14.0, 7.6))
    fig.patch.set_facecolor("white")
    ax.axis("off")
    ax.set_xlim(0.0, 100.0)
    ax.set_ylim(0.0, 86.0)

    nodes = {
        "sensor": {
            "xy": (2.0, 52.0),
            "wh": (23.0, 24.0),
            "txt": "Измерения\n$\\omega$, $i_d$, $i_q$, $M_{н}$, Pвх+",
        },
        "policy": {"xy": (27.0, 52.0), "wh": (23.0, 24.0), "txt": "MIC AI\n(политика PPO)"},
        "idref": {
            "xy": (52.0, 52.0),
            "wh": (23.0, 24.0),
            "txt": "Ограничение $\\Delta i_d$\nи расчет $i_{d,ref}$",
        },
        "plant": {"xy": (77.0, 52.0), "wh": (21.0, 24.0), "txt": "FOC +\nинвертор + АД"},
        "reward": {
            "xy": (27.0, 16.0),
            "wh": (23.0, 24.0),
            "txt": "Критерий качества:\nmin Pвх+\nпри $|e_\\omega| \\leq e_{dop}$",
        },
        "lut": {"xy": (52.0, 16.0), "wh": (23.0, 24.0), "txt": "Дистилляция\nв LUT (для MCU)"},
    }

    for node in nodes.values():
        x, y = node["xy"]
        w, h = node["wh"]
        ax.add_patch(
            Rectangle(
                (x, y),
                w,
                h,
                linewidth=3.0,
                edgecolor="black",
                facecolor="white",
                zorder=2,
            )
        )
        ax.text(x + w / 2, y + h / 2, node["txt"], ha="center", va="center", fontsize=18.0, zorder=3)

    def arrow(
        p1: tuple[float, float],
        p2: tuple[float, float],
        *,
        label: str | None = None,
        dashed: bool = False,
        label_shift: tuple[float, float] = (0.0, 0.0),
    ) -> None:
        ax.add_patch(
            FancyArrowPatch(
                p1,
                p2,
                arrowstyle="-|>",
                mutation_scale=28,
                linewidth=3.0,
                color="black",
                linestyle="--" if dashed else "-",
                zorder=4,
            )
        )
        if label:
            xm = 0.5 * (p1[0] + p2[0]) + label_shift[0]
            ym = 0.5 * (p1[1] + p2[1]) + label_shift[1]
            ax.text(
                xm,
                ym,
                label,
                fontsize=16,
                ha="center",
                va="center",
                bbox=dict(facecolor="white", edgecolor="none", pad=0.2),
                zorder=5,
            )

    # Основной контур управления.
    arrow((25.0, 64.0), (27.0, 64.0), label="$s_k$", label_shift=(0.0, 4.0))
    arrow((50.0, 64.0), (52.0, 64.0), label="$a_k$", label_shift=(0.0, 4.0))
    arrow((75.0, 64.0), (77.0, 64.0), label="$i_{d,ref}$", label_shift=(0.0, 4.0))

    # Контур обучения и ограничений.
    arrow((13.5, 52.0), (38.5, 40.0), label="оценка режима", dashed=True, label_shift=(0.0, -3.2))
    arrow((38.5, 40.0), (38.5, 52.0), label="$r_k$", dashed=True, label_shift=(-3.5, 0.0))
    arrow((38.5, 52.0), (63.5, 40.0), label="дистилляция", dashed=True, label_shift=(0.0, -3.2))
    arrow((63.5, 40.0), (63.5, 52.0), label="LUT", dashed=True, label_shift=(3.0, 0.0))

    ax.set_title("Блок-схема алгоритма MIC AI", fontsize=22, pad=10)
    fig.tight_layout()
    _save_figure(fig, out_base)
    plt.close(fig)


def main() -> None:
    if not FINAL_BEST.exists():
        raise FileNotFoundError(FINAL_BEST)

    _plot_power_saving_ru(ROOT / "fig_power_saving_ru")
    _plot_ablation_ru(ROOT / "fig_ablation_methods_ru")
    _plot_mechanical_characteristics_ru(ROOT / "fig_mech_characteristics_ru")
    _plot_power_eta_time_ru(ROOT / "fig_power_eta_time_ru", tag="start_stop")
    _plot_algorithm_block_ru(ROOT / "fig_algorithm_block_ru")
    print(str(ROOT.resolve()))


if __name__ == "__main__":
    main()
