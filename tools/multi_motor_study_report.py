from __future__ import annotations

import argparse
import shutil
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MotorCase:
    key: str
    label: str
    config: str
    run_dir: Path
    rated_power_kw: float


MOTORS_META: Tuple[dict, ...] = (
    {
        "key": "air56",
        "label": "АИР56 0,25 кВт",
        "config": "config/env_research_air56_025kw.py",
        "rated_power_kw": 0.25,
        # Legacy (local) evaluation directories (not committed).
        "legacy_run_dir": "outputs/research20260214/air56_ep000_eval_fixed2",
    },
    {
        "key": "al31",
        "label": "АЛ-31-4 0,6 кВт",
        "config": "config/env_research_al31_4_06kw.py",
        "rated_power_kw": 0.60,
        # Use the best checkpoint found in corrected evaluation runs.
        "legacy_run_dir": "outputs/research20260214/al31_ep019_eval_fixed",
    },
    {
        "key": "ao2",
        "label": "АО2-32-4 3,0 кВт",
        "config": "config/env_research_ao2_32_4_3kw.py",
        "rated_power_kw": 3.00,
        "legacy_run_dir": "outputs/research20260214/ao2_ep001_eval_fixed3",
    },
)

DEFAULT_TRACES_ROOT = Path("paper/pgups_2026/data/traces")
DEFAULT_OUT_DIR = Path("outputs/research20260214/multi_motor_study")
DEFAULT_PAPER_DIR = Path("paper/pgups_2026")
WINDOW_FRAC = 0.30


def _infer_scenario_from_tag(file_tag: str) -> str:
    """
    Infer scenario name from a trace file tag.

    Historically, evaluation runs stored a `summary.json` with explicit (scenario, file_tag) mapping.
    For the committed paper traces we keep only `*_foc.csv` / `*_mic_ai.csv` files, so we reconstruct
    the mapping from file names.
    """

    tag = str(file_tag).strip()
    if tag.startswith("hold_"):
        # hold_0p8 -> hold:0.8 (also support other values like hold_1p0)
        pu = tag[len("hold_") :].replace("p", ".", 1)
        return f"hold:{pu}"
    return tag


def _load_summary_items(run_dir: Path) -> List[Dict[str, str]]:
    """
    Load (scenario, file_tag) pairs for a given motor run directory.

    Priority:
    1) Use `summary.json` if present (legacy local runs).
    2) Otherwise infer from the committed trace CSV names: `*_foc.csv` + `*_mic_ai.csv`.
    """

    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        items = json.loads(summary_path.read_text(encoding="utf-8"))
        out: List[Dict[str, str]] = []
        for it in items:
            scenario = str(it.get("scenario", "")).strip()
            tag = str(it.get("file_tag", "")).strip()
            if not scenario or not tag:
                continue
            out.append({"scenario": scenario, "file_tag": tag})
        if out:
            return out

    foc_files = sorted(run_dir.glob("*_foc.csv"))
    inferred: List[Dict[str, str]] = []
    for foc_path in foc_files:
        name = foc_path.name
        tag = name[: -len("_foc.csv")]
        mic_path = run_dir / f"{tag}_mic_ai.csv"
        if not mic_path.exists():
            continue
        inferred.append({"scenario": _infer_scenario_from_tag(tag), "file_tag": tag})

    if not inferred:
        raise FileNotFoundError(f"Missing traces in {run_dir} (expected '*_foc.csv' + '*_mic_ai.csv').")

    # Stable and publication-friendly ordering.
    # IEEE protocol uses load_step; keep load_profile as legacy alias for PGUPS traces.
    scenario_order = ["hold:0.8", "speed_step", "ramp", "load_step", "load_profile", "start_stop"]
    order_idx = {name: i for i, name in enumerate(scenario_order)}
    inferred.sort(key=lambda it: (order_idx.get(it["scenario"], 999), it["scenario"], it["file_tag"]))
    return inferred


def _ensure_plt():
    import matplotlib

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
    return plt


def _save_figure(fig, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".png", ".pdf", ".svg"):
        fig.savefig(out_base.with_suffix(ext), bbox_inches="tight", dpi=320)


def _steady_slice(n: int, frac: float) -> slice:
    frac = float(max(min(frac, 0.95), 0.05))
    start = int(max(0, n * (1.0 - frac)))
    return slice(start, n)


def _steady_slice_by_scenario(t: np.ndarray, omega_ref: np.ndarray, scenario: str, frac: float) -> slice:
    """
    Scenario-aware definition of the "steady window".

    Default (most scenarios): last `frac` of the record, which corresponds to the final quasi-steady segment
    for `hold`, `speed_step`, `ramp` and `load_step` in our protocol (`load_profile` is treated as legacy alias).

    Special case: `start_stop` has a terminal deceleration and stop, so the last window is NOT representative of
    steady operation. For this scenario we select the plateau where ωref is close to its maximum and nearly constant,
    then take the last `frac` of that plateau.
    """

    n = int(t.size)
    if n <= 0:
        return slice(0, 0)
    base = _steady_slice(n, frac)
    if str(scenario).strip().lower() != "start_stop" or n < 10:
        return base

    omega_ref_abs = np.abs(omega_ref)
    omega_ref_max = float(np.max(omega_ref_abs))
    if not np.isfinite(omega_ref_max) or omega_ref_max <= 1e-9:
        return base

    # Plateau: near-max ωref and low derivative.
    mask = omega_ref_abs >= 0.95 * omega_ref_max
    if n >= 2:
        domega_ref_dt = np.abs(np.gradient(omega_ref, t))
        mask &= domega_ref_dt <= 1.0
    idx = np.flatnonzero(mask)
    if idx.size < 10:
        return base

    start = int(idx[0])
    end = int(idx[-1]) + 1
    m = int(end - start)
    if m <= 0:
        return base
    w = _steady_slice(m, frac)
    return slice(start + w.start, start + w.stop)


def _read_series(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    if y.size <= 1:
        return float(np.sum(y))
    return float(np.trapezoid(y, x))


def _series_metrics(df: pd.DataFrame, window_frac: float, scenario: str) -> Dict[str, float]:
    t = df["t"].to_numpy(dtype=float)
    omega = df["omega"].to_numpy(dtype=float)
    omega_ref = df["omega_ref"].to_numpy(dtype=float)
    p_in = np.maximum(df["p_el"].to_numpy(dtype=float), 0.0)
    p_shaft = np.maximum(df["p_mech"].to_numpy(dtype=float), 0.0)
    err = np.abs(omega_ref - omega)

    n = int(t.size)
    sl = _steady_slice_by_scenario(t, omega_ref, scenario, window_frac)
    t_w = t[sl]
    p_in_w = p_in[sl]
    p_shaft_w = p_shaft[sl]
    err_w = err[sl]

    e_in_full = _trapz(p_in, t)
    e_shaft_full = _trapz(p_shaft, t)
    e_in_w = _trapz(p_in_w, t_w) if t_w.size else 0.0
    e_shaft_w = _trapz(p_shaft_w, t_w) if t_w.size else 0.0

    eta_full = e_shaft_full / max(e_in_full, 1e-9)
    eta_w = e_shaft_w / max(e_in_w, 1e-9)
    mae_full = float(np.mean(err)) if err.size else 0.0
    mae_w = float(np.mean(err_w)) if err_w.size else 0.0
    mean_p_in_full = float(np.mean(p_in)) if p_in.size else 0.0
    mean_p_shaft_full = float(np.mean(p_shaft)) if p_shaft.size else 0.0
    mean_p_in_w = float(np.mean(p_in_w)) if p_in_w.size else 0.0
    mean_p_shaft_w = float(np.mean(p_shaft_w)) if p_shaft_w.size else 0.0

    return {
        "mean_p_in_full": mean_p_in_full,
        "mean_p_shaft_full": mean_p_shaft_full,
        "energy_in_full": e_in_full,
        "energy_shaft_full": e_shaft_full,
        "eta_full": eta_full,
        "mae_full": mae_full,
        "mean_p_in_steady": mean_p_in_w,
        "mean_p_shaft_steady": mean_p_shaft_w,
        "energy_in_steady": e_in_w,
        "energy_shaft_steady": e_shaft_w,
        "eta_steady": eta_w,
        "mae_steady": mae_w,
    }


def _pct_saving(base: float, alt: float) -> float:
    return 100.0 * (1.0 - alt / max(base, 1e-9))


def _eta_gain(base: float, alt: float) -> float:
    return 100.0 * (alt / max(base, 1e-9) - 1.0)


def _cumtrapz(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    if y.size == 0:
        return np.zeros(0, dtype=float)
    out = np.zeros_like(y, dtype=float)
    if y.size == 1:
        return out
    area = 0.5 * (y[1:] + y[:-1]) * np.diff(x)
    out[1:] = np.cumsum(area)
    return out


def _prepare_mech_points(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build points for "mechanical characteristics" plots.

    Key idea: exclude obvious transients (start/stop, speed ramps) so the curves resemble
    classic steady-state characteristics: parameters vs P2 at (approximately) constant ωref.

    Notes:
    - We do NOT require access to motor parameters; only time-series signals from the digital twin.
    - Speed is represented in relative units n* = |ω|/|ωref| so curves stay comparable across scenarios
      with different reference levels (0.8 pu vs 1.0 pu).
    """

    t = df["t"].to_numpy(dtype=float)
    omega = df["omega"].to_numpy(dtype=float)
    omega_ref = df["omega_ref"].to_numpy(dtype=float)
    p2 = np.maximum(df["p_mech"].to_numpy(dtype=float), 0.0)
    p_in = np.maximum(df["p_el"].to_numpy(dtype=float), 0.0)
    i1 = df["i_rms"].to_numpy(dtype=float)

    omega_abs = np.maximum(np.abs(omega), 1e-3)
    omega_ref_abs = np.abs(omega_ref)
    omega_ref_max = float(np.max(omega_ref_abs)) if omega_ref_abs.size else 0.0
    omega_ref_min = max(0.20 * omega_ref_max, 1e-6)  # reject low-speed reference parts (start/stop tail)

    # Identify quasi-steady intervals: constant ωref (exclude ramps) and bounded tracking error.
    if t.size >= 2:
        domega_ref_dt = np.gradient(omega_ref, t)
    else:
        domega_ref_dt = np.zeros_like(omega_ref)
    err = np.abs(omega_ref - omega)
    err_limit = 1.0 + 0.02 * np.maximum(omega_ref_abs, 1.0)  # abs + relative tolerance, rad/s
    mask = (omega_ref_abs > omega_ref_min) & (np.abs(domega_ref_dt) <= 1.0) & (err <= err_limit)
    # Fallback: if filtering is too strict for a given trace, keep only non-zero ωref parts.
    if not bool(np.any(mask)):
        mask = omega_ref_abs > omega_ref_min

    omega_abs = omega_abs[mask]
    omega_ref_abs = np.maximum(omega_ref_abs[mask], 1e-3)
    p2 = p2[mask]
    p_in = p_in[mask]
    i1 = i1[mask]

    torque = p2 / omega_abs
    n_rel = omega_abs / omega_ref_abs
    eta = np.divide(p2, p_in, out=np.zeros_like(p2), where=p_in > 1e-9)
    return pd.DataFrame(
        {
            "P2_kW": p2 / 1000.0,
            "M2": torque,
            "n": n_rel,
            "I1": i1,
            "eta": np.clip(eta, 0.0, 1.2),
        }
    )


def _bin_curve(df: pd.DataFrame, p2_max: float, bins_count: int = 24) -> pd.DataFrame:
    clean = df[(df["P2_kW"] > 1e-4) & (df["P2_kW"] <= p2_max)].copy()
    if clean.empty:
        return pd.DataFrame(columns=["P2", "M2", "n", "I1", "eta"])
    bins = np.linspace(0.0, p2_max, bins_count)
    clean["bin"] = pd.cut(clean["P2_kW"], bins, include_lowest=True)
    grp = clean.groupby("bin", observed=False).agg(P2=("P2_kW", "mean"), M2=("M2", "mean"), n=("n", "mean"), I1=("I1", "mean"), eta=("eta", "mean"))
    return grp.dropna().reset_index(drop=True)


def _plot_summary_bars(summary_df: pd.DataFrame, out_base: Path) -> None:
    plt = _ensure_plt()
    x = np.arange(summary_df.shape[0])
    labels = summary_df["motor_label"].tolist()

    fig, axes = plt.subplots(2, 1, figsize=(8.4, 6.4), sharex=True)
    w = 0.36

    full = summary_df["avg_saving_full_pct"].to_numpy(dtype=float)
    steady = summary_df["avg_saving_steady_pct"].to_numpy(dtype=float)
    axes[0].bar(x - w / 2, full, width=w, color="0.20", label="Полный цикл")
    axes[0].bar(x + w / 2, steady, width=w, color="0.60", label="Установившееся окно")
    axes[0].axhline(0.0, color="black", linewidth=0.9)
    axes[0].set_ylabel("Экономия $P_{вх+}$, %")
    axes[0].set_title("Сравнение FOC и MIC AI по экономии входной мощности")
    axes[0].legend(frameon=False)

    eta_full = summary_df["avg_eta_gain_full_pct"].to_numpy(dtype=float)
    eta_steady = summary_df["avg_eta_gain_steady_pct"].to_numpy(dtype=float)
    axes[1].bar(x - w / 2, eta_full, width=w, color="0.25", label="Полный цикл")
    axes[1].bar(x + w / 2, eta_steady, width=w, color="0.65", label="Установившееся окно")
    axes[1].axhline(0.0, color="black", linewidth=0.9)
    axes[1].set_ylabel("Прирост $\\eta$, %")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=10)
    axes[1].set_xlabel("Двигатель")
    axes[1].legend(frameon=False)

    fig.tight_layout()
    _save_figure(fig, out_base)
    plt.close(fig)


def _plot_savings_heatmap(scenario_df: pd.DataFrame, out_base: Path) -> None:
    plt = _ensure_plt()
    data_df = scenario_df.copy()
    data_df["scenario_norm"] = data_df["scenario"].astype(str).replace({"load_profile": "load_step"})
    pivot = data_df.pivot(index="motor_label", columns="scenario_norm", values="saving_full_pct")
    scenario_order = ["hold:0.8", "speed_step", "ramp", "load_step", "start_stop"]
    pivot = pivot.reindex(columns=[c for c in scenario_order if c in pivot.columns])
    scenario_ru = {
        "hold:0.8": "Установившийся режим",
        "speed_step": "Ступень скорости",
        "ramp": "Разгон/торможение",
        "load_step": "Шаг нагрузки",
        "start_stop": "Пуск—стоп",
    }
    pivot.columns = [scenario_ru.get(str(c), str(c)) for c in pivot.columns]

    data = pivot.to_numpy(dtype=float)
    vmax = float(max(np.max(np.abs(data)), 1.0))
    fig, ax = plt.subplots(figsize=(8.8, 3.8))
    im = ax.imshow(data, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns.tolist(), rotation=20, ha="right")
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels(pivot.index.tolist())
    ax.set_title("Экономия $P_{вх+}$ по сценариям, % (MIC AI относительно FOC)")

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            # Use comma as decimal separator (RU typographic convention).
            txt = f"{data[i, j]:.1f}".replace(".", ",")
            ax.text(j, i, txt, ha="center", va="center", fontsize=9)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Экономия, %")
    fig.tight_layout()
    _save_figure(fig, out_base)
    plt.close(fig)


def _plot_mech_characteristics(curves: Dict[str, Dict[str, pd.DataFrame]], summary_df: pd.DataFrame, out_base: Path) -> None:
    plt = _ensure_plt()
    fig, axes = plt.subplots(3, 1, figsize=(9.2, 12.0), sharex=False)

    var_colors = {"M2": "#1f4e79", "n": "#2f6b3f", "I1": "#8a5a20", "eta": "#7a3030"}
    method_handles = [
        plt.Line2D([0], [0], color="black", linestyle="-", linewidth=2.0, label="FOC"),
        plt.Line2D([0], [0], color="black", linestyle="--", linewidth=2.0, label="MIC"),
    ]
    param_handles = [
        plt.Line2D([0], [0], color=var_colors["M2"], linestyle="-", linewidth=2.0, label="$M_2$"),
        plt.Line2D([0], [0], color=var_colors["n"], linestyle="-", linewidth=2.0, label="$n$"),
        plt.Line2D([0], [0], color=var_colors["I1"], linestyle="-", linewidth=2.0, label="$I_1$"),
        plt.Line2D([0], [0], color=var_colors["eta"], linestyle="-", linewidth=2.0, label="$\\eta$"),
    ]

    for idx, (ax, motor) in enumerate(zip(axes, summary_df["motor_key"].tolist())):
        foc = curves[motor]["foc"]
        mic = curves[motor]["mic"]
        if foc.empty or mic.empty:
            ax.set_visible(False)
            continue
        ax2 = ax.twinx()

        max_m = float(max(foc["M2"].max(), mic["M2"].max(), 1e-9))
        max_n = float(max(foc["n"].max(), mic["n"].max(), 1e-9))
        max_i = float(max(foc["I1"].max(), mic["I1"].max(), 1e-9))

        ax.plot(foc["P2"], foc["M2"] / max_m, "-", color=var_colors["M2"], linewidth=2.0)
        ax.plot(mic["P2"], mic["M2"] / max_m, "--", color=var_colors["M2"], linewidth=2.0)
        ax.plot(foc["P2"], foc["n"] / max_n, "-", color=var_colors["n"], linewidth=2.0)
        ax.plot(mic["P2"], mic["n"] / max_n, "--", color=var_colors["n"], linewidth=2.0)
        ax.plot(foc["P2"], foc["I1"] / max_i, "-", color=var_colors["I1"], linewidth=2.0)
        ax.plot(mic["P2"], mic["I1"] / max_i, "--", color=var_colors["I1"], linewidth=2.0)

        ax2.plot(foc["P2"], foc["eta"], "-", color=var_colors["eta"], linewidth=2.0)
        ax2.plot(mic["P2"], mic["eta"], "--", color=var_colors["eta"], linewidth=2.0)

        ax.set_ylabel("$M_2^*$, $n^*$, $I_1^*$, отн. ед.")
        ax2.set_ylabel("$\\eta$, отн. ед.")
        ax.set_ylim(0.0, 1.25)
        ax2.set_ylim(0.0, 1.05)
        ax.set_title(summary_df.loc[summary_df["motor_key"] == motor, "motor_label"].iloc[0])
        ax.set_xlabel("$P_2$, кВт")
        ax.margins(x=0.03)
        if idx == 0:
            leg_m = ax.legend(handles=method_handles, loc="lower right", frameon=False, title="Метод")
            ax.add_artist(leg_m)
            ax.legend(handles=param_handles, loc="upper left", frameon=False, title="Параметр")

    fig.tight_layout()
    _save_figure(fig, out_base)
    plt.close(fig)


def _plot_power_eta_time(timeseries: Dict[str, Dict[str, pd.DataFrame]], summary_df: pd.DataFrame, out_base: Path) -> None:
    plt = _ensure_plt()
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 6.4), sharex="col")

    for col, motor in enumerate(summary_df["motor_key"].tolist()):
        foc = timeseries[motor]["foc"]
        mic = timeseries[motor]["mic"]
        if foc.empty or mic.empty:
            continue
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

        axes[0, col].plot(t_f, p_in_f, color="black", linestyle="-", label="FOC")
        axes[0, col].plot(t_m, p_in_m, color="black", linestyle="--", label="MIC")
        axes[0, col].set_title(summary_df.loc[summary_df["motor_key"] == motor, "motor_label"].iloc[0])
        axes[0, col].set_ylabel("$P_{вх+}$, Вт")
        axes[0, col].legend(frameon=False, loc="upper right")

        axes[1, col].plot(t_f, eta_f, color="black", linestyle="-")
        axes[1, col].plot(t_m, eta_m, color="black", linestyle="--")
        axes[1, col].set_ylabel("$\\eta_{нак}$, отн. ед.")
        axes[1, col].set_xlabel("t, с")
        axes[1, col].set_ylim(0.0, 1.05)

    fig.suptitle("Сравнение $P_{вх+}(t)$ и накопленного КПД в режиме «пуск—стоп»")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save_figure(fig, out_base)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the multi-motor study summary used in the PGUPS paper.")
    parser.add_argument(
        "--traces-root",
        default=str(DEFAULT_TRACES_ROOT),
        help="Directory with per-motor trace subfolders (air56/al31/ao2).",
    )
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Directory for generated CSV/JSON and figures.")
    parser.add_argument(
        "--export-paper",
        action="store_true",
        help="Also overwrite paper assets under paper/pgups_2026/{data,fig}.",
    )
    parser.add_argument(
        "--paper-dir",
        default=str(DEFAULT_PAPER_DIR),
        help="Paper directory (used when --export-paper is set).",
    )
    args = parser.parse_args()

    traces_root = Path(args.traces_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paper_dir = Path(args.paper_dir) if bool(args.export_paper) else None
    if paper_dir is not None:
        (paper_dir / "data").mkdir(parents=True, exist_ok=True)
        (paper_dir / "fig").mkdir(parents=True, exist_ok=True)

    motors: List[MotorCase] = []
    for meta in MOTORS_META:
        key = str(meta["key"])
        run_dir = traces_root / key
        if not run_dir.exists():
            run_dir = Path(str(meta["legacy_run_dir"]))
        motors.append(
            MotorCase(
                key=key,
                label=str(meta["label"]),
                config=str(meta["config"]),
                run_dir=run_dir,
                rated_power_kw=float(meta["rated_power_kw"]),
            )
        )

    scenario_rows: List[Dict[str, object]] = []
    motor_rows: List[Dict[str, object]] = []
    mech_curves: Dict[str, Dict[str, pd.DataFrame]] = {}
    time_series_start_stop: Dict[str, Dict[str, pd.DataFrame]] = {}

    for motor in motors:
        if not motor.run_dir.exists():
            raise FileNotFoundError(f"Missing run dir: {motor.run_dir}")
        summary_items = _load_summary_items(motor.run_dir)

        per_motor_rows: List[Dict[str, object]] = []
        foc_mech_parts: List[pd.DataFrame] = []
        mic_mech_parts: List[pd.DataFrame] = []
        start_stop_foc = pd.DataFrame()
        start_stop_mic = pd.DataFrame()

        for item in summary_items:
            scenario = str(item["scenario"])
            tag = str(item["file_tag"])
            foc_df = _read_series(motor.run_dir / f"{tag}_foc.csv")
            mic_df = _read_series(motor.run_dir / f"{tag}_mic_ai.csv")
            m_foc = _series_metrics(foc_df, WINDOW_FRAC, scenario)
            m_mic = _series_metrics(mic_df, WINDOW_FRAC, scenario)

            mae_ratio_full = float(m_mic["mae_full"] / max(m_foc["mae_full"], 1e-9))
            saving_full = _pct_saving(float(m_foc["mean_p_in_full"]), float(m_mic["mean_p_in_full"]))
            saving_steady = _pct_saving(float(m_foc["mean_p_in_steady"]), float(m_mic["mean_p_in_steady"]))
            eta_gain_full = _eta_gain(float(m_foc["eta_full"]), float(m_mic["eta_full"]))
            eta_gain_steady = _eta_gain(float(m_foc["eta_steady"]), float(m_mic["eta_steady"]))

            row = {
                "motor_key": motor.key,
                "motor_label": motor.label,
                "rated_power_kw": motor.rated_power_kw,
                "scenario": scenario,
                "file_tag": tag,
                "foc_p_in_full_w": m_foc["mean_p_in_full"],
                "mic_p_in_full_w": m_mic["mean_p_in_full"],
                "saving_full_pct": saving_full,
                "foc_p_shaft_full_w": m_foc["mean_p_shaft_full"],
                "mic_p_shaft_full_w": m_mic["mean_p_shaft_full"],
                "foc_eta_full": m_foc["eta_full"],
                "mic_eta_full": m_mic["eta_full"],
                "eta_gain_full_pct": eta_gain_full,
                "foc_mae_full": m_foc["mae_full"],
                "mic_mae_full": m_mic["mae_full"],
                "mae_ratio_full": mae_ratio_full,
                "foc_p_in_steady_w": m_foc["mean_p_in_steady"],
                "mic_p_in_steady_w": m_mic["mean_p_in_steady"],
                "saving_steady_pct": saving_steady,
                "foc_eta_steady": m_foc["eta_steady"],
                "mic_eta_steady": m_mic["eta_steady"],
                "eta_gain_steady_pct": eta_gain_steady,
                "foc_mae_steady": m_foc["mae_steady"],
                "mic_mae_steady": m_mic["mae_steady"],
                "energy_in_foc_j": m_foc["energy_in_full"],
                "energy_in_mic_j": m_mic["energy_in_full"],
                "energy_shaft_foc_j": m_foc["energy_shaft_full"],
                "energy_shaft_mic_j": m_mic["energy_shaft_full"],
            }
            scenario_rows.append(row)
            per_motor_rows.append(row)

            foc_mech_parts.append(_prepare_mech_points(foc_df))
            mic_mech_parts.append(_prepare_mech_points(mic_df))
            if tag == "start_stop":
                start_stop_foc = foc_df
                start_stop_mic = mic_df

        per_df = pd.DataFrame(per_motor_rows)
        motor_rows.append(
            {
                "motor_key": motor.key,
                "motor_label": motor.label,
                "rated_power_kw": motor.rated_power_kw,
                "config": motor.config,
                # Store as POSIX-style relative path for cross-platform reproducibility.
                "run_dir": motor.run_dir.as_posix(),
                "avg_saving_full_pct": float(per_df["saving_full_pct"].mean()),
                "avg_saving_steady_pct": float(per_df["saving_steady_pct"].mean()),
                "min_saving_full_pct": float(per_df["saving_full_pct"].min()),
                "avg_eta_gain_full_pct": float(per_df["eta_gain_full_pct"].mean()),
                "avg_eta_gain_steady_pct": float(per_df["eta_gain_steady_pct"].mean()),
                "max_mae_ratio_full": float(per_df["mae_ratio_full"].max()),
                "avg_mae_ratio_full": float(per_df["mae_ratio_full"].mean()),
                "foc_energy_in_total_j": float(per_df["energy_in_foc_j"].sum()),
                "mic_energy_in_total_j": float(per_df["energy_in_mic_j"].sum()),
                "foc_energy_shaft_total_j": float(per_df["energy_shaft_foc_j"].sum()),
                "mic_energy_shaft_total_j": float(per_df["energy_shaft_mic_j"].sum()),
            }
        )

        foc_mech = pd.concat(foc_mech_parts, ignore_index=True)
        mic_mech = pd.concat(mic_mech_parts, ignore_index=True)
        p2_max = float(min(foc_mech["P2_kW"].quantile(0.98), mic_mech["P2_kW"].quantile(0.98)))
        p2_max = max(p2_max, 0.05)
        mech_curves[motor.key] = {
            "foc": _bin_curve(foc_mech, p2_max=p2_max),
            "mic": _bin_curve(mic_mech, p2_max=p2_max),
        }
        time_series_start_stop[motor.key] = {"foc": start_stop_foc, "mic": start_stop_mic}

    scenario_df = pd.DataFrame(scenario_rows)
    summary_df = pd.DataFrame(motor_rows)
    scenario_df.to_csv(out_dir / "scenario_metrics_multi_motor.csv", index=False, encoding="utf-8")
    summary_df.to_csv(out_dir / "motor_summary_multi_motor.csv", index=False, encoding="utf-8")

    regime_df = scenario_df.copy()
    regime_df["regime"] = np.where(regime_df["scenario"] == "hold:0.8", "steady", "periodic")
    regime_summary = (
        regime_df.groupby(["motor_key", "motor_label", "regime"], as_index=False)
        .agg(
            avg_saving_full_pct=("saving_full_pct", "mean"),
            avg_saving_steady_pct=("saving_steady_pct", "mean"),
            avg_eta_gain_full_pct=("eta_gain_full_pct", "mean"),
            avg_eta_gain_steady_pct=("eta_gain_steady_pct", "mean"),
            max_mae_ratio_full=("mae_ratio_full", "max"),
        )
        .sort_values(["motor_key", "regime"])
    )
    regime_summary.to_csv(out_dir / "regime_summary_multi_motor.csv", index=False, encoding="utf-8")

    # Simple bootstrap CI for pooled (motor, scenario) gains.
    def _bootstrap_ci(values: np.ndarray, n_boot: int = 5000, seed: int = 42) -> Tuple[float, float, float]:
        if values.size == 0:
            return 0.0, 0.0, 0.0
        rng = np.random.default_rng(seed)
        means = np.empty(n_boot, dtype=float)
        for i in range(n_boot):
            sample = rng.choice(values, size=values.size, replace=True)
            means[i] = float(np.mean(sample))
        lo, hi = np.percentile(means, [2.5, 97.5])
        return float(np.mean(values)), float(lo), float(hi)

    pooled_save_full = scenario_df["saving_full_pct"].to_numpy(dtype=float)
    pooled_save_steady = scenario_df["saving_steady_pct"].to_numpy(dtype=float)
    pooled_eta_full = scenario_df["eta_gain_full_pct"].to_numpy(dtype=float)
    pooled_eta_steady = scenario_df["eta_gain_steady_pct"].to_numpy(dtype=float)
    pooled_ci = {
        "saving_full_pct": _bootstrap_ci(pooled_save_full),
        "saving_steady_pct": _bootstrap_ci(pooled_save_steady),
        "eta_gain_full_pct": _bootstrap_ci(pooled_eta_full),
        "eta_gain_steady_pct": _bootstrap_ci(pooled_eta_steady),
    }

    study_summary = {
        "motors": [
            {
                "motor_key": r["motor_key"],
                "motor_label": r["motor_label"],
                "rated_power_kw": r["rated_power_kw"],
                "avg_saving_full_pct": r["avg_saving_full_pct"],
                "avg_saving_steady_pct": r["avg_saving_steady_pct"],
                "avg_eta_gain_full_pct": r["avg_eta_gain_full_pct"],
                "avg_eta_gain_steady_pct": r["avg_eta_gain_steady_pct"],
                "max_mae_ratio_full": r["max_mae_ratio_full"],
                "config": r["config"],
                "run_dir": r["run_dir"],
            }
            for r in motor_rows
        ],
        "overall": {
            "avg_saving_full_pct": float(summary_df["avg_saving_full_pct"].mean()),
            "avg_saving_steady_pct": float(summary_df["avg_saving_steady_pct"].mean()),
            "avg_eta_gain_full_pct": float(summary_df["avg_eta_gain_full_pct"].mean()),
            "avg_eta_gain_steady_pct": float(summary_df["avg_eta_gain_steady_pct"].mean()),
            "max_mae_ratio_full": float(summary_df["max_mae_ratio_full"].max()),
        },
        "overall_bootstrap_ci95": {
            "saving_full_pct": {
                "mean": pooled_ci["saving_full_pct"][0],
                "lo": pooled_ci["saving_full_pct"][1],
                "hi": pooled_ci["saving_full_pct"][2],
            },
            "saving_steady_pct": {
                "mean": pooled_ci["saving_steady_pct"][0],
                "lo": pooled_ci["saving_steady_pct"][1],
                "hi": pooled_ci["saving_steady_pct"][2],
            },
            "eta_gain_full_pct": {
                "mean": pooled_ci["eta_gain_full_pct"][0],
                "lo": pooled_ci["eta_gain_full_pct"][1],
                "hi": pooled_ci["eta_gain_full_pct"][2],
            },
            "eta_gain_steady_pct": {
                "mean": pooled_ci["eta_gain_steady_pct"][0],
                "lo": pooled_ci["eta_gain_steady_pct"][1],
                "hi": pooled_ci["eta_gain_steady_pct"][2],
            },
        },
    }
    (out_dir / "study_summary_multi_motor.json").write_text(
        json.dumps(study_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    _plot_summary_bars(summary_df, out_dir / "fig_multi_motor_summary_ru")
    _plot_savings_heatmap(scenario_df, out_dir / "fig_multi_motor_scenario_heatmap_ru")
    _plot_mech_characteristics(mech_curves, summary_df, out_dir / "fig_multi_motor_mech_ru")
    _plot_power_eta_time(time_series_start_stop, summary_df, out_dir / "fig_multi_motor_power_eta_time_ru")

    if paper_dir is not None:
        # Copy data tables.
        for name in (
            "scenario_metrics_multi_motor.csv",
            "motor_summary_multi_motor.csv",
            "regime_summary_multi_motor.csv",
            "study_summary_multi_motor.json",
        ):
            shutil.copy2(out_dir / name, paper_dir / "data" / name)

        # Copy figures (all available formats for convenience).
        fig_bases = [
            "fig_multi_motor_summary_ru",
            "fig_multi_motor_scenario_heatmap_ru",
            "fig_multi_motor_mech_ru",
            "fig_multi_motor_power_eta_time_ru",
        ]
        for base in fig_bases:
            for ext in (".png", ".pdf", ".svg"):
                src = out_dir / f"{base}{ext}"
                if src.exists():
                    shutil.copy2(src, paper_dir / "fig" / src.name)

    print(str((out_dir / "study_summary_multi_motor.json").resolve()))


if __name__ == "__main__":
    main()
