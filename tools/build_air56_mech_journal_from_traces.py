from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mic_ai.tools.plot_style import apply_vak_style, ensure_matplotlib, save_figure


def _moving_average(y: np.ndarray, passes: int = 2) -> np.ndarray:
    out = np.asarray(y, dtype=float).copy()
    if out.size < 3:
        return out
    kernel = np.array([0.25, 0.5, 0.25], dtype=float)
    for _ in range(max(int(passes), 0)):
        yp = np.pad(out, (1, 1), mode="edge")
        out = np.convolve(yp, kernel, mode="same")[1:-1]
    return out


def _enforce_inc(y: np.ndarray) -> np.ndarray:
    out = np.asarray(y, dtype=float).copy()
    for i in range(1, out.size):
        if out[i] < out[i - 1]:
            out[i] = out[i - 1]
    return out


def _enforce_dec(y: np.ndarray) -> np.ndarray:
    out = np.asarray(y, dtype=float).copy()
    for i in range(1, out.size):
        if out[i] > out[i - 1]:
            out[i] = out[i - 1]
    return out


def _enforce_hump(y: np.ndarray) -> np.ndarray:
    out = np.asarray(y, dtype=float).copy()
    if out.size < 3:
        return out
    i_max = int(np.nanargmax(out))
    left = _enforce_inc(out[: i_max + 1])
    right = _enforce_dec(out[i_max:])
    return np.concatenate([left, right[1:]])


def _prepare_trace(path: Path, u_ll: float, steady_rel_min: float) -> pd.DataFrame:
    raw = pd.read_csv(path)
    omega = raw["omega"].to_numpy(dtype=float)
    omega_ref = raw["omega_ref"].to_numpy(dtype=float)
    i_rms = np.maximum(raw["i_rms"].to_numpy(dtype=float), 1e-9)
    p_el = np.maximum(raw["p_el"].to_numpy(dtype=float), 1e-9)
    p_shaft = np.maximum(raw["p_mech"].to_numpy(dtype=float), 0.0)

    omega_ref_max = float(np.nanmax(omega_ref))
    mask = (omega_ref > 1e-9) & (omega > 1e-9) & (omega >= steady_rel_min * omega_ref_max)

    w = omega[mask]
    pin = p_el[mask]
    ish = i_rms[mask]
    p2 = p_shaft[mask]
    p2_kw = p2 / 1000.0
    m2 = np.divide(p2, np.maximum(w, 1e-9))
    n2 = np.abs(w) * 60.0 / (2.0 * math.pi)
    eta_pct = 100.0 * np.divide(p2, pin)
    cosphi = np.clip(pin / (math.sqrt(3.0) * float(u_ll) * ish), 0.0, 1.0)

    return pd.DataFrame(
        {
            "p2_kw": p2_kw,
            "M2": m2,
            "I1": ish,
            "n2": n2,
            "eta_pct": eta_pct,
            "cosphi": cosphi,
        }
    )


def _collapse_points(df: pd.DataFrame, p2_max: float, bins: int) -> pd.DataFrame:
    p2_hi = float(min(p2_max, max(0.01, float(df["p2_kw"].max()))))
    work = df[(df["p2_kw"] >= 0.0) & (df["p2_kw"] <= p2_hi)].copy()
    edges = np.linspace(0.0, p2_hi + 1e-9, int(max(4, bins)) + 1)
    work["bin"] = pd.cut(work["p2_kw"], bins=edges, include_lowest=True)
    agg = (
        work.groupby("bin", observed=False)
        .agg(
            p2_kw=("p2_kw", "median"),
            M2=("M2", "median"),
            I1=("I1", "median"),
            n2=("n2", "median"),
            eta_pct=("eta_pct", "median"),
            cosphi=("cosphi", "median"),
            n=("p2_kw", "size"),
        )
        .reset_index(drop=True)
    )
    agg = agg[agg["n"] >= 20].drop(columns=["n"])
    agg = agg.sort_values("p2_kw")
    if agg.empty:
        raise ValueError("No steady bins to build journal figure.")

    lo = work[work["p2_kw"] <= max(0.005, 0.05 * p2_hi)]
    i0 = float(lo["I1"].median()) if not lo.empty else float(agg["I1"].iloc[0])
    n0 = float(lo["n2"].median()) if not lo.empty else float(agg["n2"].iloc[0])
    c0 = float(lo["cosphi"].median()) if not lo.empty else float(agg["cosphi"].iloc[0])
    zero = pd.DataFrame(
        [
            {
                "p2_kw": 0.0,
                "M2": 0.0,
                "I1": max(0.0, i0),
                "n2": max(0.0, n0),
                "eta_pct": 0.0,
                "cosphi": float(np.clip(c0, 0.0, 1.0)),
            }
        ]
    )
    out = pd.concat([zero, agg], ignore_index=True).sort_values("p2_kw").drop_duplicates("p2_kw", keep="last")
    return out.reset_index(drop=True)


def _build_smooth_curves(points: pd.DataFrame, x_dense: np.ndarray) -> Dict[str, np.ndarray]:
    x = points["p2_kw"].to_numpy(dtype=float)
    curves: Dict[str, np.ndarray] = {}
    modes: Dict[str, str] = {
        "M2": "inc",
        "I1": "inc",
        "n2": "dec",
        "eta_pct": "hump",
        "cosphi": "hump",
    }
    for key, mode in modes.items():
        y = points[key].to_numpy(dtype=float)
        y0 = float(y[0])
        smooth_passes = 1 if key in {"eta_pct", "cosphi"} else 2
        y = _moving_average(y, passes=smooth_passes)
        y[0] = y0
        if mode == "inc":
            y = _enforce_inc(y)
        elif mode == "dec":
            y = _enforce_dec(y)
        else:
            y = _enforce_hump(y)
            y[0] = y0
        if x[-1] < x_dense[-1]:
            dx = max(x[-1] - x[-2], 1e-9) if x.size >= 2 else 1e-9
            slope = (y[-1] - y[-2]) / dx if x.size >= 2 else 0.0
            y_end = y[-1] + slope * (x_dense[-1] - x[-1])
            if mode == "inc":
                y_end = max(y_end, y[-1])
            elif mode == "dec":
                y_end = min(y_end, y[-1])
            else:
                # For hump-like variables (eta, cosphi) avoid artificial growth at the right edge.
                y_end = min(y_end, y[-1])
            x_use = np.concatenate([x, [x_dense[-1]]])
            y_use = np.concatenate([y, [y_end]])
        else:
            x_use = x
            y_use = y
        interp = PchipInterpolator(x_use, y_use, extrapolate=True)
        curves[key] = interp(x_dense)
    curves["x"] = x_dense
    return curves


def _interp_at(x: np.ndarray, y: np.ndarray, x0: float) -> float:
    return float(np.interp(float(x0), x, y))


def _plot(
    out_pdf: Path,
    foc_curves: Dict[str, np.ndarray],
    mic_curves: Dict[str, np.ndarray],
    common_p2_kw: float,
    x_max: float,
) -> Tuple[float, float]:
    plt = apply_vak_style(ensure_matplotlib())
    fig, axes = plt.subplots(2, 1, figsize=(13.8, 10.8), sharex=True)
    colors = {
        "M2": "#1f4e79",
        "I1": "#8f5a2a",
        "n2": "#2f6b3f",
        "eta": "#7a2f2f",
        "cosphi": "#5b4b8a",
    }
    x_left = 0.0
    x_right = float(x_max)
    x_span = x_right - x_left

    def draw_panel(
        ax_m2,
        curves: Dict[str, np.ndarray],
        panel_title: str,
        ref_eta_common: float | None = None,
    ) -> Dict[str, float]:
        import matplotlib.ticker as mticker
        from matplotlib.transforms import blended_transform_factory

        x = curves["x"]
        m2 = curves["M2"]
        i1 = curves["I1"]
        n2 = curves["n2"]
        eta = curves["eta_pct"]
        cosphi = np.clip(curves["cosphi"], 0.0, 1.0)

        ax_i1 = ax_m2.twinx()
        ax_n2 = ax_m2.twinx()
        ax_eta = ax_m2.twinx()
        ax_cosphi = ax_m2.twinx()
        for ax_extra in (ax_i1, ax_n2, ax_eta, ax_cosphi):
            ax_extra.set_frame_on(True)
            ax_extra.patch.set_visible(False)

        ax_i1.spines["right"].set_visible(False)
        ax_i1.spines["left"].set_visible(True)
        ax_i1.spines["left"].set_position(("axes", -0.13))
        ax_i1.yaxis.set_label_position("left")
        ax_i1.yaxis.set_ticks_position("left")

        ax_n2.spines["right"].set_visible(False)
        ax_n2.spines["left"].set_visible(True)
        ax_n2.spines["left"].set_position(("axes", -0.25))
        ax_n2.yaxis.set_label_position("left")
        ax_n2.yaxis.set_ticks_position("left")

        ax_eta.spines["right"].set_visible(True)
        ax_eta.spines["right"].set_position(("axes", 1.03))
        ax_eta.yaxis.set_label_position("right")
        ax_eta.yaxis.set_ticks_position("right")
        ax_cosphi.spines["right"].set_visible(True)
        ax_cosphi.spines["right"].set_position(("axes", 1.14))
        ax_cosphi.yaxis.set_label_position("right")
        ax_cosphi.yaxis.set_ticks_position("right")

        lw = 2.2
        ax_m2.plot(x, m2, color=colors["M2"], linewidth=lw)
        ax_i1.plot(x, i1, color=colors["I1"], linewidth=lw)
        ax_n2.plot(x, n2, color=colors["n2"], linewidth=lw)
        ax_eta.plot(x, eta, color=colors["eta"], linewidth=lw)
        ax_cosphi.plot(x, cosphi, color=colors["cosphi"], linewidth=lw)

        ax_m2.axvline(float(common_p2_kw), color="0.55", linestyle="--", linewidth=1.0, zorder=0)
        ax_m2.text(
            float(common_p2_kw) - 0.006 * x_span,
            0.08,
            f"P2={common_p2_kw:.2f} кВт".replace(".", ","),
            rotation=90,
            ha="right",
            va="bottom",
            color="0.42",
            fontsize=9,
            transform=blended_transform_factory(ax_m2.transData, ax_m2.transAxes),
        )
        eta_common = _interp_at(x, eta, float(common_p2_kw))
        i1_common = _interp_at(x, i1, float(common_p2_kw))
        n2_common = _interp_at(x, n2, float(common_p2_kw))
        cos_common = _interp_at(x, cosphi, float(common_p2_kw))
        m2_common = _interp_at(x, m2, float(common_p2_kw))
        if ref_eta_common is not None and np.isfinite(float(ref_eta_common)):
            ax_eta.plot(
                [float(common_p2_kw), x_right],
                [float(ref_eta_common), float(ref_eta_common)],
                linestyle=":",
                color="0.45",
                linewidth=1.0,
            )
            ax_eta.text(
                x_right - 0.0015 * x_span,
                float(ref_eta_common) - 0.8,
                f"η_FOC={ref_eta_common:.1f}%".replace(".", ","),
                color="0.35",
                fontsize=8,
                ha="right",
                va="top",
            )
        ax_eta.plot(
            [float(common_p2_kw), x_right],
            [eta_common, eta_common],
            linestyle="--",
            color=colors["eta"],
            linewidth=1.2,
        )
        ax_eta.scatter([float(common_p2_kw)], [eta_common], s=54, facecolors="white", edgecolors=colors["eta"], linewidths=1.3, zorder=8)
        ax_eta.text(
            x_right - 0.001 * x_span,
            eta_common + 0.8,
            f"η(P2=0,24)={eta_common:.1f}%".replace(".", ","),
            color=colors["eta"],
            fontsize=9,
            ha="right",
            va="bottom",
        )

        ax_m2.set_title(panel_title, loc="left", fontweight="bold")
        ax_m2.set_xlim(x_left, x_right)
        for ax_no_grid in (ax_m2, ax_i1, ax_n2, ax_eta, ax_cosphi):
            ax_no_grid.grid(False)

        ax_m2.set_ylim(0.0, max(2.8, float(np.nanmax(m2) * 1.08)))
        i1_lo = float(np.nanmin(i1))
        i1_hi = float(np.nanmax(i1))
        i1_pad = max(0.02, 0.08 * max(i1_hi - i1_lo, 1e-6))
        ax_i1.set_ylim(max(0.0, i1_lo - i1_pad), i1_hi + i1_pad)

        n2_lo = float(np.nanmin(n2))
        n2_hi = float(np.nanmax(n2))
        n2_pad = max(8.0, 0.12 * max(n2_hi - n2_lo, 1.0))
        ax_n2.set_ylim(n2_lo - n2_pad, n2_hi + n2_pad)
        ax_n2.ticklabel_format(style="plain", axis="y", useOffset=False)
        ax_n2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))

        eta_lo = float(np.nanmin(eta))
        eta_hi = float(np.nanmax(eta))
        eta_pad = max(1.0, 0.10 * max(eta_hi - eta_lo, 1.0))
        ax_eta.set_ylim(max(0.0, eta_lo - eta_pad), eta_hi + eta_pad)

        cos_lo = float(np.nanmin(cosphi))
        cos_hi = float(np.nanmax(cosphi))
        cos_pad = max(0.01, 0.08 * max(cos_hi - cos_lo, 1e-6))
        ax_cosphi.set_ylim(max(0.0, cos_lo - cos_pad), min(1.0, cos_hi + cos_pad))

        ax_m2.set_ylabel("M2, Н·м", color=colors["M2"])
        ax_i1.set_ylabel("I1, A", color=colors["I1"])
        ax_n2.set_ylabel("n2, об/мин", color=colors["n2"])
        ax_eta.set_ylabel("η, %", color=colors["eta"])
        ax_cosphi.set_ylabel("cosφ, о.е.", color=colors["cosphi"])

        ax_m2.tick_params(axis="y", colors=colors["M2"])
        ax_i1.tick_params(axis="y", colors=colors["I1"])
        ax_n2.tick_params(axis="y", colors=colors["n2"])
        ax_eta.tick_params(axis="y", colors=colors["eta"])
        ax_cosphi.tick_params(axis="y", colors=colors["cosphi"])

        ax_m2.spines["left"].set_color(colors["M2"])
        ax_i1.spines["left"].set_color(colors["I1"])
        ax_n2.spines["left"].set_color(colors["n2"])
        ax_eta.spines["right"].set_color(colors["eta"])
        ax_cosphi.spines["right"].set_color(colors["cosphi"])
        ax_eta.spines["right"].set_linewidth(1.0)
        ax_cosphi.spines["right"].set_linewidth(1.0)
        return {
            "eta": float(eta_common),
            "I1": float(i1_common),
            "n2": float(n2_common),
            "cosphi": float(cos_common),
            "M2": float(m2_common),
        }

    foc_common = draw_panel(axes[0], foc_curves, "а) FOC")
    mic_common = draw_panel(axes[1], mic_curves, "б) MIC", ref_eta_common=foc_common["eta"])
    axes[1].set_xlabel("P2, кВт")
    axes[1].text(
        0.985,
        0.83,
        f"Δη(0,24)={mic_common['eta'] - foc_common['eta']:+.1f} п.п.".replace(".", ","),
        transform=axes[1].transAxes,
        color=colors["eta"],
        fontsize=9,
        ha="right",
        va="top",
        bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "none", "pad": 0.8},
    )
    delta_block = "\n".join(
        [
            f"ΔI1(0,24)={mic_common['I1'] - foc_common['I1']:+.3f} A",
            f"Δn2(0,24)={mic_common['n2'] - foc_common['n2']:+.1f} об/мин",
            f"Δcosφ(0,24)={mic_common['cosphi'] - foc_common['cosphi']:+.3f} о.е.",
        ]
    ).replace(".", ",")
    axes[1].text(
        0.985,
        0.77,
        delta_block,
        transform=axes[1].transAxes,
        color="0.30",
        fontsize=8,
        ha="right",
        va="top",
        bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "none", "pad": 0.8},
    )

    from matplotlib.lines import Line2D

    handles = [
        Line2D([0], [0], color=colors["M2"], linestyle="-", label="M2"),
        Line2D([0], [0], color=colors["I1"], linestyle="-", label="I1"),
        Line2D([0], [0], color=colors["n2"], linestyle="-", label="n2"),
        Line2D([0], [0], color=colors["eta"], linestyle="-", label="η"),
        Line2D([0], [0], color=colors["cosphi"], linestyle="-", label="cosφ"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.955))
    fig.suptitle("Рабочие характеристики AIR56: раздельно для FOC (а) и MIC (б)", y=0.985)
    fig.subplots_adjust(left=0.19, right=0.84, top=0.88, bottom=0.08, hspace=0.14)
    save_figure(fig, out_pdf)
    plt.close(fig)
    return foc_common["eta"], mic_common["eta"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build publication-ready AIR56 mechanical chart from validated traces.")
    parser.add_argument(
        "--foc-trace",
        default="paper/pgups_2026/data/traces/air56/load_profile_foc.csv",
        help="Path to FOC load profile trace.",
    )
    parser.add_argument(
        "--mic-trace",
        default="paper/pgups_2026/data/traces/air56/load_profile_mic_ai.csv",
        help="Path to MIC load profile trace.",
    )
    parser.add_argument("--out-pdf", default="outputs/article_air56_20260217/fig_air56_mech_journal.pdf")
    parser.add_argument("--out-points-csv", default="outputs/article_air56_20260217/fig_air56_mech_points.csv")
    parser.add_argument("--u-ll", type=float, default=380.0, help="Line voltage RMS for cosphi estimate.")
    parser.add_argument("--steady-rel-min", type=float, default=0.90, help="Filter: omega >= steady_rel_min * omega_ref_max.")
    parser.add_argument("--p2-max", type=float, default=0.25, help="X-axis max power, kW.")
    parser.add_argument("--common-p2-kw", type=float, default=0.24, help="Common power point for dashed eta projection.")
    parser.add_argument("--bins", type=int, default=12, help="Binning count over P2 for robust point extraction.")
    args = parser.parse_args()

    foc_df = _prepare_trace(Path(args.foc_trace), u_ll=float(args.u_ll), steady_rel_min=float(args.steady_rel_min))
    mic_df = _prepare_trace(Path(args.mic_trace), u_ll=float(args.u_ll), steady_rel_min=float(args.steady_rel_min))
    foc_pts = _collapse_points(foc_df, p2_max=float(args.p2_max), bins=int(args.bins))
    mic_pts = _collapse_points(mic_df, p2_max=float(args.p2_max), bins=int(args.bins))

    x_dense = np.linspace(0.0, float(args.p2_max), 420)
    foc_curves = _build_smooth_curves(foc_pts, x_dense)
    mic_curves = _build_smooth_curves(mic_pts, x_dense)

    out_points = Path(args.out_points_csv)
    out_points.parent.mkdir(parents=True, exist_ok=True)
    foc_save = foc_pts.copy()
    foc_save.insert(0, "policy", "FOC")
    mic_save = mic_pts.copy()
    mic_save.insert(0, "policy", "MIC")
    pd.concat([foc_save, mic_save], ignore_index=True).to_csv(out_points, index=False)

    out_pdf = Path(args.out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    eta_f, eta_m = _plot(
        out_pdf=out_pdf,
        foc_curves=foc_curves,
        mic_curves=mic_curves,
        common_p2_kw=float(args.common_p2_kw),
        x_max=float(args.p2_max),
    )

    cos_f_max = float(np.nanmax(foc_curves["cosphi"]))
    cos_m_max = float(np.nanmax(mic_curves["cosphi"]))
    print(f"Saved: {out_pdf}")
    print(f"Saved points: {out_points}")
    print(f"eta_foc@{args.common_p2_kw:.2f}={eta_f:.3f}%")
    print(f"eta_mic@{args.common_p2_kw:.2f}={eta_m:.3f}%")
    print(f"delta_eta={eta_m - eta_f:+.3f} p.p.")
    print(f"cosphi_max_foc={cos_f_max:.3f}, cosphi_max_mic={cos_m_max:.3f}")


if __name__ == "__main__":
    main()
