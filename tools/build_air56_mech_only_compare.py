from __future__ import annotations

import argparse
import math
import sys
import warnings
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mic_ai.tools.plot_style import apply_vak_style, ensure_matplotlib, save_figure
from tools.build_air56_mech_journal_from_traces import _prepare_trace, _collapse_points, _build_smooth_curves


def _smooth_mech_curve(points: pd.DataFrame, m_max: float) -> Dict[str, np.ndarray]:
    p = points.copy().sort_values("M2")
    m = p["M2"].to_numpy(dtype=float).copy()
    n = p["n2"].to_numpy(dtype=float).copy()
    if m[0] > 1e-9:
        m = np.concatenate([[0.0], m])
        n = np.concatenate([[n[0]], n])
    for i in range(1, n.size):
        if n[i] > n[i - 1]:
            n[i] = n[i - 1]
    m_dense = np.linspace(0.0, float(m_max), 420)
    if m[-1] < m_dense[-1]:
        m = np.concatenate([m, [m_dense[-1]]])
        n = np.concatenate([n, [n[-1]]])
    spline = PchipInterpolator(m, n, extrapolate=True)
    return {"M2": m_dense, "n2": spline(m_dense)}


def _interp(x: np.ndarray, y: np.ndarray, x0: float) -> float:
    return float(np.interp(float(x0), x, y))


def main() -> None:
    warnings.warn(
        "tools/build_air56_mech_only_compare.py is deprecated. "
        "Use tools/build_air56_mech_journal_from_traces.py for production figures.",
        DeprecationWarning,
        stacklevel=2,
    )
    parser = argparse.ArgumentParser(description="Build pure AIR56 mechanical characteristics for FOC vs MIC.")
    parser.add_argument("--foc-trace", default="paper/pgups_2026/data/traces/air56/load_profile_foc.csv")
    parser.add_argument("--mic-trace", default="paper/pgups_2026/data/traces/air56/load_profile_mic_ai.csv")
    parser.add_argument("--out-pdf", default="outputs/article_air56_20260217/fig_air56_mech_journal.pdf")
    parser.add_argument("--out-csv", default="outputs/article_air56_20260217/fig_air56_mech_only_points.csv")
    parser.add_argument("--u-ll", type=float, default=380.0)
    parser.add_argument("--steady-rel-min", type=float, default=0.90)
    parser.add_argument("--p2-max", type=float, default=0.25)
    parser.add_argument("--common-p2-kw", type=float, default=0.24)
    parser.add_argument("--bins", type=int, default=12)
    args = parser.parse_args()

    foc_raw = _prepare_trace(Path(args.foc_trace), u_ll=float(args.u_ll), steady_rel_min=float(args.steady_rel_min))
    mic_raw = _prepare_trace(Path(args.mic_trace), u_ll=float(args.u_ll), steady_rel_min=float(args.steady_rel_min))
    foc_pts = _collapse_points(foc_raw, p2_max=float(args.p2_max), bins=int(args.bins))
    mic_pts = _collapse_points(mic_raw, p2_max=float(args.p2_max), bins=int(args.bins))

    x_dense = np.linspace(0.0, float(args.p2_max), 420)
    foc_work = _build_smooth_curves(foc_pts, x_dense)
    mic_work = _build_smooth_curves(mic_pts, x_dense)

    m_max = float(max(foc_pts["M2"].max(), mic_pts["M2"].max()) * 1.02)
    foc_mech = _smooth_mech_curve(foc_pts, m_max=m_max)
    mic_mech = _smooth_mech_curve(mic_pts, m_max=m_max)

    m_nom = float(np.percentile(foc_pts["M2"].to_numpy(dtype=float), 90))
    n_f_nom = _interp(foc_mech["M2"], foc_mech["n2"], m_nom)
    n_m_nom = _interp(mic_mech["M2"], mic_mech["n2"], m_nom)
    eta_f_024 = _interp(foc_work["x"], foc_work["eta_pct"], float(args.common_p2_kw))
    eta_m_024 = _interp(mic_work["x"], mic_work["eta_pct"], float(args.common_p2_kw))

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    foc_save = foc_pts.copy()
    foc_save.insert(0, "policy", "FOC")
    mic_save = mic_pts.copy()
    mic_save.insert(0, "policy", "MIC")
    pd.concat([foc_save, mic_save], ignore_index=True).to_csv(out_csv, index=False)

    plt = apply_vak_style(ensure_matplotlib())
    fig, axes = plt.subplots(2, 1, figsize=(11.6, 9.2), sharex=True, sharey=True)

    c_f = "#1f4e79"
    c_m = "#8f5a2a"
    c_ref = "0.55"

    ax = axes[0]
    ax.plot(foc_mech["M2"], foc_mech["n2"], color=c_f, linewidth=2.4, label="FOC")
    ax.plot(mic_mech["M2"], mic_mech["n2"], color=c_ref, linestyle="--", linewidth=1.9, label="MIC (сравнение)")
    ax.set_title("а) FOC: механическая характеристика n2=f(M2)", loc="left", fontweight="bold")

    ax = axes[1]
    ax.plot(mic_mech["M2"], mic_mech["n2"], color=c_m, linewidth=2.4, label="MIC")
    ax.plot(foc_mech["M2"], foc_mech["n2"], color=c_ref, linestyle="--", linewidth=1.9, label="FOC (сравнение)")
    ax.set_title("б) MIC: механическая характеристика n2=f(M2)", loc="left", fontweight="bold")

    n_all = np.concatenate([foc_mech["n2"], mic_mech["n2"]])
    n_lo = float(np.nanmin(n_all))
    n_hi = float(np.nanmax(n_all))
    n_pad = max(6.0, 0.1 * (n_hi - n_lo))
    for ax in axes:
        ax.grid(False)
        ax.set_xlim(0.0, m_max)
        ax.set_ylim(n_lo - n_pad, n_hi + n_pad)
        ax.axvline(m_nom, color="0.65", linestyle=":", linewidth=1.0)
    axes[0].set_ylabel("n2, об/мин")
    axes[1].set_ylabel("n2, об/мин")
    axes[1].set_xlabel("M2, Н·м")

    delta_text = "\n".join(
        [
            f"Δn2(Mном)={n_m_nom - n_f_nom:+.1f} об/мин",
            f"η_FOC(P2=0,24)={eta_f_024:.1f}%",
            f"η_MIC(P2=0,24)={eta_m_024:.1f}%",
            f"Δη(0,24)={eta_m_024 - eta_f_024:+.1f} п.п.",
        ]
    ).replace(".", ",")
    axes[1].text(
        0.985,
        0.10,
        delta_text,
        transform=axes[1].transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="0.20",
        bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none", "pad": 0.9},
    )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.955))
    fig.suptitle("AIR56: сравнение механических характеристик FOC и MIC", y=0.985)
    fig.subplots_adjust(left=0.10, right=0.96, top=0.90, bottom=0.08, hspace=0.16)

    out_pdf = Path(args.out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_pdf)
    plt.close(fig)

    print(f"Saved: {out_pdf}")
    print(f"Saved points: {out_csv}")
    print(f"delta_n2_nom={n_m_nom - n_f_nom:+.3f} rpm")
    print(f"delta_eta_024={eta_m_024 - eta_f_024:+.3f} p.p.")


if __name__ == "__main__":
    main()
