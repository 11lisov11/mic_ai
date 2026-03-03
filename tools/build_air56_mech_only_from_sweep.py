from __future__ import annotations

import argparse
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


def _prepare_policy(df: pd.DataFrame, policy: str) -> pd.DataFrame:
    out = df[df["policy"] == policy].copy()
    out = out.sort_values("m2")
    out = out[(out["m2"] >= 0.0) & np.isfinite(out["m2"]) & np.isfinite(out["n2_rpm"])]
    out = out.drop_duplicates("m2", keep="last")
    return out.reset_index(drop=True)


def _smooth_mech(m: np.ndarray, n: np.ndarray, m_dense: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=float).copy()
    n = np.asarray(n, dtype=float).copy()
    if m.size == 0:
        return np.zeros_like(m_dense)
    if m[0] > 1e-9:
        m = np.concatenate([[0.0], m])
        n = np.concatenate([[n[0]], n])
    for i in range(1, n.size):
        if n[i] > n[i - 1]:
            n[i] = n[i - 1]
    if m[-1] < m_dense[-1]:
        m = np.concatenate([m, [m_dense[-1]]])
        n = np.concatenate([n, [n[-1]]])
    spl = PchipInterpolator(m, n, extrapolate=True)
    return spl(m_dense)


def _interp(x: np.ndarray, y: np.ndarray, x0: float) -> float:
    return float(np.interp(float(x0), x, y))


def main() -> None:
    warnings.warn(
        "tools/build_air56_mech_only_from_sweep.py is deprecated. "
        "Use tools/build_air56_mech_journal_from_traces.py for production figures.",
        DeprecationWarning,
        stacklevel=2,
    )
    parser = argparse.ArgumentParser(description="Build natural mechanical characteristics from load sweep CSV.")
    parser.add_argument("--sweep-csv", default="outputs/article_air56_20260217_natural/load_characteristics.csv")
    parser.add_argument("--out-pdf", default="outputs/article_air56_20260217/fig_air56_mech_journal.pdf")
    parser.add_argument("--out-csv", default="outputs/article_air56_20260217/fig_air56_mech_only_points.csv")
    parser.add_argument("--common-p2-kw", type=float, default=0.24)
    args = parser.parse_args()

    src = pd.read_csv(Path(args.sweep_csv))
    src = src.rename(columns={"M2": "m2", "n2": "n2_rpm"})
    for col in ("m2", "n2_rpm", "eta_pct", "p2_kw"):
        if col not in src.columns:
            raise ValueError(f"Missing column in sweep CSV: {col}")

    foc = _prepare_policy(src, "FOC")
    mic = _prepare_policy(src, "MIC_AI")
    if foc.empty or mic.empty:
        raise ValueError("No FOC/MIC rows found in sweep CSV.")

    m_max = float(max(foc["m2"].max(), mic["m2"].max()) * 1.02)
    m_dense = np.linspace(0.0, m_max, 420)
    n_f = _smooth_mech(foc["m2"].to_numpy(), foc["n2_rpm"].to_numpy(), m_dense)
    n_m = _smooth_mech(mic["m2"].to_numpy(), mic["n2_rpm"].to_numpy(), m_dense)

    m_nom = float(np.percentile(foc["m2"].to_numpy(dtype=float), 90))
    n_f_nom = _interp(m_dense, n_f, m_nom)
    n_m_nom = _interp(m_dense, n_m, m_nom)
    eta_f = _interp(foc["p2_kw"].to_numpy(dtype=float), foc["eta_pct"].to_numpy(dtype=float), float(args.common_p2_kw))
    eta_m = _interp(mic["p2_kw"].to_numpy(dtype=float), mic["eta_pct"].to_numpy(dtype=float), float(args.common_p2_kw))

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([foc.assign(policy="FOC"), mic.assign(policy="MIC")], ignore_index=True).to_csv(out_csv, index=False)

    plt = apply_vak_style(ensure_matplotlib())
    fig, axes = plt.subplots(2, 1, figsize=(11.6, 9.2), sharex=True, sharey=True)

    c_f = "#1f4e79"
    c_m = "#8f5a2a"
    c_ref = "0.50"

    axes[0].plot(m_dense, n_f, color=c_f, linewidth=2.5, label="FOC")
    axes[0].plot(m_dense, n_m, color=c_ref, linewidth=1.8, linestyle="--", label="MIC (сравнение)")
    axes[0].set_title("а) FOC: естественная механическая характеристика n2=f(M2)", loc="left", fontweight="bold")

    axes[1].plot(m_dense, n_m, color=c_m, linewidth=2.5, label="MIC")
    axes[1].plot(m_dense, n_f, color=c_ref, linewidth=1.8, linestyle="--", label="FOC (сравнение)")
    axes[1].set_title("б) MIC: естественная механическая характеристика n2=f(M2)", loc="left", fontweight="bold")

    n_all = np.concatenate([n_f, n_m])
    n_lo = float(np.nanmin(n_all))
    n_hi = float(np.nanmax(n_all))
    n_pad = max(8.0, 0.10 * (n_hi - n_lo))
    for ax in axes:
        ax.grid(False)
        ax.set_xlim(0.0, m_max)
        ax.set_ylim(n_lo - n_pad, n_hi + n_pad)
        ax.axvline(m_nom, color="0.65", linestyle=":", linewidth=1.0)
    axes[0].set_ylabel("n2, об/мин")
    axes[1].set_ylabel("n2, об/мин")
    axes[1].set_xlabel("M2, Н·м")

    txt = "\n".join(
        [
            f"Δn2(Mном)={n_m_nom - n_f_nom:+.1f} об/мин",
            f"η_FOC(P2=0,24)={eta_f:.1f}%",
            f"η_MIC(P2=0,24)={eta_m:.1f}%",
            f"Δη(0,24)={eta_m - eta_f:+.1f} п.п.",
        ]
    ).replace(".", ",")
    axes[1].text(
        0.985,
        0.10,
        txt,
        transform=axes[1].transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="0.20",
        bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none", "pad": 0.9},
    )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.955))
    fig.suptitle("AIR56: сравнение естественных механических характеристик FOC и MIC", y=0.985)
    fig.subplots_adjust(left=0.10, right=0.96, top=0.90, bottom=0.08, hspace=0.16)

    out_pdf = Path(args.out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_pdf)
    plt.close(fig)

    print(f"Saved: {out_pdf}")
    print(f"Saved points: {out_csv}")
    print(f"delta_n2_nom={n_m_nom - n_f_nom:+.3f} rpm")
    print(f"delta_eta_024={eta_m - eta_f:+.3f} p.p.")


if __name__ == "__main__":
    main()
