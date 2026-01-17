from __future__ import annotations

from pathlib import Path


def ensure_matplotlib():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def apply_vak_style(plt, font_family: str = "serif"):
    family = str(font_family or "serif").lower()
    if family in {"sans", "sans-serif", "sans_serif", "arial", "helvetica"}:
        font_key = "font.sans-serif"
        font_list = ["DejaVu Sans", "Arial", "Helvetica"]
        math_fontset = "dejavusans"
    else:
        font_key = "font.serif"
        font_list = ["Times New Roman", "DejaVu Serif", "STIXGeneral"]
        math_fontset = "stix"

    plt.rcParams.update(
        {
            "font.family": font_list,
            font_key: font_list,
            "mathtext.fontset": math_fontset,
            "font.size": 12,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "lines.linewidth": 2.0,
            "lines.markersize": 5.5,
            "figure.dpi": 200,
            "savefig.dpi": 300,
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


def save_figure(fig, path: Path) -> None:
    base = path.with_suffix("")
    base.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".png", ".pdf", ".svg"):
        fig.savefig(base.with_suffix(ext), bbox_inches="tight")
