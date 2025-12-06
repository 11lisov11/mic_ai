"""
Вспомогательные настройки оформления для Matplotlib.
"""

from __future__ import annotations

import matplotlib.pyplot as plt


def apply_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.figsize": (10, 6),
            "axes.grid": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.alpha": 0.6,
        }
    )


__all__ = ["apply_style"]
