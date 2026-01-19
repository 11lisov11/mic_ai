# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def _ensure_backend() -> None:
    matplotlib.use("Agg", force=True)


def plot_timeseries(series: Dict[str, np.ndarray], out_path: Path, title: str | None = None) -> None:
    _ensure_backend()
    fig, axes = plt.subplots(3, 1, figsize=(7.4, 6.8), sharex=True)

    axes[0].plot(series["t"], series["omega"], color="black", label="omega")
    axes[0].plot(series["t"], series["omega_ref"], color="0.35", linestyle="--", label="omega_ref")
    axes[0].set_ylabel("omega, rad/s")
    axes[0].legend(frameon=False)

    axes[1].plot(series["t"], series["i_q_ref"], color="black", label="i_q_ref")
    axes[1].plot(series["t"], series["i_q"], color="0.35", linestyle="--", label="i_q")
    axes[1].set_ylabel("i_q, A")
    axes[1].legend(frameon=False)

    err = series["omega_ref"] - series["omega"]
    axes[2].plot(series["t"], err, color="black")
    axes[2].set_ylabel("error, rad/s")
    axes[2].set_xlabel("t, s")

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
