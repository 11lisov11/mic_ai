"""
Построение графиков по результатам симуляции, сохранённым в NPZ.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from outputs.styles import apply_style


def _unique_png(prefix: str, directory: Path, idx_hint: int | None = None) -> Path:
    """
    Вернуть уникальный путь PNG в формате <prefix>_<n>.png.
    Если idx_hint задан, сначала используется он.
    """
    if idx_hint is not None:
        candidate = directory / f"{prefix}_{idx_hint}.png"
        if not candidate.exists():
            return candidate
    idx = 1
    while True:
        candidate = directory / f"{prefix}_{idx}.png"
        if not candidate.exists():
            return candidate
        idx += 1


def _load_meta(data: np.lib.npyio.NpzFile) -> dict:
    meta_raw = data.get("meta")
    if meta_raw is None:
        return {}
    try:
        meta_bytes = meta_raw.item()
    except Exception:
        meta_bytes = meta_raw
    if isinstance(meta_bytes, bytes):
        meta_str = meta_bytes.decode("utf-8", errors="ignore")
    else:
        meta_str = str(meta_bytes)
    try:
        return json.loads(meta_str)
    except json.JSONDecodeError:
        return {}


def _format_meta(meta: dict) -> str:
    if not meta:
        return ""
    sim = meta.get("sim", {})
    motor = meta.get("motor", {})
    inverter = meta.get("inverter", {})
    parts = [
        f"mode: {sim.get('mode', '?')}",
        f"scenario: {sim.get('scenario_name', '?')}",
        f"t_end: {sim.get('t_end', '?')} s",
        f"dt: {sim.get('dt', '?')}",
        f"load: {sim.get('load_torque', '?')} Nm",
        f"Vdc: {inverter.get('Vdc', '?')} V",
        f"p: {motor.get('p', '?')}",
    ]
    return "\n".join(parts)


def _extract_index_from_data_name(result_path: Path) -> int | None:
    m = re.search(r"data_(\d+)", result_path.stem)
    if m:
        return int(m.group(1))
    return None


def _annotate(ax, text: str) -> None:
    if not text:
        return
    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
    )


def plot_run(result_path: str, save_dir: str = "outputs/figures") -> None:
    apply_style()
    data = np.load(result_path)
    meta_raw = _load_meta(data)
    meta = _format_meta(meta_raw)

    idx_hint = _extract_index_from_data_name(Path(result_path))

    t = data["t"]
    omega_m = data["omega_m"]
    omega_ref = data["omega_ref"]
    torque = data["T_e"]
    load_torque = data["load_torque"]
    i_a, i_b, i_c = data["i_a"], data["i_b"], data["i_c"]
    p_in, p_out = data["P_in"], data["P_out"]

    save_dir_path = Path(save_dir)
    save_dir_path.mkdir(parents=True, exist_ok=True)

    # Speed
    fig, ax = plt.subplots()
    ax.plot(t, omega_m, label="omega_m")
    ax.plot(t, omega_ref, label="omega_ref", linestyle="--")
    ax.set_xlabel("t, s")
    ax.set_ylabel("Speed, rad/s")
    ax.legend()
    _annotate(ax, meta)
    fig.tight_layout()
    fig.savefig(_unique_png("speed", save_dir_path, idx_hint))
    plt.close(fig)

    # Torque
    fig, ax = plt.subplots()
    ax.plot(t, torque, label="T_e")
    ax.plot(t, load_torque, label="load", linestyle="--")
    ax.set_xlabel("t, s")
    ax.set_ylabel("Torque, Nm")
    ax.legend()
    _annotate(ax, meta)
    fig.tight_layout()
    fig.savefig(_unique_png("torque", save_dir_path, idx_hint))
    plt.close(fig)

    # Currents
    fig, ax = plt.subplots()
    ax.plot(t, i_a, label="i_a")
    ax.plot(t, i_b, label="i_b")
    ax.plot(t, i_c, label="i_c")
    ax.set_xlabel("t, s")
    ax.set_ylabel("Current, A")
    ax.legend()
    _annotate(ax, meta)
    fig.tight_layout()
    fig.savefig(_unique_png("currents", save_dir_path, idx_hint))
    plt.close(fig)

    # Power
    fig, ax = plt.subplots()
    ax.plot(t, p_in, label="P_in")
    ax.plot(t, p_out, label="P_out")
    ax.set_xlabel("t, s")
    ax.set_ylabel("Power, W")
    ax.legend()
    _annotate(ax, meta)
    fig.tight_layout()
    fig.savefig(_unique_png("power", save_dir_path, idx_hint))
    plt.close(fig)


__all__ = ["plot_run"]
