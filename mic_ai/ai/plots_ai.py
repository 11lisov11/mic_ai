from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from outputs.styles import apply_style


def _load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _collect_params(ident: Dict) -> List[Dict[str, float]]:
    est = ident.get("estimated_params", {}) or ident.get("estimated", {}) or {}
    true = ident.get("true_params_if_available") or ident.get("true_params") or {}
    keys = ["Rs", "Rr", "Ls", "Lr", "Lm", "J", "B"]
    out = []
    for k in keys:
        if k in est:
            out.append({"name": k, "est": float(est.get(k, 0.0)), "true": float(true.get(k, np.nan))})
    return out


def _pick_eps(episodes: List[Dict], count: int = 3) -> List[Dict]:
    if not episodes:
        return []
    idxs = [0, len(episodes) // 2, len(episodes) - 1]
    seen = set()
    picked = []
    for i in idxs:
        if i in seen or i < 0 or i >= len(episodes):
            continue
        picked.append(episodes[i])
        seen.add(i)
    return picked[:count]


def plot_ident_and_learning(ident_path: str, episodes_path: str, output_path: str) -> None:
    ident = _load_json(ident_path)
    episodes_data = _load_json(episodes_path)
    episodes = episodes_data.get("episodes", [])
    params = _collect_params(ident)

    apply_style()
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))

    # 1) Params bar chart
    ax0 = axes[0]
    names = [p["name"] for p in params]
    est_vals = [p["est"] for p in params]
    true_vals = [p["true"] for p in params]
    x = np.arange(len(names))
    width = 0.35
    if names:
        ax0.bar(x - width / 2, est_vals, width, label="Estimated")
        ax0.bar(x + width / 2, true_vals, width, label="True")
        for i, (e, t) in enumerate(zip(est_vals, true_vals)):
            if np.isnan(t) or t == 0:
                continue
            rel = abs(e - t) / abs(t) * 100.0
            ax0.text(x[i], max(e, t), f"{rel:.1f}%", ha="center", va="bottom", fontsize=8)
    ax0.set_title("Identification: estimated vs true")
    ax0.set_xticks(x, names)
    ax0.legend()
    ax0.grid(True, axis="y")

    # 2) Learning curves
    ax1 = axes[1]
    ep_idx = [ep.get("episode_idx", i) for i, ep in enumerate(episodes)]
    mean_err = [ep.get("mean_speed_error", 0.0) for ep in episodes]
    total_reward = [ep.get("total_reward", 0.0) for ep in episodes]
    ax1.plot(ep_idx, mean_err, marker="o", color="tab:blue", label="mean |e_w|")
    ax1.plot(ep_idx, total_reward, marker="s", color="tab:orange", label="total_reward")
    ax1.set_ylabel("Value")
    ax1.grid(True)
    ax1.set_xlabel("Episode")
    ax1.set_title("Learning curves (error ↓, reward ↑)")

    # 3) Trajectories for selected episodes
    ax2 = axes[2]
    picked = _pick_eps(episodes)
    colors = plt.cm.viridis(np.linspace(0, 1, max(len(picked), 1)))
    for idx, ep in enumerate(picked):
        t = np.asarray(ep.get("t", []), dtype=float)
        omega = np.asarray(ep.get("omega", []), dtype=float)
        omega_ref = np.asarray(ep.get("omega_ref", []), dtype=float)
        if t.size == 0:
            continue
        label = f"ep{ep.get('episode_idx', idx)}"
        ax2.plot(t, omega, color=colors[idx], label=f"{label} omega")
        ax2.plot(t, omega_ref, color=colors[idx], linestyle="--", alpha=0.7, label=f"{label} omega_ref")
    ax2.set_title("Trajectories (early/mid/late episodes)")
    ax2.set_xlabel("t, s")
    ax2.set_ylabel("Speed, rad/s")
    ax2.legend()
    ax2.grid(True)

    fig.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    plt.close(fig)


__all__ = ["plot_ident_and_learning"]
