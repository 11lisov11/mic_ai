# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

from motor_phys_ai.utils.metrics import calc_metrics, weighted_score


def _load_csv(path: Path) -> Dict[str, np.ndarray]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"CSV is empty: {path}")
    header = lines[0].split(",")
    cols: Dict[str, list[float]] = {name: [] for name in header}
    for line in lines[1:]:
        parts = line.split(",")
        if len(parts) != len(header):
            continue
        for key, value in zip(header, parts):
            cols[key].append(float(value))
    return {key: np.asarray(values, dtype=float) for key, values in cols.items()}


def _parse_map(text: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            continue
        k, v = part.split(":", 1)
        mapping[k.strip()] = v.strip()
    return mapping


def _plot_compare(
    pdf: PdfPages,
    scenario: str,
    physmod: Dict[str, np.ndarray],
    pi: Dict[str, np.ndarray],
    mic: Dict[str, np.ndarray],
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(7.4, 7.0), sharex=True)

    axes[0].plot(physmod["t"], physmod["omega"], color="black", label="PhysMod")
    axes[0].plot(pi["t"], pi["omega"], color="0.4", linestyle="--", label="PI")
    axes[0].plot(mic["t"], mic["omega"], color="0.6", linestyle=":", label="MIC")
    axes[0].plot(mic["t"], mic["omega_ref"], color="0.7", linestyle="-.", label="omega_ref")
    axes[0].set_ylabel("omega, rad/s")
    axes[0].legend(frameon=False, ncol=2)

    axes[1].plot(physmod["t"], physmod["i_q"], color="black", label="PhysMod i_q")
    axes[1].plot(pi["t"], pi["i_q"], color="0.4", linestyle="--", label="PI i_q")
    if "i_rms" in mic:
        axes[1].plot(mic["t"], mic["i_rms"], color="0.6", linestyle=":", label="MIC i_rms")
    else:
        axes[1].plot(mic["t"], mic.get("i_q", np.zeros_like(mic["t"])), color="0.6", linestyle=":", label="MIC i_q")
    axes[1].set_ylabel("current, A")
    axes[1].legend(frameon=False, ncol=2)

    err_phys = physmod["omega_ref"] - physmod["omega"]
    err_pi = pi["omega_ref"] - pi["omega"]
    err_mic = mic["omega_ref"] - mic["omega"]
    axes[2].plot(physmod["t"], err_phys, color="black", label="PhysMod err")
    axes[2].plot(pi["t"], err_pi, color="0.4", linestyle="--", label="PI err")
    axes[2].plot(mic["t"], err_mic, color="0.6", linestyle=":", label="MIC err")
    axes[2].set_ylabel("error, rad/s")
    axes[2].set_xlabel("t, s")
    axes[2].legend(frameon=False, ncol=2)

    fig.suptitle(f"Scenario: {scenario}")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _plot_table(pdf: PdfPages, title: str, table_rows: list[Tuple[str, Dict[str, float]]]) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    ax.axis("off")
    ax.set_title(title)
    columns = ["Scenario", "PhysMod", "PI", "MIC"]
    cell_text = []
    for scenario, row in table_rows:
        cell_text.append(
            [
                scenario,
                f"{row.get('PhysMod', 0.0):.3f}",
                f"{row.get('PI', 0.0):.3f}",
                f"{row.get('MIC', 0.0):.3f}",
            ]
        )
    ax.table(cellText=cell_text, colLabels=columns, loc="center")
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare PhysMod/PI vs MIC AI.")
    parser.add_argument("--phys-run", required=True, help="Path to motor_phys_ai run directory with summary.json.")
    parser.add_argument("--mic-dir", required=True, help="Path to mic_ai CSV outputs.")
    parser.add_argument(
        "--mic-map",
        default="step:speed_step_0p2,load:load_step_0p2,drift:ramp_0p2",
        help="Scenario mapping: phys:mic case tag list.",
    )
    parser.add_argument("--out-dir", default="outputs/compare_phys_vs_mic")
    parser.add_argument("--speed-tol-rel", type=float, default=0.05)
    parser.add_argument("--w-speed", type=float, default=1.0)
    parser.add_argument("--w-energy", type=float, default=0.1)
    parser.add_argument("--w-stability", type=float, default=5.0)
    args = parser.parse_args()

    phys_dir = Path(args.phys_run)
    mic_dir = Path(args.mic_dir)
    if not (phys_dir / "summary.json").exists():
        raise FileNotFoundError(f"summary.json not found in {phys_dir}")

    mapping = _parse_map(args.mic_map)
    summary = json.loads((phys_dir / "summary.json").read_text(encoding="utf-8"))
    results = summary.get("results", {})

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / "summary.pdf"

    table_rows_score: list[Tuple[str, Dict[str, float]]] = []
    table_rows_power: list[Tuple[str, Dict[str, float]]] = []
    with PdfPages(pdf_path) as pdf:
        for scenario, data in results.items():
            scenario_key = scenario.split(":", 1)[0]
            physmod_csv = phys_dir / f"{scenario}_physmod.csv"
            pi_csv = phys_dir / f"{scenario}_pi.csv"
            if not physmod_csv.exists() or not pi_csv.exists():
                continue
            mic_tag = mapping.get(scenario, mapping.get(scenario_key, scenario_key))
            mic_csv = mic_dir / f"{mic_tag}_mic_ai.csv"
            if not mic_csv.exists():
                continue

            physmod = _load_csv(physmod_csv)
            pi = _load_csv(pi_csv)
            mic = _load_csv(mic_csv)

            mic_metrics = calc_metrics(
                mic["t"],
                mic["omega"],
                mic["omega_ref"],
                i_rms=mic.get("i_rms"),
                iq_limit=None,
                speed_tol_rel=float(args.speed_tol_rel),
            )
            mic_score = weighted_score(mic_metrics, args.w_speed, args.w_energy, args.w_stability)
            mic_pel = np.maximum(mic.get("p_el", np.zeros_like(mic["t"])), 0.0)
            mic_pel_mean = float(np.mean(mic_pel)) if mic_pel.size else 0.0

            row = {
                "PhysMod": float(data.get("PhysMod", {}).get("score", 0.0)),
                "PI": float(data.get("PI", {}).get("score", 0.0)),
                "MIC": float(mic_score),
            }
            table_rows_score.append((scenario, row))

            row_power = {
                "PhysMod": float(data.get("PhysMod", {}).get("p_el_mean", 0.0)),
                "PI": float(data.get("PI", {}).get("p_el_mean", 0.0)),
                "MIC": float(mic_pel_mean),
            }
            table_rows_power.append((scenario, row_power))

            _plot_compare(pdf, scenario, physmod, pi, mic)

        if table_rows_score:
            _plot_table(pdf, "Score (ниже лучше)", table_rows_score)
            _plot_table(pdf, "Средняя потребляемая мощность, Вт", table_rows_power)
        else:
            fig, ax = plt.subplots(figsize=(7.4, 4.2))
            ax.axis("off")
            ax.text(0.5, 0.5, "Нет совпадающих сценариев для сравнения.", ha="center", va="center")
            pdf.savefig(fig)
            plt.close(fig)

    print(f"Saved summary to {pdf_path}")


if __name__ == "__main__":
    main()
