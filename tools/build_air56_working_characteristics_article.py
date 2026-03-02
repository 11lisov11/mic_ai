from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd=ROOT, check=True)


def _interp_common(df: pd.DataFrame, policy: str, col: str, x_common: float) -> float:
    d = df[df["policy"] == policy].sort_values("p2_kw")
    x = d["p2_kw"].to_numpy(dtype=float)
    y = d[col].to_numpy(dtype=float)
    if x.size == 0:
        return float("nan")
    return float(np.interp(float(x_common), x, y))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build AIR56 article-ready working characteristics (FOC vs MIC).")
    parser.add_argument("--env-config", default="config/env_research_air56_025kw.py")
    parser.add_argument("--out-dir", default="outputs/article_air56_20260302")
    parser.add_argument("--fig-dir", default="paper/pgups_2026/fig")
    parser.add_argument("--omega-ref-pu", type=float, default=1.0)
    parser.add_argument("--i-max", type=float, default=2.2)
    parser.add_argument("--mic-id-ref-low", type=float, default=1.20)
    parser.add_argument("--mic-id-ref-high", type=float, default=1.50)
    parser.add_argument("--load-points", type=int, default=15)
    parser.add_argument("--common-p2-kw", type=float, default=0.24)
    args = parser.parse_args()

    out_dir = ROOT / args.out_dir
    fig_dir = ROOT / args.fig_dir
    fig_base = fig_dir / "working_characteristics_air56_foc_mic"
    table_csv = fig_dir / "working_characteristics_air56_foc_mic_table.csv"
    note_txt = fig_dir / "working_characteristics_air56_foc_mic_cosphi_fix_note.txt"
    summary_json = fig_dir / "working_characteristics_air56_foc_mic_summary.json"

    fig_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "mic_ai.tools.drive_characteristics_ai",
        "--env-config",
        str(args.env_config),
        "--omega-ref-pu",
        str(args.omega_ref_pu),
        "--load-points",
        str(args.load_points),
        "--speed-pu",
        str(args.omega_ref_pu),
        "--t-end",
        "4.0",
        "--dt",
        "0.0005",
        "--window-frac",
        "0.25",
        "--i-max",
        str(args.i_max),
        "--mic-id-ref-low",
        str(args.mic_id_ref_low),
        "--mic-id-ref-high",
        str(args.mic_id_ref_high),
        "--speed-tol",
        "0.35",
        "--plot-air56-journal",
        "--journal-drop-zero-load",
        "--journal-common-p2-kw",
        str(args.common_p2_kw),
        "--journal-out-base",
        str(fig_base),
        "--export-abc-traces",
        "--out-dir",
        str(out_dir),
    ]
    _run(cmd)

    src_csv = out_dir / "load_characteristics.csv"
    if not src_csv.exists():
        raise FileNotFoundError(src_csv)
    df = pd.read_csv(src_csv)

    table = (
        df[
            [
                "policy",
                "load_factor",
                "p2_kw",
                "m2",
                "n2_rpm",
                "i_rms",
                "p_el",
                "eta_pct",
                "cos_phi",
                "cos_phi_method",
            ]
        ]
        .rename(
            columns={
                "policy": "policy",
                "load_factor": "load_factor",
                "p2_kw": "P2_kW",
                "m2": "M2_Nm",
                "n2_rpm": "n2_rpm",
                "i_rms": "I1_A",
                "p_el": "P1_W",
                "eta_pct": "eta_pct",
                "cos_phi": "cosphi",
                "cos_phi_method": "cosphi_method",
            }
        )
        .sort_values(["policy", "load_factor"])
    )
    table.to_csv(table_csv, index=False)

    eta_f = _interp_common(df, "FOC", "eta_pct", float(args.common_p2_kw))
    eta_m = _interp_common(df, "MIC_AI", "eta_pct", float(args.common_p2_kw))
    i1_f = _interp_common(df, "FOC", "i_rms", float(args.common_p2_kw))
    i1_m = _interp_common(df, "MIC_AI", "i_rms", float(args.common_p2_kw))
    n2_f = _interp_common(df, "FOC", "n2_rpm", float(args.common_p2_kw))
    n2_m = _interp_common(df, "MIC_AI", "n2_rpm", float(args.common_p2_kw))
    p1_f = _interp_common(df, "FOC", "p_el", float(args.common_p2_kw))
    p1_m = _interp_common(df, "MIC_AI", "p_el", float(args.common_p2_kw))

    summary = {
        "source_csv": str(src_csv),
        "figure_base": str(fig_base),
        "table_csv": str(table_csv),
        "common_p2_kw": float(args.common_p2_kw),
        "eta_foc_pct_at_common": eta_f,
        "eta_mic_pct_at_common": eta_m,
        "eta_gain_pp_at_common": float(eta_m - eta_f),
        "i1_foc_A_at_common": i1_f,
        "i1_mic_A_at_common": i1_m,
        "i1_delta_A_at_common": float(i1_m - i1_f),
        "n2_foc_rpm_at_common": n2_f,
        "n2_mic_rpm_at_common": n2_m,
        "n2_delta_rpm_at_common": float(n2_m - n2_f),
        "p1_foc_W_at_common": p1_f,
        "p1_mic_W_at_common": p1_m,
        "p1_saving_pct_at_common": float(100.0 * (1.0 - p1_m / max(p1_f, 1e-9))),
        "cosphi_range_foc": [
            float(df[df["policy"] == "FOC"]["cos_phi"].min()),
            float(df[df["policy"] == "FOC"]["cos_phi"].max()),
        ],
        "cosphi_range_mic": [
            float(df[df["policy"] == "MIC_AI"]["cos_phi"].min()),
            float(df[df["policy"] == "MIC_AI"]["cos_phi"].max()),
        ],
    }
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    note_txt.write_text(
        (
            "Исправление cosφ: ранее использовалась одна формула полной мощности "
            "без проверки типа напряжения (фазное/линейное), что могло искажать форму кривой. "
            "Теперь cosφ считается по мгновенным v_abc/i_abc на стационарном окне с диагностикой "
            "phase vs line-line и авто-выбором метода.\n"
        ),
        encoding="utf-8",
    )

    print(f"Figure: {fig_base}.(png|pdf|svg)")
    print(f"Table: {table_csv}")
    print(f"Summary: {summary_json}")
    print(f"Note: {note_txt}")


if __name__ == "__main__":
    main()
