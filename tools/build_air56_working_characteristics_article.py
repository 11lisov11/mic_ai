from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd=ROOT, check=True)


def _load_config_module(env_config: str):
    cfg_path = Path(env_config)
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    cfg_path = cfg_path.resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Env config not found: {cfg_path}")
    spec = importlib.util.spec_from_file_location("build_air56_env_cfg", str(cfg_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load env config: {cfg_path}")
    root_str = str(ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def _interp_common(df: pd.DataFrame, policy: str, col: str, x_common: float) -> float:
    d = df[df["policy"] == policy].sort_values("p2_kw")
    x = d["p2_kw"].to_numpy(dtype=float)
    y = d[col].to_numpy(dtype=float)
    if x.size == 0:
        return float("nan")
    return float(np.interp(float(x_common), x, y))


def main() -> None:
    warnings.warn(
        "tools/build_air56_working_characteristics_article.py is deprecated. "
        "Use tools/build_air56_mech_journal_from_traces.py (figures) and "
        "tools/validate_theory_working_characteristics.py (validation).",
        DeprecationWarning,
        stacklevel=2,
    )
    parser = argparse.ArgumentParser(description="Build AIR56 article-ready working characteristics (FOC vs MIC).")
    parser.add_argument("--env-config", default="config/env_research_air56_025kw.py")
    parser.add_argument("--out-dir", default="outputs/article_air56_20260302")
    parser.add_argument("--fig-dir", default="paper/pgups_2026/fig")
    parser.add_argument("--mic-policy", choices=["ai", "rule", "fixed"], default="ai")
    parser.add_argument("--ai-mode", choices=["ai_id_ref", "ai_voltage", "foc_assist"], default="ai_id_ref")
    parser.add_argument("--ai-checkpoint", default=None, help="Path to AI checkpoint for MIC policy.")
    parser.add_argument("--delta-id-max", type=float, default=None, help="Override delta-id-max for ai_id_ref mode.")
    parser.add_argument("--ai-id-relative", action="store_true", help="Force relative id_ref action for ai_id_ref.")
    parser.add_argument("--ai-id-absolute", action="store_true", help="Force absolute id_ref action for ai_id_ref.")
    parser.add_argument("--mic-id-ref", type=float, default=None, help="Fixed id_ref for MIC when --mic-policy=fixed.")
    parser.add_argument("--omega-ref-pu", type=float, default=1.0)
    parser.add_argument("--i-max", type=float, default=2.2)
    parser.add_argument("--mic-id-ref-low", type=float, default=1.20)
    parser.add_argument("--mic-id-ref-high", type=float, default=1.50)
    parser.add_argument("--load-points", type=int, default=15)
    parser.add_argument("--common-p2-kw", type=float, default=0.24)
    parser.add_argument(
        "--journal-max-speed-err-rel",
        type=float,
        default=0.2,
        help="Forwarded to drive_characteristics_ai; <=0 disables speed-error filtering for journal figure.",
    )
    parser.add_argument(
        "--journal-max-n2-step-rpm",
        type=float,
        default=0.0,
        help="Forwarded to drive_characteristics_ai; <=0 disables abrupt n2-step clipping for journal figure.",
    )
    parser.add_argument("--journal-formats", default="pdf", help="Formats for journal figure: comma-separated (e.g. 'pdf').")
    parser.add_argument("--figure-only", action="store_true", help="Build only figure; do not export table/summary/note.")
    args = parser.parse_args()
    env_mod = _load_config_module(str(args.env_config))
    if bool(args.ai_id_relative) and bool(args.ai_id_absolute):
        raise ValueError("Choose only one: --ai-id-relative or --ai-id-absolute.")

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
        "--speed-tol",
        "0.35",
        "--plot-air56-journal",
        "--journal-drop-zero-load",
        "--journal-common-p2-kw",
        str(args.common_p2_kw),
        "--journal-max-speed-err-rel",
        str(args.journal_max_speed_err_rel),
        "--journal-max-n2-step-rpm",
        str(args.journal_max_n2_step_rpm),
        "--journal-out-base",
        str(fig_base),
        "--journal-formats",
        str(args.journal_formats),
        "--export-abc-traces",
        "--out-dir",
        str(out_dir),
    ]

    mic_policy = str(args.mic_policy).lower()
    if mic_policy == "rule":
        cmd += [
            "--mic-id-ref-low",
            str(args.mic_id_ref_low),
            "--mic-id-ref-high",
            str(args.mic_id_ref_high),
        ]
    elif mic_policy == "fixed":
        if args.mic_id_ref is None:
            raise ValueError("For --mic-policy=fixed provide --mic-id-ref.")
        cmd += ["--mic-id-ref", str(float(args.mic_id_ref))]
    else:
        ai_checkpoint = str(args.ai_checkpoint) if args.ai_checkpoint else str(getattr(env_mod, "ai_eval_checkpoint_path", ""))
        if not ai_checkpoint:
            raise ValueError("AI policy selected, but checkpoint is missing. Set --ai-checkpoint or ai_eval_checkpoint_path in env config.")
        ai_ckpt_path = Path(ai_checkpoint)
        if not ai_ckpt_path.is_absolute():
            ai_ckpt_path = (ROOT / ai_ckpt_path).resolve()
        if not ai_ckpt_path.exists():
            raise FileNotFoundError(f"AI checkpoint not found: {ai_ckpt_path}")

        cmd += [
            "--ai-mode",
            str(args.ai_mode),
            "--ai-checkpoint",
            str(ai_ckpt_path),
        ]
        if str(args.ai_mode) == "ai_id_ref":
            rel_default = bool(getattr(env_mod, "ai_eval_id_ref_relative", False))
            use_relative = rel_default
            if bool(args.ai_id_relative):
                use_relative = True
            elif bool(args.ai_id_absolute):
                use_relative = False
            delta_default = float(getattr(env_mod, "ai_eval_delta_id_max", 0.3))
            delta_id_max = float(args.delta_id_max) if args.delta_id_max is not None else delta_default
            if use_relative:
                cmd.append("--ai-id-relative")
            cmd += ["--delta-id-max", str(delta_id_max)]

    _run(cmd)

    # If only figure is requested, remove extra per-figure files in fig_dir.
    if bool(args.figure_only):
        for extra in (table_csv, note_txt, summary_json):
            if extra.exists():
                extra.unlink()
        print(f"Figure: {fig_base}.pdf")
        return

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

    print(f"Figure: {fig_base}.({args.journal_formats})")
    print(f"Table: {table_csv}")
    print(f"Summary: {summary_json}")
    print(f"Note: {note_txt}")


if __name__ == "__main__":
    main()
