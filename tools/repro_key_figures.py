from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce key PGUPS figures in one command.")
    parser.add_argument("--traces-root", default="paper/pgups_2026/data/traces")
    parser.add_argument("--multi-out-dir", default="outputs/plan_repro_study")
    parser.add_argument("--air56-out-dir", default="outputs/article_air56_20260302")
    parser.add_argument("--air56-fig-dir", default="paper/pgups_2026/fig")
    parser.add_argument("--air56-common-p2-kw", type=float, default=0.236)
    parser.add_argument("--build-nominal-win", action="store_true")
    parser.add_argument("--nominal-source-dir", default="outputs/_tmp_bench_reg")
    parser.add_argument("--nominal-case", default="speed_step_0p2")
    parser.add_argument("--nominal-out-dir", default="outputs/paper_win_nominal_speed_step")
    args = parser.parse_args()

    _run(
        [
            sys.executable,
            "tools/multi_motor_study_report.py",
            "--traces-root",
            str(args.traces_root),
            "--out-dir",
            str(args.multi_out_dir),
        ]
    )

    _run(
        [
            sys.executable,
            "tools/build_air56_working_characteristics_article.py",
            "--mic-policy",
            "ai",
            "--out-dir",
            str(args.air56_out_dir),
            "--fig-dir",
            str(args.air56_fig_dir),
            "--common-p2-kw",
            str(float(args.air56_common_p2_kw)),
            "--journal-formats",
            "png,pdf,svg",
            "--figure-only",
        ]
    )

    nominal_built = False
    if bool(args.build_nominal_win):
        src = Path(args.nominal_source_dir)
        if (src / f"{args.nominal_case}_foc.csv").exists() and (src / f"{args.nominal_case}_mic_ai.csv").exists():
            _run(
                [
                    sys.executable,
                    "-m",
                    "mic_ai.tools.plot_nominal_win",
                    "--source-dir",
                    str(args.nominal_source_dir),
                    "--case",
                    str(args.nominal_case),
                    "--out-dir",
                    str(args.nominal_out_dir),
                ]
            )
            nominal_built = True

    report = {
        "multi_out_dir": str((ROOT / args.multi_out_dir).resolve()),
        "air56_out_dir": str((ROOT / args.air56_out_dir).resolve()),
        "air56_figure_pdf": str((ROOT / args.air56_fig_dir / "working_characteristics_air56_foc_mic.pdf").resolve()),
        "air56_figure_png": str((ROOT / args.air56_fig_dir / "working_characteristics_air56_foc_mic.png").resolve()),
        "air56_figure_svg": str((ROOT / args.air56_fig_dir / "working_characteristics_air56_foc_mic.svg").resolve()),
        "nominal_win_built": bool(nominal_built),
    }
    out_report = ROOT / "outputs" / "key_figures_repro_report.json"
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(out_report)


if __name__ == "__main__":
    main()
