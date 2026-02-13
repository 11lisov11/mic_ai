from __future__ import annotations

"""
Run scenario_compare + guardrails in one command.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


def _build_scenario_compare_cmd(args: argparse.Namespace) -> List[str]:
    cmd = [
        args.python,
        "-m",
        "mic_ai.tools.scenario_compare",
        "--env-config",
        str(args.env_config),
        "--out-dir",
        str(args.out_dir),
        "--window-frac",
        str(args.window_frac),
        "--error-tol-rel",
        str(args.error_tol_rel),
        "--error-tol-abs",
        str(args.error_tol_abs),
    ]
    if args.dt is not None:
        cmd += ["--dt", str(args.dt)]
    if args.t_end is not None:
        cmd += ["--t-end", str(args.t_end)]
    if args.load_torque is not None:
        cmd += ["--load-torque", str(args.load_torque)]
    if args.scenarios:
        cmd += ["--scenarios", str(args.scenarios)]
    if args.include_v3:
        cmd += ["--include-v3"]
    if args.use_total_power:
        cmd += ["--use-total-power"]

    if args.mic_id_ref_low is not None and args.mic_id_ref_high is not None:
        cmd += [
            "--mic-id-ref-low",
            str(args.mic_id_ref_low),
            "--mic-id-ref-high",
            str(args.mic_id_ref_high),
            "--mic-id-ref-speed-tol-rel",
            str(args.mic_id_ref_speed_tol_rel),
            "--mic-id-ref-omega-min",
            str(args.mic_id_ref_omega_min),
        ]
    elif args.ai_checkpoint:
        cmd += [
            "--ai-checkpoint",
            str(args.ai_checkpoint),
            "--ai-id-relative" if args.ai_id_relative else "",
            "--delta-id-max",
            str(args.delta_id_max),
        ]
    else:
        raise ValueError("Provide either mic-id-ref-low/high or --ai-checkpoint")

    # Filter empty tokens (from conditional flags above).
    cmd = [c for c in cmd if c]
    return cmd


def _build_guardrails_cmd(args: argparse.Namespace, summary_path: Path) -> List[str]:
    cmd = [
        args.python,
        "-m",
        "mic_ai.tools.guardrails_check",
        "--summary",
        str(summary_path),
        "--min-power-saving-pct",
        str(args.min_power_saving_pct),
        "--max-err-failures",
        str(args.max_err_failures),
    ]
    if args.min_eta_gain_pct is not None:
        cmd += ["--min-eta-gain-pct", str(args.min_eta_gain_pct)]
    if args.no_require_err_ok:
        cmd += ["--no-require-err-ok"]
    if args.guardrails_report:
        cmd += ["--write-report", str(args.guardrails_report)]
    return cmd


def _build_compare_cmd(args: argparse.Namespace, summary_path: Path) -> List[str]:
    cmd = [
        args.python,
        "-m",
        "mic_ai.tools.compare_summary",
        "--baseline",
        str(args.baseline_summary),
        "--current",
        str(summary_path),
        "--max-err-rel",
        str(args.compare_max_err_rel),
        "--max-err-abs",
        str(args.compare_max_err_abs),
        "--max-power-rel",
        str(args.compare_max_power_rel),
        "--max-power-abs",
        str(args.compare_max_power_abs),
    ]
    if args.compare_no_require_err_ok:
        cmd += ["--no-require-err-ok"]
    if args.compare_report:
        cmd += ["--report", str(args.compare_report)]
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description="Run scenario_compare + guardrails")
    parser.add_argument("--env-config", required=True, help="Env config path (.py)")
    parser.add_argument("--out-dir", default="outputs/bench_run")
    parser.add_argument("--python", default=sys.executable)

    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--t-end", type=float, default=None)
    parser.add_argument("--load-torque", type=float, default=None)
    parser.add_argument("--scenarios", default="speed_step,ramp,load_step,start_stop")
    parser.add_argument("--window-frac", type=float, default=0.25)
    parser.add_argument("--error-tol-rel", type=float, default=0.05)
    parser.add_argument("--error-tol-abs", type=float, default=0.0)
    parser.add_argument("--include-v3", action="store_true")
    parser.add_argument("--use-total-power", action="store_true")

    # MIC rule-based options
    parser.add_argument("--mic-id-ref-low", type=float, default=None)
    parser.add_argument("--mic-id-ref-high", type=float, default=None)
    parser.add_argument("--mic-id-ref-speed-tol-rel", type=float, default=0.05)
    parser.add_argument("--mic-id-ref-omega-min", type=float, default=0.1)

    # AI options
    parser.add_argument("--ai-checkpoint", default=None)
    parser.add_argument("--ai-id-relative", action="store_true")
    parser.add_argument("--delta-id-max", type=float, default=0.1)

    # Guardrails options
    parser.add_argument("--min-power-saving-pct", type=float, default=0.0)
    parser.add_argument("--min-eta-gain-pct", type=float, default=None)
    parser.add_argument("--max-err-failures", type=int, default=0)
    parser.add_argument("--no-require-err-ok", action="store_true")
    parser.add_argument("--guardrails-report", default=None)

    # Baseline compare options
    parser.add_argument("--baseline-summary", default=None, help="Baseline summary.json for regression check")
    parser.add_argument("--compare-max-err-rel", type=float, default=0.1)
    parser.add_argument("--compare-max-err-abs", type=float, default=0.0)
    parser.add_argument("--compare-max-power-rel", type=float, default=0.1)
    parser.add_argument("--compare-max-power-abs", type=float, default=0.0)
    parser.add_argument("--compare-no-require-err-ok", action="store_true")
    parser.add_argument("--compare-report", default=None)

    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    scenario_cmd = _build_scenario_compare_cmd(args)
    if not args.dry_run:
        subprocess.run(scenario_cmd, check=True)
    else:
        print("[dry-run] scenario_compare:", " ".join(scenario_cmd))

    summary_path = out_dir / "summary.json"
    if not summary_path.exists():
        raise SystemExit(f"Missing summary.json at {summary_path}")

    guard_cmd = _build_guardrails_cmd(args, summary_path)
    if not args.dry_run:
        subprocess.run(guard_cmd, check=True)
    else:
        print("[dry-run] guardrails:", " ".join(guard_cmd))

    compare_cmd = None
    if args.baseline_summary:
        compare_cmd = _build_compare_cmd(args, summary_path)
        if not args.dry_run:
            subprocess.run(compare_cmd, check=True)
        else:
            print("[dry-run] compare_summary:", " ".join(compare_cmd))

    report = {
        "scenario_cmd": scenario_cmd,
        "guardrails_cmd": guard_cmd,
        "compare_cmd": compare_cmd,
        "summary_path": str(summary_path),
        "out_dir": str(out_dir),
    }
    (out_dir / "benchmark_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
