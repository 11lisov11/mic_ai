from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.common_utils import parse_csv_list as _parse_csv_list_shared
from tools.common_utils import parse_int_list as _parse_int_list_shared


DEFAULT_EXTENDED_SEEDS = "101,202,303,404,505,606,707,808"
DEFAULT_SCENARIOS = "speed_step,ramp,load_step,start_stop"
DEFAULT_MOTORS = "air56,al31,ao2"
DEFAULT_PERTURB_LEVELS = "0.2,0.4"

METRIC_FIELDS = (
    "avg_power_saving_pct",
    "avg_eta_gain_pct",
    "err_failures",
    "start_stop_power_saving_pct",
    "worst_current_peak_ratio",
    "worst_current_mean_ratio",
    "avg_controller_speed_err",
)

WORST_KIND = {
    "avg_power_saving_pct": "min",
    "avg_eta_gain_pct": "min",
    "start_stop_power_saving_pct": "min",
    "err_failures": "max",
    "worst_current_peak_ratio": "max",
    "worst_current_mean_ratio": "max",
    "avg_controller_speed_err": "max",
}


def _parse_csv_list(text: str) -> List[str]:
    return _parse_csv_list_shared(text)


def _parse_int_list(text: str) -> List[int]:
    return _parse_int_list_shared(text)


def _parse_float_list(text: str) -> List[float]:
    values: List[float] = []
    for token in _parse_csv_list_shared(text):
        values.append(float(token))
    return values


def _run(cmd: List[str], *, cwd: Path, dry_run: bool) -> None:
    print("[step27-extended] run:", " ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=cwd, check=True)


def _to_float(v: object) -> float:
    try:
        return float(v)
    except Exception:
        return 0.0


def _build_stats(df: pd.DataFrame) -> pd.DataFrame:
    grouped_rows: List[Dict[str, object]] = []
    keys = ["run_tag", "perturb_level", "motor", "controller"]
    for key_vals, part in df.groupby(keys, dropna=False):
        run_tag, perturb_level, motor, controller = key_vals
        row: Dict[str, object] = {
            "run_tag": str(run_tag),
            "perturb_level": float(perturb_level),
            "motor": str(motor),
            "controller": str(controller),
            "samples": int(len(part)),
        }
        for metric in METRIC_FIELDS:
            series = pd.to_numeric(part[metric], errors="coerce")
            row[f"{metric}_mean"] = float(series.mean())
            row[f"{metric}_std"] = float(series.std(ddof=0))
            row[f"{metric}_min"] = float(series.min())
            row[f"{metric}_max"] = float(series.max())
            worst_kind = str(WORST_KIND.get(metric, "min"))
            row[f"{metric}_worst"] = float(series.min() if worst_kind == "min" else series.max())
        grouped_rows.append(row)
    if not grouped_rows:
        return pd.DataFrame()
    return pd.DataFrame(grouped_rows).sort_values(keys).reset_index(drop=True)


def _build_stress_report(stats: pd.DataFrame) -> pd.DataFrame:
    if stats.empty:
        return pd.DataFrame()
    mic = stats[stats["controller"].astype(str).str.upper() == "MIC"].copy()
    if mic.empty:
        return pd.DataFrame()
    out_rows: List[Dict[str, object]] = []
    for key_vals, part in mic.groupby(["perturb_level", "motor"], dropna=False):
        perturb_level, motor = key_vals
        out_rows.append(
            {
                "perturb_level": float(perturb_level),
                "motor": str(motor),
                "power_mean_mean": float(pd.to_numeric(part["avg_power_saving_pct_mean"], errors="coerce").mean()),
                "power_worst_min": float(pd.to_numeric(part["avg_power_saving_pct_worst"], errors="coerce").min()),
                "eta_mean_mean": float(pd.to_numeric(part["avg_eta_gain_pct_mean"], errors="coerce").mean()),
                "eta_worst_min": float(pd.to_numeric(part["avg_eta_gain_pct_worst"], errors="coerce").min()),
                "err_worst_max": float(pd.to_numeric(part["err_failures_worst"], errors="coerce").max()),
                "speed_err_worst_max": float(pd.to_numeric(part["avg_controller_speed_err_worst"], errors="coerce").max()),
            }
        )
    if not out_rows:
        return pd.DataFrame()
    return pd.DataFrame(out_rows).sort_values(["perturb_level", "motor"]).reset_index(drop=True)


def _render_md(
    *,
    out_path: Path,
    run_manifest_rows: List[Dict[str, object]],
    stats: pd.DataFrame,
    stress: pd.DataFrame,
) -> None:
    lines: List[str] = []
    lines.append("# Step27 Extended Reproducibility Report")
    lines.append("")
    lines.append("## Runs")
    for row in run_manifest_rows:
        lines.append(
            "- tag={tag}, perturb_level={level:.3f}, out_dir=`{out_dir}`".format(
                tag=row.get("run_tag", ""),
                level=float(row.get("perturb_level", 0.0)),
                out_dir=row.get("run_out_dir", ""),
            )
        )
    lines.append("")

    lines.append("## MIC stats (mean/std/min/max/worst)")
    mic = stats[stats["controller"].astype(str).str.upper() == "MIC"].copy() if not stats.empty else pd.DataFrame()
    if mic.empty:
        lines.append("- no MIC rows")
    else:
        lines.append("| run_tag | motor | perturb_level | power_mean | power_std | power_worst | eta_mean | eta_std | eta_worst | err_worst | speed_err_worst |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for _, row in mic.sort_values(["run_tag", "motor"]).iterrows():
            lines.append(
                "| {run_tag} | {motor} | {perturb_level:.3f} | {avg_power_saving_pct_mean:+.3f} | {avg_power_saving_pct_std:.3f} | {avg_power_saving_pct_worst:+.3f} | {avg_eta_gain_pct_mean:+.3f} | {avg_eta_gain_pct_std:.3f} | {avg_eta_gain_pct_worst:+.3f} | {err_failures_worst:.2f} | {avg_controller_speed_err_worst:.3f} |".format(
                    **{k: row[k] for k in row.index}
                )
            )
    lines.append("")

    lines.append("## Stress sweep summary")
    if stress.empty:
        lines.append("- stress rows are empty")
    else:
        lines.append("| perturb_level | motor | power_mean_mean | power_worst_min | eta_mean_mean | eta_worst_min | err_worst_max | speed_err_worst_max |")
        lines.append("|---:|---|---:|---:|---:|---:|---:|---:|")
        for _, row in stress.iterrows():
            lines.append(
                "| {perturb_level:.3f} | {motor} | {power_mean_mean:+.3f} | {power_worst_min:+.3f} | {eta_mean_mean:+.3f} | {eta_worst_min:+.3f} | {err_worst_max:.2f} | {speed_err_worst_max:.3f} |".format(
                    **{k: row[k] for k in row.index}
                )
            )
    lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run step27 extended-seed reproducibility with perturbation sweep and build stats.")
    parser.add_argument("--out-dir", default="outputs/step27_extended_repro")
    parser.add_argument("--motors", default=DEFAULT_MOTORS)
    parser.add_argument("--seeds", default=DEFAULT_EXTENDED_SEEDS)
    parser.add_argument("--scenarios", default=DEFAULT_SCENARIOS)
    parser.add_argument("--perturb-levels", default=DEFAULT_PERTURB_LEVELS, help="Comma list. Baseline run is always included with level=0.0.")
    parser.add_argument("--window-frac", type=float, default=0.25)
    parser.add_argument("--error-tol-rel", type=float, default=0.05)
    parser.add_argument("--error-tol-abs", type=float, default=0.0)
    parser.add_argument("--foc-feedback-mode", default="encoder", choices=["encoder", "sensorless"])
    parser.add_argument("--mic-feedback-mode", default="sensorless", choices=["encoder", "sensorless"])
    parser.add_argument("--mic-mode", default="ai", choices=["ai", "rule"])
    parser.add_argument("--checkpoint-registry", default="config/checkpoint_registry.json")
    parser.add_argument("--skip-air56-tune", dest="skip_air56_tune", action="store_true")
    parser.add_argument("--allow-air56-tune", dest="skip_air56_tune", action="store_false")
    parser.add_argument("--foc-disable-lut", dest="foc_disable_lut", action="store_true")
    parser.add_argument("--allow-foc-lut", dest="foc_disable_lut", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    parser.set_defaults(skip_air56_tune=True)
    parser.set_defaults(foc_disable_lut=True)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    out_dir = Path(str(args.out_dir)).expanduser()
    if not out_dir.is_absolute():
        out_dir = (root / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    motors = ",".join(_parse_csv_list(str(args.motors)))
    seeds = ",".join(str(x) for x in _parse_int_list(str(args.seeds)))
    scenarios = ",".join(_parse_csv_list(str(args.scenarios)))
    perturb_levels = sorted(set(float(max(0.0, float(x))) for x in _parse_float_list(str(args.perturb_levels))))
    run_levels = [0.0, *[x for x in perturb_levels if x > 0.0]]

    run_manifest_rows: List[Dict[str, object]] = []
    frames: List[pd.DataFrame] = []

    for level in run_levels:
        run_tag = "baseline" if level <= 0.0 else f"perturb_{str(level).replace('.', 'p')}"
        run_out_dir = out_dir / "runs" / run_tag
        run_out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            "tools/step27_pipeline.py",
            "--motors",
            motors,
            "--seeds",
            seeds,
            "--scenarios",
            scenarios,
            "--out-dir",
            str(run_out_dir),
            "--window-frac",
            str(float(args.window_frac)),
            "--error-tol-rel",
            str(float(args.error_tol_rel)),
            "--error-tol-abs",
            str(float(args.error_tol_abs)),
            "--foc-feedback-mode",
            str(args.foc_feedback_mode),
            "--mic-feedback-mode",
            str(args.mic_feedback_mode),
            "--mic-mode",
            str(args.mic_mode),
            "--checkpoint-registry",
            str(args.checkpoint_registry),
            "--use-total-power",
        ]
        if bool(args.skip_air56_tune):
            cmd.append("--skip-air56-tune")
        if bool(args.foc_disable_lut):
            cmd.append("--foc-disable-lut")
        else:
            cmd.append("--allow-foc-lut")
        if level > 0.0:
            cmd.extend(["--seed-perturbation", "--seed-perturb-level", str(float(level))])

        run_manifest_rows.append(
            {
                "run_tag": run_tag,
                "perturb_level": float(level),
                "run_out_dir": str(run_out_dir),
                "command": cmd,
            }
        )
        _run(cmd, cwd=root, dry_run=bool(args.dry_run))
        if bool(args.dry_run):
            continue
        per_seed_csv = run_out_dir / "step27_per_seed_metrics.csv"
        if not per_seed_csv.exists():
            raise FileNotFoundError(per_seed_csv)
        run_df = pd.read_csv(per_seed_csv).copy()
        run_df["run_tag"] = str(run_tag)
        run_df["perturb_level"] = float(level)
        frames.append(run_df)

    if bool(args.dry_run):
        return
    if not frames:
        raise ValueError("No run data collected.")

    per_seed_all = pd.concat(frames, ignore_index=True)
    stats = _build_stats(per_seed_all)
    stress = _build_stress_report(stats)

    out_per_seed_csv = out_dir / "step27_extended_per_seed_metrics.csv"
    out_stats_csv = out_dir / "step27_extended_stats.csv"
    out_stress_csv = out_dir / "step27_extended_stress_sweep.csv"
    out_json = out_dir / "step27_extended_manifest.json"
    out_md = out_dir / "step27_extended_report.md"

    per_seed_all.to_csv(out_per_seed_csv, index=False)
    stats.to_csv(out_stats_csv, index=False)
    stress.to_csv(out_stress_csv, index=False)

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "motors": motors,
        "seeds": seeds,
        "scenarios": scenarios,
        "perturb_levels": run_levels,
        "mic_mode": str(args.mic_mode),
        "skip_air56_tune": bool(args.skip_air56_tune),
        "runs": run_manifest_rows,
        "files": {
            "extended_per_seed_csv": str(out_per_seed_csv),
            "extended_stats_csv": str(out_stats_csv),
            "extended_stress_csv": str(out_stress_csv),
            "extended_report_md": str(out_md),
        },
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _render_md(out_path=out_md, run_manifest_rows=run_manifest_rows, stats=stats, stress=stress)

    print(f"saved: {out_per_seed_csv}")
    print(f"saved: {out_stats_csv}")
    print(f"saved: {out_stress_csv}")
    print(f"saved: {out_json}")
    print(f"saved: {out_md}")


if __name__ == "__main__":
    main()
