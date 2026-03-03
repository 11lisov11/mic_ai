from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


MODE_DIRS = (
    "mode1_foc_encoder_vs_mic_sensorless",
    "mode2_foc_sensorless_vs_mic_sensorless",
)


def _load_mode_stats(step28_dir: Path, mode_dir: str) -> pd.DataFrame:
    path = step28_dir / mode_dir / "step27_stats_motor_controller.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    df["mode"] = mode_dir
    return df


def _to_float(row: pd.Series, key: str) -> float:
    return float(pd.to_numeric(row.get(key, 0.0), errors="coerce"))


def _air56_acceptance(metrics: Dict[str, float]) -> Dict[str, bool]:
    mean_pass = bool(
        metrics["avg_power_saving_pct_mean"] > 0.5
        and metrics["avg_eta_gain_pct_mean"] >= 0.0
        and metrics["err_failures_mean"] <= 2.0
        and metrics["start_stop_power_saving_pct_mean"] >= -0.5
    )
    worst_pass = bool(
        metrics["avg_power_saving_pct_min"] > 0.5
        and metrics["avg_eta_gain_pct_min"] >= 0.0
        and metrics["err_failures_max"] <= 2.0
        and metrics["start_stop_power_saving_pct_min"] >= -0.5
    )
    return {"mean_pass": mean_pass, "worst_case_pass": worst_pass, "acceptance_pass": bool(mean_pass and worst_pass)}


def _generic_acceptance(metrics: Dict[str, float]) -> Dict[str, bool]:
    mean_pass = bool(
        metrics["avg_power_saving_pct_mean"] >= 0.0
        and metrics["avg_eta_gain_pct_mean"] >= 0.0
        and metrics["err_failures_mean"] <= 2.0
    )
    worst_pass = bool(
        metrics["avg_power_saving_pct_min"] >= 0.0
        and metrics["avg_eta_gain_pct_min"] >= 0.0
        and metrics["err_failures_max"] <= 2.0
    )
    return {"mean_pass": mean_pass, "worst_case_pass": worst_pass, "acceptance_pass": bool(mean_pass and worst_pass)}


def _build_motor_report_rows(df: pd.DataFrame, motor: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    part = df[(df["motor"].astype(str) == motor) & (df["controller"].astype(str).str.upper() == "MIC")]
    for _, row in part.iterrows():
        rows.append(
            {
                "motor": str(motor),
                "mode": str(row["mode"]),
                "samples": int(_to_float(row, "samples")),
                "avg_power_saving_pct_mean": _to_float(row, "avg_power_saving_pct_mean"),
                "avg_power_saving_pct_min": _to_float(row, "avg_power_saving_pct_min"),
                "avg_eta_gain_pct_mean": _to_float(row, "avg_eta_gain_pct_mean"),
                "avg_eta_gain_pct_min": _to_float(row, "avg_eta_gain_pct_min"),
                "err_failures_mean": _to_float(row, "err_failures_mean"),
                "err_failures_max": _to_float(row, "err_failures_max"),
                "start_stop_power_saving_pct_mean": _to_float(row, "start_stop_power_saving_pct_mean"),
                "start_stop_power_saving_pct_min": _to_float(row, "start_stop_power_saving_pct_min"),
            }
        )
    return rows


def _aggregate_worst(rows: List[Dict[str, object]]) -> Dict[str, float]:
    if not rows:
        return {
            "avg_power_saving_pct_mean": float("nan"),
            "avg_power_saving_pct_min": float("nan"),
            "avg_eta_gain_pct_mean": float("nan"),
            "avg_eta_gain_pct_min": float("nan"),
            "err_failures_mean": float("nan"),
            "err_failures_max": float("nan"),
            "start_stop_power_saving_pct_mean": float("nan"),
            "start_stop_power_saving_pct_min": float("nan"),
        }
    return {
        "avg_power_saving_pct_mean": min(float(r["avg_power_saving_pct_mean"]) for r in rows),
        "avg_power_saving_pct_min": min(float(r["avg_power_saving_pct_min"]) for r in rows),
        "avg_eta_gain_pct_mean": min(float(r["avg_eta_gain_pct_mean"]) for r in rows),
        "avg_eta_gain_pct_min": min(float(r["avg_eta_gain_pct_min"]) for r in rows),
        "err_failures_mean": max(float(r["err_failures_mean"]) for r in rows),
        "err_failures_max": max(float(r["err_failures_max"]) for r in rows),
        "start_stop_power_saving_pct_mean": min(float(r["start_stop_power_saving_pct_mean"]) for r in rows),
        "start_stop_power_saving_pct_min": min(float(r["start_stop_power_saving_pct_min"]) for r in rows),
    }


def _score_row(r: Dict[str, object]) -> float:
    penalty = 0.0
    penalty += max(0.0, -float(r["avg_power_saving_pct_mean"])) * 20.0
    penalty += max(0.0, -float(r["avg_eta_gain_pct_mean"])) * 15.0
    penalty += max(0.0, float(r["err_failures_mean"]) - 2.0) * 10.0
    return float(penalty)


def _write_md(path: Path, *, motor: str, rows: List[Dict[str, object]], agg: Dict[str, float], acc: Dict[str, bool]) -> None:
    lines: List[str] = []
    lines.append(f"# {motor.upper()} tuning report (from frozen step28)")
    lines.append("")
    lines.append("| mode | power_mean % | power_min % | eta_mean % | eta_min % | err_mean | err_max | start_stop_mean % | start_stop_min % |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            "| {mode} | {avg_power_saving_pct_mean:+.3f} | {avg_power_saving_pct_min:+.3f} | {avg_eta_gain_pct_mean:+.3f} | {avg_eta_gain_pct_min:+.3f} | {err_failures_mean:.2f} | {err_failures_max:.2f} | {start_stop_power_saving_pct_mean:+.3f} | {start_stop_power_saving_pct_min:+.3f} |".format(
                **r
            )
        )
    lines.append("")
    lines.append("## Worst-case across modes")
    lines.append(f"- avg_power_saving_pct_mean: `{agg['avg_power_saving_pct_mean']:+.3f}`")
    lines.append(f"- avg_power_saving_pct_min: `{agg['avg_power_saving_pct_min']:+.3f}`")
    lines.append(f"- avg_eta_gain_pct_mean: `{agg['avg_eta_gain_pct_mean']:+.3f}`")
    lines.append(f"- avg_eta_gain_pct_min: `{agg['avg_eta_gain_pct_min']:+.3f}`")
    lines.append(f"- err_failures_mean: `{agg['err_failures_mean']:.2f}`")
    lines.append(f"- err_failures_max: `{agg['err_failures_max']:.2f}`")
    lines.append(f"- start_stop_power_saving_pct_mean: `{agg['start_stop_power_saving_pct_mean']:+.3f}`")
    lines.append(f"- start_stop_power_saving_pct_min: `{agg['start_stop_power_saving_pct_min']:+.3f}`")
    lines.append("")
    lines.append("## Acceptance")
    lines.append(f"- mean_pass: `{acc['mean_pass']}`")
    lines.append(f"- worst_case_pass: `{acc['worst_case_pass']}`")
    lines.append(f"- acceptance_pass: `{acc['acceptance_pass']}`")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build per-motor tuning reports from frozen step28 statistics.")
    parser.add_argument("--step28-dir", required=True)
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--motors", default="air56,al31,ao2")
    args = parser.parse_args()

    step28_dir = Path(args.step28_dir).expanduser().resolve()
    if not step28_dir.exists():
        raise FileNotFoundError(step28_dir)
    out_dir = Path(args.out_dir).expanduser().resolve() if str(args.out_dir).strip() else (step28_dir / "derived_ieee")
    out_dir.mkdir(parents=True, exist_ok=True)

    dfs = [_load_mode_stats(step28_dir, m) for m in MODE_DIRS]
    df = pd.concat(dfs, ignore_index=True)
    motors = [m.strip().lower() for m in str(args.motors).split(",") if m.strip()]

    summary_rows: List[Dict[str, object]] = []
    for motor in motors:
        mode_rows = _build_motor_report_rows(df, motor)
        if not mode_rows:
            continue
        for r in mode_rows:
            r["score"] = _score_row(r)
        rank_rows = sorted(mode_rows, key=lambda x: (float(x["score"]), -float(x["avg_power_saving_pct_mean"])))

        agg = _aggregate_worst(mode_rows)
        acc = _air56_acceptance(agg) if motor == "air56" else _generic_acceptance(agg)
        summary_rows.append(
            {
                "motor": motor,
                **agg,
                "mean_pass": bool(acc["mean_pass"]),
                "worst_case_pass": bool(acc["worst_case_pass"]),
                "acceptance_pass": bool(acc["acceptance_pass"]),
            }
        )

        rank_csv = out_dir / f"motor_{motor}_search_rank.csv"
        pd.DataFrame(rank_rows).to_csv(rank_csv, index=False)
        report_md = out_dir / f"motor_{motor}_tuning_report.md"
        _write_md(report_md, motor=motor, rows=rank_rows, agg=agg, acc=acc)
        print(f"saved: {rank_csv}")
        print(f"saved: {report_md}")

    summary_csv = out_dir / "motor_tuning_acceptance_summary.csv"
    summary_json = out_dir / "motor_tuning_acceptance_summary.json"
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
    summary_json.write_text(json.dumps({"rows": summary_rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved: {summary_csv}")
    print(f"saved: {summary_json}")


if __name__ == "__main__":
    main()
