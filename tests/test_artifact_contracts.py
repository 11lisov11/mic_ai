from __future__ import annotations

from pathlib import Path

import pandas as pd


STEP28_TAG = "20260304_al31_robust_rand009_nodrift_v3"


def _require_columns(path: Path, required: set[str]) -> None:
    assert path.exists(), f"missing file: {path}"
    df = pd.read_csv(path)
    assert required.issubset(set(df.columns)), f"{path} missing columns: {sorted(required - set(df.columns))}"


def test_step27_per_seed_contract() -> None:
    path = Path("paper/ieee_2026/data/step28") / STEP28_TAG / "mode1_foc_encoder_vs_mic_sensorless" / "step27_per_seed_metrics.csv"
    required = {
        "motor",
        "seed",
        "controller",
        "avg_power_saving_pct",
        "avg_eta_gain_pct",
        "err_failures",
        "start_stop_power_saving_pct",
        "worst_current_peak_ratio",
        "worst_current_mean_ratio",
        "avg_controller_speed_err",
    }
    _require_columns(path, required)


def test_step27_stats_contract() -> None:
    path = Path("paper/ieee_2026/data/step28") / STEP28_TAG / "mode1_foc_encoder_vs_mic_sensorless" / "step27_stats_motor_controller.csv"
    required = {
        "motor",
        "controller",
        "samples",
        "avg_power_saving_pct_mean",
        "avg_power_saving_pct_std",
        "avg_power_saving_pct_min",
        "avg_eta_gain_pct_mean",
        "avg_eta_gain_pct_std",
        "avg_eta_gain_pct_min",
        "err_failures_mean",
        "err_failures_max",
        "start_stop_power_saving_pct_mean",
        "start_stop_power_saving_pct_min",
    }
    _require_columns(path, required)


def test_step28_summary_contract() -> None:
    path = Path("paper/ieee_2026/data/step28") / STEP28_TAG / "step28_ieee_summary.csv"
    required = {
        "mode",
        "controller",
        "avg_power_saving_pct_mean",
        "avg_power_saving_pct_std",
        "avg_power_saving_pct_min",
        "avg_eta_gain_pct_mean",
        "avg_eta_gain_pct_std",
        "avg_eta_gain_pct_min",
        "err_failures_mean",
        "err_failures_max",
        "start_stop_power_saving_pct_mean",
        "start_stop_power_saving_pct_min",
        "stable_vs_previous",
        "table_sha256",
        "air56_mean_pass",
        "air56_worst_case_pass",
    }
    _require_columns(path, required)


def test_theory_validation_summary_contract() -> None:
    path = Path("paper/ieee_2026/data/theory_validation") / STEP28_TAG / "theory_validation_summary.csv"
    required = {
        "motor",
        "csv_path",
        "passed",
        "hard_fail_count",
        "warn_fail_count",
        "report_json",
        "report_md",
    }
    _require_columns(path, required)
