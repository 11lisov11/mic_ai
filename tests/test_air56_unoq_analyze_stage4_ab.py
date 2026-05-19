from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import air56_unoq_analyze_stage4_ab
from tools.air56_unoq_analyze_stage4_ab import analyze_run, build_stage4_summary, build_stage4_summary_from_paths


HEADER = "t_ms,omega_meas,omega_ref,p_in,i_rms,guard_fail,fallback_event,thermal_fault\n"


def _write_csv(path: Path, *, p_in: float, tracking_error: float = 0.2, i_rms: float = 1.5, guard: int = 0, fallback: int = 0, thermal: int = 0) -> Path:
    rows = [
        f"0,{144.5 - tracking_error},144.5,{p_in},{i_rms},{guard},{fallback},{thermal}",
        f"10,{144.5 + tracking_error},144.5,{p_in},{i_rms},{guard},{fallback},{thermal}",
        f"20,{144.5},144.5,{p_in},{i_rms},{guard},{fallback},{thermal}",
    ]
    path.write_text(HEADER + "\n".join(rows) + "\n", encoding="utf-8")
    return path


def _passing_paths(tmp_path: Path) -> dict[str, Path]:
    return {
        "foc_no": _write_csv(tmp_path / "foc_no.csv", p_in=100.0, tracking_error=0.3, i_rms=1.4),
        "foc_load": _write_csv(tmp_path / "foc_load.csv", p_in=130.0, tracking_error=0.4, i_rms=2.0),
        "ai_no": _write_csv(tmp_path / "ai_no.csv", p_in=95.0, tracking_error=0.25, i_rms=1.3),
        "ai_load": _write_csv(tmp_path / "ai_load.csv", p_in=120.0, tracking_error=0.35, i_rms=1.9),
    }


def test_stage4_ab_analyzer_builds_passing_summary(tmp_path: Path) -> None:
    paths = _passing_paths(tmp_path)

    summary = build_stage4_summary_from_paths(
        foc_no_load_csv=paths["foc_no"],
        foc_load_step_csv=paths["foc_load"],
        ai_no_load_csv=paths["ai_no"],
        ai_load_step_csv=paths["ai_load"],
        max_current_rms_a=2.5,
    )

    assert summary["schema"] == air56_unoq_analyze_stage4_ab.SCHEMA
    assert summary["passed"] is True
    assert summary["documented"] is True
    assert summary["guard_fail_delta"] == 0
    assert summary["tracking_guard_regression"] is False
    assert summary["current_thermal_limit_ok"] is True
    assert summary["fallback_oscillation"] is False
    assert summary["power_saving_pct"] > 0.0


def test_stage4_ab_analyzer_detects_power_tracking_and_guard_regressions(tmp_path: Path) -> None:
    foc_no = _write_csv(tmp_path / "foc_no.csv", p_in=100.0, tracking_error=0.1)
    foc_load = _write_csv(tmp_path / "foc_load.csv", p_in=100.0, tracking_error=0.1)
    ai_no = _write_csv(tmp_path / "ai_no.csv", p_in=120.0, tracking_error=3.0, guard=1)
    ai_load = _write_csv(tmp_path / "ai_load.csv", p_in=120.0, tracking_error=3.0, guard=1)

    summary = build_stage4_summary_from_paths(
        foc_no_load_csv=foc_no,
        foc_load_step_csv=foc_load,
        ai_no_load_csv=ai_no,
        ai_load_step_csv=ai_load,
        max_current_rms_a=2.5,
    )

    assert summary["passed"] is False
    assert summary["power_saving_pct"] < 0.0
    assert summary["tracking_guard_regression"] is True
    assert summary["guard_fail_delta"] > 0


def test_stage4_ab_analyzer_detects_current_and_fallback_regressions(tmp_path: Path) -> None:
    paths = _passing_paths(tmp_path)
    paths["ai_load"] = _write_csv(tmp_path / "ai_load_bad.csv", p_in=120.0, i_rms=9.0, fallback=1, thermal=1)

    summary = build_stage4_summary_from_paths(
        foc_no_load_csv=paths["foc_no"],
        foc_load_step_csv=paths["foc_load"],
        ai_no_load_csv=paths["ai_no"],
        ai_load_step_csv=paths["ai_load"],
        max_current_rms_a=2.5,
    )

    assert summary["passed"] is False
    assert summary["current_thermal_limit_ok"] is False
    assert summary["fallback_oscillation"] is True
    assert summary["metrics"]["ai_thermal_fault_count"] == 3
    assert summary["metrics"]["ai_fallback_event_count"] == 3


def test_stage4_ab_summary_accepts_explicit_run_metrics() -> None:
    foc_no = air56_unoq_analyze_stage4_ab.RunMetrics("foc_no", 2, 0.01, 100.0, 1.0, 1, 1.0, 0, 0, 0)
    foc_load = air56_unoq_analyze_stage4_ab.RunMetrics("foc_load", 2, 0.01, 100.0, 1.0, 1, 1.0, 0, 0, 0)
    ai_no = air56_unoq_analyze_stage4_ab.RunMetrics("ai_no", 2, 0.01, 90.0, 1.0, 0, 1.0, 0, 0, 0)
    ai_load = air56_unoq_analyze_stage4_ab.RunMetrics("ai_load", 2, 0.01, 90.0, 1.0, 0, 1.0, 0, 0, 0)

    summary = build_stage4_summary(
        foc_no_load=foc_no,
        foc_load_step=foc_load,
        ai_no_load=ai_no,
        ai_load_step=ai_load,
        max_current_rms_a=2.0,
        min_power_saving_pct=5.0,
    )

    assert summary["passed"] is True
    assert summary["guard_fail_delta"] == -2
    assert round(summary["power_saving_pct"], 2) == 10.0


def test_stage4_ab_cli_writes_summary_and_returns_status(tmp_path: Path, monkeypatch) -> None:
    paths = _passing_paths(tmp_path)
    out_json = tmp_path / "stage4.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "air56_unoq_analyze_stage4_ab.py",
            "--foc-no-load-csv",
            str(paths["foc_no"]),
            "--foc-load-step-csv",
            str(paths["foc_load"]),
            "--ai-no-load-csv",
            str(paths["ai_no"]),
            "--ai-load-step-csv",
            str(paths["ai_load"]),
            "--max-current-rms-a",
            "2.5",
            "--out-json",
            str(out_json),
        ],
    )

    assert air56_unoq_analyze_stage4_ab.main() == 0
    assert json.loads(out_json.read_text(encoding="utf-8"))["passed"] is True

    paths["ai_load"] = _write_csv(tmp_path / "ai_load_fail.csv", p_in=140.0)
    monkeypatch.setattr(
        "sys.argv",
        [
            "air56_unoq_analyze_stage4_ab.py",
            "--foc-no-load-csv",
            str(paths["foc_no"]),
            "--foc-load-step-csv",
            str(paths["foc_load"]),
            "--ai-no-load-csv",
            str(paths["ai_no"]),
            "--ai-load-step-csv",
            str(paths["ai_load"]),
            "--max-current-rms-a",
            "2.5",
            "--out-json",
            str(out_json),
        ],
    )
    assert air56_unoq_analyze_stage4_ab.main() == 1


def test_stage4_ab_analyzer_rejects_invalid_csv(tmp_path: Path) -> None:
    empty = tmp_path / "empty.csv"
    empty.write_text(HEADER, encoding="utf-8")
    with pytest.raises(ValueError, match="CSV log is empty"):
        analyze_run("bad", empty)

    missing = tmp_path / "missing.csv"
    missing.write_text("t_ms,p_in\n0,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="misses required columns"):
        analyze_run("bad", missing)

    bad_value = tmp_path / "bad_value.csv"
    bad_value.write_text(HEADER + "0,bad,144.5,100,1,0,0,0\n10,144.5,144.5,100,1,0,0,0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid numeric value"):
        analyze_run("bad", bad_value)

    non_finite = tmp_path / "non_finite.csv"
    non_finite.write_text(HEADER + "0,nan,144.5,100,1,0,0,0\n10,144.5,144.5,100,1,0,0,0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite numeric value"):
        analyze_run("bad", non_finite)

    one_sample = tmp_path / "one_sample.csv"
    one_sample.write_text(HEADER + "0,144.5,144.5,100,1,0,0,0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="at least two samples"):
        analyze_run("bad", one_sample)

    zero_power = tmp_path / "zero_power.csv"
    zero_power.write_text(HEADER + "0,144.5,144.5,0,1,0,0,0\n10,144.5,144.5,0,1,0,0,0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mean p_in must be positive"):
        analyze_run("bad", zero_power)


def test_stage4_ab_helper_edges() -> None:
    helper = air56_unoq_analyze_stage4_ab

    assert helper._truthy(True) is True
    assert helper._truthy(0) is False
    assert helper._truthy(1.0) is True
    assert helper._duration_s([10.0]) == 0.0
    assert helper._fallback_transitions([True]) == 0
    assert helper._weighted_mean([], "mean_p_in_w") == float("inf")
