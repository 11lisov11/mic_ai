from __future__ import annotations

from tools.step27_pipeline import Air56Acceptance, SeedPerturbationSettings, _build_report_markdown


def _acceptance() -> Air56Acceptance:
    return Air56Acceptance(
        min_avg_power_saving_pct=0.5,
        min_avg_eta_gain_pct=0.0,
        max_err_failures=2.0,
        min_start_stop_power_saving_pct=-0.5,
    )


def _seed_settings() -> SeedPerturbationSettings:
    return SeedPerturbationSettings(enabled=True, level=0.2)


def test_report_omits_air56_section_when_air56_not_requested() -> None:
    report = _build_report_markdown(
        motors=["ao2"],
        scenarios=["speed_step"],
        seeds=[101],
        seed_perturbation=_seed_settings(),
        acceptance=_acceptance(),
        tuning={"selected_metrics": {"tag": "x"}},
        global_stats=[],
        motor_stats=[],
        air56_accept={"mean_pass": False, "worst_case_pass": False},
        reproducibility={"table_sha256": "abc", "stable_vs_previous": None},
    )
    assert "## AIR56 Acceptance Criteria" not in report
    assert "## AIR56 Tuned Candidate" not in report


def test_report_keeps_air56_section_when_air56_requested() -> None:
    report = _build_report_markdown(
        motors=["air56", "ao2"],
        scenarios=["speed_step"],
        seeds=[101],
        seed_perturbation=_seed_settings(),
        acceptance=_acceptance(),
        tuning={"selected_metrics": {"tag": "x", "objective": "p_in"}},
        global_stats=[],
        motor_stats=[],
        air56_accept={"mean_pass": True, "worst_case_pass": True},
        reproducibility={"table_sha256": "abc", "stable_vs_previous": None},
    )
    assert "## AIR56 Acceptance Criteria" in report
    assert "## AIR56 Tuned Candidate" in report
