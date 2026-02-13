import json
import subprocess
import sys
from pathlib import Path

from mic_ai.tools.compare_summary import compare_summaries


def test_compare_summary_pass() -> None:
    baseline = [
        {"scenario": "s1", "file_tag": "s1", "mic_mean_err": 1.0, "mic_p_el_pos": 10.0, "err_ok": True},
    ]
    current = [
        {"scenario": "s1", "file_tag": "s1", "mic_mean_err": 1.05, "mic_p_el_pos": 10.5, "err_ok": True},
    ]
    ok, report = compare_summaries(baseline, current, max_err_rel=0.1, max_power_rel=0.1)
    assert ok is True
    assert report["passed"] is True


def test_compare_summary_failures() -> None:
    baseline = [
        {"scenario": "s1", "file_tag": "s1", "mic_mean_err": 1.0, "mic_p_el_pos": 10.0, "err_ok": True},
    ]
    current = [
        {"scenario": "s1", "file_tag": "s1", "mic_mean_err": 1.3, "mic_p_el_pos": 12.0, "err_ok": False},
    ]
    ok, report = compare_summaries(baseline, current, max_err_rel=0.1, max_power_rel=0.1)
    assert ok is False
    reasons = {f["reason"] for f in report["failures"]}
    assert "err_ok_false" in reasons
    assert "err_increase" in reasons
    assert "power_increase" in reasons


def test_compare_summary_cli(tmp_path: Path) -> None:
    baseline = [
        {"scenario": "s1", "file_tag": "s1", "mic_mean_err": 1.0, "mic_p_el_pos": 10.0, "err_ok": True},
    ]
    current = [
        {"scenario": "s1", "file_tag": "s1", "mic_mean_err": 1.2, "mic_p_el_pos": 10.0, "err_ok": True},
    ]
    baseline_path = tmp_path / "baseline.json"
    current_path = tmp_path / "current.json"
    report_path = tmp_path / "report.json"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    current_path.write_text(json.dumps(current), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mic_ai.tools.compare_summary",
            "--baseline",
            str(baseline_path),
            "--current",
            str(current_path),
            "--max-err-rel",
            "0.1",
            "--report",
            str(report_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["passed"] is False
