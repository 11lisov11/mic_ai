from mic_ai.tools.guardrails_check import evaluate_summary


def test_guardrails_pass() -> None:
    rows = [
        {"err_ok": True, "power_saving_pct": 5.0, "eta_gain_pct": 1.0},
        {"err_ok": True, "power_saving_pct": 2.0, "eta_gain_pct": 0.5},
    ]
    ok, report = evaluate_summary(rows, min_power_saving_pct=1.0, max_err_failures=0)
    assert ok
    assert report["failures"] == 0


def test_guardrails_fail() -> None:
    rows = [
        {"err_ok": False, "power_saving_pct": -1.0, "eta_gain_pct": -0.2},
    ]
    ok, report = evaluate_summary(rows, min_power_saving_pct=0.0, max_err_failures=0)
    assert not ok
    assert report["failures"] > 0
