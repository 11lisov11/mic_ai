from mic_ai.tools import run_benchmark


class DummyArgs:
    def __init__(self):
        self.python = "python"
        self.env_config = "config/env_demo_true_motor1.py"
        self.out_dir = "outputs/bench_run"
        self.window_frac = 0.25
        self.error_tol_rel = 0.05
        self.error_tol_abs = 0.0
        self.dt = None
        self.t_end = None
        self.load_torque = None
        self.scenarios = "speed_step"
        self.include_v3 = True
        self.use_total_power = True
        self.mic_id_ref_low = 1.0
        self.mic_id_ref_high = 1.4
        self.mic_id_ref_speed_tol_rel = 0.05
        self.mic_id_ref_omega_min = 0.1
        self.ai_checkpoint = None
        self.ai_id_relative = False
        self.delta_id_max = 0.1
        self.min_power_saving_pct = 0.0
        self.min_eta_gain_pct = None
        self.max_err_failures = 0
        self.no_require_err_ok = False
        self.guardrails_report = None
        self.baseline_summary = "benchmarks/baseline_summary_physical_motor1.json"
        self.compare_max_err_rel = 0.1
        self.compare_max_err_abs = 0.0
        self.compare_max_power_rel = 0.1
        self.compare_max_power_abs = 0.0
        self.compare_no_require_err_ok = False
        self.compare_report = None


def test_build_scenario_compare_cmd_rule() -> None:
    args = DummyArgs()
    cmd = run_benchmark._build_scenario_compare_cmd(args)
    assert "mic_ai.tools.scenario_compare" in cmd
    assert "--mic-id-ref-low" in cmd
    assert "--include-v3" in cmd


def test_build_guardrails_cmd() -> None:
    args = DummyArgs()
    cmd = run_benchmark._build_guardrails_cmd(args, summary_path="summary.json")  # type: ignore[arg-type]
    assert "mic_ai.tools.guardrails_check" in cmd


def test_build_compare_cmd() -> None:
    args = DummyArgs()
    cmd = run_benchmark._build_compare_cmd(args, summary_path="summary.json")  # type: ignore[arg-type]
    assert "mic_ai.tools.compare_summary" in cmd
    assert "--baseline" in cmd
