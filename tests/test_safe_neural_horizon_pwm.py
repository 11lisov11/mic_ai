from __future__ import annotations

import json
import math

from config.env import create_default_env
from control.safe_neural_horizon_pwm import NeuralHorizonConfig, SafeNeuralHorizonPwmController
from models.induction_motor_alpha_beta import AlphaBetaInductionMotorModel, AlphaBetaMotorParams, AlphaBetaMotorState
from models.two_level_inverter import (
    TwoLevelInverterParams,
    alpha_beta_voltage,
    phase_voltages,
    switch_events,
    vector_bits,
    vector_id_from_bits,
)
from safety.ai_pwm_gateway import (
    AIPwmRequest,
    AIPwmSafetyGateway,
    FaultFlag,
    GatewayLimits,
    has_direct_leg_transition,
    has_shoot_through,
    transition_waveform,
)
from tools.run_safe_neural_horizon_pwm_study import pareto_front, run_fault_injection_matrix, run_matrix, run_study
from tools.build_safe_neural_horizon_pwm_report import build_report
from tools.package_safe_neural_horizon_pwm_release import package_release
from tools.check_safe_neural_horizon_pwm_release import analyze_release
from tools.build_safe_neural_horizon_pwm_figures import build_figures


def _motor_params() -> AlphaBetaMotorParams:
    return AlphaBetaMotorParams.from_motor_params(create_default_env().motor)


def _inverter_params() -> TwoLevelInverterParams:
    return TwoLevelInverterParams(Vdc=300.0, f_pwm=10_000.0, dead_time_s=1e-6, min_pulse_s=2e-6)


def _safe_request(vector_id: int = 3) -> AIPwmRequest:
    return AIPwmRequest(
        vector_id=vector_id,
        dwell_s=100e-6,
        confidence=0.9,
        predicted_i_abs=0.5,
        measured_i_abs=0.4,
        vdc=300.0,
        tj_c=40.0,
        predicted_risk=0.1,
    )


def test_alpha_beta_motor_step_is_finite() -> None:
    params = _motor_params()
    model = AlphaBetaInductionMotorModel(params)
    step = model.step(10.0, -5.0, load_torque_nm=0.0, dt=1e-4)
    assert math.isfinite(step.state.psi_s_alpha)
    assert math.isfinite(step.currents.i_s_alpha)
    assert math.isfinite(step.torque_nm)


def test_two_level_inverter_vector_mapping_and_voltage() -> None:
    assert vector_bits(0b101) == (1, 0, 1)
    assert vector_id_from_bits((1, 0, 1)) == 0b101
    va, vb, vc = phase_voltages(0b100, 300.0)
    assert va > 0.0
    assert vb < 0.0
    assert vc < 0.0
    assert abs(va + vb + vc) < 1e-9
    alpha, beta = alpha_beta_voltage(0b100, _inverter_params())
    assert alpha > 0.0
    assert math.isfinite(beta)
    assert switch_events(0b000, 0b111) == 3


def test_gateway_transition_waveforms_never_shoot_through() -> None:
    for prev in range(8):
        for nxt in range(8):
            wave = transition_waveform(prev, nxt, dead_time_ticks=3)
            assert not has_shoot_through(wave)
            assert not has_direct_leg_transition(wave)


def test_gateway_transition_detector_flags_missing_deadtime_path() -> None:
    unsafe_path = transition_waveform(0b100, 0b011, dead_time_ticks=0)
    safe_path = transition_waveform(0b100, 0b011, dead_time_ticks=2)
    assert not has_shoot_through(unsafe_path)
    assert has_direct_leg_transition(unsafe_path)
    assert not has_direct_leg_transition(safe_path)


def test_gateway_accepts_safe_vector_and_blocks_invalid_with_latch() -> None:
    gateway = AIPwmSafetyGateway(GatewayLimits())
    decision = gateway.evaluate(_safe_request(2))
    assert decision.accepted is True
    assert decision.pwm_enabled is True
    assert decision.vector_id == 2

    bad = gateway.evaluate(_safe_request(99))
    assert bad.accepted is False
    assert bad.pwm_enabled is False
    assert bad.fault_latched is True
    assert FaultFlag.INVALID_VECTOR_FAULT in bad.fault_flags


def test_gateway_latches_deadtime_misconfiguration() -> None:
    gateway = AIPwmSafetyGateway(GatewayLimits(dead_time_s=0.0))
    decision = gateway.evaluate(_safe_request(2))
    assert decision.accepted is False
    assert decision.pwm_enabled is False
    assert decision.fault_latched is True
    assert FaultFlag.DEADTIME_FAULT in decision.fault_flags


def test_gateway_soft_fault_falls_back_without_latching() -> None:
    gateway = AIPwmSafetyGateway(GatewayLimits(i_soft_a=1.0, i_trip_a=4.0))
    decision = gateway.evaluate(_safe_request(4))
    assert decision.accepted is True
    soft = gateway.evaluate(
        AIPwmRequest(
            vector_id=5,
            dwell_s=100e-6,
            confidence=0.9,
            predicted_i_abs=1.1,
            measured_i_abs=0.5,
            vdc=300.0,
            tj_c=40.0,
            predicted_risk=0.1,
        )
    )
    assert soft.accepted is False
    assert soft.pwm_enabled is True
    assert soft.vector_id == 4
    assert soft.fault_latched is False
    assert FaultFlag.CURRENT_SOFT_FAULT in soft.fault_flags


def test_controller_step_uses_gateway_and_returns_safe_decision() -> None:
    motor = _motor_params()
    inverter = _inverter_params()
    gateway = AIPwmSafetyGateway(
        GatewayLimits(
            t_pwm_s=inverter.t_pwm_s,
            min_pulse_s=inverter.min_pulse_s,
            i_soft_a=20.0,
            i_trip_a=30.0,
            vdc_min_v=50.0,
            vdc_max_v=500.0,
        )
    )
    controller = SafeNeuralHorizonPwmController(
        motor,
        inverter,
        gateway,
        NeuralHorizonConfig(horizon=2, dt_s=inverter.t_pwm_s, max_branching=4),
    )
    result = controller.step(omega_ref=50.0, load_torque_nm=0.1, measured_i_abs=0.0, vdc=inverter.Vdc)
    assert 0 <= result.vector_id <= 7
    assert result.decision.gates.shoot_through is False
    assert result.confidence > 0.0
    assert math.isfinite(result.metrics["cost"])


def test_controller_h4_sequence_selection_is_bounded() -> None:
    motor = _motor_params()
    inverter = _inverter_params()
    gateway = AIPwmSafetyGateway(GatewayLimits(i_soft_a=20.0, i_trip_a=30.0, vdc_max_v=500.0))
    controller = SafeNeuralHorizonPwmController(
        motor,
        inverter,
        gateway,
        NeuralHorizonConfig(horizon=4, dt_s=inverter.t_pwm_s, max_branching=3),
    )
    sequence, metrics = controller.select_sequence(
        omega_ref=25.0,
        load_torque_nm=0.0,
        feedback_requested=False,
    )
    assert len(sequence) == 4
    assert all(0 <= vector_id <= 7 for vector_id in sequence)
    assert math.isfinite(metrics["cost"])


def test_controller_reports_applied_losses_after_gateway_disable() -> None:
    motor = _motor_params()
    inverter = _inverter_params()
    gateway = AIPwmSafetyGateway(GatewayLimits(i_trip_a=0.25, vdc_max_v=500.0))
    controller = SafeNeuralHorizonPwmController(
        motor,
        inverter,
        gateway,
        NeuralHorizonConfig(horizon=2, dt_s=inverter.t_pwm_s, max_branching=4),
    )
    measured_state = AlphaBetaMotorState(psi_s_alpha=0.2, psi_r_alpha=0.1, omega_m=10.0)
    result = controller.step(
        omega_ref=50.0,
        load_torque_nm=0.1,
        measured_state=measured_state,
        measured_i_abs=1.0,
        vdc=inverter.Vdc,
    )
    assert result.decision.pwm_enabled is False
    assert FaultFlag.OC_FAULT in result.decision.fault_flags
    assert result.metrics["loss_w"] == 0.0
    assert result.metrics["switch_events"] == 0.0
    assert result.metrics["planned_loss_w"] >= result.metrics["loss_w"]


def test_gateway_fault_injection_matrix() -> None:
    cases = [
        (_safe_request(9), FaultFlag.INVALID_VECTOR_FAULT, True),
        (
            AIPwmRequest(1, 1e-7, 0.9, 0.5, 0.4, 300.0, 40.0, 0.1),
            FaultFlag.MIN_PULSE_FAULT,
            False,
        ),
        (
            AIPwmRequest(1, 100e-6, 0.1, 0.5, 0.4, 300.0, 40.0, 0.1),
            FaultFlag.AI_CONFIDENCE_FAULT,
            False,
        ),
        (
            AIPwmRequest(1, 100e-6, 0.9, 0.5, 5.0, 300.0, 40.0, 0.1),
            FaultFlag.OC_FAULT,
            True,
        ),
        (
            AIPwmRequest(1, 100e-6, 0.9, 0.5, 0.4, 20.0, 40.0, 0.1),
            FaultFlag.UNDERVOLTAGE_FAULT,
            True,
        ),
        (
            AIPwmRequest(1, 100e-6, 0.9, 0.5, 0.4, 300.0, 130.0, 0.1),
            FaultFlag.OVERTEMP_FAULT,
            True,
        ),
        (
            AIPwmRequest(1, 100e-6, 0.9, 0.5, 0.4, 300.0, 40.0, 0.1, watchdog_ok=False),
            FaultFlag.WATCHDOG_FAULT,
            True,
        ),
    ]
    for request, expected, should_latch in cases:
        gateway = AIPwmSafetyGateway(GatewayLimits(i_trip_a=4.0, vdc_min_v=40.0, vdc_max_v=500.0))
        decision = gateway.evaluate(request)
        assert expected in decision.fault_flags
        assert decision.fault_latched is should_latch


def test_safe_neural_horizon_pwm_study_quick_smoke() -> None:
    payload = run_study(mc=2, steps=40, seed=11, quick=True)
    assert payload["hardware_claim"] is False
    controllers = payload["controllers"]
    assert "safe_neural_horizon_pwm_h2" in controllers
    assert "pareto_front" in payload
    for metrics in controllers.values():
        assert metrics["safety_violations"]["worst"] == 0.0


def test_safe_neural_horizon_pwm_matrix_smoke() -> None:
    payload = run_matrix(mc=1, steps=20, seed=5, quick=True, scenarios=["start_no_load"], include_ablation=True)
    assert payload["hardware_claim"] is False
    assert payload["fault_injection"]["all_gateway_cases_no_shoot_through"] is True
    assert payload["fault_injection"]["raw_shoot_through_detector_triggered"] is True
    scenario = payload["matrix"]["start_no_load"]
    assert "foc_svm_key_proxy" in scenario
    assert "safe_neural_horizon_pwm_h2" in scenario
    assert scenario["safe_neural_horizon_pwm_h2"]["safety_violations"]["worst"] == 0.0
    assert payload["ablation"]["pareto_front"]


def test_gateway_fault_injection_matrix_summary() -> None:
    payload = run_fault_injection_matrix()
    assert payload["all_gateway_cases_no_shoot_through"] is True
    assert payload["raw_shoot_through_detector_triggered"] is True
    assert payload["cases"]["invalid_vector"]["fault_latched"] is True
    assert payload["cases"]["low_confidence"]["accepted"] is False
    assert payload["cases"]["raw_shoot_through_request_emulation"]["blocked_by_interface"] is True
    assert payload["cases"]["no_deadtime_transition_emulation"]["direct_leg_transition_without_deadtime"] is True
    assert payload["cases"]["no_deadtime_transition_emulation"]["safe_deadtime_path_valid"] is True
    assert payload["cases"]["no_deadtime_transition_emulation"]["blocked_by_gateway_deadtime_path"] is True


def test_pareto_front_keeps_nondominated_controller() -> None:
    controllers = {
        "bad": {
            "mean_abs_speed_error": {"mean": 2.0},
            "mean_current_abs": {"mean": 2.0},
            "torque_ripple_proxy": {"mean": 2.0},
            "switch_events": {"mean": 2.0},
            "feedback_usage_ratio": {"mean": 2.0},
            "fallback_count": {"mean": 2.0},
        },
        "good": {
            "mean_abs_speed_error": {"mean": 1.0},
            "mean_current_abs": {"mean": 1.0},
            "torque_ripple_proxy": {"mean": 1.0},
            "switch_events": {"mean": 1.0},
            "feedback_usage_ratio": {"mean": 1.0},
            "fallback_count": {"mean": 1.0},
        },
    }
    assert pareto_front(controllers) == ["good"]


def test_build_safe_neural_horizon_pwm_report_from_matrix() -> None:
    payload = run_matrix(mc=1, steps=12, seed=3, quick=True, scenarios=["load_step"], include_ablation=True)
    report = build_report(payload)
    assert "Safe Neural Horizon PWM Host Research Report" in report
    assert "hardware_claim: `False`" in report
    assert "load_step" in report
    assert "Fault Injection" in report


def test_package_safe_neural_horizon_pwm_release(tmp_path) -> None:
    payload = run_matrix(mc=1, steps=8, seed=3, quick=True, scenarios=["load_step"], include_ablation=False)
    input_json = tmp_path / "result.json"
    input_json.write_text(__import__("json").dumps(payload), encoding="utf-8")
    out_dir = tmp_path / "release"
    manifest = package_release(input_json=input_json, out_dir=out_dir, tag="test_tag")
    assert manifest["hardware_claim"] is False
    assert (out_dir / "safe_neural_horizon_pwm_report.md").exists()
    assert (out_dir / "safe_neural_horizon_pwm_article_draft.md").exists()
    assert (out_dir / "WHAT_IS_NOT_DONE.md").exists()
    assert (out_dir / "HOST_ACCEPTANCE_SUMMARY.json").exists()
    assert (out_dir / "figures" / "safe_neural_horizon_pwm_summary.csv").exists()
    assert (out_dir / "figures" / "fig_speed_error_vs_current.svg").exists()
    assert (out_dir / "HOST_RELEASE_MANIFEST.json").exists()


def test_check_safe_neural_horizon_pwm_release_and_figures(tmp_path) -> None:
    payload = run_matrix(mc=1, steps=8, seed=4, quick=True, scenarios=["load_step"], include_ablation=False)
    input_json = tmp_path / "result.json"
    input_json.write_text(__import__("json").dumps(payload), encoding="utf-8")
    check = analyze_release(input_json)
    assert check["host_release_ready"] is False
    assert "missing scenarios" in "\n".join(check["failures"])
    files = build_figures(input_json, tmp_path / "figures")
    assert len(files) == 4
    assert all(path.exists() for path in files)


def test_release_checker_requires_packaged_artifacts_in_manifest(tmp_path) -> None:
    payload = run_matrix(mc=1, steps=8, seed=6, quick=True, scenarios=["load_step"], include_ablation=False)
    input_json = tmp_path / "result.json"
    input_json.write_text(json.dumps(payload), encoding="utf-8")
    out_dir = tmp_path / "release"
    package_release(input_json=input_json, out_dir=out_dir, tag="test_tag")

    manifest_path = out_dir / "HOST_RELEASE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"] = [
        item for item in manifest["files"] if item["path"] != "safe_neural_horizon_pwm_report.md"
    ]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    check = analyze_release(out_dir)
    assert check["checks"]["required_release_files_present"] is False
    assert "manifest missing required release files" in "\n".join(check["failures"])


def test_release_checker_rejects_unsafe_manifest_paths(tmp_path) -> None:
    payload = run_matrix(mc=1, steps=8, seed=7, quick=True, scenarios=["load_step"], include_ablation=False)
    input_json = tmp_path / "result.json"
    input_json.write_text(json.dumps(payload), encoding="utf-8")
    out_dir = tmp_path / "release"
    package_release(input_json=input_json, out_dir=out_dir, tag="test_tag")

    manifest_path = out_dir / "HOST_RELEASE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"].append({"path": "../evil.txt", "bytes": 0, "sha256": ""})
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    check = analyze_release(out_dir)
    assert check["checks"]["manifest_paths_safe"] is False
    assert "unsafe manifest path" in "\n".join(check["failures"])
