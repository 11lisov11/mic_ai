from __future__ import annotations

import json
import math
from pathlib import Path

from config.env import create_default_env
from control.deadbeat_current_baseline import DeadbeatCurrentBaselineConfig, DeadbeatCurrentBaselineController
from control.dtc_baseline import DtcHysteresisBaselineConfig, DtcHysteresisBaselineController
from control.dtc_svm_baseline import DtcSvmBaselineConfig, DtcSvmBaselineController
from control.fcs_mpc_baseline import FcsMpcOneStepBaselineConfig, FcsMpcOneStepBaselineController
from control.foc_svm_key_baseline import FocSvmKeyBaselineConfig, FocSvmKeyBaselineController
from control.protected_ai_pwm_h1_baseline import ProtectedAiPwmH1BaselineController, protected_h1_config
from control.safe_neural_horizon_pwm import NeuralHorizonConfig, SafeNeuralHorizonPwmController
from control.sensorless_adaptive_foc_baseline import (
    SensorlessAdaptiveFocBaselineConfig,
    SensorlessAdaptiveFocBaselineController,
)
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
from tools.build_safe_neural_horizon_pwm_baseline_stress import build_baseline_stress
from tools.package_safe_neural_horizon_pwm_release import package_release
from tools.build_safe_neural_horizon_pwm_trace_evidence import build_trace_evidence
from tools.build_safe_neural_horizon_pwm_twin_evidence import build_twin_evidence
from tools.check_safe_neural_horizon_pwm_baselines import analyze_baselines
from tools.check_safe_neural_horizon_pwm_release import MC_SMOKE_REQUIRED_CONTROLLERS, analyze_release
from tools.check_safe_neural_horizon_pwm_novelty import COMPARISON_CONTROLLERS, analyze_novelty
from tools.check_safe_neural_horizon_pwm_theory import analyze_theory
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


def _mc_smoke_payload(trials: int, *, hardware_claim: bool = False) -> dict:
    return {
        "study": "Safe Neural Horizon PWM",
        "status": "host_simulation_only",
        "hardware_claim": hardware_claim,
        "mc_trials": trials,
        "steps_per_trial": 120,
        "controllers": {
            name: {
                "safety_violations": {"worst": 0.0},
                "fault_latch_count": {"worst": 0.0},
                "failure_count": 0,
            }
            for name in sorted(MC_SMOKE_REQUIRED_CONTROLLERS)
        },
    }


def _baseline_stress_payload() -> dict:
    return {
        "status": "safe_neural_horizon_pwm_baseline_stress_evidence",
        "hardware_claim": False,
        "mc_trials": 3,
        "steps_per_trial": 60,
        "scenarios": ["load_step", "overload", "dc_sag", "sensor_delay", "shock_load"],
        "controllers": {
            name: {
                "scenario_count": 5,
                "safety_violations_worst": 0.0,
                "unexpected_failure_count": 0,
                "finite_metrics": True,
                "stress_ready": True,
            }
            for name in sorted(COMPARISON_CONTROLLERS)
        },
        "baseline_stress_ready": True,
        "publication_tuning_claim": False,
    }


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


def test_gateway_latches_invalid_limit_configuration() -> None:
    cases = [
        GatewayLimits(i_soft_a=4.0, i_trip_a=4.0),
        GatewayLimits(i_soft_a=5.0, i_trip_a=4.0),
        GatewayLimits(i_soft_a=float("nan"), i_trip_a=4.0),
        GatewayLimits(vdc_min_v=500.0, vdc_max_v=100.0),
        GatewayLimits(confidence_min=1.5),
        GatewayLimits(max_switch_events_per_window=-1),
    ]
    for limits in cases:
        gateway = AIPwmSafetyGateway(limits)
        decision = gateway.evaluate(_safe_request(2))
        assert decision.accepted is False
        assert decision.pwm_enabled is False
        assert decision.fault_latched is True
        assert FaultFlag.LIMIT_CONFIG_FAULT in decision.fault_flags or FaultFlag.NONFINITE_FAULT in decision.fault_flags


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


def test_foc_svm_key_baseline_selects_legal_active_vector() -> None:
    params = _motor_params()
    inverter = _inverter_params()
    limits = GatewayLimits(
        t_pwm_s=inverter.t_pwm_s,
        dead_time_s=inverter.dead_time_s,
        min_pulse_s=inverter.min_pulse_s,
        i_soft_a=20.0,
        i_trip_a=25.0,
        vdc_min_v=40.0,
        vdc_max_v=500.0,
        confidence_min=0.3,
        risk_max=2.0,
    )
    controller = FocSvmKeyBaselineController(
        params,
        inverter,
        AIPwmSafetyGateway(limits),
        FocSvmKeyBaselineConfig(dt_s=inverter.t_pwm_s),
    )
    result = controller.step(
        omega_ref=40.0,
        load_torque_nm=0.1,
        measured_state=AlphaBetaMotorState(),
        measured_i_abs=0.0,
        vdc=inverter.Vdc,
    )
    assert 0 <= result.vector_id <= 7
    assert result.vector_id not in {0, 7}
    assert result.decision.gates.shoot_through is False
    assert result.metrics["id_ref"] >= 0.0
    assert abs(result.metrics["iq_ref"]) <= params.i_limit


def test_fcs_mpc_one_step_baseline_selects_legal_vector() -> None:
    params = _motor_params()
    inverter = _inverter_params()
    limits = GatewayLimits(
        t_pwm_s=inverter.t_pwm_s,
        dead_time_s=inverter.dead_time_s,
        min_pulse_s=inverter.min_pulse_s,
        i_soft_a=20.0,
        i_trip_a=25.0,
        vdc_min_v=40.0,
        vdc_max_v=500.0,
        confidence_min=0.3,
        risk_max=2.0,
    )
    controller = FcsMpcOneStepBaselineController(
        params,
        inverter,
        AIPwmSafetyGateway(limits),
        FcsMpcOneStepBaselineConfig(dt_s=inverter.t_pwm_s),
    )
    result = controller.step(
        omega_ref=40.0,
        load_torque_nm=0.1,
        measured_state=AlphaBetaMotorState(),
        measured_i_abs=0.0,
        vdc=inverter.Vdc,
    )
    assert 0 <= result.vector_id <= 7
    assert result.decision.gates.shoot_through is False
    assert math.isfinite(result.metrics["cost"])
    assert result.metrics["candidate_current"] >= 0.0


def test_dtc_hysteresis_baseline_selects_legal_vector() -> None:
    params = _motor_params()
    inverter = _inverter_params()
    limits = GatewayLimits(
        t_pwm_s=inverter.t_pwm_s,
        dead_time_s=inverter.dead_time_s,
        min_pulse_s=inverter.min_pulse_s,
        i_soft_a=20.0,
        i_trip_a=25.0,
        vdc_min_v=40.0,
        vdc_max_v=500.0,
        confidence_min=0.3,
        risk_max=2.0,
    )
    controller = DtcHysteresisBaselineController(
        params,
        inverter,
        AIPwmSafetyGateway(limits),
        DtcHysteresisBaselineConfig(dt_s=inverter.t_pwm_s),
    )
    result = controller.step(
        omega_ref=40.0,
        load_torque_nm=0.1,
        measured_state=AlphaBetaMotorState(),
        measured_i_abs=0.0,
        vdc=inverter.Vdc,
    )
    assert 0 <= result.vector_id <= 7
    assert result.decision.gates.shoot_through is False
    assert math.isfinite(result.metrics["cost"])
    assert result.metrics["flux_hysteresis_cmd"] >= 0.0


def test_dtc_svm_baseline_selects_legal_vector() -> None:
    params = _motor_params()
    inverter = _inverter_params()
    limits = GatewayLimits(
        t_pwm_s=inverter.t_pwm_s,
        dead_time_s=inverter.dead_time_s,
        min_pulse_s=inverter.min_pulse_s,
        i_soft_a=20.0,
        i_trip_a=25.0,
        vdc_min_v=40.0,
        vdc_max_v=500.0,
        confidence_min=0.3,
        risk_max=2.0,
    )
    controller = DtcSvmBaselineController(
        params,
        inverter,
        AIPwmSafetyGateway(limits),
        DtcSvmBaselineConfig(dt_s=inverter.t_pwm_s),
    )
    result = controller.step(
        omega_ref=40.0,
        load_torque_nm=0.1,
        measured_state=AlphaBetaMotorState(),
        measured_i_abs=0.0,
        vdc=inverter.Vdc,
    )
    assert 0 <= result.vector_id <= 7
    assert result.decision.gates.shoot_through is False
    assert math.isfinite(result.metrics["v_alpha_ref"])
    assert math.isfinite(result.metrics["v_beta_ref"])
    assert math.isfinite(result.metrics["torque_error"])
    assert math.isfinite(result.metrics["flux_error"])


def test_deadbeat_current_baseline_selects_legal_vector() -> None:
    params = _motor_params()
    inverter = _inverter_params()
    limits = GatewayLimits(
        t_pwm_s=inverter.t_pwm_s,
        dead_time_s=inverter.dead_time_s,
        min_pulse_s=inverter.min_pulse_s,
        i_soft_a=20.0,
        i_trip_a=25.0,
        vdc_min_v=40.0,
        vdc_max_v=500.0,
        confidence_min=0.3,
        risk_max=2.0,
    )
    controller = DeadbeatCurrentBaselineController(
        params,
        inverter,
        AIPwmSafetyGateway(limits),
        DeadbeatCurrentBaselineConfig(dt_s=inverter.t_pwm_s),
    )
    result = controller.step(
        omega_ref=40.0,
        load_torque_nm=0.1,
        measured_state=AlphaBetaMotorState(),
        measured_i_abs=0.0,
        vdc=inverter.Vdc,
    )
    assert 0 <= result.vector_id <= 7
    assert result.decision.gates.shoot_through is False
    assert math.isfinite(result.metrics["v_alpha_deadbeat"])
    assert math.isfinite(result.metrics["v_beta_deadbeat"])
    assert math.isfinite(result.metrics["candidate_current_error"])


def test_sensorless_adaptive_foc_baseline_uses_observer_not_measured_speed() -> None:
    params = _motor_params()
    inverter = _inverter_params()
    limits = GatewayLimits(
        t_pwm_s=inverter.t_pwm_s,
        dead_time_s=inverter.dead_time_s,
        min_pulse_s=inverter.min_pulse_s,
        i_soft_a=20.0,
        i_trip_a=25.0,
        vdc_min_v=40.0,
        vdc_max_v=500.0,
        confidence_min=0.3,
        risk_max=2.0,
    )
    controller = SensorlessAdaptiveFocBaselineController(
        params,
        inverter,
        AIPwmSafetyGateway(limits),
        SensorlessAdaptiveFocBaselineConfig(dt_s=inverter.t_pwm_s),
    )
    measured = AlphaBetaMotorState(
        psi_s_alpha=0.08,
        psi_s_beta=0.02,
        psi_r_alpha=0.07,
        psi_r_beta=0.01,
        omega_m=123.0,
    )
    result = controller.step(
        omega_ref=40.0,
        load_torque_nm=0.1,
        measured_state=measured,
        measured_i_abs=0.0,
        vdc=inverter.Vdc,
    )
    assert 0 <= result.vector_id <= 7
    assert result.decision.gates.shoot_through is False
    assert math.isfinite(result.metrics["omega_hat"])
    assert math.isfinite(result.metrics["rs_scale"])
    assert abs(result.metrics["omega_hat"] - measured.omega_m) > 1.0


def test_protected_ai_pwm_h1_baseline_keeps_prior_one_step_policy() -> None:
    params = _motor_params()
    inverter = _inverter_params()
    limits = GatewayLimits(
        t_pwm_s=inverter.t_pwm_s,
        dead_time_s=inverter.dead_time_s,
        min_pulse_s=inverter.min_pulse_s,
        i_soft_a=20.0,
        i_trip_a=25.0,
        vdc_min_v=40.0,
        vdc_max_v=500.0,
        confidence_min=0.25,
        risk_max=1.4,
    )
    cfg = protected_h1_config(dt_s=inverter.t_pwm_s, feedback_period=5)
    controller = ProtectedAiPwmH1BaselineController(params, inverter, AIPwmSafetyGateway(limits), cfg)
    result = controller.step(
        omega_ref=40.0,
        load_torque_nm=0.1,
        measured_state=AlphaBetaMotorState(),
        measured_i_abs=0.0,
        vdc=inverter.Vdc,
    )
    assert controller.cfg.horizon == 1
    assert 0 <= result.vector_id <= 7
    assert result.decision.gates.shoot_through is False
    assert result.metrics["prior_protected_h1_baseline"] == 1.0
    assert result.metrics["horizon"] == 1.0


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
    gateway = AIPwmSafetyGateway(GatewayLimits(i_soft_a=0.2, i_trip_a=0.25, vdc_max_v=500.0))
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
        gateway = AIPwmSafetyGateway(GatewayLimits(i_soft_a=3.0, i_trip_a=4.0, vdc_min_v=40.0, vdc_max_v=500.0))
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
    assert "protected_ai_pwm_h1_baseline" in scenario
    assert "fcs_mpc_one_step_baseline" in scenario
    assert "foc_svm_key_baseline" in scenario
    assert "dtc_hysteresis_baseline" in scenario
    assert "dtc_svm_baseline" in scenario
    assert "deadbeat_current_baseline" in scenario
    assert "sensorless_adaptive_foc_baseline" in scenario
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
    mc100_json = tmp_path / "mc100.json"
    mc100_json.write_text(json.dumps(_mc_smoke_payload(100)), encoding="utf-8")
    mc500_json = tmp_path / "mc500.json"
    mc500_json.write_text(json.dumps(_mc_smoke_payload(500)), encoding="utf-8")
    baseline_stress_json = tmp_path / "baseline_stress.json"
    baseline_stress_json.write_text(json.dumps(_baseline_stress_payload()), encoding="utf-8")
    trace_dir = tmp_path / "trace_evidence_src"
    build_trace_evidence(
        out_dir=trace_dir,
        steps=32,
        controllers=["safe_neural_horizon_pwm_h2", "protected_ai_pwm_h1_baseline", "foc_svm_key_baseline"],
    )
    twin_dir = tmp_path / "twin_evidence_src"
    build_twin_evidence(out_dir=twin_dir, train_episodes=3, val_episodes=2, steps=55, hidden_features=12)
    out_dir = tmp_path / "release"
    manifest = package_release(
        input_json=input_json,
        out_dir=out_dir,
        tag="test_tag",
        mc100_json=mc100_json,
        mc500_json=mc500_json,
        baseline_stress_json=baseline_stress_json,
        trace_dir=trace_dir,
        twin_dir=twin_dir,
    )
    assert manifest["hardware_claim"] is False
    assert (out_dir / "safe_neural_horizon_pwm_report.md").exists()
    assert (out_dir / "safe_neural_horizon_pwm_article_draft.md").exists()
    assert (out_dir / "safe_neural_horizon_pwm_baseline_stress_evidence.json").exists()
    assert (out_dir / "safe_neural_horizon_pwm_baseline_strength_audit.json").exists()
    assert (out_dir / "safe_neural_horizon_pwm_novelty_audit.json").exists()
    assert (out_dir / "safe_neural_horizon_pwm_theory_completion_audit.json").exists()
    assert (out_dir / "safe_neural_horizon_pwm_mc100_smoke.json").exists()
    assert (out_dir / "safe_neural_horizon_pwm_mc500_publication_smoke.json").exists()
    assert (out_dir / "WHAT_IS_NOT_DONE.md").exists()
    assert (out_dir / "HOST_ACCEPTANCE_SUMMARY.json").exists()
    assert (out_dir / "figures" / "safe_neural_horizon_pwm_summary.csv").exists()
    assert (out_dir / "figures" / "fig_speed_error_vs_current.svg").exists()
    assert (out_dir / "trace_evidence" / "trace_summary.json").exists()
    assert (out_dir / "trace_evidence" / "trace_summary.csv").exists()
    assert (out_dir / "trace_evidence" / "figures" / "fig_trace_fft_thd.svg").exists()
    assert (out_dir / "twin_evidence" / "twin_training_summary.json").exists()
    assert (out_dir / "twin_evidence" / "residual_twin_weights.json").exists()
    assert (out_dir / "HOST_RELEASE_MANIFEST.json").exists()
    disk_manifest = json.loads((out_dir / "HOST_RELEASE_MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest == disk_manifest


def test_build_safe_neural_horizon_pwm_trace_evidence(tmp_path) -> None:
    payload = build_trace_evidence(
        out_dir=tmp_path / "trace",
        steps=40,
        controllers=["safe_neural_horizon_pwm_h2", "protected_ai_pwm_h1_baseline", "fcs_mpc_one_step_baseline"],
    )
    assert payload["hardware_claim"] is False
    assert payload["trace_evidence_ready"] is True
    assert (tmp_path / "trace" / "trace_summary.csv").exists()
    assert (tmp_path / "trace" / "figures" / "fig_trace_speed.svg").exists()
    assert (tmp_path / "trace" / "figures" / "fig_trace_fft_thd.svg").exists()
    for row in payload["summary"]:
        assert math.isfinite(float(row["current_thd_like"]))
        assert math.isfinite(float(row["torque_thd_like"]))
        assert float(row["safety_violations"]) == 0.0


def test_build_safe_neural_horizon_pwm_twin_evidence(tmp_path) -> None:
    payload = build_twin_evidence(
        out_dir=tmp_path / "twin",
        train_episodes=4,
        val_episodes=2,
        steps=55,
        hidden_features=12,
    )
    assert payload["hardware_claim"] is False
    assert payload["trained_domain_randomized_twin_ready"] is True
    assert payload["identified_domain_randomized_twin_ready"] is True
    assert (tmp_path / "twin" / "twin_training_summary.json").exists()
    assert (tmp_path / "twin" / "residual_twin_weights.json").exists()
    for horizon in ("1", "5", "10", "50"):
        row = payload["theta_conditioned_multi_step"][horizon]
        assert float(row["theta_conditioned_twin_rmse"]) >= 0.0
        assert float(row["improvement_pct"]) > 0.0


def test_build_safe_neural_horizon_pwm_baseline_stress() -> None:
    payload = build_baseline_stress(mc=1, steps=12, seed=13, scenarios=["load_step"])
    assert payload["hardware_claim"] is False
    assert payload["baseline_stress_ready"] is True
    assert payload["publication_tuning_claim"] is False
    assert set(payload["controllers"]) == set(COMPARISON_CONTROLLERS)
    for row in payload["controllers"].values():
        assert row["safety_violations_worst"] == 0.0
        assert row["unexpected_failure_count"] == 0
        assert row["stress_ready"] is True


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


def test_safe_neural_horizon_pwm_tracked_release_supports_host_novelty_claim() -> None:
    release_dir = Path("paper/safe_neural_horizon_pwm_2026/20260522_host_release")
    audit = analyze_novelty(release_dir)
    assert audit["host_novelty_claim_supported"] is True
    assert audit["checks"]["deadtime_path_detector_triggered"] is True
    assert "MCU/HIL/bench readiness" in audit["not_allowed_claims"]


def test_safe_neural_horizon_pwm_tracked_release_has_baseline_strength_audit() -> None:
    release_dir = Path("paper/safe_neural_horizon_pwm_2026/20260522_host_release")
    audit = analyze_baselines(release_dir)
    assert audit["host_baseline_scaffold_ready"] is True
    assert audit["publication_strong_baselines_ready"] is False
    assert audit["stress_evidence_ready"] is True
    assert audit["baseline_count"] == 7
    for row in audit["baselines"].values():
        assert row["source_marker_present"] is True
        assert row["matrix_coverage_ready"] is True
        assert row["safety_ready"] is True
        assert row["pareto_participation_count"] > 0
        assert row["stress_evidence_ready"] is True
        assert row["baseline_scaffold_ready"] is True
        assert row["publication_tuned_ready"] is False


def test_safe_neural_horizon_pwm_tracked_release_supports_host_theory_scaffold() -> None:
    release_dir = Path("paper/safe_neural_horizon_pwm_2026/20260522_host_release")
    audit = analyze_theory(release_dir)
    assert audit["host_theory_scaffold_ready"] is True
    assert audit["publication_theory_complete"] is False
    assert audit["checks"]["first_mc100_smoke"] is True
    assert audit["checks"]["foc_svm_key_baseline_ready"] is True
    assert audit["checks"]["fcs_mpc_one_step_baseline_ready"] is True
    assert audit["checks"]["dtc_hysteresis_baseline_ready"] is True
    assert audit["checks"]["dtc_svm_baseline_ready"] is True
    assert audit["checks"]["deadbeat_current_baseline_ready"] is True
    assert audit["checks"]["sensorless_adaptive_foc_baseline_ready"] is True
    assert audit["checks"]["protected_ai_pwm_h1_baseline_ready"] is True
    assert audit["checks"]["named_baseline_comparison_matrix"] is True
    assert "proxy_comparison_matrix" not in audit["checks"]
    assert audit["checks"]["trace_fft_thd_evidence_ready"] is True
    assert audit["checks"]["publication_plots_fft_thd_ready"] is True
    assert audit["checks"]["domain_randomized_twin_evidence_ready"] is True
    assert audit["checks"]["baseline_strength_audit_ready"] is True
    assert audit["checks"]["baseline_stress_evidence_ready"] is True
    assert audit["checks"]["trained_domain_randomized_twin_ready"] is True
    assert audit["checks"]["publication_mc500_ready"] is True
    assert audit["checks"]["strong_baselines_ready"] is False


def test_release_checker_requires_packaged_artifacts_in_manifest(tmp_path) -> None:
    payload = run_matrix(mc=1, steps=8, seed=6, quick=True, scenarios=["load_step"], include_ablation=False)
    input_json = tmp_path / "result.json"
    input_json.write_text(json.dumps(payload), encoding="utf-8")
    mc100_json = tmp_path / "mc100.json"
    mc100_json.write_text(json.dumps(_mc_smoke_payload(100)), encoding="utf-8")
    mc500_json = tmp_path / "mc500.json"
    mc500_json.write_text(json.dumps(_mc_smoke_payload(500)), encoding="utf-8")
    baseline_stress_json = tmp_path / "baseline_stress.json"
    baseline_stress_json.write_text(json.dumps(_baseline_stress_payload()), encoding="utf-8")
    out_dir = tmp_path / "release"
    package_release(
        input_json=input_json,
        out_dir=out_dir,
        tag="test_tag",
        mc100_json=mc100_json,
        mc500_json=mc500_json,
        baseline_stress_json=baseline_stress_json,
    )

    manifest_path = out_dir / "HOST_RELEASE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"] = [
        item for item in manifest["files"] if item["path"] != "safe_neural_horizon_pwm_report.md"
    ]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    check = analyze_release(out_dir)
    assert check["checks"]["required_release_files_present"] is False
    assert "manifest missing required release files" in "\n".join(check["failures"])


def test_release_checker_requires_acceptance_summary(tmp_path) -> None:
    payload = run_matrix(mc=1, steps=8, seed=8, quick=True, scenarios=["load_step"], include_ablation=False)
    input_json = tmp_path / "result.json"
    input_json.write_text(json.dumps(payload), encoding="utf-8")
    mc100_json = tmp_path / "mc100.json"
    mc100_json.write_text(json.dumps(_mc_smoke_payload(100)), encoding="utf-8")
    mc500_json = tmp_path / "mc500.json"
    mc500_json.write_text(json.dumps(_mc_smoke_payload(500)), encoding="utf-8")
    baseline_stress_json = tmp_path / "baseline_stress.json"
    baseline_stress_json.write_text(json.dumps(_baseline_stress_payload()), encoding="utf-8")
    out_dir = tmp_path / "release"
    package_release(
        input_json=input_json,
        out_dir=out_dir,
        tag="test_tag",
        mc100_json=mc100_json,
        mc500_json=mc500_json,
        baseline_stress_json=baseline_stress_json,
    )

    (out_dir / "HOST_ACCEPTANCE_SUMMARY.json").unlink()

    check = analyze_release(out_dir)
    assert check["checks"]["acceptance_summary_present"] is False
    assert "missing HOST_ACCEPTANCE_SUMMARY.json" in "\n".join(check["failures"])


def test_release_and_theory_reject_fake_mc500_evidence(tmp_path) -> None:
    payload = run_matrix(mc=1, steps=8, seed=9, quick=True, scenarios=["load_step"], include_ablation=False)
    input_json = tmp_path / "result.json"
    input_json.write_text(json.dumps(payload), encoding="utf-8")
    mc100_json = tmp_path / "mc100.json"
    mc100_json.write_text(json.dumps(_mc_smoke_payload(100)), encoding="utf-8")
    bad_mc500 = _mc_smoke_payload(500, hardware_claim=True)
    bad_mc500["controllers"].pop("safe_neural_horizon_pwm_h2")
    mc500_json = tmp_path / "mc500.json"
    mc500_json.write_text(json.dumps(bad_mc500), encoding="utf-8")
    baseline_stress_json = tmp_path / "baseline_stress.json"
    baseline_stress_json.write_text(json.dumps(_baseline_stress_payload()), encoding="utf-8")
    out_dir = tmp_path / "release"
    package_release(
        input_json=input_json,
        out_dir=out_dir,
        tag="test_tag",
        mc100_json=mc100_json,
        mc500_json=mc500_json,
        baseline_stress_json=baseline_stress_json,
    )

    release_check = analyze_release(out_dir)
    assert release_check["checks"]["mc500_publication_content_ready"] is False
    failures = "\n".join(release_check["failures"])
    assert "MC500: hardware_claim must be false" in failures
    assert "MC500: missing controllers" in failures

    theory = analyze_theory(out_dir)
    assert theory["checks"]["publication_mc500_ready"] is False
    mc500_criterion = [item for item in theory["criteria"] if item["key"] == "publication_mc500_evidence"][0]
    assert "MC500: hardware_claim must be false" in "\n".join(mc500_criterion["missing"])


def test_release_checker_rejects_unsafe_manifest_paths(tmp_path) -> None:
    payload = run_matrix(mc=1, steps=8, seed=7, quick=True, scenarios=["load_step"], include_ablation=False)
    input_json = tmp_path / "result.json"
    input_json.write_text(json.dumps(payload), encoding="utf-8")
    mc100_json = tmp_path / "mc100.json"
    mc100_json.write_text(json.dumps(_mc_smoke_payload(100)), encoding="utf-8")
    mc500_json = tmp_path / "mc500.json"
    mc500_json.write_text(json.dumps(_mc_smoke_payload(500)), encoding="utf-8")
    baseline_stress_json = tmp_path / "baseline_stress.json"
    baseline_stress_json.write_text(json.dumps(_baseline_stress_payload()), encoding="utf-8")
    out_dir = tmp_path / "release"
    package_release(
        input_json=input_json,
        out_dir=out_dir,
        tag="test_tag",
        mc100_json=mc100_json,
        mc500_json=mc500_json,
        baseline_stress_json=baseline_stress_json,
    )

    manifest_path = out_dir / "HOST_RELEASE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"].append({"path": "../evil.txt", "bytes": 0, "sha256": ""})
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    check = analyze_release(out_dir)
    assert check["checks"]["manifest_paths_safe"] is False
    assert "unsafe manifest path" in "\n".join(check["failures"])
