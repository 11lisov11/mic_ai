from __future__ import annotations

from mic_ai.ai.id_ref_supervisor import AiIdRefSupervisor, AiIdRefSupervisorConfig


def test_supervisor_gate_blocks_at_low_speed() -> None:
    cfg = AiIdRefSupervisorConfig(enabled=True, omega_min_pu=0.2, dither_amp=0.1)
    sup = AiIdRefSupervisor(cfg, omega_nominal=100.0)
    action, gate = sup.adjust_action(ai_action=0.3, omega_ref=10.0, omega=10.0)
    assert gate is False
    assert abs(action - 0.3) < 1e-9


def test_supervisor_bias_moves_down_when_positive_gradient() -> None:
    cfg = AiIdRefSupervisorConfig(
        enabled=True,
        speed_tol_rel=0.05,
        omega_min_pu=0.1,
        update_steps=2,
        dither_amp=0.1,
        bias_step=0.05,
        bias_max=0.2,
        objective="p_in",
    )
    sup = AiIdRefSupervisor(cfg, omega_nominal=100.0)

    # phase +1 window -> higher objective
    for _ in range(2):
        _, gate = sup.adjust_action(ai_action=0.0, omega_ref=80.0, omega=80.0)
        sup.update(p_in_pos=5.0, p_shaft_pos=1.0, gate_open=gate)

    # phase -1 window -> lower objective
    for _ in range(2):
        _, gate = sup.adjust_action(ai_action=0.0, omega_ref=80.0, omega=80.0)
        sup.update(p_in_pos=3.0, p_shaft_pos=1.0, gate_open=gate)

    assert sup.bias < 0.0
