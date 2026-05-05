from __future__ import annotations

from types import SimpleNamespace

from tools.air56_unoq_bridge import (
    Air56IdRefParams,
    Air56BridgeRuntimeState,
    _action_to_id_ref,
    _build_obs,
    _clamp_rate,
    _compute_gate_scale,
    _estimate_load_nm_from_iq,
    _load_id_ref_params,
    _load_supervisor,
    _parse_host_port,
    _process_telemetry_step,
    _resolve_existing_file,
    _send_fallback_command,
    _should_switch_secondary,
    _status_fault,
    _validate_id_ref_window,
    _validate_runtime_args,
)
from tools.uno_q_protocol import CURRENT_SCALE, Telemetry
from tools.uno_q_protocol import Command


class FakeSupervisor:
    def __init__(self) -> None:
        self.adjust_calls: list[tuple[float, float, float]] = []
        self.update_calls: list[dict[str, float | bool]] = []

    def adjust_action(self, action: float, *, omega_ref: float, omega: float) -> tuple[float, bool]:
        self.adjust_calls.append((action, omega_ref, omega))
        return action, True

    def update(self, *, p_in_pos: float, p_shaft_pos: float, gate_open: bool) -> None:
        self.update_calls.append(
            {
                "p_in_pos": p_in_pos,
                "p_shaft_pos": p_shaft_pos,
                "gate_open": gate_open,
            }
        )


def _params() -> Air56IdRefParams:
    return Air56IdRefParams(
        id_ref_base=1.35,
        id_ref_min=1.10,
        id_ref_max=1.70,
        id_ref_alpha=0.2,
        delta_id_max=0.1,
        gate_speed_tol_abs=0.0,
        gate_speed_tol_rel=0.1,
        gate_min_scale=0.1,
        gate_exponent=1.0,
        ai_id_relative=True,
        allow_positive_delta=True,
    )


def _bundle(name: str, params: Air56IdRefParams | None = None, supervisor=None):
    return SimpleNamespace(name=name, agent=object(), params=params or _params(), supervisor=supervisor)


def _telem(
    *,
    t_ms: int = 10,
    omega_meas: float = 100.0,
    omega_ref: float = 100.0,
    i_d: float = 1.35,
    i_q: float = 0.6,
    status: int = 0,
    p_in: float = 40.0,
) -> Telemetry:
    return Telemetry(
        t_ms=t_ms,
        omega_meas=omega_meas,
        omega_ref=omega_ref,
        i_d=i_d,
        i_q=i_q,
        v_dc=24.0,
        i_rms=1.5,
        p_in=p_in,
        status=status,
    )


def test_estimate_load_nm_from_iq_uses_absolute_current() -> None:
    assert _estimate_load_nm_from_iq(-0.8, 2.5) == 2.0


def test_parse_host_port() -> None:
    assert _parse_host_port("127.0.0.1:9000") == ("127.0.0.1", 9000)


def test_parse_host_port_rejects_invalid_endpoints() -> None:
    bad_cases = [
        "127.0.0.1",
        ":9000",
        "127.0.0.1:0",
        "127.0.0.1:70000",
        "127.0.0.1:notaport",
    ]
    for endpoint in bad_cases:
        try:
            _parse_host_port(endpoint)
        except ValueError:
            pass
        else:  # pragma: no cover - keeps failures readable without pytest import dependency here
            raise AssertionError(f"{endpoint!r} was accepted")


def test_status_fault_mask_semantics() -> None:
    assert not _status_fault(0, 0)
    assert _status_fault(1, 0)
    assert not _status_fault(0x01, 0x02)
    assert _status_fault(0x02, 0x02)


def test_compute_gate_scale_handles_abs_rel_and_exponent() -> None:
    params = _params()
    scale, tol = _compute_gate_scale(omega_ref=100.0, omega_meas=96.0, params=params)
    assert tol == 10.0
    assert 0.5 < scale < 1.0

    no_gate = _params()
    no_gate = Air56IdRefParams(**{**no_gate.__dict__, "gate_speed_tol_abs": None, "gate_speed_tol_rel": None})
    scale, tol = _compute_gate_scale(omega_ref=100.0, omega_meas=0.0, params=no_gate)
    assert scale == 1.0
    assert tol == 0.0


def test_action_to_id_ref_blocks_demagnetization_on_large_error() -> None:
    cmd, gate_scale, _gate_tol = _action_to_id_ref(
        action=-1.0,
        prev_id_ref=1.35,
        omega_ref=100.0,
        omega_meas=0.0,
        params=_params(),
    )
    assert gate_scale <= 0.1
    assert cmd >= 1.35


def test_action_to_id_ref_absolute_mode_gates_below_base() -> None:
    params = Air56IdRefParams(
        id_ref_base=1.35,
        id_ref_min=1.10,
        id_ref_max=1.70,
        id_ref_alpha=1.0,
        delta_id_max=0.1,
        gate_speed_tol_abs=10.0,
        gate_speed_tol_rel=None,
        gate_min_scale=0.0,
        gate_exponent=1.0,
        ai_id_relative=False,
        allow_positive_delta=True,
    )
    cmd, gate_scale, _gate_tol = _action_to_id_ref(
        action=-1.0,
        prev_id_ref=1.35,
        omega_ref=100.0,
        omega_meas=95.0,
        params=params,
    )
    assert gate_scale == 0.5
    assert 1.10 < cmd < 1.35


def test_clamp_rate_limits_both_directions() -> None:
    assert _clamp_rate(1.0, 2.0, 0.2) == 1.2
    assert _clamp_rate(1.0, 0.0, 0.2) == 0.8
    assert _clamp_rate(1.0, 1.1, 0.2) == 1.1
    assert _clamp_rate(1.0, 2.0, 0.0) == 1.0
    assert _clamp_rate(1.0, 2.0, -0.2) == 1.0


def test_build_obs_computes_normalized_runtime_features() -> None:
    telem = Telemetry(
        t_ms=1,
        omega_meas=100.0,
        omega_ref=120.0,
        i_d=1.2,
        i_q=0.6,
        v_dc=24.0,
        i_rms=1.3,
        p_in=40.0,
        status=0,
    )
    obs = _build_obs(
        telem=telem,
        omega_base=200.0,
        i_base=2.0,
        pole_pairs=2,
        rr=0.5,
        lr_total=1.0,
        load_torque_nm=0.4,
        load_base_nm=0.8,
    )
    assert obs["omega_norm"] == 0.5
    assert obs["omega_ref_norm"] == 0.6
    assert obs["err_norm"] == 0.1
    assert obs["id_norm"] == 0.6
    assert obs["iq_norm"] == 0.3
    assert obs["load_torque_norm"] == 0.5
    assert obs["omega_syn_norm"] > 1.0


def test_should_switch_secondary_requires_real_positive_load_jump() -> None:
    assert _should_switch_secondary(
        load_est_nm=0.60,
        prev_load_est_nm=0.40,
        speed_err=8.0,
        omega_ref=100.0,
        threshold_nm=0.05,
        positive_only=True,
        speed_err_threshold_rel=0.05,
        speed_err_threshold_abs=0.0,
    )
    assert not _should_switch_secondary(
        load_est_nm=0.42,
        prev_load_est_nm=0.40,
        speed_err=8.0,
        omega_ref=100.0,
        threshold_nm=0.05,
        positive_only=True,
        speed_err_threshold_rel=0.05,
        speed_err_threshold_abs=0.0,
    )


def test_should_switch_secondary_can_use_absolute_delta() -> None:
    assert _should_switch_secondary(
        load_est_nm=0.20,
        prev_load_est_nm=0.40,
        speed_err=8.0,
        omega_ref=100.0,
        threshold_nm=0.05,
        positive_only=False,
        speed_err_threshold_rel=0.05,
        speed_err_threshold_abs=0.0,
    )


def test_load_id_ref_params_reads_env_config_attrs() -> None:
    env_cfg = SimpleNamespace(
        foc=SimpleNamespace(id_ref=1.35),
        ai_eval_id_ref_alpha=0.2,
        ai_eval_delta_id_max=0.3,
        ai_eval_id_ref_gate_speed_tol=2.0,
        ai_eval_id_ref_gate_speed_tol_rel=0.1,
        ai_eval_id_ref_gate_min_scale=0.2,
        ai_eval_id_ref_gate_exponent=2.0,
        ai_eval_id_ref_relative=True,
        ai_eval_id_ref_allow_positive_delta=False,
    )
    params = _load_id_ref_params(env_cfg, "ai_eval_", id_ref_min=1.1, id_ref_max=1.7)
    assert params.id_ref_base == 1.35
    assert params.gate_speed_tol_abs == 2.0
    assert params.gate_speed_tol_rel == 0.1
    assert params.ai_id_relative
    assert not params.allow_positive_delta


def test_load_supervisor_returns_none_when_disabled() -> None:
    assert _load_supervisor(SimpleNamespace(ai_eval_supervisor_enabled=False), "ai_eval_supervisor_enabled", "ai_eval_", omega_nominal=100.0) is None


def test_process_telemetry_step_uses_primary_and_rate_limit(monkeypatch) -> None:
    monkeypatch.setattr("tools.air56_unoq_bridge._infer_action_scalar", lambda _agent, _obs: -1.0)
    state = Air56BridgeRuntimeState(last_id_ref=1.35)

    step = _process_telemetry_step(
        telem=_telem(t_ms=10),
        primary=_bundle("primary"),
        secondary=None,
        state=state,
        omega_nominal=200.0,
        i_base=2.0,
        pole_pairs=2,
        rr=0.5,
        lr_total=1.0,
        load_est_gain=1.0,
        load_base_nm=1.0,
        load_delta_threshold=0.05,
        positive_only=True,
        latch_steps=2,
        speed_err_threshold_rel=0.0,
        speed_err_threshold_abs=0.0,
        fault_mask=0,
        disable_on_fault=True,
        disable_on_guard=True,
        cmd_rate_limit_a_per_s=1.0,
        id_min=1.10,
        id_max=1.70,
    )

    assert step.active_name == "primary"
    assert step.command.enable_ai == 1
    assert abs(step.command.id_ref - 1.34) < 1e-12
    assert state.last_id_ref == step.command.id_ref
    assert state.prev_t_ms == 10
    assert state.last_telem_t_ms == 10


def test_process_telemetry_step_switches_to_secondary(monkeypatch) -> None:
    monkeypatch.setattr("tools.air56_unoq_bridge._infer_action_scalar", lambda _agent, _obs: 0.0)
    state = Air56BridgeRuntimeState(last_id_ref=1.35, prev_load_est_nm=0.1)

    step = _process_telemetry_step(
        telem=_telem(t_ms=20, omega_meas=80.0, omega_ref=100.0, i_q=1.0),
        primary=_bundle("primary"),
        secondary=_bundle("secondary"),
        state=state,
        omega_nominal=200.0,
        i_base=2.0,
        pole_pairs=2,
        rr=0.5,
        lr_total=1.0,
        load_est_gain=1.0,
        load_base_nm=1.0,
        load_delta_threshold=0.05,
        positive_only=True,
        latch_steps=2,
        speed_err_threshold_rel=0.05,
        speed_err_threshold_abs=0.0,
        fault_mask=0,
        disable_on_fault=False,
        disable_on_guard=False,
        cmd_rate_limit_a_per_s=0.0,
        id_min=1.10,
        id_max=1.70,
    )

    assert step.active_name == "secondary"
    assert state.secondary_left == 1


def test_process_telemetry_step_fault_disables_ai(monkeypatch) -> None:
    monkeypatch.setattr("tools.air56_unoq_bridge._infer_action_scalar", lambda _agent, _obs: -1.0)
    state = Air56BridgeRuntimeState(last_id_ref=1.35)

    step = _process_telemetry_step(
        telem=_telem(status=0x2),
        primary=_bundle("primary"),
        secondary=None,
        state=state,
        omega_nominal=200.0,
        i_base=2.0,
        pole_pairs=2,
        rr=0.5,
        lr_total=1.0,
        load_est_gain=1.0,
        load_base_nm=1.0,
        load_delta_threshold=0.05,
        positive_only=True,
        latch_steps=0,
        speed_err_threshold_rel=0.0,
        speed_err_threshold_abs=0.0,
        fault_mask=0x2,
        disable_on_fault=True,
        disable_on_guard=False,
        cmd_rate_limit_a_per_s=0.0,
        id_min=1.10,
        id_max=1.70,
    )

    assert step.command.enable_ai == 0
    assert step.command.id_ref == 1.35


def test_process_telemetry_step_supervisor_adjusts_and_updates(monkeypatch) -> None:
    monkeypatch.setattr("tools.air56_unoq_bridge._infer_action_scalar", lambda _agent, _obs: -0.5)
    supervisor = FakeSupervisor()
    state = Air56BridgeRuntimeState(last_id_ref=1.35)

    step = _process_telemetry_step(
        telem=_telem(omega_meas=100.0, omega_ref=100.0, i_q=0.5, p_in=-1.0),
        primary=_bundle("primary", supervisor=supervisor),
        secondary=None,
        state=state,
        omega_nominal=200.0,
        i_base=2.0,
        pole_pairs=2,
        rr=0.5,
        lr_total=1.0,
        load_est_gain=2.0,
        load_base_nm=1.0,
        load_delta_threshold=0.05,
        positive_only=True,
        latch_steps=0,
        speed_err_threshold_rel=0.0,
        speed_err_threshold_abs=0.0,
        fault_mask=0,
        disable_on_fault=False,
        disable_on_guard=False,
        cmd_rate_limit_a_per_s=0.0,
        id_min=1.10,
        id_max=1.70,
    )

    assert step.command.enable_ai == 1
    assert supervisor.adjust_calls == [(-0.5, 100.0, 100.0)]
    assert supervisor.update_calls == [{"p_in_pos": 0.0, "p_shaft_pos": 100.0, "gate_open": True}]


def test_resolve_existing_file_accepts_relative_repo_file() -> None:
    path = _resolve_existing_file("config/env_research_air56_025kw.py", "--config")
    assert path.is_file()
    assert path.name == "env_research_air56_025kw.py"


def test_validate_runtime_args_rejects_unsafe_cli_ranges() -> None:
    base = {
        "baud": 921600,
        "serial_timeout": 0.05,
        "fault_mask": 0,
        "id_min": 1.1,
        "id_max": 1.7,
        "cmd_rate_limit_a_per_s": 12.0,
        "load_est_gain": None,
        "log_every": 50,
    }
    _validate_runtime_args(SimpleNamespace(**base))

    bad_cases = [
        ("baud", 0, "--baud must be positive"),
        ("serial_timeout", 0.0, "--serial-timeout must be positive"),
        ("fault_mask", -1, "--fault-mask must be in uint16 range"),
        ("fault_mask", 0x10000, "--fault-mask must be in uint16 range"),
        ("id_min", 2.0, "--id-min must be <= --id-max"),
        ("cmd_rate_limit_a_per_s", -0.1, "--cmd-rate-limit-a-per-s must be non-negative"),
        ("load_est_gain", -0.1, "--load-est-gain must be non-negative"),
        ("log_every", -1, "--log-every must be non-negative"),
    ]
    for field, value, message in bad_cases:
        data = dict(base)
        data[field] = value
        try:
            _validate_runtime_args(SimpleNamespace(**data))
        except ValueError as exc:
            assert message in str(exc)
        else:  # pragma: no cover - keeps failures readable without pytest import dependency here
            raise AssertionError(f"{field}={value!r} was accepted")


def test_validate_id_ref_window_requires_fallback_inside_launch_limits() -> None:
    _validate_id_ref_window(id_ref_base=1.35, id_min=1.1, id_max=1.7)

    for lo, hi in [(1.4, 1.7), (1.1, 1.3)]:
        try:
            _validate_id_ref_window(id_ref_base=1.35, id_min=lo, id_max=hi)
        except ValueError as exc:
            assert "FOC base id_ref must be inside" in str(exc)
        else:  # pragma: no cover
            raise AssertionError(f"id_ref window [{lo}, {hi}] accepted without base id_ref")


def test_send_fallback_command_disables_ai() -> None:
    class FakeTransport:
        def __init__(self) -> None:
            self.payloads: list[bytes] = []

        def send(self, payload: bytes) -> None:
            self.payloads.append(payload)

    transport = FakeTransport()
    _send_fallback_command(transport, t_ms=10, id_ref_base=1.35, crc=True)  # type: ignore[arg-type]
    assert len(transport.payloads) == 1

    cmd = Command.unpack(transport.payloads[0])
    assert cmd.t_ms == 10
    assert cmd.enable_ai == 0
    assert abs(cmd.id_ref - 1.35) < 1.0 / CURRENT_SCALE


def test_send_fallback_command_respects_dry_run() -> None:
    class FakeTransport:
        def __init__(self) -> None:
            self.payloads: list[bytes] = []

        def send(self, payload: bytes) -> None:
            self.payloads.append(payload)

    transport = FakeTransport()
    _send_fallback_command(transport, t_ms=10, id_ref_base=1.35, crc=True, dry_run=True)  # type: ignore[arg-type]
    assert transport.payloads == []
