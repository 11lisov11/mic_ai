from __future__ import annotations

import socket
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

from tools.air56_unoq_bridge import (
    Air56IdRefParams,
    Air56PolicyBundle,
    Air56BridgeRuntimeState,
    BaseTransport,
    SerialTransport,
    UdpTransport,
    _action_to_id_ref,
    _build_obs,
    _build_bundle,
    _build_startup_self_check_report,
    _clamp_rate,
    _clip_action_scalar,
    _compute_gate_scale,
    _estimate_load_nm_from_iq,
    _finite_float,
    _infer_action_scalar,
    _load_id_ref_params,
    _load_supervisor,
    _parse_host_port,
    _process_telemetry_step,
    _resolve_existing_file,
    _send_fallback_command,
    _should_switch_secondary,
    _status_fault,
    _validate_id_ref_window,
    _validate_id_ref_params,
    _validate_runtime_args,
)
from tools.uno_q_protocol import CURRENT_SCALE, Telemetry
from tools.uno_q_protocol import Command


def _free_udp_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
    finally:
        sock.close()


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


def test_base_transport_contract_methods() -> None:
    transport = BaseTransport()
    try:
        transport.recv(1)
    except NotImplementedError:
        pass
    else:  # pragma: no cover
        raise AssertionError("BaseTransport.recv did not raise")

    try:
        transport.send(b"x")
    except NotImplementedError:
        pass
    else:  # pragma: no cover
        raise AssertionError("BaseTransport.send did not raise")

    transport.close()


def test_udp_transport_roundtrip_and_size_filter() -> None:
    listen_port = _free_udp_port()
    send_port = _free_udp_port()
    rx_peer = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    tx_peer = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    transport = UdpTransport(f"127.0.0.1:{listen_port}", f"127.0.0.1:{send_port}")
    try:
        rx_peer.bind(("127.0.0.1", send_port))
        tx_peer.sendto(b"x", ("127.0.0.1", listen_port))
        tx_peer.sendto(b"abcd", ("127.0.0.1", listen_port))
        assert transport.recv(4) == b"abcd"

        transport.send(b"cmd")
        payload, addr = rx_peer.recvfrom(16)
        assert payload == b"cmd"
        assert addr[0] == "127.0.0.1"
    finally:
        transport.close()
        rx_peer.close()
        tx_peer.close()


def test_serial_transport_uses_pyserial_contract(monkeypatch) -> None:
    created: list[object] = []

    class FakeSerial:
        def __init__(self, *, port: str, baudrate: int, timeout: float) -> None:
            self.port = port
            self.baudrate = baudrate
            self.timeout = timeout
            self.read_chunks = [b"a", b"", b"bc", b"d"]
            self.writes: list[bytes] = []
            self.flushed = False
            self.closed = False
            created.append(self)

        def read(self, size: int) -> bytes:
            if not self.read_chunks:
                return b""
            chunk = self.read_chunks.pop(0)
            return chunk[:size]

        def write(self, payload: bytes) -> None:
            self.writes.append(payload)

        def flush(self) -> None:
            self.flushed = True

        def close(self) -> None:
            self.closed = True

    monkeypatch.setitem(sys.modules, "serial", SimpleNamespace(Serial=FakeSerial))
    transport = SerialTransport("COM99", 115200, 0.25)
    assert transport.recv(4) == b"abcd"
    transport.send(b"ok")
    transport.close()

    fake = created[0]
    assert fake.port == "COM99"
    assert fake.baudrate == 115200
    assert fake.timeout == 0.25
    assert fake.writes == [b"ok"]
    assert fake.flushed
    assert fake.closed


def test_serial_transport_recv_times_out_on_incomplete_frame(monkeypatch) -> None:
    created: list[object] = []

    class EmptySerial:
        def __init__(self, *, port: str, baudrate: int, timeout: float) -> None:
            self.port = port
            self.baudrate = baudrate
            self.timeout = timeout
            self.closed = False
            created.append(self)

        def read(self, size: int) -> bytes:
            return b""

        def write(self, payload: bytes) -> None:
            raise AssertionError("timeout test must not write")

        def flush(self) -> None:
            raise AssertionError("timeout test must not flush")

        def close(self) -> None:
            self.closed = True

    monkeypatch.setitem(sys.modules, "serial", SimpleNamespace(Serial=EmptySerial))
    transport = SerialTransport("COM99", 115200, 0.001)
    try:
        try:
            transport.recv(4)
        except TimeoutError as exc:
            assert "serial frame timeout" in str(exc)
        else:  # pragma: no cover
            raise AssertionError("incomplete serial frame did not time out")
    finally:
        transport.close()
    assert created[0].closed


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

    exp_gate = Air56IdRefParams(**{**_params().__dict__, "gate_exponent": 2.0, "gate_min_scale": 0.0})
    scale, tol = _compute_gate_scale(omega_ref=100.0, omega_meas=95.0, params=exp_gate)
    assert tol == 10.0
    assert abs(scale - 0.25) < 1e-12


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


def test_action_to_id_ref_blocks_positive_relative_delta_when_disabled() -> None:
    params = Air56IdRefParams(**{**_params().__dict__, "allow_positive_delta": False})
    cmd, gate_scale, _gate_tol = _action_to_id_ref(
        action=1.0,
        prev_id_ref=1.35,
        omega_ref=100.0,
        omega_meas=100.0,
        params=params,
    )
    assert gate_scale == 1.0
    assert cmd == 1.35


def test_clip_action_scalar_rejects_nan_instead_of_maxing_command() -> None:
    assert _clip_action_scalar(-2.0) == -1.0
    assert _clip_action_scalar(2.0) == 1.0
    try:
        _clip_action_scalar(float("nan"))
    except ValueError as exc:
        assert "AI action must be finite" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("NaN AI action was accepted")


def test_action_to_id_ref_rejects_non_finite_action() -> None:
    try:
        _action_to_id_ref(
            action=float("nan"),
            prev_id_ref=1.35,
            omega_ref=100.0,
            omega_meas=100.0,
            params=_params(),
        )
    except ValueError as exc:
        assert "AI action must be finite" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("NaN AI action reached id_ref mapping")


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


def test_should_switch_secondary_rejects_small_absolute_delta_and_speed_gate() -> None:
    assert not _should_switch_secondary(
        load_est_nm=0.43,
        prev_load_est_nm=0.40,
        speed_err=8.0,
        omega_ref=100.0,
        threshold_nm=0.05,
        positive_only=False,
        speed_err_threshold_rel=0.05,
        speed_err_threshold_abs=0.0,
    )
    assert not _should_switch_secondary(
        load_est_nm=0.60,
        prev_load_est_nm=0.40,
        speed_err=2.0,
        omega_ref=100.0,
        threshold_nm=0.05,
        positive_only=True,
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


def test_validate_id_ref_params_rejects_bad_config_values() -> None:
    _validate_id_ref_params(_params())

    bad_cases = [
        ("id_ref_alpha", float("nan"), "id_ref_alpha must be finite"),
        ("id_ref_alpha", 1.1, "id_ref_alpha must be in [0, 1]"),
        ("delta_id_max", float("nan"), "delta_id_max must be finite"),
        ("delta_id_max", -0.1, "delta_id_max must be non-negative"),
        ("gate_speed_tol_abs", float("nan"), "gate_speed_tol_abs must be finite"),
        ("gate_speed_tol_abs", -0.1, "gate_speed_tol_abs must be non-negative"),
        ("gate_speed_tol_rel", float("nan"), "gate_speed_tol_rel must be finite"),
        ("gate_speed_tol_rel", -0.1, "gate_speed_tol_rel must be non-negative"),
        ("gate_min_scale", float("nan"), "gate_min_scale must be finite"),
        ("gate_min_scale", 1.1, "gate_min_scale must be in [0, 1]"),
        ("gate_exponent", float("nan"), "gate_exponent must be finite"),
        ("gate_exponent", -0.1, "gate_exponent must be non-negative"),
    ]
    for field, value, message in bad_cases:
        data = {**_params().__dict__, field: value}
        try:
            _validate_id_ref_params(Air56IdRefParams(**data))
        except ValueError as exc:
            assert message in str(exc)
        else:  # pragma: no cover
            raise AssertionError(f"{field}={value!r} was accepted")


def test_action_to_id_ref_rejects_invalid_params_before_command_mapping() -> None:
    data = {**_params().__dict__, "gate_min_scale": float("nan")}
    try:
        _action_to_id_ref(
            action=0.0,
            prev_id_ref=1.35,
            omega_ref=100.0,
            omega_meas=100.0,
            params=Air56IdRefParams(**data),
        )
    except ValueError as exc:
        assert "gate_min_scale must be finite" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("invalid id_ref params reached command mapping")


def test_load_supervisor_returns_none_when_disabled() -> None:
    assert _load_supervisor(SimpleNamespace(ai_eval_supervisor_enabled=False), "ai_eval_supervisor_enabled", "ai_eval_", omega_nominal=100.0) is None


def test_load_supervisor_builds_enabled_supervisor() -> None:
    env_cfg = SimpleNamespace(
        ai_eval_supervisor_enabled=True,
        ai_eval_sup_speed_tol_rel=0.04,
        ai_eval_sup_speed_tol_abs=1.0,
        ai_eval_sup_omega_min=0.2,
        ai_eval_sup_update=4,
        ai_eval_sup_dither=0.03,
        ai_eval_sup_step=0.02,
        ai_eval_sup_bias_max=0.1,
        ai_eval_sup_objective="p_in",
        ai_eval_sup_shaft_eps=2.0,
        ai_eval_sup_reset_decay=0.9,
        ai_eval_sup_objective_clip=4.0,
        ai_eval_sup_idle_enable=True,
        ai_eval_sup_idle_omega_min=0.05,
        ai_eval_sup_idle_action=-0.8,
        ai_eval_sup_idle_blend=0.7,
        ai_eval_sup_idle_exit_boost=3,
        ai_eval_sup_idle_exit_action=0.6,
        ai_eval_sup_idle_bias_decay=0.8,
    )
    supervisor = _load_supervisor(env_cfg, "ai_eval_supervisor_enabled", "ai_eval_sup_", omega_nominal=100.0)
    assert supervisor is not None


def test_infer_action_scalar_uses_agent_network_and_clips() -> None:
    class FakeAgent:
        def _to_tensor(self, obs: dict[str, float]) -> torch.Tensor:
            assert obs == {"omega_norm": 1.0}
            return torch.tensor([1.0], dtype=torch.float32)

        class Net:
            def __call__(self, state_t: torch.Tensor):
                assert tuple(state_t.shape) == (1, 1)
                return torch.tensor([[2.0]], dtype=torch.float32), None, None

        net = Net()

    assert _infer_action_scalar(FakeAgent(), {"omega_norm": 1.0}) == 1.0  # type: ignore[arg-type]


def test_build_bundle_wires_agent_params_and_supervisor(monkeypatch, tmp_path: Path) -> None:
    checkpoint = tmp_path / "actor.pth"
    checkpoint.write_bytes(b"not-used")
    fake_agent = object()
    fake_supervisor = object()

    monkeypatch.setattr("tools.air56_unoq_bridge._load_agent", lambda path: fake_agent)
    monkeypatch.setattr("tools.air56_unoq_bridge._load_id_ref_params", lambda env, prefix, id_ref_min, id_ref_max: _params())
    monkeypatch.setattr("tools.air56_unoq_bridge._load_supervisor", lambda env, enabled_attr, prefix, omega_nominal: fake_supervisor)

    bundle = _build_bundle(
        name="primary",
        checkpoint=checkpoint,
        env_cfg=SimpleNamespace(),
        prefix="ai_eval_",
        enabled_attr="ai_eval_supervisor_enabled",
        supervisor_prefix="ai_eval_sup_",
        id_ref_min=1.1,
        id_ref_max=1.7,
        omega_nominal=100.0,
    )
    assert isinstance(bundle, Air56PolicyBundle)
    assert bundle.name == "primary"
    assert bundle.agent is fake_agent
    assert bundle.supervisor is fake_supervisor


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


def test_process_telemetry_step_guard_disables_ai(monkeypatch) -> None:
    monkeypatch.setattr("tools.air56_unoq_bridge._infer_action_scalar", lambda _agent, _obs: -1.0)
    state = Air56BridgeRuntimeState(last_id_ref=1.35)

    step = _process_telemetry_step(
        telem=_telem(omega_meas=50.0, omega_ref=100.0),
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
        fault_mask=0,
        disable_on_fault=False,
        disable_on_guard=True,
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


def test_resolve_existing_file_rejects_missing_file(tmp_path: Path) -> None:
    try:
        _resolve_existing_file("missing.txt", "--missing", root=tmp_path)
    except FileNotFoundError as exc:
        assert "--missing does not exist" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("missing file accepted")


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
        ("serial_timeout", float("nan"), "--serial-timeout must be finite"),
        ("serial_timeout", 0.0, "--serial-timeout must be positive"),
        ("fault_mask", -1, "--fault-mask must be in uint16 range"),
        ("fault_mask", 0x10000, "--fault-mask must be in uint16 range"),
        ("id_min", float("nan"), "--id-min must be finite"),
        ("id_min", -0.1, "--id-min must be non-negative"),
        ("id_max", float("nan"), "--id-max must be finite"),
        ("id_min", 2.0, "--id-min must be <= --id-max"),
        ("cmd_rate_limit_a_per_s", float("nan"), "--cmd-rate-limit-a-per-s must be finite"),
        ("cmd_rate_limit_a_per_s", -0.1, "--cmd-rate-limit-a-per-s must be non-negative"),
        ("load_est_gain", float("nan"), "--load-est-gain must be finite"),
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

    non_finite_cases = [
        {"id_ref_base": float("nan"), "id_min": 1.1, "id_max": 1.7, "message": "id_ref_base must be finite"},
        {"id_ref_base": 1.35, "id_min": float("nan"), "id_max": 1.7, "message": "id_min must be finite"},
        {"id_ref_base": 1.35, "id_min": 1.1, "id_max": float("nan"), "message": "id_max must be finite"},
    ]
    for case in non_finite_cases:
        try:
            _validate_id_ref_window(
                id_ref_base=case["id_ref_base"],
                id_min=case["id_min"],
                id_max=case["id_max"],
            )
        except ValueError as exc:
            assert case["message"] in str(exc)
        else:  # pragma: no cover
            raise AssertionError(f"non-finite id_ref window accepted: {case}")

    for lo, hi, message in [(-0.1, 1.7, "--id-min must be non-negative"), (1.8, 1.7, "--id-min must be <= --id-max")]:
        try:
            _validate_id_ref_window(id_ref_base=1.35, id_min=lo, id_max=hi)
        except ValueError as exc:
            assert message in str(exc)
        else:  # pragma: no cover
            raise AssertionError(f"invalid id_ref window [{lo}, {hi}] accepted")

    for lo, hi in [(1.4, 1.7), (1.1, 1.3)]:
        try:
            _validate_id_ref_window(id_ref_base=1.35, id_min=lo, id_max=hi)
        except ValueError as exc:
            assert "FOC base id_ref must be inside" in str(exc)
        else:  # pragma: no cover
            raise AssertionError(f"id_ref window [{lo}, {hi}] accepted without base id_ref")


def test_finite_float_rejects_non_finite_values() -> None:
    assert _finite_float(1.25, "field") == 1.25
    for value in (float("nan"), float("inf"), float("-inf")):
        try:
            _finite_float(value, "field")
        except ValueError as exc:
            assert "field must be finite" in str(exc)
        else:  # pragma: no cover
            raise AssertionError(f"non-finite value accepted: {value!r}")


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


def test_send_fallback_command_ignores_missing_transport() -> None:
    _send_fallback_command(None, t_ms=10, id_ref_base=1.35, crc=True)


def test_send_fallback_command_respects_dry_run() -> None:
    class FakeTransport:
        def __init__(self) -> None:
            self.payloads: list[bytes] = []

        def send(self, payload: bytes) -> None:
            self.payloads.append(payload)

    transport = FakeTransport()
    _send_fallback_command(transport, t_ms=10, id_ref_base=1.35, crc=True, dry_run=True)  # type: ignore[arg-type]
    assert transport.payloads == []


def test_startup_self_check_report_records_deploy_contract(tmp_path: Path) -> None:
    config = tmp_path / "env.py"
    primary = tmp_path / "primary.pth"
    secondary = tmp_path / "secondary.pth"
    for path in (config, primary, secondary):
        path.write_bytes(b"x")

    report = _build_startup_self_check_report(
        config_path=config,
        primary_checkpoint=primary,
        secondary_checkpoint=secondary,
        transport_name="serial",
        crc=True,
        disable_on_fault=True,
        disable_on_guard=False,
        id_min=1.1,
        id_max=1.7,
        fallback_id_ref=1.35,
    )

    assert report["config_exists"] is True
    assert report["primary_checkpoint_exists"] is True
    assert report["secondary_checkpoint_required"] is True
    assert report["secondary_checkpoint_exists"] is True
    assert report["transport_ready"] is True
    assert report["crc_enabled"] is True
    assert report["disable_on_fault"] is True
    assert report["fallback_inside_limits"] is True
