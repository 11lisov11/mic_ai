from types import SimpleNamespace

from tools.uno_q_ai_bridge import (
    _action_to_float,
    _apply_gates,
    _clamp_id_ref,
    _parse_host_port,
    _status_fault,
    _validate_runtime_args,
)


def test_parse_host_port_rejects_invalid_endpoints() -> None:
    assert _parse_host_port("127.0.0.1:9000") == ("127.0.0.1", 9000)
    for endpoint in ("127.0.0.1", ":9000", "127.0.0.1:0", "127.0.0.1:70000", "127.0.0.1:x"):
        try:
            _parse_host_port(endpoint)
        except ValueError:
            pass
        else:  # pragma: no cover
            raise AssertionError(f"{endpoint!r} was accepted")


def test_action_to_float_clips_and_rejects_non_finite() -> None:
    assert _action_to_float([2.0]) == 1.0
    assert _action_to_float([-2.0]) == -1.0
    for action in (float("nan"), [float("nan")]):
        try:
            _action_to_float(action)
        except ValueError as exc:
            assert "AI action must be finite" in str(exc)
        else:  # pragma: no cover
            raise AssertionError("NaN action was accepted")


def test_clamp_id_ref_rejects_non_finite() -> None:
    assert _clamp_id_ref(2.0, 1.0, 1.5) == 1.5
    try:
        _clamp_id_ref(float("nan"), 1.0, 1.5)
    except ValueError as exc:
        assert "id_ref must be finite" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("NaN id_ref was accepted")


def test_validate_runtime_args_rejects_unsafe_ranges() -> None:
    base = {
        "fault_mask": 0,
        "id_min": 1.0,
        "id_max": 1.5,
        "delta_id_max": 0.1,
        "speed_tol": 0.5,
        "speed_tol_rel": 0.1,
        "load_torque": None,
    }
    _validate_runtime_args(SimpleNamespace(**base))

    bad_cases = [
        ("fault_mask", -1, "--fault-mask must be in uint16 range"),
        ("fault_mask", 0x10000, "--fault-mask must be in uint16 range"),
        ("id_min", float("nan"), "--id-min must be finite"),
        ("id_min", -0.1, "--id-min must be non-negative"),
        ("id_max", float("nan"), "--id-max must be finite"),
        ("id_min", 2.0, "--id-min must be <= --id-max"),
        ("delta_id_max", float("nan"), "--delta-id-max must be finite"),
        ("delta_id_max", -0.1, "--delta-id-max must be non-negative"),
        ("speed_tol", float("nan"), "--speed-tol must be finite"),
        ("speed_tol", -0.1, "--speed-tol must be non-negative"),
        ("speed_tol_rel", float("nan"), "--speed-tol-rel must be finite"),
        ("speed_tol_rel", -0.1, "--speed-tol-rel must be non-negative"),
        ("load_torque", float("nan"), "--load-torque must be finite"),
    ]
    for field, value, message in bad_cases:
        data = dict(base)
        data[field] = value
        try:
            _validate_runtime_args(SimpleNamespace(**data))
        except ValueError as exc:
            assert message in str(exc)
        else:  # pragma: no cover
            raise AssertionError(f"{field}={value!r} was accepted")


def test_status_fault_mask() -> None:
    assert _status_fault(0, 0) is False
    assert _status_fault(1, 0) is True
    assert _status_fault(0x1, 0x2) is False
    assert _status_fault(0x2, 0x2) is True


def test_apply_gates_guard_disable() -> None:
    enable_ai, id_cmd, gated = _apply_gates(
        speed_err=2.0,
        tol=1.0,
        status=0,
        id_ref_base=1.0,
        id_ref_cmd=1.5,
        disable_on_guard=True,
        disable_on_fault=False,
        fault_mask=0,
    )
    assert enable_ai == 0
    assert id_cmd == 1.0
    assert gated is True


def test_apply_gates_fault_disable() -> None:
    enable_ai, id_cmd, gated = _apply_gates(
        speed_err=0.1,
        tol=1.0,
        status=0x1,
        id_ref_base=1.0,
        id_ref_cmd=1.5,
        disable_on_guard=False,
        disable_on_fault=True,
        fault_mask=0,
    )
    assert enable_ai == 0
    assert id_cmd == 1.0
    assert gated is True


def test_apply_gates_pass_through() -> None:
    enable_ai, id_cmd, gated = _apply_gates(
        speed_err=0.1,
        tol=1.0,
        status=0,
        id_ref_base=1.0,
        id_ref_cmd=1.5,
        disable_on_guard=False,
        disable_on_fault=False,
        fault_mask=0,
    )
    assert enable_ai == 1
    assert id_cmd == 1.5
    assert gated is False
