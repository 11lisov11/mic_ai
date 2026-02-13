from tools.uno_q_ai_bridge import _apply_gates, _status_fault


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
