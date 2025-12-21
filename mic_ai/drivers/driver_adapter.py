from __future__ import annotations

from dataclasses import dataclass
import inspect
import math
from typing import Callable, Iterable, Optional, Tuple

from models.transformations import abc_to_dq, dq_to_abc


_TWO_PI_OVER_60 = 2.0 * math.pi / 60.0


def _is_finite(value: float) -> bool:
    return math.isfinite(float(value))


def _as_float(value: object, name: str) -> float:
    try:
        return float(value)
    except Exception as exc:
        raise ValueError(f"{name} must be a float-like value, got {value!r}") from exc


def _call_with_duration(
    func: Callable[..., object],
    vd: float,
    vq: float,
    duration_s: float | None,
) -> bool:
    if duration_s is None:
        func(vd, vq)
        return True
    try:
        func(vd, vq, duration_s=duration_s)
        return True
    except TypeError:
        pass
    try:
        func(vd, vq, duration=duration_s)
        return True
    except TypeError:
        pass
    try:
        func(vd, vq, dt=duration_s)
        return True
    except TypeError:
        func(vd, vq)
        return False


@dataclass(frozen=True)
class DriverCapabilities:
    has_lock_rotor: bool
    has_torque_channel: bool


class DriverAdapter:
    """
    Adapter that maps a raw hardware driver to the identification contract.
    """

    def __init__(self, raw_driver: object) -> None:
        self._raw = raw_driver
        self._driver = getattr(raw_driver, "driver", None)

    def _targets(self) -> Iterable[object]:
        if self._driver is not None:
            yield self._driver
        yield self._raw

    def _resolve_method(self, names: Tuple[str, ...]) -> Optional[Callable[..., object]]:
        for target in self._targets():
            for name in names:
                if hasattr(target, name):
                    method = getattr(target, name)
                    if callable(method):
                        return method
        return None

    def _resolve_attr(self, names: Tuple[str, ...]) -> Optional[object]:
        for target in self._targets():
            for name in names:
                if hasattr(target, name):
                    return getattr(target, name)
        return None

    @property
    def dt(self) -> float:
        attr = self._resolve_attr(("dt", "sample_time", "sample_time_s", "control_dt", "period_s"))
        if attr is not None:
            return _as_float(attr, "dt")
        getter = self._resolve_method(("get_dt", "get_sample_time", "get_period_s"))
        if getter is not None:
            return _as_float(getter(), "dt")
        raise NotImplementedError("Driver must provide dt (attribute or get_dt()).")

    @property
    def capabilities(self) -> DriverCapabilities:
        return DriverCapabilities(
            has_lock_rotor=self._resolve_method(("lock_rotor", "set_rotor_locked", "set_lock_rotor")) is not None,
            has_torque_channel=self._resolve_method(
                ("set_torque_command", "set_torque_ref", "set_iq_ref", "set_current_dq")
            )
            is not None,
        )

    @property
    def supports_voltage_command(self) -> bool:
        if self._resolve_method(("set_voltage_dq", "set_voltage_dq_ref", "set_dq_voltage")) is not None:
            return True
        return self._resolve_method(("set_voltage_abc", "set_phase_voltages")) is not None and self._can_read_theta_e()

    def reset(self) -> None:
        reset = self._resolve_method(("reset", "reset_driver", "reset_device"))
        if reset is not None:
            reset()

    def lock_rotor(self, enable: bool) -> None:
        method = self._resolve_method(("lock_rotor", "set_rotor_locked", "set_lock_rotor"))
        if method is None:
            raise NotImplementedError("Driver does not support rotor lock control.")
        method(bool(enable))

    def set_voltage_dq(self, vd: float, vq: float, *, duration_s: float | None = None) -> None:
        vd = _as_float(vd, "vd")
        vq = _as_float(vq, "vq")
        method = self._resolve_method(("set_voltage_dq", "set_voltage_dq_ref", "set_dq_voltage"))
        if method is None:
            abc_method = self._resolve_method(("set_voltage_abc", "set_phase_voltages"))
            theta_e = self._read_theta_e()
            if abc_method is None:
                raise NotImplementedError("Driver does not support dq voltage commands.")
            v_a, v_b, v_c = dq_to_abc(vd, vq, theta_e)
            abc_method(v_a, v_b, v_c)
            if duration_s is not None:
                self._step_or_sleep(duration_s)
            return

        used_duration = _call_with_duration(method, vd, vq, duration_s)
        if duration_s is not None and not used_duration:
            self._step_or_sleep(duration_s)

    def _step_or_sleep(self, duration_s: float) -> None:
        step = self._resolve_method(("step", "advance", "tick"))
        if step is None:
            raise NotImplementedError(
                "Driver does not support step/advance and set_voltage_dq has no duration support."
            )
        try:
            sig = inspect.signature(step)
            if "duration_s" in sig.parameters:
                step(duration_s=duration_s)
                return
            if "dt" in sig.parameters:
                step(dt=duration_s)
                return
            step()
        except TypeError:
            step()

    def read_currents_dq(self) -> Tuple[float, float]:
        method = self._resolve_method(
            ("read_currents_dq", "get_currents_dq", "read_current_dq", "get_current_dq", "read_id_iq", "get_id_iq")
        )
        if method is not None:
            values = method()
            return _validate_tuple(values, 2, "read_currents_dq")
        i_abc = self.read_phase_currents_abc()
        theta_e = self._read_theta_e()
        i_d, i_q = abc_to_dq(i_abc[0], i_abc[1], i_abc[2], theta_e)
        return float(i_d), float(i_q)

    def read_phase_currents_abc(self) -> Tuple[float, float, float]:
        method = self._resolve_method(
            ("read_phase_currents_abc", "read_currents_abc", "get_phase_currents_abc", "get_currents_abc")
        )
        if method is not None:
            values = method()
            return _validate_tuple(values, 3, "read_phase_currents_abc")
        attr = self._resolve_attr(("i_abc", "currents_abc", "phase_currents"))
        if attr is not None:
            return _validate_tuple(attr, 3, "phase_currents")
        raise NotImplementedError("Driver does not provide phase currents (abc).")

    def read_mech_speed(self) -> float:
        method = self._resolve_method(
            ("read_mech_speed", "get_mech_speed", "read_speed", "read_speed_rad_s", "read_omega")
        )
        if method is not None:
            return _as_float(method(), "omega_m")
        method = self._resolve_method(("read_speed_rpm", "get_speed_rpm"))
        if method is not None:
            return _as_float(method(), "speed_rpm") * _TWO_PI_OVER_60
        attr = self._resolve_attr(("omega_m", "w_mech", "omega"))
        if attr is not None:
            return _as_float(attr, "omega_m")
        raise NotImplementedError("Driver does not provide mechanical speed.")

    def read_torque(self) -> float | None:
        method = self._resolve_method(("read_torque", "get_torque", "read_torque_nm"))
        if method is not None:
            return _as_float(method(), "torque")
        attr = self._resolve_attr(("torque", "torque_e", "last_torque"))
        if attr is not None:
            return _as_float(attr, "torque")
        return None

    def set_iq_ref(self, iq: float) -> None:
        method = self._resolve_method(("set_iq_ref", "set_current_iq"))
        if method is None:
            raise NotImplementedError("Driver does not support iq reference commands.")
        method(_as_float(iq, "iq"))

    def set_torque_command(self, torque: float) -> None:
        method = self._resolve_method(("set_torque_command", "set_torque_ref"))
        if method is not None:
            method(_as_float(torque, "torque"))
            return
        raise NotImplementedError("Driver does not support torque commands.")

    def _read_theta_e(self) -> float:
        method = self._resolve_method(
            ("read_theta_e", "read_electrical_angle", "get_theta_e", "get_electrical_angle", "read_theta")
        )
        if method is not None:
            return _as_float(method(), "theta_e")
        attr = self._resolve_attr(("theta_e", "electrical_angle", "theta"))
        if attr is not None:
            return _as_float(attr, "theta_e")
        raise NotImplementedError("Driver does not provide electrical angle for dq/abc conversion.")

    def _can_read_theta_e(self) -> bool:
        return self._resolve_method(
            ("read_theta_e", "read_electrical_angle", "get_theta_e", "get_electrical_angle", "read_theta")
        ) is not None or self._resolve_attr(("theta_e", "electrical_angle", "theta")) is not None


def _validate_tuple(value: object, length: int, name: str) -> Tuple[float, ...]:
    if isinstance(value, tuple) or isinstance(value, list):
        if len(value) != length:
            raise ValueError(f"{name} must return {length} values, got {len(value)}")
        values = tuple(_as_float(v, name) for v in value)
        if not all(_is_finite(v) for v in values):
            raise ValueError(f"{name} returned non-finite values: {values}")
        return values
    raise ValueError(f"{name} must return a tuple/list of length {length}")


def check_driver_conformance(adapter: DriverAdapter) -> None:
    """
    Validate driver contract availability and signal shapes.
    """
    dt = adapter.dt
    if dt <= 0.0 or not _is_finite(dt):
        raise ValueError(f"Driver dt must be positive, got {dt}")

    if not adapter.supports_voltage_command:
        raise NotImplementedError("Driver missing dq (or abc+theta) voltage command support.")

    i_d, i_q = adapter.read_currents_dq()
    if not _is_finite(i_d) or not _is_finite(i_q):
        raise ValueError(f"read_currents_dq returned invalid values: {(i_d, i_q)}")

    omega = adapter.read_mech_speed()
    if not _is_finite(omega):
        raise ValueError(f"read_mech_speed returned invalid value: {omega}")

    if adapter.capabilities.has_lock_rotor is False:
        raise NotImplementedError("Driver missing lock_rotor capability.")
