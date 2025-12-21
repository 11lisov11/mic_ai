from __future__ import annotations

from typing import Tuple

from mic_ai.drivers.driver_adapter import DriverAdapter


class IdentSignalInterface:
    """Unified IO contract for identification through a driver adapter."""

    def __init__(self, driver: object | DriverAdapter) -> None:
        self._driver = driver if isinstance(driver, DriverAdapter) else DriverAdapter(driver)
        self._last_torque_mode: str | None = None

    @property
    def dt(self) -> float:
        return self._driver.dt

    @property
    def torque_command_mode(self) -> str | None:
        return self._last_torque_mode

    def reset(self) -> None:
        self._driver.reset()

    def lock_rotor(self, enabled: bool = True) -> None:
        self._driver.lock_rotor(bool(enabled))

    def set_voltage_dq(self, u_d: float, u_q: float) -> None:
        self._driver.set_voltage_dq(u_d, u_q)

    def apply_voltage_step(self, u_d: float, u_q: float) -> None:
        self._driver.set_voltage_dq(u_d, u_q, duration_s=self.dt)

    def apply_torque_step(self, torque_cmd: float) -> None:
        try:
            self._driver.set_torque_command(float(torque_cmd))
            self._last_torque_mode = "torque_command"
            return
        except NotImplementedError:
            pass
        try:
            self._driver.set_iq_ref(float(torque_cmd))
            self._last_torque_mode = "iq_ref"
            return
        except NotImplementedError:
            pass
        raise NotImplementedError("Driver does not support torque or iq commands.")

    def read_currents_dq(self) -> Tuple[float, float]:
        return self._driver.read_currents_dq()

    def read_mech_speed(self) -> float:
        return self._driver.read_mech_speed()

    def read_torque(self) -> float | None:
        return self._driver.read_torque()


__all__ = ["IdentSignalInterface"]
