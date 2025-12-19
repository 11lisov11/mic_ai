from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ServoMode = Literal["torque", "torque_mode", "speed", "speed_mode"]


def _normalize_mode(mode: str) -> str:
    mode_lower = mode.strip().lower()
    if mode_lower in ("torque", "torque_mode"):
        return "torque"
    if mode_lower in ("speed", "speed_mode"):
        return "speed"
    raise ValueError(f"Unsupported servo mode: {mode!r}")


def _clip(value: float, limit: float) -> float:
    if limit <= 0.0:
        raise ValueError("limit must be positive")
    return max(-limit, min(limit, value))


@dataclass(frozen=True)
class ServoLoadConfig:
    """Configuration for the servo load (dynamometer) model."""

    mode: ServoMode
    tau_load: float
    t_max: float
    speed_kp: float = 0.0
    speed_ki: float = 0.0
    speed_int_limit: float = 0.0


class ServoLoadModel:
    """First-order servo load model with torque and optional speed modes.

    torque mode:
        command = T_cmd, output T_load with first-order dynamics and saturation
    speed mode:
        command = omega_ref, simple PI controller generates torque command
    """

    def __init__(self, config: ServoLoadConfig) -> None:
        self._config = config
        self._mode = _normalize_mode(config.mode)
        self._t_load = 0.0
        self._speed_int = 0.0
        self._validate_config()

    def reset(self) -> None:
        """Reset internal state."""
        self._t_load = 0.0
        self._speed_int = 0.0

    def step(self, dt: float, omega: float, command: float) -> float:
        """Advance the model by one step.

        Args:
            dt: Simulation step [s].
            omega: Current mechanical speed [rad/s].
            command: T_cmd (torque mode) or omega_ref (speed mode).
        """
        if dt <= 0.0:
            raise ValueError("dt must be positive")

        t_cmd = float(command)
        if self._mode == "speed":
            err = float(command) - float(omega)
            self._speed_int += err * dt
            if self._config.speed_int_limit > 0.0:
                self._speed_int = _clip(self._speed_int, self._config.speed_int_limit)
            t_cmd = self._config.speed_kp * err + self._config.speed_ki * self._speed_int

        t_cmd = _clip(t_cmd, self._config.t_max)
        if self._config.tau_load <= 0.0:
            self._t_load = t_cmd
        else:
            alpha = dt / self._config.tau_load
            self._t_load += alpha * (t_cmd - self._t_load)
        self._t_load = _clip(self._t_load, self._config.t_max)
        return self._t_load

    @property
    def t_load(self) -> float:
        """Current load torque [N*m]."""
        return self._t_load

    def _validate_config(self) -> None:
        if self._config.t_max <= 0.0:
            raise ValueError("t_max must be positive")
        if self._config.tau_load < 0.0:
            raise ValueError("tau_load must be non-negative")
        if self._mode == "speed" and self._config.speed_kp == 0.0 and self._config.speed_ki == 0.0:
            raise ValueError("speed mode requires speed_kp and/or speed_ki")


if __name__ == "__main__":
    config = ServoLoadConfig(mode="torque", tau_load=0.1, t_max=2.0)
    model = ServoLoadModel(config)
    dt = 0.01
    for k in range(10):
        t_load = model.step(dt, omega=0.0, command=10.0)
        print(f"step={k:02d} T_load={t_load:.3f} N*m")
