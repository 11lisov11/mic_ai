from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class SensorChannelConfig:
    """Noise/quantization/delay settings for a single sensor channel."""

    sigma: float = 0.0
    quant: float = 0.0
    delay_steps: int = 0


@dataclass(frozen=True)
class SensorConfig:
    """Full sensor configuration (encoder, currents, DC link)."""

    omega: SensorChannelConfig = field(default_factory=SensorChannelConfig)
    currents: SensorChannelConfig = field(default_factory=SensorChannelConfig)
    vdc: SensorChannelConfig = field(default_factory=SensorChannelConfig)
    seed: int | None = None


@dataclass(frozen=True)
class SensorReading:
    """Measured signals returned by SensorModel."""

    omega: float
    i_abc: tuple[float, float, float]
    vdc: float


class _DelayLine:
    def __init__(self, steps: int, initial: object) -> None:
        if steps < 0:
            raise ValueError("delay_steps must be non-negative")
        self._steps = steps
        self._queue: Deque[object] = deque([initial] * (steps + 1), maxlen=steps + 1)

    def push(self, value: object) -> object:
        self._queue.append(value)
        return self._queue[0]


class SensorModel:
    """Applies noise, quantization, and delay to true signals."""

    def __init__(self, config: SensorConfig) -> None:
        self._config = config
        self._rng = np.random.default_rng(config.seed)
        self._omega_delay = _DelayLine(config.omega.delay_steps, 0.0)
        self._i_abc_delay = _DelayLine(config.currents.delay_steps, (0.0, 0.0, 0.0))
        self._vdc_delay = _DelayLine(config.vdc.delay_steps, 0.0)

    def reset(self) -> None:
        self._omega_delay = _DelayLine(self._config.omega.delay_steps, 0.0)
        self._i_abc_delay = _DelayLine(self._config.currents.delay_steps, (0.0, 0.0, 0.0))
        self._vdc_delay = _DelayLine(self._config.vdc.delay_steps, 0.0)

    def measure(self, omega: float, i_abc: Sequence[float], vdc: float) -> SensorReading:
        omega_meas = self._apply_channel(float(omega), self._config.omega)
        i_abc_meas = tuple(
            self._apply_channel(float(val), self._config.currents) for val in i_abc
        )
        vdc_meas = self._apply_channel(float(vdc), self._config.vdc)

        omega_out = float(self._omega_delay.push(omega_meas))
        i_abc_out = self._i_abc_delay.push(i_abc_meas)
        vdc_out = float(self._vdc_delay.push(vdc_meas))

        if not isinstance(i_abc_out, Iterable):
            raise ValueError("invalid delayed current signal type")
        i_abc_tuple = tuple(float(val) for val in i_abc_out)
        return SensorReading(omega=omega_out, i_abc=i_abc_tuple, vdc=vdc_out)

    def _apply_channel(self, value: float, cfg: SensorChannelConfig) -> float:
        if cfg.sigma > 0.0:
            value += float(self._rng.normal(0.0, cfg.sigma))
        if cfg.quant > 0.0:
            value = round(value / cfg.quant) * cfg.quant
        return value


__all__ = ["SensorChannelConfig", "SensorConfig", "SensorReading", "SensorModel"]
