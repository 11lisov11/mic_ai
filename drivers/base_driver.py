from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional


class BaseDriver(ABC):
    """Abstract driver contract for sim/hw backends."""

    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> None:
        """Reset driver state; optionally set a deterministic seed."""

    @abstractmethod
    def set_mode(self, mode: str) -> None:
        """Set control mode: "FOC" or "MIC"."""

    @abstractmethod
    def set_limits(self, limits: Dict[str, float]) -> None:
        """Set safety limits (i_max, v_max, dv_dt, omega_max, t_max)."""

    @abstractmethod
    def apply_action(self, vd: float, vq: float) -> None:
        """Provide the next dq voltage action."""

    @abstractmethod
    def step(self) -> None:
        """Advance one control tick."""

    @abstractmethod
    def read_obs(self) -> Dict[str, float]:
        """Read current observations."""

    @abstractmethod
    def get_last_fault(self) -> Optional[str]:
        """Return last fault reason (if any)."""

    @abstractmethod
    def close(self) -> None:
        """Release resources (if any)."""
