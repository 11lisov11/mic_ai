from __future__ import annotations
from typing import Protocol, Tuple

class BaseController(Protocol):
    """
    Protocol for motor controllers.
    """
    
    def reset(self) -> None:
        """Reset internal state."""
        ...

    def step(
        self,
        t: float,
        omega_ref: float,
        omega_m: float,
        i_abc: Tuple[float, float, float],
        torque_e: float,
        theta_mech: float
    ) -> Tuple[float, float, float, float]:
        """
        Compute dq voltage commands.

        Args:
            t: current simulation time.
            omega_ref: mechanical speed reference (rad/s).
            omega_m: measured mechanical speed (rad/s).
            i_abc: measured phase currents (A).
            torque_e: measured/estimated torque (Nm).
            theta_mech: mechanical angle (rad).

        Returns:
            v_d: d-axis voltage command (V).
            v_q: q-axis voltage command (V).
            theta_e: electrical angle for coordinate transformation (rad).
            omega_syn: synchronous electrical speed (rad/s).
        """
        ...
